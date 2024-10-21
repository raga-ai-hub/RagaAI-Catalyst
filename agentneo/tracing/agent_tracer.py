import asyncio
import psutil
import json
import functools
from datetime import datetime
from .user_interaction_tracer import UserInteractionTracer
from ..data import AgentCallModel, ToolCallModel, LLMCallModel, UserInteractionModel
import pdb

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AgentTracerMixin:
    def trace_agent(self, name: str):
        def decorator(func_or_class):
            if isinstance(func_or_class, type):
                # If it's a class, wrap all its methods as tools
                for attr_name, attr_value in func_or_class.__dict__.items():
                    if callable(attr_value) and not attr_name.startswith("__"):
                        setattr(
                            func_or_class,
                            attr_name,
                            self.trace_tool(f"{name}.{attr_name}")(attr_value),
                        )
                # Wrap the class initialization
                original_init = func_or_class.__init__

                @functools.wraps(original_init)
                def wrapped_init(self_instance, *args, **kwargs):
                    self_instance._agent_name = name
                    self_instance._agent_start_time = datetime.now()
                    self_instance._agent_start_memory = (
                        psutil.Process().memory_info().rss
                    )
                    self_instance._agent_tracer = (
                        self  # Use the current tracer instance
                    )

                    self_instance._agent_id = self._start_agent_call(name, args, kwargs)
                    self.current_agent_id.set(self_instance._agent_id)
                    original_init(self_instance, *args, **kwargs)

                func_or_class.__init__ = wrapped_init

                # Wrap the class destruction
                original_del = (
                    func_or_class.__del__ if hasattr(func_or_class, "__del__") else None
                )

                def wrapped_del(self_instance):
                    if hasattr(self_instance, "_agent_name"):
                        try:
                            self._end_agent_call(self_instance._agent_id)
                        except Exception as e:
                            logging.error(f"Error finalizing agent call: {e}")
                    if original_del:
                        original_del(self_instance)

                func_or_class.__del__ = wrapped_del

                return func_or_class
            else:

                @functools.wraps(func_or_class)
                async def async_wrapper(*args, **kwargs):
                    return await self._trace_agent_call_async(
                        func_or_class, name, *args, **kwargs
                    )

                @functools.wraps(func_or_class)
                def sync_wrapper(*args, **kwargs):
                    return self._trace_agent_call_sync(
                        func_or_class, name, *args, **kwargs
                    )

                return (
                    async_wrapper
                    if asyncio.iscoroutinefunction(func_or_class)
                    else sync_wrapper
                )

        return decorator

    def _start_agent_call(self, name, args, kwargs):
        start_time = datetime.now()
        agent_call = AgentCallModel(
            project_id=self.project_id,
            trace_id=self.trace_id,
            name=name,
            start_time=start_time,
            llm_call_ids=json.dumps([]),
            tool_call_ids=json.dumps([]),
            user_interaction_ids=json.dumps([]),
        )
        with self.Session() as session:
            session.add(agent_call)
            session.commit()
            return agent_call.id

    def _end_agent_call(self, agent_id):
        with self.Session() as session:
            agent_call = session.query(AgentCallModel).get(agent_id)
            if agent_call:
                agent_call.end_time = datetime.now()
                agent_call.llm_call_ids = json.dumps(
                    self.current_llm_call_ids.get() or []
                )
                agent_call.tool_call_ids = json.dumps(
                    self.current_tool_call_ids.get() or []
                )
                agent_call.user_interaction_ids = json.dumps(
                    self.current_user_interaction_ids.get() or []
                )

                try:
                    session.add(agent_call)
                    session.commit()
                    logger.debug(
                        f"Successfully updated and committed AgentCallModel with id {agent_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error committing AgentCallModel with id {agent_id}: {str(e)}"
                    )
                    session.rollback()

                self.trace_data.setdefault("agent_calls", []).append(
                    {
                        "id": agent_call.id,
                        "name": agent_call.name,
                        "start_time": agent_call.start_time,
                        "end_time": agent_call.end_time,
                        "llm_call_ids": agent_call.llm_call_ids,
                        "tool_call_ids": agent_call.tool_call_ids,
                        "user_interaction_ids": agent_call.user_interaction_ids,
                    }
                )
            else:
                print(f"Warning: AgentCallModel with id {agent_id} not found")

    def _trace_agent_call_sync(self, func, name, *args, **kwargs):
        agent_id = self._start_agent_call(name, args, kwargs)

        token_agent = self.current_agent_id.set(agent_id)
        token_llm = self.current_llm_call_ids.set([])
        token_tool = self.current_tool_call_ids.set([])
        token_user = self.current_user_interaction_ids.set([])

        user_interaction_tracer = UserInteractionTracer(self)

        try:
            with user_interaction_tracer.capture():
                result = func(*args, **kwargs)

            self._end_agent_call(agent_id)

        except Exception as e:
            self._log_error(e, "agent", name, agent_id)
            raise
        finally:
            self.current_agent_id.reset(token_agent)
            self.current_llm_call_ids.reset(token_llm)
            self.current_tool_call_ids.reset(token_tool)
            self.current_user_interaction_ids.reset(token_user)

        return result

    async def _trace_agent_call_async(self, func, name, *args, **kwargs):
        agent_id = self._start_agent_call(name, args, kwargs)

        token_agent = self.current_agent_id.set(agent_id)
        token_llm = self.current_llm_call_ids.set([])
        token_tool = self.current_tool_call_ids.set([])
        token_user = self.current_user_interaction_ids.set([])

        user_interaction_tracer = UserInteractionTracer(self)

        try:
            async with user_interaction_tracer.async_capture():
                result = await func(*args, **kwargs)

            self._end_agent_call(agent_id)

        except Exception as e:
            self._log_error(e, "agent", name, agent_id)
            raise
        finally:
            self.current_agent_id.reset(token_agent)
            self.current_llm_call_ids.reset(token_llm)
            self.current_tool_call_ids.reset(token_tool)
            self.current_user_interaction_ids.reset(token_user)

        return result

    def _serialize_params(self, args, kwargs):
        def _serialize(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            else:
                return str(obj)

        # Handle 'self' argument for class methods
        if args and isinstance(args[0], object) and hasattr(args[0], "__dict__"):
            args = ("self (instance)",) + args[1:]

        serialized_args = {f"arg_{i}": _serialize(arg) for i, arg in enumerate(args)}
        serialized_kwargs = {k: _serialize(v) for k, v in kwargs.items()}
        return {**serialized_args, **serialized_kwargs}

    def _args_to_dict(self, args):
        return {f"arg_{i}": arg for i, arg in enumerate(args)}
