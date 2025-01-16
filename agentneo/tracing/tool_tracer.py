import asyncio
import psutil
import json
import functools
from datetime import datetime
from .network_tracer import NetworkTracer, patch_aiohttp_trace_config
from .user_interaction_tracer import UserInteractionTracer
from ..data import ToolCallModel
from functools import wraps


class ToolTracerMixin:
    def trace_tool(self, name: str, description: str = ""):
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._trace_tool_call_async(
                    func, name, description, *args, **kwargs
                )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._trace_tool_call_sync(
                    func, name, description, *args, **kwargs
                )

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def _trace_tool_call_sync(self, func, name, description, *args, **kwargs):
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        agent_id = self.current_agent_id.get()

        # Activate network tracing
        self.network_tracer.activate_patches()

        try:
            result = func(*args, **kwargs)

            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = end_memory - start_memory

            serialized_params = self._serialize_params(args, kwargs)
            tool_call = ToolCallModel(
                project_id=self.project_id,
                trace_id=self.trace_id,
                agent_id=agent_id,
                name=name,
                input_parameters=json.dumps(serialized_params),
                output=str(result),
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                memory_used=memory_used,
                network_calls=self.network_tracer.network_calls,
            )
            with self.Session() as session:
                session.add(tool_call)
                session.commit()
                tool_call_id = tool_call.id

            # Append tool_call_id to current_tool_call_ids
            tool_call_ids = self.current_tool_call_ids.get()
            if tool_call_ids is None:
                tool_call_ids = []
                self.current_tool_call_ids.set(tool_call_ids)
            tool_call_ids.append(tool_call_id)

            self.trace_data.setdefault("tool_calls", []).append(
                {
                    "id": tool_call_id,
                    "name": name,
                    "description": description,
                    "input_parameters": json.dumps(serialized_params),
                    "output": str(result),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "memory_used": memory_used,
                    "network_calls": self.network_tracer.network_calls,
                    "agent_id": agent_id,
                }
            )

            return result
        except Exception as e:
            self._log_error(e, "tool", name)
            raise
        finally:
            self.network_tracer.deactivate_patches()

    async def _trace_tool_call_async(self, func, name, description, *args, **kwargs):
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        agent_id = self.current_agent_id.get()

        # Initialize the UserInteractionTracer
        user_interaction_tracer = UserInteractionTracer(self)

        # Activate network tracing
        self.network_tracer.activate_patches()

        try:
            async with user_interaction_tracer.async_capture():
                if asyncio.iscoroutinefunction(func):
                    trace_config = await patch_aiohttp_trace_config(self.network_tracer)
                    result = await func(*args, **kwargs, trace_config=trace_config)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)

            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = end_memory - start_memory

            serialized_params = self._serialize_params(args, kwargs)
            tool_call = ToolCallModel(
                project_id=self.project_id,
                trace_id=self.trace_id,
                agent_id=agent_id,
                name=name,
                input_parameters=json.dumps(serialized_params),
                output=str(result),
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                memory_used=memory_used,
                network_calls=self.network_tracer.network_calls,
            )
            with self.Session() as session:
                session.add(tool_call)
                session.commit()
                tool_call_id = tool_call.id

            # Append tool_call_id to current_tool_call_ids if available
            tool_call_ids = self.current_tool_call_ids.get()
            if tool_call_ids is not None:
                tool_call_ids.append(tool_call_id)

            self.trace_data.setdefault("tool_calls", []).append(
                {
                    "id": tool_call_id,
                    "name": name,
                    "description": description,
                    "input_parameters": json.dumps(serialized_params),
                    "output": str(result),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "memory_used": memory_used,
                    "network_calls": self.network_tracer.network_calls,
                    "agent_id": agent_id,
                }
            )

            return result
        except Exception as e:
            self._log_error(e, "tool", name)
            raise
        finally:
            self.network_tracer.deactivate_patches()

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
        
    def _serialize_result(self, result):
        """
        Helper method to serialize function result for tracing.
        """
        try:
            return self._safe_serialize(result)
        except Exception:
            return str(result)

    def _safe_serialize(self, obj):
        """
        Safely serialize an object to JSON-serializable format.
        """
        if isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    # Decorators for tracing Pinecone operations
    def trace_pinecone_upsert(self, func):
        """
        Decorator to trace Pinecone upsert operations, capturing database interactions,
        query results, and performance metrics.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_memory = psutil.Process().memory_info().rss
            agent_id = self.current_agent_id.get()

            self.network_tracer.activate_patches()

            try:
                result = func(*args, **kwargs)

                end_time = datetime.now()
                end_memory = psutil.Process().memory_info().rss

                memory_used = end_memory - start_memory

                serialized_params = self._serialize_params(args, kwargs)
                serialized_output = self._serialize_result(result)

                tool_call = ToolCallModel(
                    project_id=self.project_id,
                    trace_id=self.trace_id,
                    agent_id=agent_id,
                    name="Pinecone Upsert",
                    input_parameters=json.dumps(serialized_params),
                    output=json.dumps(serialized_output),
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    memory_used=memory_used,
                    network_calls=self.network_tracer.network_calls,
                )

                with self.Session() as session:
                    session.add(tool_call)
                    session.commit()

                return result
            except Exception as e:
                self._log_error(e, "Pinecone Upsert", func.__name__)
                raise
            finally:
                self.network_tracer.deactivate_patches()

        return wrapper

    def trace_pinecone_create_query_vector(self, func):
        """
        Decorator to trace the creation of query vectors for Pinecone.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_memory = psutil.Process().memory_info().rss
            agent_id = self.current_agent_id.get()

            self.network_tracer.activate_patches()

            try:
                result = func(*args, **kwargs)

                end_time = datetime.now()
                end_memory = psutil.Process().memory_info().rss

                memory_used = end_memory - start_memory

                serialized_params = self._serialize_params(args, kwargs)
                serialized_output = self._serialize_result(result)

                tool_call = ToolCallModel(
                    project_id=self.project_id,
                    trace_id=self.trace_id,
                    agent_id=agent_id,
                    name="Pinecone Create Query Vector",
                    input_parameters=json.dumps(serialized_params),
                    output=json.dumps(serialized_output),
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    memory_used=memory_used,
                    network_calls=self.network_tracer.network_calls,
                )

                with self.Session() as session:
                    session.add(tool_call)
                    session.commit()

                return result
            except Exception as e:
                self._log_error(e, "Pinecone Create Query Vector", func.__name__)
                raise
            finally:
                self.network_tracer.deactivate_patches()

        return wrapper

    def trace_pinecone_similarity_search(self, func):
        """
        Decorator to trace Pinecone similarity search operations, capturing database interactions,
        query results, and performance metrics.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_memory = psutil.Process().memory_info().rss
            agent_id = self.current_agent_id.get()

            self.network_tracer.activate_patches()

            try:
                result = func(*args, **kwargs)

                end_time = datetime.now()
                end_memory = psutil.Process().memory_info().rss

                memory_used = end_memory - start_memory

                serialized_params = self._serialize_params(args, kwargs)
                serialized_output = self._serialize_result(result)

                tool_call = ToolCallModel(
                    project_id=self.project_id,
                    trace_id=self.trace_id,
                    agent_id=agent_id,
                    name="Pinecone Similarity Search",
                    input_parameters=json.dumps(serialized_params),
                    output=json.dumps(serialized_output),
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    memory_used=memory_used,
                    network_calls=self.network_tracer.network_calls,
                )

                with self.Session() as session:
                    session.add(tool_call)
                    session.commit()

                return result
            except Exception as e:
                self._log_error(e, "Pinecone Similarity Search", func.__name__)
                raise
            finally:
                self.network_tracer.deactivate_patches()

        return wrapper

    def _log_error(self, exception, component_name, operation_name):
        """
        Helper method to log errors during tracing.
        """
        # Implement your logging logic here
        print(f"Error in {component_name} '{operation_name}': {exception}")
        
        
        
    def wrap_langchain_tool(self, tool_input, tool_name=None, **tool_kwargs):
        try:
            from langchain.tools import Tool as LangChainTool
            from langchain.tools.base import BaseTool
        except ImportError as exc:
            raise ImportError(
                "LangChain is not installed. Please install it using: "
                "pip install agentneo[langchain]"
            ) from exc

        # Check if tool_input is already an instance
        if isinstance(tool_input, BaseTool):
            tool_instance = tool_input
        else:
            # Assume it's a class and try to instantiate it
            tool_instance = tool_input(**tool_kwargs)

        # Get the name, prioritizing the provided tool_name
        if tool_name:
            name = tool_name
        elif hasattr(tool_instance, "name"):
            name = tool_instance.name
        elif hasattr(tool_input, "__name__"):
            name = tool_input.__name__
        else:
            name = type(tool_instance).__name__

        @self.trace_tool(name)
        @wraps(getattr(tool_instance, "invoke", tool_instance))
        def wrapped_tool(tool_input, **kwargs):
            if hasattr(tool_instance, "invoke"):
                return tool_instance.invoke(tool_input, **kwargs)
            else:
                return tool_instance(tool_input, **kwargs)

        return LangChainTool(
            name=name,
            func=wrapped_tool,
            description=getattr(tool_instance, "description", ""),
        )