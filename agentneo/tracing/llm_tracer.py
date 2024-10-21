import asyncio
import psutil
import json
import wrapt
import functools
from datetime import datetime
import os

from .user_interaction_tracer import UserInteractionTracer
from .utils import calculate_cost, load_model_costs, convert_usage_to_dict
from .llm_utils import extract_llm_output
from ..data import LLMCallModel


class LLMTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = []
        self.model_costs = load_model_costs()

    def instrument_llm_calls(self):
        # Use wrapt to register post-import hooks
        wrapt.register_post_import_hook(self.patch_openai_methods, "openai")
        wrapt.register_post_import_hook(self.patch_litellm_methods, "litellm")

    def patch_openai_methods(self, module):
        if hasattr(module, "OpenAI"):
            client_class = getattr(module, "OpenAI")
            self.wrap_openai_client_methods(client_class)
        if hasattr(module, "AsyncOpenAI"):
            async_client_class = getattr(module, "AsyncOpenAI")
            self.wrap_openai_client_methods(async_client_class)

    def wrap_openai_client_methods(self, client_class):
        original_init = client_class.__init__

        @functools.wraps(original_init)
        def patched_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            self.wrap_method(client_self.chat.completions, "create")
            if hasattr(client_self.chat.completions, "acreate"):
                self.wrap_method(client_self.chat.completions, "acreate")

        setattr(client_class, "__init__", patched_init)

    def patch_litellm_methods(self, module):
        # Wrap methods in litellm
        self.wrap_method(module, "completion")
        self.wrap_method(module, "acompletion")

    def wrap_method(self, obj, method_name):
        original_method = getattr(obj, method_name)

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            return self.trace_llm_call(wrapped, *args, **kwargs)

        wrapped_method = wrapper(original_method)
        setattr(obj, method_name, wrapped_method)
        self.patches.append((obj, method_name, original_method))

    def trace_llm_call(self, original_func, *args, **kwargs):
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss

        agent_id = self.current_agent_id.get()
        llm_call_name = self.current_llm_call_name.get() or original_func.__name__

        try:
            result = original_func(*args, **kwargs)

            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            sanitized_args = self._sanitize_api_keys(args)
            sanitized_kwargs = self._sanitize_api_keys(kwargs)

            model_name = self._extract_model_name(sanitized_kwargs)
            try:
                model = (
                    os.path.join("groq", model_name) if result.x_groq else model_name
                )
            except:
                model = model_name

            llm_call = self.process_llm_result(
                result,
                llm_call_name,
                model,
                self._extract_input(sanitized_args, sanitized_kwargs),
                start_time,
                end_time,
                memory_used,
                agent_id,
            )

            # Append llm_call_id to current_llm_call_ids
            llm_call_ids = self.current_llm_call_ids.get()
            if llm_call_ids is None:
                llm_call_ids = []
                self.current_llm_call_ids.set(llm_call_ids)
            llm_call_ids.append(llm_call.id)

            return result
        except Exception as e:
            self._log_error(e, "llm", llm_call_name)
            raise

    def process_llm_result(
        self, result, name, model, prompt, start_time, end_time, memory_used, agent_id
    ):
        llm_data = extract_llm_output(result)
        agent_id = self.current_agent_id.get()

        token_usage = {"input": 0, "completion": 0, "reasoning": 0}

        if hasattr(result, "usage"):
            usage = result.usage
            token_usage["input"] = getattr(usage, "prompt_tokens", 0)
            token_usage["completion"] = getattr(usage, "completion_tokens", 0)
            token_usage["reasoning"] = getattr(usage, "reasoning_tokens", 0)

        # Default cost values if the model or default is not found
        default_cost = {"input": 0.0, "output": 0.0, "reasoning": 0.0}

        # Try to get the model-specific cost, fall back to default, then to default_cost
        model_cost = self.model_costs.get(
            model, self.model_costs.get("default", default_cost)
        )

        cost = {
            "input": token_usage["input"] * model_cost["input_cost_per_token"],
            "output": token_usage["completion"] * model_cost["output_cost_per_token"],
            "reasoning": token_usage["reasoning"]
            * model_cost.get("reasoning_cost_per_token", 0),
        }

        llm_call = LLMCallModel(
            project_id=self.project_id,
            trace_id=self.trace_id,
            agent_id=agent_id,
            name=name,
            model=llm_data.model_name,
            input_prompt=str(prompt),
            output=str(llm_data.output_response),
            tool_call=(
                str(llm_data.tool_call) if llm_data.tool_call else llm_data.tool_call
            ),
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            token_usage=json.dumps(token_usage),
            cost=json.dumps(cost),
            memory_used=memory_used,
        )

        with self.Session() as session:
            session.add(llm_call)
            session.commit()
            session.refresh(llm_call)

            # Create a dictionary with all the necessary information
            llm_call_data = {
                "id": llm_call.id,
                "name": name,
                "model": llm_data.model_name,
                "input_prompt": prompt,
                "output": llm_data.output_response,
                "tool_call": llm_data.tool_call,
                "start_time": start_time,
                "end_time": end_time,
                "duration": (end_time - start_time).total_seconds(),
                "token_usage": token_usage,
                "cost": cost,
                "memory_used": memory_used,
                "agent_id": agent_id,
            }

        if agent_id:
            llm_call_ids = self.current_llm_call_ids.get()
            if llm_call_ids is None:
                llm_call_ids = []
                self.current_llm_call_ids.set(llm_call_ids)
            llm_call_ids.append(llm_call.id)

        # Append the data to trace_data outside the session
        self.trace_data.setdefault("llm_calls", []).append(llm_call_data)

        return llm_call

    def _extract_model_name(self, kwargs):
        return kwargs.get("model", "unknown")

    def _extract_input(self, args, kwargs):
        if "prompt" in kwargs:
            return kwargs["prompt"]
        elif "messages" in kwargs:
            return kwargs["messages"]
        else:
            return args[0] if args else ""

    def _sanitize_api_keys(self, data):
        # Implement sanitization logic to remove API keys from data
        return data

    def unpatch_llm_calls(self):
        for obj, method_name, original_method in self.patches:
            setattr(obj, method_name, original_method)
        self.patches.clear()

    def trace_llm(self, name: str):
        def decorator(func_or_class):
            if isinstance(func_or_class, type):
                for attr_name, attr_value in func_or_class.__dict__.items():
                    if callable(attr_value) and not attr_name.startswith("__"):
                        setattr(
                            func_or_class,
                            attr_name,
                            self.trace_llm(f"{name}.{attr_name}")(attr_value),
                        )
                return func_or_class
            else:

                @functools.wraps(func_or_class)
                async def async_wrapper(*args, **kwargs):
                    token = self.current_llm_call_name.set(name)
                    try:
                        return await func_or_class(*args, **kwargs)
                    finally:
                        self.current_llm_call_name.reset(token)

                @functools.wraps(func_or_class)
                def sync_wrapper(*args, **kwargs):
                    token = self.current_llm_call_name.set(name)
                    try:
                        return func_or_class(*args, **kwargs)
                    finally:
                        self.current_llm_call_name.reset(token)

                return (
                    async_wrapper
                    if asyncio.iscoroutinefunction(func_or_class)
                    else sync_wrapper
                )

        return decorator
