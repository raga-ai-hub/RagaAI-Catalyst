from typing import Optional, Any, Dict, List
import asyncio
import psutil
import wrapt
import functools
import json
import os
import time
from datetime import datetime
import uuid
import contextvars
import traceback
import importlib
import sys
import logging

try:
    from llama_index.core.base.llms.types import ChatResponse,TextBlock, ChatMessage
except ImportError:
    logging.warning("Failed to import ChatResponse, TextBlock, ChatMessage. Some features from llamaindex may not work. Please upgrade to the latest version of llama_index or version (>=0.12)")
from .base import BaseTracer
from ..utils.llm_utils import (
    extract_model_name,
    extract_parameters,
    extract_token_usage,
    extract_input_data,
    calculate_llm_cost,
    sanitize_api_keys,
    sanitize_input,
    extract_llm_output,
    num_tokens_from_messages,
    get_model_cost
)
from ..utils.unique_decorator import generate_unique_hash
from ..utils.file_name_tracker import TrackName
from ..utils.span_attributes import SpanAttributes

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


class LLMTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.patches = []
        try:
            self.model_costs = get_model_cost()
        except Exception as e:
            self.model_costs = {
                "default": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
            }
        self.MAX_PARAMETERS_TO_DISPLAY = 10
        self.current_llm_call_name = contextvars.ContextVar(
            "llm_call_name", default=None
        )
        self.component_network_calls = {}
        self.component_user_interaction = {}
        self.current_component_id = None
        self.total_tokens = 0
        self.total_cost = 0.0
        self.llm_data = {}

        self.auto_instrument_llm = False
        self.auto_instrument_user_interaction = False
        self.auto_instrument_file_io = False
        self.auto_instrument_network = False

    def check_package_available(self, package_name):
        """Check if a package is available in the environment"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def validate_openai_key(self):
        """Validate if OpenAI API key is available"""
        return bool(os.getenv("OPENAI_API_KEY"))

    def instrument_llm_calls(self):
        """Enable LLM instrumentation"""
        self.auto_instrument_llm = True
        # Check currently loaded modules
        if "vertexai" in sys.modules:
            self.patch_vertex_ai_methods(sys.modules["vertexai"])
        if "openai" in sys.modules and self.validate_openai_key():
            self.patch_openai_methods(sys.modules["openai"])
            self.patch_openai_beta_methods(sys.modules["openai"])
        if "litellm" in sys.modules:
            self.patch_litellm_methods(sys.modules["litellm"])
        if "anthropic" in sys.modules:
            self.patch_anthropic_methods(sys.modules["anthropic"])
        if "google.generativeai" in sys.modules:
            self.patch_google_genai_methods(sys.modules["google.generativeai"])
        if "langchain_google_vertexai" in sys.modules:
            self.patch_langchain_google_methods(sys.modules["langchain_google_vertexai"])
        if "langchain_google_genai" in sys.modules:
            self.patch_langchain_google_methods(sys.modules["langchain_google_genai"])

        if "langchain_openai" in sys.modules:
            self.patch_langchain_openai_methods(sys.modules["langchain_openai"])
        if "langchain_anthropic" in sys.modules:
            self.patch_langchain_anthropic_methods(sys.modules["langchain_anthropic"])

        if "llama_index" in sys.modules:
            self.patch_llama_index_methods(sys.modules["llama_index"])

        # Register hooks for future imports with availability checks
        if self.check_package_available("vertexai"):
            wrapt.register_post_import_hook(self.patch_vertex_ai_methods, "vertexai")
            wrapt.register_post_import_hook(
                self.patch_vertex_ai_methods, "vertexai.generative_models"
            )

        if self.check_package_available("openai") and self.validate_openai_key():
            wrapt.register_post_import_hook(self.patch_openai_methods, "openai")
            wrapt.register_post_import_hook(self.patch_openai_beta_methods, "openai")

        if self.check_package_available("litellm"):
            wrapt.register_post_import_hook(self.patch_litellm_methods, "litellm")

        if self.check_package_available("anthropic"):
            wrapt.register_post_import_hook(self.patch_anthropic_methods, "anthropic")

        if self.check_package_available("google.generativeai"):
            wrapt.register_post_import_hook(
                self.patch_google_genai_methods, "google.generativeai"
            )

        # Add hooks for LangChain integrations with availability checks
        if self.check_package_available("langchain_google_vertexai"):
            wrapt.register_post_import_hook(
                self.patch_langchain_google_methods, "langchain_google_vertexai"
            )


        # Add hooks for llama-index
        wrapt.register_post_import_hook(self.patch_llama_index_methods, "llama_index")
        
        if self.check_package_available("langchain_google_genai"):
            wrapt.register_post_import_hook(
                self.patch_langchain_google_methods, "langchain_google_genai"
            )

        if self.check_package_available("langchain_openai"):
            wrapt.register_post_import_hook(
                self.patch_langchain_openai_methods, "langchain_openai"
            )
        if self.check_package_available("langchain_anthropic"):
            wrapt.register_post_import_hook(
                self.patch_langchain_anthropic_methods, "langchain_anthropic"
            )

    def instrument_user_interaction_calls(self):
        """Enable user interaction instrumentation for LLM calls"""
        self.auto_instrument_user_interaction = True

    def instrument_network_calls(self):
        """Enable network instrumentation for LLM calls"""
        self.auto_instrument_network = True

    def instrument_file_io_calls(self):
        """Enable file IO instrumentation for LLM calls"""
        self.auto_instrument_file_io = True

    def patch_llama_index_methods(self, module):
        """Patch llama-index LLM methods"""
        try:
            # Handle OpenAI LLM from llama-index
            if hasattr(module, "llms"):
                # OpenAI
                if hasattr(module.llms, "openai"):
                    openai_module = module.llms.openai
                    if hasattr(openai_module, "OpenAI"):
                        llm_class = getattr(openai_module, "OpenAI")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")
                        self.wrap_method(llm_class, "stream_chat")
                        # self.wrap_method(llm_class, "stream_achat")
                        self.wrap_method(llm_class, "stream_complete")
                        # self.wrap_method(llm_class, "stream_acomplete")

                # Anthropic
                if hasattr(module.llms, "anthropic"):
                    anthropic_module = module.llms.anthropic
                    if hasattr(anthropic_module, "Anthropic"):
                        llm_class = getattr(anthropic_module, "Anthropic")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")
                        self.wrap_method(llm_class, "stream_chat")
                        # self.wrap_method(llm_class, "stream_achat")

                # Azure OpenAI
                if hasattr(module.llms, "azure_openai"):
                    azure_module = module.llms.azure_openai
                    if hasattr(azure_module, "AzureOpenAI"):
                        llm_class = getattr(azure_module, "AzureOpenAI")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")
                        self.wrap_method(llm_class, "stream_chat")
                        # self.wrap_method(llm_class, "stream_achat")

                # LiteLLM
                if hasattr(module.llms, "litellm"):
                    litellm_module = module.llms.litellm
                    if hasattr(litellm_module, "LiteLLM"):
                        llm_class = getattr(litellm_module, "LiteLLM")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")

                # Vertex AI
                if hasattr(module.llms, "vertex"):
                    vertex_module = module.llms.vertex
                    if hasattr(vertex_module, "Vertex"):
                        llm_class = getattr(vertex_module, "Vertex")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")

                # Gemini
                if hasattr(module.llms, "gemini"):
                    gemini_module = module.llms.gemini
                    if hasattr(gemini_module, "Gemini"):
                        llm_class = getattr(gemini_module, "Gemini")
                        self.wrap_method(llm_class, "complete")
                        self.wrap_method(llm_class, "acomplete")
                        self.wrap_method(llm_class, "chat")
                        self.wrap_method(llm_class, "achat")

        except Exception as e:
            # Log the error but continue execution
            print(f"Warning: Failed to patch llama-index methods: {str(e)}")

    def patch_openai_methods(self, module):
        try:
            if hasattr(module, "OpenAI"):
                client_class = getattr(module, "OpenAI")
                self.wrap_openai_client_methods(client_class)
            if hasattr(module, "AsyncOpenAI"):
                async_client_class = getattr(module, "AsyncOpenAI")
                self.wrap_openai_client_methods(async_client_class)
        except Exception as e:
            # Log the error but continue execution
            print(f"Warning: Failed to patch OpenAI methods: {str(e)}")

    def patch_langchain_openai_methods(self, module):
        try:
            if hasattr(module, 'ChatOpenAI'):
                client_class = getattr(module, "ChatOpenAI")

                if hasattr(client_class, "invoke"):
                    self.wrap_langchain_openai_method(client_class, f"{client_class.__name__}.invoke")
                elif hasattr(client_class, "run"):
                    self.wrap_langchain_openai_method(client_class, f"{client_class.__name__}.run")
            if hasattr(module, 'AsyncChatOpenAI'):
                if hasattr(client_class, "ainvoke"):
                    self.wrap_langchain_openai_method(client_class, f"{client_class.__name__}.ainvoke")
                elif hasattr(client_class, "arun"):
                    self.wrap_langchain_openai_method(client_class, f"{client_class.__name__}.arun")
        except Exception as e:
            # Log the error but continue execution
            print(f"Warning: Failed to patch OpenAI methods: {str(e)}")

    def patch_langchain_anthropic_methods(self, module):
        try:
            if hasattr(module, 'ChatAnthropic'):
                client_class = getattr(module, "ChatAnthropic")
                if hasattr(client_class, "invoke"):
                    self.wrap_langchain_anthropic_method(client_class, f"{client_class.__name__}.invoke")
                if hasattr(client_class, "ainvoke"):
                    self.wrap_langchain_anthropic_method(client_class, f"{client_class.__name__}.ainvoke")
            if hasattr(module, 'AsyncChatAnthropic'):
                async_client_class = getattr(module, "AsyncChatAnthropic")
                if hasattr(async_client_class, "ainvoke"):
                    self.wrap_langchain_anthropic_method(async_client_class, f"{async_client_class.__name__}.ainvoke")
                if hasattr(async_client_class, "arun"):
                    self.wrap_langchain_anthropic_method(async_client_class, f"{async_client_class.__name__}.arun")
        except Exception as e:
            # Log the error but continue execution
            print(f"Warning: Failed to patch Anthropic methods: {str(e)}")

    def patch_openai_beta_methods(self, openai_module):
        """
        Patch the new openai.beta endpoints (threads, runs, messages, etc.)
        so that calls like openai.beta.threads.create(...) or
        openai.beta.threads.runs.create(...) are automatically traced.
        """
        # Make sure openai_module has a 'beta' attribute
        openai_module.api_type = "openai"
        if not hasattr(openai_module, "beta"):
            return

        beta_module = openai_module.beta

        # Patch openai.beta.threads
        import openai
        openai.api_type = "openai"
        if hasattr(beta_module, "threads"):
            threads_obj = beta_module.threads
            # Patch top-level methods on openai.beta.threads
            for method_name in ["create", "list"]:
                if hasattr(threads_obj, method_name):
                    self.wrap_method(threads_obj, method_name)

            # Patch the nested objects: messages, runs
            if hasattr(threads_obj, "messages"):
                messages_obj = threads_obj.messages
                for method_name in ["create", "list"]:
                    if hasattr(messages_obj, method_name):
                        self.wrap_method(messages_obj, method_name)

            if hasattr(threads_obj, "runs"):
                runs_obj = threads_obj.runs
                for method_name in ["create", "retrieve", "list"]:
                    if hasattr(runs_obj, method_name):
                        self.wrap_method(runs_obj, method_name)

    def patch_anthropic_methods(self, module):
        if hasattr(module, "Anthropic"):
            client_class = getattr(module, "Anthropic")
            self.wrap_anthropic_client_methods(client_class)

    def patch_google_genai_methods(self, module):
        # Patch direct Google GenerativeAI usage
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_genai_model_methods(model_class)

        # Patch LangChain integration
        if hasattr(module, "ChatGoogleGenerativeAI"):
            chat_class = getattr(module, "ChatGoogleGenerativeAI")
            # Wrap invoke method to capture messages
            original_invoke = chat_class.invoke

            def patched_invoke(self, messages, *args, **kwargs):
                # Store messages in the instance for later use
                self._last_messages = messages
                return original_invoke(self, messages, *args, **kwargs)

            chat_class.invoke = patched_invoke

            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "_generate")
            if hasattr(chat_class, "_agenerate"):
                self.wrap_method(chat_class, "_agenerate")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

    def patch_vertex_ai_methods(self, module):
        # Patch the GenerativeModel class
        if hasattr(module, "generative_models"):
            gen_models = getattr(module, "generative_models")
            if hasattr(gen_models, "GenerativeModel"):
                model_class = getattr(gen_models, "GenerativeModel")
                self.wrap_vertex_model_methods(model_class)

        # Also patch the class directly if available
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_vertex_model_methods(model_class)

    def wrap_vertex_model_methods(self, model_class):
        # Patch both sync and async methods
        self.wrap_method(model_class, "generate_content")
        if hasattr(model_class, "generate_content_async"):
            self.wrap_method(model_class, "generate_content_async")

    def patch_litellm_methods(self, module):
        self.wrap_method(module, "completion")
        self.wrap_method(module, "acompletion")

    def patch_langchain_google_methods(self, module):
        """Patch LangChain's Google integration methods"""
        if hasattr(module, "ChatVertexAI"):
            chat_class = getattr(module, "ChatVertexAI")
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "_generate")
            if hasattr(chat_class, "_agenerate"):
                self.wrap_method(chat_class, "_agenerate")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

        if hasattr(module, "ChatGoogleGenerativeAI"):
            chat_class = getattr(module, "ChatGoogleGenerativeAI")
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "_generate")
            if hasattr(chat_class, "_agenerate"):
                self.wrap_method(chat_class, "_agenerate")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

    def wrap_openai_client_methods(self, client_class):
        original_init = client_class.__init__

        @functools.wraps(original_init)
        def patched_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            # Check if this is AsyncOpenAI or OpenAI
            is_async = "AsyncOpenAI" in client_class.__name__

            if is_async:
                # Patch async methods for AsyncOpenAI
                if hasattr(client_self.chat.completions, "create"):
                    original_create = client_self.chat.completions.create

                    @functools.wraps(original_create)
                    async def wrapped_create(*args, **kwargs):
                        return await self.trace_llm_call(
                            original_create, *args, **kwargs
                        )

                    client_self.chat.completions.create = wrapped_create
            else:
                # Patch sync methods for OpenAI
                if hasattr(client_self.chat.completions, "create"):
                    original_create = client_self.chat.completions.create

                    @functools.wraps(original_create)
                    def wrapped_create(*args, **kwargs):
                        return self.trace_llm_call_sync(
                            original_create, *args, **kwargs
                        )

                    client_self.chat.completions.create = wrapped_create

        setattr(client_class, "__init__", patched_init)

    def wrap_langchain_openai_method(self, client_class, method_name):
        method = method_name.split(".")[-1]
        original_init = getattr(client_class, method)

        @functools.wraps(original_init)
        def patched_init(*args, **kwargs):
            # Check if this is AsyncOpenAI or OpenAI
            is_async = "AsyncChatOpenAI" in client_class.__name__

            if is_async:
                return self.trace_llm_call(original_init, *args, **kwargs)
            else:
                return self.trace_llm_call_sync(original_init, *args, **kwargs)

        setattr(client_class, method, patched_init)

    def wrap_langchain_anthropic_method(self, client_class, method_name):
        original_init = getattr(client_class, method_name)

        @functools.wraps(original_init)
        def patched_init(*args, **kwargs):
            is_async = "AsyncChatAnthropic" in client_class.__name__

            if is_async:
                return self.trace_llm_call(original_init, *args, **kwargs)
            else:
                return self.trace_llm_call_sync(original_init, *args, **kwargs)

        setattr(client_class, method_name, patched_init)

    def wrap_anthropic_client_methods(self, client_class):
        original_init = client_class.__init__

        @functools.wraps(original_init)
        def patched_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            self.wrap_method(client_self.messages, "create")
            if hasattr(client_self.messages, "acreate"):
                self.wrap_method(client_self.messages, "acreate")

        setattr(client_class, "__init__", patched_init)

    def wrap_genai_model_methods(self, model_class):
        original_init = model_class.__init__

        @functools.wraps(original_init)
        def patched_init(model_self, *args, **kwargs):
            original_init(model_self, *args, **kwargs)
            self.wrap_method(model_self, "generate_content")
            if hasattr(model_self, "generate_content_async"):
                self.wrap_method(model_self, "generate_content_async")

        setattr(model_class, "__init__", patched_init)

    def wrap_method(self, obj, method_name):
        """
        Wrap a method with tracing functionality.
        Works for both class methods and instance methods.
        """
        # If obj is a class, we need to patch both the class and any existing instances
        if isinstance(obj, type):
            # Store the original class method
            original_method = getattr(obj, method_name)

            @wrapt.decorator
            def wrapper(wrapped, instance, args, kwargs):
                if asyncio.iscoroutinefunction(wrapped):
                    return self.trace_llm_call(wrapped, *args, **kwargs)
                return self.trace_llm_call_sync(wrapped, *args, **kwargs)

            # Wrap the class method
            wrapped_method = wrapper(original_method)
            setattr(obj, method_name, wrapped_method)
            self.patches.append((obj, method_name, original_method))

        else:
            # For instance methods
            original_method = getattr(obj, method_name)

            @wrapt.decorator
            def wrapper(wrapped, instance, args, kwargs):
                if asyncio.iscoroutinefunction(wrapped):
                    return self.trace_llm_call(wrapped, *args, **kwargs)
                return self.trace_llm_call_sync(wrapped, *args, **kwargs)

            wrapped_method = wrapper(original_method)
            setattr(obj, method_name, wrapped_method)
            self.patches.append((obj, method_name, original_method))

    def create_llm_component(
            self,
            component_id,
            hash_id,
            name,
            llm_type,
            version,
            memory_used,
            start_time,
            input_data,
            output_data,
            cost={},
            usage={},
            error=None,
            parameters={},
    ):
        try:
            # Update total metrics
            self.total_tokens += usage.get("total_tokens", 0)
            self.total_cost += cost.get("total_cost", 0)

            network_calls = []
            if self.auto_instrument_network:
                network_calls = self.component_network_calls.get(component_id, [])

            interactions = []
            if self.auto_instrument_user_interaction:
                input_output_interactions = []
                for interaction in self.component_user_interaction.get(component_id, []):
                    if interaction["interaction_type"] in ["input", "output"]:
                        input_output_interactions.append(interaction)
                interactions.extend(input_output_interactions)
            if self.auto_instrument_file_io:
                file_io_interactions = []
                for interaction in self.component_user_interaction.get(component_id, []):
                    if interaction["interaction_type"] in ["file_read", "file_write"]:
                        file_io_interactions.append(interaction)
                interactions.extend(file_io_interactions)

            parameters_to_display = {}
            if "run_manager" in parameters:
                parameters_obj = parameters["run_manager"]
                if hasattr(parameters_obj, "metadata"):
                    metadata = parameters_obj.metadata
                    # parameters = {'metadata': metadata}
                    parameters_to_display.update(metadata)

            # Add only those keys in parameters that are single values and not objects, dict or list
            for key, value in parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    parameters_to_display[key] = value

            # Limit the number of parameters to display
            parameters_to_display = dict(
                list(parameters_to_display.items())[: self.MAX_PARAMETERS_TO_DISPLAY]
            )

            # Set the Context and GT
            span_gt = None
            span_context = None
            if name in self.span_attributes_dict:
                span_gt = self.span_attributes_dict[name].gt
                span_context = self.span_attributes_dict[name].context

                logger.debug(f"span context {span_context}, span_gt {span_gt}")

            # Tags
            tags = []
            if name in self.span_attributes_dict:
                tags = self.span_attributes_dict[name].tags or []

            # Get End Time
            end_time = datetime.now().astimezone().isoformat()

            # Metrics
            metrics = []
            if name in self.span_attributes_dict:
                raw_metrics = self.span_attributes_dict[name].metrics or []
                for metric in raw_metrics:
                    base_metric_name = metric["name"]
                    counter = sum(1 for x in self.visited_metrics if x.startswith(base_metric_name))
                    metric_name = f'{base_metric_name}_{counter}' if counter > 0 else base_metric_name
                    self.visited_metrics.append(metric_name)
                    metric["name"] = metric_name
                    metrics.append(metric)

            # TODO TO check i/p and o/p is according or not
            input = input_data["args"] if hasattr(input_data, "args") else input_data
            output = output_data.output_response if output_data else None
            prompt = self.convert_to_content(input)
            response = self.convert_to_content(output)

            # TODO: Execute & Add the User requested metrics here
            formatted_metrics = BaseTracer.get_formatted_metric(self.span_attributes_dict, self.project_id, name)
            if formatted_metrics:
                metrics.extend(formatted_metrics)

            component = {
                "id": component_id,
                "hash_id": hash_id,
                "source_hash_id": None,
                "type": "llm",
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "error": error,
                "parent_id": self.current_agent_id.get(),
                "info": {
                    "model": llm_type,
                    "version": version,
                    "memory_used": memory_used,
                    "cost": cost,
                    "tokens": usage,
                    "tags": tags,
                    **parameters_to_display,
                },
                "extra_info": parameters,
                "data": {
                    "input": input,
                    "output": output,
                    "memory_used": memory_used,
                },
                "metrics": metrics,
                "network_calls": network_calls,
                "interactions": interactions,
            }

            # Assign context and gt if available
            component["data"]["gt"] = span_gt
            component["data"]["context"] = span_context

            # Reset the SpanAttributes context variable
            self.span_attributes_dict[name] = SpanAttributes(name)

            return component
        except Exception as e:
            raise Exception("Failed to create LLM component")

    def convert_to_content(self, input_data):
        try:
            if isinstance(input_data, dict):
                messages = input_data.get("kwargs", {}).get("messages", [])
            elif isinstance(input_data, list):
                if len(input_data)>0 and isinstance(input_data[0]['content'],ChatResponse):
                    extracted_messages = []
                    for item in input_data:
                        chat_response = item.get('content')
                        if hasattr(chat_response, 'message') and hasattr(chat_response.message, 'blocks'):
                            for block in chat_response.message.blocks:
                                if hasattr(block, 'text'):
                                    extracted_messages.append(block.text)
                    messages=extracted_messages
                    if isinstance(messages,list):
                        return "\n".join(messages)
                elif len(input_data)>0 and isinstance(input_data[0]['content'],TextBlock):
                    return " ".join(block.text for item in input_data for block in item['content'] if isinstance(block, TextBlock))
                elif len(input_data)>0 and isinstance(input_data[0]['content'],ChatMessage):
                    return " ".join(block.text for block in input_data[0]['content'].blocks if isinstance(block, TextBlock)) 
                else:
                    messages = input_data
            elif isinstance(input_data,ChatResponse):
                messages=input_data['content']
            else:
                return ""
            res=""
            res="\n".join(msg.get("content", "").strip() for msg in messages if msg.get("content"))
        except Exception as e:
            res=str(input_data)
        return res

    def process_content(content):
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            # Handle list of content blocks
            text_parts = []
            for block in content:
                if hasattr(block, 'text'):
                    # Handle TextBlock-like objects
                    text_parts.append(block.text.strip())
                elif isinstance(block, dict) and 'text' in block:
                    # Handle dictionary with text field
                    text_parts.append(block['text'].strip())
            return " ".join(text_parts)
        elif isinstance(content, dict):
            # Handle dictionary content
            return content.get('text', '').strip()
        return ""

    
    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        self.component_network_calls[component_id] = []
        self.current_component_id = component_id

    def end_component(self, component_id):
        """Stop tracking network calls for a component"""
        self.current_component_id = None

    async def trace_llm_call(self, original_func, *args, **kwargs):
        """Trace an LLM API call"""
        if not self.is_active:
            return await original_func(*args, **kwargs)

        if not self.auto_instrument_llm:
            return await original_func(*args, **kwargs)

        start_time = datetime.now().astimezone().isoformat()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash(original_func, args, kwargs)

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the LLM call
            result = await original_func(*args, **kwargs)

            # Calculate resource usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract token usage and calculate cost
            model_name = extract_model_name(args, kwargs, result)
            if 'stream' in kwargs:
                stream = kwargs['stream']
                if stream:
                    prompt_messages = kwargs['messages']
                    # Create response message for streaming case
                    response_message = {"role": "assistant", "content": result} if result else {"role": "assistant",
                                                                                                "content": ""}
                    token_usage = num_tokens_from_messages(model_name, prompt_messages, response_message)
                else:
                    token_usage = extract_token_usage(result)
            else:
                token_usage = extract_token_usage(result)
            cost = calculate_llm_cost(token_usage, model_name, self.model_costs, self.model_custom_cost)
            parameters = extract_parameters(kwargs)
            input_data = extract_input_data(args, kwargs, result)

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__

            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type=model_name,
                version=None,
                memory_used=memory_used,
                start_time=start_time,
                input_data=input_data,
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage,
                parameters=parameters,
            )

            self.add_component(llm_component)
            self.llm_data = llm_component

            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__

            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type="unknown",
                version=None,
                memory_used=0,
                start_time=start_time,
                input_data=extract_input_data(args, kwargs, None),
                output_data=None,
                error=error_component,
            )

            self.add_component(llm_component)

            raise

    def trace_llm_call_sync(self, original_func, *args, **kwargs):
        """Sync version of trace_llm_call"""
        if not self.is_active:
            if asyncio.iscoroutinefunction(original_func):
                return asyncio.run(original_func(*args, **kwargs))
            return original_func(*args, **kwargs)

        if not self.auto_instrument_llm:
            return original_func(*args, **kwargs)

        start_time = datetime.now().astimezone().isoformat()
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash(original_func, args, kwargs)

        # Start tracking network calls for this component
        self.start_component(component_id)

        # Calculate resource usage
        start_memory = psutil.Process().memory_info().rss

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(original_func):
                result = asyncio.run(original_func(*args, **kwargs))
            else:
                result = original_func(*args, **kwargs)

            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract token usage and calculate cost
            model_name = extract_model_name(args, kwargs, result)

            if 'stream' in kwargs:
                stream = kwargs['stream']
                if stream:
                    prompt_messages = kwargs['messages']
                    # Create response message for streaming case
                    response_message = {"role": "assistant", "content": result} if result else {"role": "assistant",
                                                                                                "content": ""}
                    token_usage = num_tokens_from_messages(model_name, prompt_messages, response_message)
                else:
                    token_usage = extract_token_usage(result)
            else:
                token_usage = extract_token_usage(result)
            cost = calculate_llm_cost(token_usage, model_name, self.model_costs, self.model_custom_cost)
            parameters = extract_parameters(kwargs)
            input_data = extract_input_data(args, kwargs, result)

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__

            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type=model_name,
                version=None,
                memory_used=memory_used,
                start_time=start_time,
                input_data=input_data,
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage,
                parameters=parameters,
            )
            self.llm_data = llm_component
            self.add_component(llm_component)

            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__

            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type="unknown",
                version=None,
                memory_used=memory_used,
                start_time=start_time,
                input_data=extract_input_data(args, kwargs, None),
                output_data=None,
                error=error_component,
            )
            self.llm_data = llm_component
            self.add_component(llm_component, is_error=True)

            raise

    def trace_llm(
            self,
            name: str = None,
            tags: List[str] = [],
            metadata: Dict[str, Any] = {},
            metrics: List[Dict[str, Any]] = [],
            feedback: Optional[Any] = None,
    ):

        start_memory = psutil.Process().memory_info().rss
        start_time = datetime.now().astimezone().isoformat()

        if name not in self.span_attributes_dict:
            self.span_attributes_dict[name] = SpanAttributes(name)
        if tags:
            self.span(name).add_tags(tags)
        if metadata:
            self.span(name).add_metadata(metadata)
        if metrics:
            if isinstance(metrics, dict):
                metrics = [metrics]
            try:
                for metric in metrics:
                    self.span(name).add_metrics(
                        name=metric["name"],
                        score=metric["score"],
                        reasoning=metric.get("reasoning", ""),
                        cost=metric.get("cost", None),
                        latency=metric.get("latency", None),
                        metadata=metric.get("metadata", {}),
                        config=metric.get("config", {}),
                    )
            except ValueError as e:
                logger.error(f"Validation Error: {e}")
            except Exception as e:
                logger.error(f"Error adding metric: {e}")

        if feedback:
            self.span(name).add_feedback(feedback)

        self.current_llm_call_name.set(name)

        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                gt = kwargs.get("gt") if kwargs else None
                if gt is not None:
                    span = self.span(name)
                    span.add_gt(gt)
                self.current_llm_call_name.set(name)
                if not self.is_active:
                    return await func(*args, **kwargs)

                component_id = str(uuid.uuid4())
                parent_agent_id = self.current_agent_id.get()
                self.start_component(component_id)

                error_info = None
                result = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_component = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }

                    # End tracking network calls for this component
                    self.end_component(component_id)

                    end_memory = psutil.Process().memory_info().rss
                    memory_used = max(0, end_memory - start_memory)

                    llm_component = self.create_llm_component(
                        component_id=component_id,
                        hash_id=generate_unique_hash(func, args, kwargs),
                        name=name,
                        llm_type="unknown",
                        version=None,
                        memory_used=memory_used,
                        start_time=start_time,
                        input_data=extract_input_data(args, kwargs, None),
                        output_data=None,
                        error=error_component,
                    )
                    self.llm_data = llm_component
                    self.add_component(llm_component, is_error=True)

                    raise
                finally:

                    llm_component = self.llm_data
                    if (name is not None) or (name != ""):
                        llm_component["name"] = name

                    if name in self.span_attributes_dict:
                        span_gt = self.span_attributes_dict[name].gt
                        if span_gt is not None:
                            llm_component["data"]["gt"] = span_gt
                        span_context = self.span_attributes_dict[name].context
                        if span_context:
                            llm_component["data"]["context"] = span_context

                    if error_info:
                        llm_component["error"] = error_info["error"]

                    self.end_component(component_id)

                    # metrics
                    metrics = []
                    if name in self.span_attributes_dict:
                        raw_metrics = self.span_attributes_dict[name].metrics or []
                        for metric in raw_metrics:
                            base_metric_name = metric["name"]
                            counter = sum(1 for x in self.visited_metrics if x.startswith(base_metric_name))
                            metric_name = f'{base_metric_name}_{counter}' if counter > 0 else base_metric_name
                            self.visited_metrics.append(metric_name)
                            metric["name"] = metric_name
                            metrics.append(metric)
                    llm_component["metrics"] = metrics
                    if parent_agent_id:
                        children = self.agent_children.get()
                        children.append(llm_component)
                        self.agent_children.set(children)
                    else:
                        self.add_component(llm_component)

                    llm_component["interactions"] = self.component_user_interaction.get(
                        component_id, []
                    )
                    self.add_component(llm_component)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                gt = kwargs.get("gt") if kwargs else None
                if gt is not None:
                    span = self.span(name)
                    span.add_gt(gt)
                self.current_llm_call_name.set(name)
                if not self.is_active:
                    return func(*args, **kwargs)

                component_id = str(uuid.uuid4())
                parent_agent_id = self.current_agent_id.get()
                self.start_component(component_id)

                start_time = datetime.now().astimezone().isoformat()
                error_info = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_component = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }

                    # End tracking network calls for this component
                    self.end_component(component_id)

                    end_memory = psutil.Process().memory_info().rss
                    memory_used = max(0, end_memory - start_memory)

                    llm_component = self.create_llm_component(
                        component_id=component_id,
                        hash_id=generate_unique_hash(func, args, kwargs),
                        name=name,
                        llm_type="unknown",
                        version=None,
                        memory_used=memory_used,
                        start_time=start_time,
                        input_data=extract_input_data(args, kwargs, None),
                        output_data=None,
                        error=error_component,
                    )
                    self.llm_data = llm_component
                    self.add_component(llm_component, is_error=True)

                    raise
                finally:
                    llm_component = self.llm_data
                    if (name is not None) or (name != ""):
                        llm_component["name"] = name

                    if error_info:
                        llm_component["error"] = error_info["error"]

                    self.end_component(component_id)
                    metrics = []
                    if name in self.span_attributes_dict:
                        raw_metrics = self.span_attributes_dict[name].metrics or []
                        for metric in raw_metrics:
                            base_metric_name = metric["name"]
                            counter = sum(1 for x in self.visited_metrics if x.startswith(base_metric_name))
                            metric_name = f'{base_metric_name}_{counter}' if counter > 0 else base_metric_name
                            self.visited_metrics.append(metric_name)
                            metric["name"] = metric_name
                            metrics.append(metric)
                    llm_component["metrics"] = metrics
                    if parent_agent_id:
                        children = self.agent_children.get()
                        children.append(llm_component)
                        self.agent_children.set(children)
                    else:
                        self.add_component(llm_component)

                    llm_component["interactions"] = self.component_user_interaction.get(
                        component_id, []
                    )
                    self.add_component(llm_component)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def unpatch_llm_calls(self):
        # Remove all patches
        for obj, method_name, original_method in self.patches:
            try:
                setattr(obj, method_name, original_method)
            except Exception as e:
                print(f"Error unpatching {method_name}: {str(e)}")
        self.patches = []
