from typing import Any, Dict, List, Optional, Union, Sequence

import attr
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish, BaseMessage
from datetime import datetime
import json
import os
from uuid import UUID
from functools import wraps
import asyncio
from langchain_core.documents import Document
import logging
import tempfile
import sys
import importlib
from importlib.util import find_spec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangchainTracer(BaseCallbackHandler):
    """
    An enhanced callback handler for LangChain that traces all actions and saves them to a JSON file.
    Includes improved error handling, async support, and configuration options.
    """

    def __init__(
        self,
        output_path: str = tempfile.gettempdir(),
        trace_all: bool = True,
        save_interval: Optional[int] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the tracer with enhanced configuration options.

        Args:
            output_path (str): Directory where trace files will be saved
            trace_all (bool): Whether to trace all components or only specific ones
            save_interval (Optional[int]): Interval in seconds to auto-save traces
            log_level (int): Logging level for the tracer
        """
        super().__init__()
        self.output_path = output_path
        self.trace_all = trace_all
        self.save_interval = save_interval
        self._active = False
        self._original_inits = {}
        self._original_methods = {}
        self.additional_metadata = {}
        self._save_task = None
        self._current_query = None
        self.filepath = None
        self.model_names = {}  # Store model names by component instance
        logger.setLevel(log_level)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.reset_trace()
        

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        
        self.stop()
        if exc_type:
            logger.error(f"Error in context manager: {exc_val}")
            return False
        return True

    def reset_trace(self):
        """Reset the current trace to initial state with enhanced structure"""
        self.current_trace: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "actions": [],
            "llm_calls": [],
            "chain_starts": [],
            "chain_ends": [],
            "agent_actions": [],
            "chat_model_calls": [],
            "retriever_actions": [],
            "tokens": [],
            "errors": [],
            "query": self._current_query,
            "metadata": {
                "version": "2.0",
                "trace_all": self.trace_all,
                "save_interval": self.save_interval,
            },
        }

    async def _periodic_save(self):
        """Periodically save traces if save_interval is set"""
        while self._active and self.save_interval:
            await asyncio.sleep(self.save_interval)
            await self._async_save_trace()

    async def _async_save_trace(self, force: bool = False):
        """Asynchronously save the current trace to a JSON file"""
        if not self.current_trace["start_time"] and not force:
            return

        try:
            self.current_trace["end_time"] = datetime.now()
            
            # Use the query from the trace or fallback to a default
            safe_query = self._current_query or "unknown"
            
            # Sanitize the query for filename
            safe_query = ''.join(c for c in safe_query if c.isalnum() or c.isspace())[:50].strip()

            # Add a timestamp to ensure unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langchain_callback_traces.json"
            filepath = os.path.join(self.output_path, filename)
            self.filepath = filepath

            trace_to_save = self.current_trace.copy()
            trace_to_save["start_time"] = str(trace_to_save["start_time"])
            trace_to_save["end_time"] = str(trace_to_save["end_time"])

            # Save if there are meaningful events or if force is True
            if (
                len(trace_to_save["llm_calls"]) > 0
                or len(trace_to_save["chain_starts"]) > 0
                or len(trace_to_save["chain_ends"]) > 0
                or len(trace_to_save["errors"]) > 0
                or force
            ):
                async with asyncio.Lock():
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(trace_to_save, f, indent=2, default=str)

                logger.info(f"Trace saved to: {filepath}")
                
                # Reset the current query after saving
                self._current_query = None
                
                # Reset the trace
                self.reset_trace()

        except Exception as e:
            logger.error(f"Error saving trace: {e}")
            self.on_error(e, context="save_trace")

    def _save_trace(self, force: bool = False):
        """Synchronous version of trace saving"""
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self._async_save_trace(force))
        else:
            asyncio.run(self._async_save_trace(force))

    def _create_safe_wrapper(self, original_func, component_name, method_name):
        """Create a safely wrapped version of an original function with enhanced error handling"""

        @wraps(original_func)
        def wrapped(*args, **kwargs):
            if not self._active:
                return original_func(*args, **kwargs)

            try:
                # Deep copy kwargs to avoid modifying the original
                kwargs_copy = kwargs.copy() if kwargs is not None else {}
                
                # Handle different calling conventions
                if 'callbacks' not in kwargs_copy:
                    kwargs_copy['callbacks'] = [self]
                elif self not in kwargs_copy['callbacks']:
                    kwargs_copy['callbacks'].append(self)

                # Store model name if available
                if component_name in ["OpenAI", "ChatOpenAI_LangchainOpenAI", "ChatOpenAI_ChatModels",
                                    "ChatVertexAI", "VertexAI", "ChatGoogleGenerativeAI", "ChatAnthropic", 
                                    "ChatLiteLLM", "ChatBedrock", "AzureChatOpenAI", "ChatAnthropicVertex"]:
                    instance = args[0] if args else None
                    model_name = kwargs.get('model_name') or kwargs.get('model') or kwargs.get('model_id')

                    if instance and model_name:
                        self.model_names[id(instance)] = model_name
                
                # Try different method signatures
                try:
                    # First, try calling with modified kwargs
                    return original_func(*args, **kwargs_copy)
                except TypeError:
                    # If that fails, try without kwargs
                    try:
                        return original_func(*args)
                    except Exception as e:
                        # If all else fails, use original call
                        logger.error(f"Failed to invoke {component_name} with modified callbacks: {e}")
                        return original_func(*args, **kwargs)
            
            except Exception as e:
                # Log any errors that occur during the function call
                logger.error(f"Error in {component_name} wrapper: {e}")
                
                # Record the error using the tracer's error handling method
                self.on_error(e, context=f"wrapper_{component_name}")
                
                # Fallback to calling the original function without modifications
                return original_func(*args, **kwargs)
            
        @wraps(original_func)
        def wrapped_invoke(*args, **kwargs):
            if not self._active:
                return original_func(*args, **kwargs)
            
            try:
                # Deep copy kwargs to avoid modifying the original
                kwargs_copy = kwargs.copy() if kwargs is not None else {}

                # Handle different calling conventions
                if 'config' not in kwargs_copy:
                    kwargs_copy['config'] = {'callbacks': [self]}
                elif 'callbacks' not in kwargs_copy['config']:
                    kwargs_copy['config']['callbacks'] = [self]
                elif self not in kwargs_copy['config']['callbacks']:
                    kwargs_copy['config']['callbacks'].append(self)

                # Store model name if available
                if component_name in ["OpenAI", "ChatOpenAI_LangchainOpenAI", "ChatOpenAI_ChatModels",
                                    "ChatVertexAI", "VertexAI", "ChatGoogleGenerativeAI", "ChatAnthropic", 
                                    "ChatLiteLLM", "ChatBedrock", "AzureChatOpenAI", "ChatAnthropicVertex"]:
                    instance = args[0] if args else None
                    model_name = kwargs.get('model_name') or kwargs.get('model') or kwargs.get('model_id')

                    if instance and model_name:
                        self.model_names[id(instance)] = model_name
                
                # Try different method signatures
                try:
                    # First, try calling with modified kwargs
                    return original_func(*args, **kwargs_copy)
                except TypeError:
                    # If that fails, try without kwargs
                    try:
                        return original_func(*args)
                    except Exception as e:
                        # If all else fails, use original call
                        logger.error(f"Failed to invoke {component_name} with modified callbacks: {e}")
                        return original_func(*args, **kwargs)
            
            except Exception as e:
                # Log any errors that occur during the function call
                logger.error(f"Error in {component_name} wrapper: {e}")
                
                # Record the error using the tracer's error handling method
                self.on_error(e, context=f"wrapper_{component_name}")
                
                # Fallback to calling the original function without modifications
                return original_func(*args, **kwargs)
        
        if method_name == 'invoke':
            return wrapped_invoke
        return wrapped


    def _monkey_patch(self):
        """Enhanced monkey-patching with comprehensive component support"""
        components_to_patch = {}
        
        try:
            from langchain.llms import OpenAI
            components_to_patch["OpenAI"] = (OpenAI, "__init__")
        except ImportError:
            logger.debug("OpenAI not available for patching")

        try:
            from langchain_aws import ChatBedrock
            components_to_patch["ChatBedrock"] = (ChatBedrock, "__init__")
        except ImportError:
            logger.debug("ChatBedrock not available for patching")
            
        try:
            from langchain_google_vertexai import ChatVertexAI
            components_to_patch["ChatVertexAI"] = (ChatVertexAI, "__init__")
        except ImportError:
            logger.debug("ChatVertexAI not available for patching")

        try:
            from langchain_google_vertexai import VertexAI
            components_to_patch["VertexAI"] = (VertexAI, "__init__")
        except ImportError:
            logger.debug("VertexAI not available for patching")

        try:
            from langchain_google_vertexai.model_garden import ChatAnthropicVertex
            components_to_patch["ChatAnthropicVertex"] = (ChatAnthropicVertex, "__init__")
        except ImportError:
            logger.debug("ChatAnthropicVertex not available for patching")
            
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            components_to_patch["ChatGoogleGenerativeAI"] = (ChatGoogleGenerativeAI, "__init__")
        except ImportError:
            logger.debug("ChatGoogleGenerativeAI not available for patching")
            
        try:
            from langchain_anthropic import ChatAnthropic
            components_to_patch["ChatAnthropic"] = (ChatAnthropic, "__init__")
        except ImportError:
            logger.debug("ChatAnthropic not available for patching")
            
        try:
            from langchain_community.chat_models import ChatLiteLLM
            components_to_patch["ChatLiteLLM"] = (ChatLiteLLM, "__init__")
        except ImportError:
            logger.debug("ChatLiteLLM not available for patching")
            
        try:
            from langchain_openai import ChatOpenAI as ChatOpenAI_LangchainOpenAI
            components_to_patch["ChatOpenAI_LangchainOpenAI"] = (ChatOpenAI_LangchainOpenAI, "__init__")
        except ImportError:
            logger.debug("ChatOpenAI (from langchain_openai) not available for patching")

        try:
            from langchain_openai import AzureChatOpenAI
            components_to_patch["AzureChatOpenAI"] = (AzureChatOpenAI, "__init__")
        except ImportError:
            logger.debug("AzureChatOpenAI (from langchain_openai) not available for patching")
            
        try:
            from langchain.chat_models import ChatOpenAI as ChatOpenAI_ChatModels
            components_to_patch["ChatOpenAI_ChatModels"] = (ChatOpenAI_ChatModels, "__init__")
        except ImportError:
            logger.debug("ChatOpenAI (from langchain.chat_models) not available for patching")
            
        try:
            from langchain.chains import create_retrieval_chain, RetrievalQA
            from langchain_core.runnables import RunnableBinding
            from langchain_core.runnables import RunnableSequence
            from langchain.chains import ConversationalRetrievalChain
            components_to_patch["RetrievalQA"] = (RetrievalQA, "from_chain_type")
            components_to_patch["create_retrieval_chain"] = (create_retrieval_chain, None)
            components_to_patch['RetrievalQA.invoke'] = (RetrievalQA, 'invoke')
            components_to_patch["RunnableBinding"] = (RunnableBinding, "invoke")
            components_to_patch["RunnableSequence"] = (RunnableSequence, "invoke")
            components_to_patch["ConversationalRetrievalChain"] = (ConversationalRetrievalChain, "invoke")
        except ImportError:
            logger.debug("Langchain chains not available for patching")

        for name, (component, method_name) in components_to_patch.items():
            try:
                if method_name == "__init__":
                    original = component.__init__
                    self._original_inits[name] = original
                    component.__init__ = self._create_safe_wrapper(original, name, method_name)
                elif method_name:
                    original = getattr(component, method_name)
                    self._original_methods[name] = original
                    if isinstance(original, classmethod):
                        wrapped = classmethod(
                            self._create_safe_wrapper(original.__func__, name, method_name)
                        )
                    else:
                        wrapped = self._create_safe_wrapper(original, name, method_name)
                    setattr(component, method_name, wrapped)
                else:
                    self._original_methods[name] = component
                    globals()[name] = self._create_safe_wrapper(component, name, method_name)
            except Exception as e:
                logger.error(f"Error patching {name}: {e}")
                self.on_error(e, context=f"patch_{name}")

    def _restore_original_methods(self):
        """Restore all original methods and functions with enhanced error handling"""
        # Dynamically import only what we need based on what was patched
        imported_components = {}
        
        if self._original_inits or self._original_methods:
            for name in list(self._original_inits.keys()) + list(self._original_methods.keys()):
                try:
                    if name == "OpenAI":
                        from langchain.llms import OpenAI
                        imported_components[name] = OpenAI
                    elif name == "ChatVertexAI":
                        from langchain_google_vertexai import ChatVertexAI
                        imported_components[name] = ChatVertexAI
                    elif name == "VertexAI":
                        from langchain_google_vertexai import VertexAI
                        imported_components[name] = VertexAI
                    elif name == "ChatGoogleGenerativeAI":
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        imported_components[name] = ChatGoogleGenerativeAI
                    elif name == "ChatAnthropic":
                        from langchain_anthropic import ChatAnthropic
                        imported_components[name] = ChatAnthropic
                    elif name == "ChatBedrock":
                        from langchain_aws import ChatBedrock
                        imported_components[name] = ChatBedrock
                    elif name == "AzureChatOpenAI":
                        from langchain_openai import AzureChatOpenAI
                        imported_components[name] = AzureChatOpenAI
                    elif name == "ChatAnthropicVertex":
                        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
                        imported_components[name] = ChatAnthropicVertex
                    elif name == "ChatLiteLLM":
                        from langchain_community.chat_models import ChatLiteLLM
                        imported_components[name] = ChatLiteLLM
                    elif name == "ChatOpenAI_LangchainOpenAI":
                        from langchain_openai import ChatOpenAI as ChatOpenAI_LangchainOpenAI
                        imported_components[name] = ChatOpenAI_LangchainOpenAI
                    elif name == "ChatOpenAI_ChatModels":
                        from langchain.chat_models import ChatOpenAI as ChatOpenAI_ChatModels
                        imported_components[name] = ChatOpenAI_ChatModels
                    elif name in ["RetrievalQA", "create_retrieval_chain", 'RetrievalQA.invoke', "RunnableBinding", "RunnableSequence","ConversationalRetrievalChain"]:
                        from langchain.chains import create_retrieval_chain, RetrievalQA
                        from langchain_core.runnables import RunnableBinding
                        from langchain_core.runnables import RunnableSequence
                        from langchain.chains import ConversationalRetrievalChain
                        imported_components["RetrievalQA"] = RetrievalQA
                        imported_components["create_retrieval_chain"] = create_retrieval_chain
                        imported_components["RunnableBinding"] = RunnableBinding
                        imported_components["RunnableSequence"] = RunnableSequence
                        imported_components["ConversationalRetrievalChain"] = ConversationalRetrievalChain
                except ImportError:
                    logger.debug(f"{name} not available for restoration")

        for name, original in self._original_inits.items():
            try:
                if name in imported_components:
                    component = imported_components[name]
                    component.__init__ = original
            except Exception as e:
                logger.error(f"Error restoring {name}: {e}")
                self.on_error(e, context=f"restore_{name}")

        # Restore original methods and functions
        for name, original in self._original_methods.items():
            try:
                if "." in name:
                    module_name, method_name = name.rsplit(".", 1)
                    if module_name in imported_components:
                        module = imported_components[module_name]
                        setattr(module, method_name, original)
                else:
                    if name in imported_components:
                        globals()[name] = original
            except Exception as e:
                logger.error(f"Error restoring {name}: {e}")
                self.on_error(e, context=f"restore_{name}")

    def start(self):
        """Start tracing with enhanced error handling and async support"""
        try:
            self.reset_trace()
            self.current_trace["start_time"] = datetime.now()
            self._active = True
            self._monkey_patch()

            if self.save_interval:
                loop = asyncio.get_event_loop()
                self._save_task = loop.create_task(self._periodic_save())

            logger.info("Tracing started")
        except Exception as e:
            logger.error(f"Error starting tracer: {e}")
            self.on_error(e, context="start")
            raise

    def stop(self):
        """Stop tracing with enhanced cleanup"""
        try:
            self._active = False
            if self._save_task:
                self._save_task.cancel()
            self._restore_original_methods()
            # self._save_trace(force=True)

            return self.current_trace.copy(), self.additional_metadata

            logger.info("Tracing stopped")
        except Exception as e:
            logger.error(f"Error stopping tracer: {e}")
            self.on_error(e, context="stop")
            raise
        finally:
            self._original_inits.clear()
            self._original_methods.clear()

    def force_save(self):
        """Force save the current trace"""
        self._save_trace(force=True)

    # Callback methods with enhanced error handling and logging
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.current_trace["start_time"]:
                self.current_trace["start_time"] = datetime.now()

            self.current_trace["llm_calls"].append(
                {
                    "timestamp": datetime.now(),
                    "event": "llm_start",
                    "serialized": serialized,
                    "prompts": prompts,
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="llm_start")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self.current_trace["llm_calls"].append(
                {
                    "timestamp": datetime.now(),
                    "event": "llm_end",
                    "response": response.dict(),
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )

            # Calculate latency
            end_time = datetime.now()
            latency = (end_time - self.current_trace["start_time"]).total_seconds()

            # Check if values are there in llm_output
            model = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            # Try to get model name from llm_output first
            if response and response.llm_output:
                try:
                    model = response.llm_output.get("model_name")
                    if not model:
                        model = response.llm_output.get("model", "")
                except Exception as e:
                    # logger.debug(f"Error getting model name: {e}")
                    model = ""

            # Add model name
            if not model:
                try:
                    model = response.llm_output.get("model_name")
                    if not model:
                        model = response.llm_output.get("model", "")
                except Exception as e:
                    # logger.debug(f"Error getting model name: {e}")
                    model = ""


            # Add token usage
            try:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage=={}:
                    try:
                        token_usage = response.llm_output.get("usage")
                    except Exception as e:
                        # logger.debug(f"Error getting token usage: {e}")
                        token_usage = {}
                    
                if token_usage !={}:
                    prompt_tokens = token_usage.get("prompt_tokens", 0)
                    if prompt_tokens==0:
                        prompt_tokens = token_usage.get("input_tokens", 0)
                    completion_tokens = token_usage.get("completion_tokens", 0)
                    if completion_tokens==0:
                        completion_tokens = token_usage.get("output_tokens", 0)

                    total_tokens = prompt_tokens + completion_tokens
            except Exception as e:
                # logger.debug(f"Error getting token usage: {e}")
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            # Check if values are there in 
            if prompt_tokens == 0 and completion_tokens == 0:
                try:
                    usage_data = response.generations[0][0].message.usage_metadata
                    prompt_tokens = usage_data.get("input_tokens", 0)
                    completion_tokens = usage_data.get("output_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens
                except Exception as e:
                    # logger.debug(f"Error getting usage data: {e}")
                    try:
                        usage_data = response.generations[0][0].generation_info['usage_metadata']
                        prompt_tokens = usage_data.get("prompt_token_count", 0)
                        completion_tokens = usage_data.get("candidates_token_count", 0)
                        total_tokens = prompt_tokens + completion_tokens
                    except Exception as e:
                        # logger.debug(f"Error getting token usage: {e}")
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0

            # If no model name in llm_output, try to get it from stored model names
            try:
                if model == "":
                    model = list(self.model_names.values())[0]
            except Exception as e:
                model=""

            self.additional_metadata = {
                'latency': latency,
                'model_name': model,
                'tokens': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': total_tokens
                }
            }

        except Exception as e:
            self.on_error(e, context="llm_end")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            messages_dict = [
                [
                    {
                        "type": msg.type,
                        "content": msg.content,
                        "additional_kwargs": msg.additional_kwargs,
                    }
                    for msg in batch
                ]
                for batch in messages
            ]

            self.current_trace["chat_model_calls"].append(
                {
                    "timestamp": datetime.now(),
                    "event": "chat_model_start",
                    "serialized": serialized,
                    "messages": messages_dict,
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="chat_model_start")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            context = ""
            query = ""
            if isinstance(inputs, dict):
                if "context" in inputs:
                    if isinstance(inputs["context"], Document):
                        context = inputs["context"].page_content
                    elif isinstance(inputs["context"], list):
                        context = "\n".join(
                            doc.page_content if isinstance(doc, Document) else str(doc)
                            for doc in inputs["context"]
                        )
                    elif isinstance(inputs["context"], str):
                        context = inputs["context"]

                query = inputs.get("question", inputs.get("input", ""))
                
                # Set the current query
                self._current_query = query
                
                chain_event = {
                    "timestamp": datetime.now(),
                    "serialized": serialized,
                    "context": context,
                    "query": inputs.get("question", inputs.get("input", "")),
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }

                self.current_trace["chain_starts"].append(chain_event)
        except Exception as e:
            self.on_error(e, context="chain_start")

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            self.current_trace["chain_ends"].append(
                {
                    "timestamp": datetime.now(),
                    "outputs": outputs,
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="chain_end")

    def on_agent_action(self, action: AgentAction, run_id: UUID, **kwargs: Any) -> None:
        try:
            self.current_trace["agent_actions"].append(
                {
                    "timestamp": datetime.now(),
                    "action": action.dict(),
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="agent_action")

    def on_agent_finish(self, finish: AgentFinish, run_id: UUID, **kwargs: Any) -> None:
        try:
            self.current_trace["agent_actions"].append(
                {
                    "timestamp": datetime.now(),
                    "event": "agent_finish",
                    "finish": finish.dict(),
                    "run_id": str(run_id),
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="agent_finish")

    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            retriever_event = {
                "timestamp": datetime.now(),
                "event": "retriever_start",
                "serialized": serialized,
                "query": query,
                "run_id": str(run_id),
                "additional_kwargs": kwargs,
            }

            self.current_trace["retriever_actions"].append(retriever_event)
        except Exception as e:
            self.on_error(e, context="retriever_start")

    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        try:
            processed_documents = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]

            retriever_event = {
                "timestamp": datetime.now(),
                "event": "retriever_end",
                "documents": processed_documents,
                "run_id": str(run_id),
                "additional_kwargs": kwargs,
            }

            self.current_trace["retriever_actions"].append(retriever_event)
        except Exception as e:
            self.on_error(e, context="retriever_end")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        try:
            self.current_trace["tokens"].append(
                {
                    "timestamp": datetime.now(),
                    "event": "new_token",
                    "token": token,
                    "additional_kwargs": kwargs,
                }
            )
        except Exception as e:
            self.on_error(e, context="llm_new_token")

    def on_error(self, error: Exception, context: str = "", **kwargs: Any) -> None:
        """Enhanced error handling with context"""
        try:
            error_event = {
                "timestamp": datetime.now(),
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context,
                "additional_kwargs": kwargs,
            }
            self.current_trace["errors"].append(error_event)
            logger.error(f"Error in {context}: {error}")
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self.on_error(error, context="chain", **kwargs)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.on_error(error, context="llm", **kwargs)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.on_error(error, context="tool", **kwargs)

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> None:
        self.on_error(error, context="retriever", **kwargs)
