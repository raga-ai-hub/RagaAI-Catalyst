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
        self._current_query = None  # Add this line to track the current query
        self.filepath = None
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
            "query": self._current_query,  # Add this line to include the query in the trace
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

    def _create_safe_wrapper(self, original_func, component_name):
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

        return wrapped


    def _monkey_patch(self):
        """Enhanced monkey-patching with comprehensive component support"""
        from langchain.llms import OpenAI
        # from langchain_groq import ChatGroq
        # from langchain_google_genai import ChatGoogleGenerativeAI
        # from langchain_anthropic import ChatAnthropic
        from langchain_community.chat_models import ChatLiteLLM
        # from langchain_cohere import ChatCohere
        from langchain_openai import ChatOpenAI as ChatOpenAI_LangchainOpenAI
        from langchain.chat_models import ChatOpenAI as ChatOpenAI_ChatModels
        from langchain.chains import create_retrieval_chain, RetrievalQA

        components_to_patch = {
            "OpenAI": (OpenAI, "__init__"),
            # "ChatGroq": (ChatGroq, "__init__"),
            # "ChatGoogleGenerativeAI": (ChatGoogleGenerativeAI, "__init__"),
            # "ChatAnthropic": (ChatAnthropic, "__init__"),
            "ChatLiteLLM": (ChatLiteLLM, "__init__"),
            # "ChatCohere": (ChatCohere, "__init__"),
            "ChatOpenAI_LangchainOpenAI": (ChatOpenAI_LangchainOpenAI, "__init__"),
            "ChatOpenAI_ChatModels": (ChatOpenAI_ChatModels, "__init__"),
            "RetrievalQA": (RetrievalQA, "from_chain_type"),
            "create_retrieval_chain": (create_retrieval_chain, None),
        }

        for name, (component, method_name) in components_to_patch.items():
            try:
                if method_name == "__init__":
                    original = component.__init__
                    self._original_inits[name] = original
                    component.__init__ = self._create_safe_wrapper(original, name)
                elif method_name:
                    original = getattr(component, method_name)
                    self._original_methods[name] = original
                    if isinstance(original, classmethod):
                        wrapped = classmethod(
                            self._create_safe_wrapper(original.__func__, name)
                        )
                    else:
                        wrapped = self._create_safe_wrapper(original, name)
                    setattr(component, method_name, wrapped)
                else:
                    self._original_methods[name] = component
                    globals()[name] = self._create_safe_wrapper(component, name)
            except Exception as e:
                logger.error(f"Error patching {name}: {e}")
                self.on_error(e, context=f"patch_{name}")

    def _restore_original_methods(self):
        """Restore all original methods and functions with enhanced error handling"""
        from langchain.llms import OpenAI
        # from langchain_groq import ChatGroq
        # from langchain_google_genai import ChatGoogleGenerativeAI
        # from langchain_anthropic import ChatAnthropic
        from langchain_community.chat_models import ChatLiteLLM
        # from langchain_cohere import ChatCohere
        from langchain_openai import ChatOpenAI as ChatOpenAI_LangchainOpenAI
        from langchain.chat_models import ChatOpenAI as ChatOpenAI_ChatModels
        from langchain.chains import create_retrieval_chain, RetrievalQA


        for name, original in self._original_inits.items():
            try:
                component = eval(name)
                component.__init__ = original
            except Exception as e:
                logger.error(f"Error restoring {name}: {e}")
                self.on_error(e, context=f"restore_{name}")

        for name, original in self._original_methods.items():
            try:
                if "." in name:
                    module_name, method_name = name.rsplit(".", 1)
                    module = eval(module_name)
                    setattr(module, method_name, original)
                else:
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

            end_time = datetime.now()
            self.additional_metadata["latency"] = (end_time - self.current_trace["start_time"]).total_seconds()

            if response and response.llm_output:
                self.additional_metadata["model_name"] = response.llm_output.get("model_name", "")
                self.additional_metadata["tokens"] = {}
                if response.llm_output.get("token_usage"):
                    self.additional_metadata["tokens"]["total"] = response.llm_output["token_usage"].get("total_tokens", 0)
                    self.additional_metadata["tokens"]["prompt"] = response.llm_output["token_usage"].get("prompt_tokens", 0)
                    self.additional_metadata["tokens"]["completion"] = response.llm_output["token_usage"].get("completion_tokens", 0)
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
