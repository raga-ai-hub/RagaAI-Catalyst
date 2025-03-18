import os
import uuid
import datetime
import logging
import asyncio
import aiohttp
import requests
from litellm import model_cost

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from ragaai_catalyst.tracers.langchain_callback import LangchainTracer
from ragaai_catalyst.tracers.utils.convert_langchain_callbacks_output import convert_langchain_callbacks_output

from ragaai_catalyst.tracers.utils.langchain_tracer_extraction_logic import langchain_tracer_extraction
from ragaai_catalyst.tracers.upload_traces import UploadTraces
import tempfile
import json
import numpy as np
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from ragaai_catalyst.tracers.exporters.file_span_exporter import FileSpanExporter
from ragaai_catalyst.tracers.exporters.raga_exporter import RagaExporter
from ragaai_catalyst.tracers.instrumentators import (
    LangchainInstrumentor,
    OpenAIInstrumentor,
    LlamaIndexInstrumentor,
)
from ragaai_catalyst.tracers.utils import get_unique_key
# from ragaai_catalyst.tracers.llamaindex_callback import LlamaIndexTracer
from ragaai_catalyst.tracers.llamaindex_instrumentation import LlamaIndexInstrumentationTracer
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing import AgenticTracing
from ragaai_catalyst.tracers.agentic_tracing.tracers.llm_tracer import LLMTracerMixin
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter
from ragaai_catalyst.tracers.agentic_tracing.utils.file_name_tracker import TrackName

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)

class Tracer(AgenticTracing):
    NUM_PROJECTS = 99999
    TIMEOUT = 10
    def __init__(
        self,
        project_name,
        dataset_name,
        trace_name=None,
        tracer_type=None,
        pipeline=None,
        metadata=None,
        description=None,
        upload_timeout=30,  # Default timeout of 30 seconds
        update_llm_cost=True,  # Parameter to control model cost updates
        auto_instrumentation={ # to control automatic instrumentation of different components
            'llm':True,
            'tool':True,
            'agent':True,
            'user_interaction':True,
            'file_io':True,
            'network':True,
            'custom':True
        },
        interval_time=2,
        # auto_instrumentation=True/False  # to control automatic instrumentation of everything

    ):
        """
        Initializes a Tracer object. 

        Args:
            project_name (str): The name of the project.
            dataset_name (str): The name of the dataset.
            tracer_type (str, optional): The type of tracer. Defaults to None.
            pipeline (dict, optional): The pipeline configuration. Defaults to None.
            metadata (dict, optional): The metadata. Defaults to None.
            description (str, optional): The description. Defaults to None.
            upload_timeout (int, optional): The upload timeout in seconds. Defaults to 30.
            update_llm_cost (bool, optional): Whether to update model costs from GitHub. Defaults to True.
        """

        user_detail = {
            "project_name": project_name,
            "project_id": None,  # Will be set after project validation
            "dataset_name": dataset_name,
            "interval_time": interval_time,
            "trace_name": trace_name if trace_name else f"trace_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "trace_user_detail": {"metadata": metadata} if metadata else {}
        }

        # take care of auto_instrumentation
        if isinstance(auto_instrumentation, bool):
            if tracer_type.startswith("agentic/"):
                auto_instrumentation = {
                    "llm": False,
                    "tool": False,
                    "agent": False,
                    "user_interaction": False,
                    "file_io": False,
                    "network": False,
                    "custom": False
                }
            elif auto_instrumentation:
                auto_instrumentation = {
                    "llm": True,
                    "tool": True,
                    "agent": True,
                    "user_interaction": True,
                    "file_io": True,
                    "network": True,
                    "custom": True
                }
            else:
                auto_instrumentation = {
                    "llm": False,
                    "tool": False,
                    "agent": False,
                    "user_interaction": False,
                    "file_io": False,
                    "network": False,
                    "custom": False
                }
        elif isinstance(auto_instrumentation, dict):
            auto_instrumentation = {k: v for k, v in auto_instrumentation.items()}
            for key in ["llm", "tool", "agent", "user_interaction", "file_io", "network", "custom"]:
                if key not in auto_instrumentation:
                    auto_instrumentation[key] = True
        self.model_custom_cost = {}
        super().__init__(user_detail=user_detail, auto_instrumentation=auto_instrumentation)

        self.project_name = project_name
        self.dataset_name = dataset_name
        self.tracer_type = tracer_type
        self.metadata = self._improve_metadata(metadata, tracer_type)
        # self.metadata["total_cost"] = 0.0
        # self.metadata["total_tokens"] = 0
        self.pipeline = pipeline
        self.description = description
        self.upload_timeout = upload_timeout
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 30
        self.num_projects = 99999
        self.start_time = datetime.datetime.now().astimezone().isoformat()
        self.model_cost_dict = model_cost
        self.user_context = ""  # Initialize user_context to store context from add_context
        self.file_tracker = TrackName()
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/llm/projects?size={self.num_projects}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Projects list retrieved successfully")

            project_list = [
                project["name"] for project in response.json()["data"]["content"]
            ]
            if project_name not in project_list:
                raise ValueError("Project not found. Please enter a valid project name")
            
            self.project_id = [
                project["id"] for project in response.json()["data"]["content"] if project["name"] == project_name
            ][0]
            # super().__init__(user_detail=self._pass_user_data())
            # self.file_tracker = TrackName()
            self._pass_user_data()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve projects list: {e}")
            raise

        if tracer_type == "langchain":
            # self.raga_client = RagaExporter(project_name=self.project_name, dataset_name=self.dataset_name)

            # self._tracer_provider = self._setup_provider()
            # self._instrumentor = self._setup_instrumentor(tracer_type)
            # self.is_instrumented = False
            # self._upload_task = None
            self._upload_task = None
        elif tracer_type == "llamaindex":
            self._upload_task = None
            self.llamaindex_tracer = None
        # Handle agentic tracers
        elif tracer_type == "agentic" or tracer_type.startswith("agentic/"):
            
            # Setup instrumentors based on tracer type
            instrumentors = []

            # Add LLM Instrumentors
            if tracer_type in ['agentic/crewai']:
                try:
                    from openinference.instrumentation.vertexai import VertexAIInstrumentor
                    instrumentors.append((VertexAIInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("VertexAI not available in environment")
                try:
                    from openinference.instrumentation.anthropic import AnthropicInstrumentor
                    instrumentors.append((AnthropicInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("Anthropic not available in environment")
                try:
                    from openinference.instrumentation.groq import GroqInstrumentor
                    instrumentors.append((GroqInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("Groq not available in environment")
                try:
                    from openinference.instrumentation.litellm import LiteLLMInstrumentor
                    instrumentors.append((LiteLLMInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("LiteLLM not available in environment")
                try:
                    from openinference.instrumentation.mistralai import MistralAIInstrumentor
                    instrumentors.append((MistralAIInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("MistralAI not available in environment")
                try:
                    from openinference.instrumentation.openai import OpenAIInstrumentor
                    instrumentors.append((OpenAIInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("OpenAI not available in environment")
                try:
                    from openinference.instrumentation.bedrock import BedrockInstrumentor
                    instrumentors.append((BedrockInstrumentor, []))
                except (ImportError, ModuleNotFoundError):
                    logger.debug("Bedrock not available in environment")
            
            # If tracer_type is just "agentic", try to instrument all available packages
            if tracer_type == "agentic":
                logger.info("Attempting to instrument all available agentic packages")
                
                # Try to import and add all known instrumentors
                try:
                    # LlamaIndex
                    try:
                        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
                        instrumentors.append((LlamaIndexInstrumentor, []))
                        logger.info("Instrumenting LlamaIndex...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("LlamaIndex not available in environment")
                    
                    # LangChain
                    try:
                        from openinference.instrumentation.langchain import LangChainInstrumentor
                        instrumentors.append((LangChainInstrumentor, []))
                        logger.info("Instrumenting LangChain...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("LangChain not available in environment")
                    
                    # CrewAI
                    try:
                        from openinference.instrumentation.crewai import CrewAIInstrumentor
                        instrumentors.append((CrewAIInstrumentor, []))
                        logger.info("Instrumenting CrewAI...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("CrewAI not available in environment")
                    
                    # Haystack
                    try:
                        from openinference.instrumentation.haystack import HaystackInstrumentor
                        instrumentors.append((HaystackInstrumentor, []))
                        logger.info("Instrumenting Haystack...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("Haystack not available in environment")
                    
                    # AutoGen
                    try:
                        from openinference.instrumentation.autogen import AutogenInstrumentor
                        instrumentors.append((AutogenInstrumentor, []))
                        logger.info("Instrumenting AutoGen...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("AutoGen not available in environment")
                    
                    # Smolagents
                    try:
                        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
                        instrumentors.append((SmolagentsInstrumentor, []))
                        logger.info("Instrumenting Smolagents...")
                    except (ImportError, ModuleNotFoundError):
                        logger.debug("Smolagents not available in environment")
                    
                    if not instrumentors:
                        logger.warning("No agentic packages found in environment to instrument")
                        self._upload_task = None
                        return
                    
                except Exception as e:
                    logger.error(f"Error during auto-instrumentation: {str(e)}")
                    self._upload_task = None
                    return
            
            # Handle specific framework instrumentation
            elif tracer_type == "agentic/llamaindex":
                from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
                instrumentors += [(LlamaIndexInstrumentor, [])] 
            
            elif tracer_type == "agentic/langchain" or tracer_type == "agentic/langgraph":
                from openinference.instrumentation.langchain import LangChainInstrumentor
                instrumentors += [(LangChainInstrumentor, [])]
            
            elif tracer_type == "agentic/crewai":
                from openinference.instrumentation.crewai import CrewAIInstrumentor
                from openinference.instrumentation.langchain import LangChainInstrumentor
                instrumentors += [(CrewAIInstrumentor, []), (LangChainInstrumentor, [])]
            
            elif tracer_type == "agentic/haystack":
                from openinference.instrumentation.haystack import HaystackInstrumentor
                instrumentors += [(HaystackInstrumentor, [])]
            
            elif tracer_type == "agentic/autogen":
                from openinference.instrumentation.autogen import AutogenInstrumentor
                instrumentors += [(AutogenInstrumentor, [])]
            
            elif tracer_type == "agentic/smolagents":
                from openinference.instrumentation.smolagents import SmolagentsInstrumentor
                instrumentors += [(SmolagentsInstrumentor, [])]
            
            else:
                # Unknown agentic tracer type
                logger.warning(f"Unknown agentic tracer type: {tracer_type}")
                self._upload_task = None
                return
                
            # Common setup for all agentic tracers
            self._setup_agentic_tracer(instrumentors)
        else:
            self._upload_task = None
            # raise ValueError (f"Currently supported tracer types are 'langchain' and 'llamaindex'.")

    def set_model_cost(self, cost_config):
        """
        Set custom cost values for a specific model.

        Args:
            cost_config (dict): Dictionary containing model cost configuration with keys:
                - model_name (str): Name of the model
                - input_cost_per_token (float): Cost per input token
                - output_cost_per_token (float): Cost per output token

        Example:
            tracer.set_model_cost({
                "model_name": "gpt-4",
                "input_cost_per_million_token": 6,
                "output_cost_per_million_token": 2.40
            })
        """
        if not isinstance(cost_config, dict):
            raise TypeError("cost_config must be a dictionary")

        required_keys = {"model_name", "input_cost_per_million_token", "output_cost_per_million_token"}
        if not all(key in cost_config for key in required_keys):
            raise ValueError(f"cost_config must contain all required keys: {required_keys}")

        model_name = cost_config["model_name"]
        self.model_custom_cost[model_name] = {
            "input_cost_per_token": float(cost_config["input_cost_per_million_token"])/ 1000000,
            "output_cost_per_token": float(cost_config["output_cost_per_million_token"]) /1000000
        }

    def set_dataset_name(self, dataset_name):
        """
        Reinitialize the Tracer with a new dataset name while keeping all other parameters the same.
        If using agentic/llamaindex tracer with dynamic exporter, update the exporter's dataset_name property.
        
        Args:
            dataset_name (str): The new dataset name to set
        """
        # If we have a dynamic exporter, update its dataset_name property
        if self.tracer_type == "agentic/llamaindex" and hasattr(self, "dynamic_exporter"):
            # Update the dataset name in the dynamic exporter
            self.dynamic_exporter.dataset_name = dataset_name
            logger.debug(f"Updated dynamic exporter's dataset_name to {dataset_name}")
            
            # Update the instance variable
            self.dataset_name = dataset_name
            
            # Update user_details with new dataset_name
            self.user_details = self._pass_user_data()
            
            # Also update the user_details in the dynamic exporter
            self.dynamic_exporter.user_details = self.user_details
        else:
            # Store current parameters
            current_params = {
                'project_name': self.project_name,
                'tracer_type': self.tracer_type,
                'pipeline': self.pipeline,
                'metadata': self.metadata,
                'description': self.description,
                'upload_timeout': self.upload_timeout
            }
            
            # Reinitialize self with new dataset_name and stored parameters
            self.__init__(
                dataset_name=dataset_name,
                **current_params
            )

    def _improve_metadata(self, metadata, tracer_type):
        if metadata is None:
            metadata = {}
        metadata.setdefault("log_source", f"{tracer_type}_tracer")
        metadata.setdefault("recorded_on", str(datetime.datetime.now()))
        return metadata

    def _add_unique_key(self, data, key_name):
        data[key_name] = get_unique_key(data)
        return data

    def _setup_provider(self):
        self.filespanx = FileSpanExporter(
            project_name=self.project_name,
            metadata=self.metadata,
            pipeline=self.pipeline,
            raga_client=self.raga_client,
        )
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(self.filespanx))
        return tracer_provider

    def _setup_instrumentor(self, tracer_type):
        instrumentors = {
            "langchain": LangchainInstrumentor,
            "openai": OpenAIInstrumentor,
            "llama_index": LlamaIndexInstrumentor,
        }
        if tracer_type not in instrumentors:
            raise ValueError(f"Invalid tracer type: {tracer_type}")
        return instrumentors[tracer_type]().get()

    @contextmanager
    def trace(self):
        """
        Synchronous context manager for tracing.
        Usage:
            with tracer.trace():
                # Your code here
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self):
        """Start the tracer."""
        if self.tracer_type == "langchain":
            # if not self.is_instrumented:
            #     self._instrumentor().instrument(tracer_provider=self._tracer_provider)
            #     self.is_instrumented = True
            # print(f"Tracer started for project: {self.project_name}")
            self.langchain_tracer = LangchainTracer()
            return self.langchain_tracer.start()
        elif self.tracer_type == "llamaindex":
            self.llamaindex_tracer = LlamaIndexInstrumentationTracer(self._pass_user_data())
            return self.llamaindex_tracer.start()
        else:
            super().start()
            return self

    def stop(self):
        """Stop the tracer and initiate trace upload."""
        if self.tracer_type == "langchain":
            # if not self.is_instrumented:
            #     logger.warning("Tracer was not started. No traces to upload.")
            #     return "No traces to upload"

            # print("Stopping tracer and initiating trace upload...")
            # self._cleanup()
            # self._upload_task = self._run_async(self._upload_traces())
            # self.is_active = False
            # self.dataset_name = None
            
            user_detail = self._pass_user_data()
            data, additional_metadata = self.langchain_tracer.stop()

            # Add cost if possible
            if additional_metadata.get('model_name'):
                try:
                    model_cost_data = self.model_cost_dict[additional_metadata['model_name']]
                    if 'tokens' in additional_metadata and all(k in additional_metadata['tokens'] for k in ['prompt', 'completion']):
                        prompt_cost = additional_metadata["tokens"]["prompt"]*model_cost_data["input_cost_per_token"]
                        completion_cost = additional_metadata["tokens"]["completion"]*model_cost_data["output_cost_per_token"]
                        additional_metadata["cost"] = prompt_cost + completion_cost 

                        additional_metadata["prompt_tokens"] = float(additional_metadata["tokens"].get("prompt", 0.0))
                        additional_metadata["completion_tokens"] = float(additional_metadata["tokens"].get("completion", 0.0))

                        logger.debug("Metadata added successfully")
                    else:
                        logger.warning("Token information missing in additional_metadata")

                    if 'cost' in additional_metadata:
                        additional_metadata["cost"] = float(additional_metadata["cost"])
                    else:
                        additional_metadata["cost"] = 0.0
                        logger.warning("Total cost information not available")


                except Exception as e:
                    logger.warning(f"Error adding cost: {e}")
            else:
                logger.debug("Model name not available in additional_metadata, skipping cost calculation")
            

            # Safely remove tokens and cost dictionaries if they exist
            additional_metadata.pop("tokens", None)
            # additional_metadata.pop("cost", None)
            
            # Safely merge metadata
            combined_metadata = {}
            if user_detail.get('trace_user_detail', {}).get('metadata'):
                combined_metadata.update(user_detail['trace_user_detail']['metadata'])
            if additional_metadata:
                combined_metadata.update(additional_metadata)

            langchain_traces = langchain_tracer_extraction(data, self.user_context)
            final_result = convert_langchain_callbacks_output(langchain_traces)
            
            # Safely set required fields in final_result
            if final_result and isinstance(final_result, list) and len(final_result) > 0:
                final_result[0]['project_name'] = user_detail.get('project_name', '')
                final_result[0]['trace_id'] = str(uuid.uuid4())
                final_result[0]['session_id'] = None
                final_result[0]['metadata'] = combined_metadata
                final_result[0]['pipeline'] = user_detail.get('trace_user_detail', {}).get('pipeline')

                filepath_3 = os.path.join(os.getcwd(), "final_result.json")
                with open(filepath_3, 'w') as f:
                    json.dump(final_result, f, indent=2)
                
                # print(filepath_3)
            else:
                logger.warning("No valid langchain traces found in final_result")

            # additional_metadata_keys = list(additional_metadata.keys()) if additional_metadata else None
            additional_metadata_dict = additional_metadata if additional_metadata else {}

            UploadTraces(json_file_path=filepath_3,
                         project_name=self.project_name,
                         project_id=self.project_id,
                         dataset_name=self.dataset_name,
                         user_detail=self._pass_user_data(),
                         base_url=self.base_url
                         ).upload_traces(additional_metadata_keys=additional_metadata_dict)
            
            return 

        elif self.tracer_type == "llamaindex":
            if self.llamaindex_tracer is None:
                raise ValueError("LlamaIndex tracer was not started")

            user_detail = self._pass_user_data()
            converted_back_to_callback = self.llamaindex_tracer.stop()

            filepath_3 = os.path.join(os.getcwd(), "llama_final_result.json")
            with open(filepath_3, 'w') as f:
                json.dump(converted_back_to_callback, f, default=str, indent=2)

            if converted_back_to_callback:
                UploadTraces(json_file_path=filepath_3,
                             project_name=self.project_name,
                             project_id=self.project_id,
                             dataset_name=self.dataset_name,
                             user_detail=user_detail,
                             base_url=self.base_url
                             ).upload_traces()
            return 
        else:
            super().stop()

    def get_upload_status(self):
        """Check the status of the trace upload."""
        if self.tracer_type == "langchain":
            if self._upload_task is None:
                return "No upload task in progress."
            if self._upload_task.done():
                try:
                    result = self._upload_task.result()
                    return f"Upload completed: {result}"
                except Exception as e:
                    return f"Upload failed: {str(e)}"
            return "Upload in progress..."

    def _run_async(self, coroutine):
        """Run an asynchronous coroutine in a separate thread."""
        loop = asyncio.new_event_loop()
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: loop.run_until_complete(coroutine))
        return future

    async def _upload_traces(self):
        """
        Asynchronously uploads traces to the RagaAICatalyst server.

        This function uploads the traces generated by the RagaAICatalyst client to the RagaAICatalyst server. It uses the `aiohttp` library to make an asynchronous HTTP request to the server. The function first checks if the `RAGAAI_CATALYST_TOKEN` environment variable is set. If not, it raises a `ValueError` with the message "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces.".

        The function then uses the `asyncio.wait_for` function to wait for the `check_and_upload_files` method of the `raga_client` object to complete. The `check_and_upload_files` method is called with the `session` object and a list of file paths to be uploaded. The `timeout` parameter is set to the value of the `upload_timeout` attribute of the `Tracer` object.

        If the upload is successful, the function returns the string "Files uploaded successfully" if the `upload_stat` variable is truthy, otherwise it returns the string "No files to upload".

        If the upload times out, the function returns a string with the message "Upload timed out after {self.upload_timeout} seconds".

        If any other exception occurs during the upload, the function returns a string with the message "Upload failed: {str(e)}", where `{str(e)}` is the string representation of the exception.

        Parameters:
            None

        Returns:
            A string indicating the status of the upload.
        """
        async with aiohttp.ClientSession() as session:
            if not os.getenv("RAGAAI_CATALYST_TOKEN"):
                raise ValueError(
                    "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces."
                )

            try:
                upload_stat = await asyncio.wait_for(
                    self.raga_client.check_and_upload_files(
                        session=session,
                        file_paths=[self.filespanx.sync_file],
                    ),
                    timeout=self.upload_timeout,
                )
                return (
                    "Files uploaded successfully"
                    if upload_stat
                    else "No files to upload"
                )
            except asyncio.TimeoutError:
                return f"Upload timed out after {self.upload_timeout} seconds"
            except Exception as e:
                return f"Upload failed: {str(e)}"

    def _cleanup(self):
        """
        Cleans up the tracer by uninstrumenting the instrumentor, shutting down the tracer provider,
        and resetting the instrumentation flag. This function is called when the tracer is no longer
        needed.

        Parameters:
            self (Tracer): The Tracer instance.

        Returns:
            None
        """
        if self.is_instrumented:
            try:
                self._instrumentor().uninstrument()
                self._tracer_provider.shutdown()
                self.is_instrumented = False
                print("Tracer provider shut down successfully")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {str(e)}")

        # Reset instrumentation flag
        self.is_instrumented = False
        # Note: We're not resetting all attributes here to allow for upload status checking

    def _pass_user_data(self):
        user_detail = {
            "project_name":self.project_name, 
            "project_id": self.project_id,
            "dataset_name":self.dataset_name, 
            "trace_user_detail" : {
                "project_id": self.project_id,
                "trace_id": "",
                "session_id": None,
                "trace_type": self.tracer_type,
                "traces": [],
                "metadata": self.metadata,
                "pipeline": {
                    "llm_model": (getattr(self, "pipeline", {}) or {}).get("llm_model", ""),
                    "vector_store": (getattr(self, "pipeline", {}) or {}).get("vector_store", ""),
                    "embed_model": (getattr(self, "pipeline", {}) or {}).get("embed_model", "")
                    }
                }
            }
        return user_detail

    def update_dynamic_exporter(self, **kwargs):
        """
        Update the dynamic exporter's properties.
        
        Args:
            **kwargs: Keyword arguments to update. Can include any of the following:
                - files_to_zip: List of files to zip
                - project_name: Project name
                - project_id: Project ID
                - dataset_name: Dataset name
                - user_details: User details
                - base_url: Base URL for API
                - custom_model_cost: Dictionary of custom model costs
                
        Raises:
            AttributeError: If the tracer_type is not an agentic tracer or if the dynamic_exporter is not initialized.
        """
        if not self.tracer_type.startswith("agentic/") or not hasattr(self, "dynamic_exporter"):
            raise AttributeError("This method is only available for agentic tracers with a dynamic exporter.")
            
        for key, value in kwargs.items():
            if hasattr(self.dynamic_exporter, key):
                setattr(self.dynamic_exporter, key, value)
                logger.debug(f"Updated dynamic exporter's {key} to {value}")
            else:
                logger.warning(f"Dynamic exporter has no attribute '{key}'")
                
    def _setup_agentic_tracer(self, instrumentors):
        """
        Common setup for all agentic tracers.
        
        Args:
            instrumentors (list): List of tuples (instrumentor_class, args) to be instrumented
        """
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from ragaai_catalyst.tracers.exporters.dynamic_trace_exporter import DynamicTraceExporter
        
        # Get the code_files
        self.file_tracker.trace_main_file()
        list_of_unique_files = self.file_tracker.get_unique_files()

        # Create a dynamic exporter that allows property updates
        self.dynamic_exporter = DynamicTraceExporter(
            files_to_zip=list_of_unique_files,
            project_name=self.project_name,
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            user_details=self.user_details,
            base_url=self.base_url,
            custom_model_cost=self.model_custom_cost
        )
        
        # Set up tracer provider
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(self.dynamic_exporter))
        
        # Instrument all specified instrumentors
        for instrumentor_class, args in instrumentors:
            instrumentor_class().instrument(tracer_provider=tracer_provider, *args)
            
    def update_file_list(self):
        """
        Update the file list in the dynamic exporter with the latest tracked files.
        This is useful when new files are added to the project during execution.
        
        Raises:
            AttributeError: If the tracer_type is not 'agentic/llamaindex' or if the dynamic_exporter is not initialized.
        """
        if not self.tracer_type.startswith("agentic/") or not hasattr(self, "dynamic_exporter"):
            raise AttributeError("This method is only available for agentic tracers with a dynamic exporter.")
            
        # Get the latest list of unique files
        list_of_unique_files = self.file_tracker.get_unique_files()
        
        # Update the dynamic exporter's files_to_zip property
        self.dynamic_exporter.files_to_zip = list_of_unique_files
        logger.debug(f"Updated dynamic exporter's files_to_zip with {len(list_of_unique_files)} files")
    
    def add_context(self, context):
        """
        Add context information to the trace. This method is only supported for 'langchain' and 'llamaindex' tracer types.

        Args:
            context: Additional context information to be added to the trace. Can be a string.

        Raises:
            ValueError: If tracer_type is not 'langchain' or 'llamaindex'.
        """
        if self.tracer_type not in ["langchain", "llamaindex"]:
            raise ValueError("add_context is only supported for 'langchain' and 'llamaindex' tracer types")
        
        # Convert string context to string if needed
        if isinstance(context, str):
            self.user_context = context
        else:
            raise TypeError("context must be a string")
