import os
import datetime
import logging
import asyncio
import aiohttp
import requests
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

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
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing import AgenticTracing, TrackName
from ragaai_catalyst.tracers.agentic_tracing.tracers.llm_tracer import LLMTracerMixin

logger = logging.getLogger(__name__)


class Tracer(AgenticTracing):
    NUM_PROJECTS = 100
    TIMEOUT = 10
    def __init__(
        self,
        project_name,
        dataset_name,
        tracer_type=None,
        pipeline=None,
        metadata=None,
        description=None,
        upload_timeout=30,  # Default timeout of 30 seconds
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

        Returns:
            None
        """
        # Set auto_instrument_llm to True to enable automatic LLM tracing
        user_detail = {
            "project_name": project_name,
            "project_id": None,  # Will be set after project validation
            "dataset_name": dataset_name,
            "trace_user_detail": {"metadata": metadata} if metadata else {}
        }
        super().__init__(user_detail=user_detail, auto_instrument_llm=True)
        self.is_active = True
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
        self.timeout = 10
        self.num_projects = 100
        self.start_time = datetime.datetime.now(datetime.timezone.utc)

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
            self.raga_client = RagaExporter(project_name=self.project_name, dataset_name=self.dataset_name)

            self._tracer_provider = self._setup_provider()
            self._instrumentor = self._setup_instrumentor(tracer_type)
            self.is_instrumented = False
            self._upload_task = None
        elif tracer_type == "llamaindex":
            self._upload_task = None
            from ragaai_catalyst.tracers.llamaindex_callback import LlamaIndexTracer

        else:
            self._upload_task = None
            # raise ValueError (f"Currently supported tracer types are 'langchain' and 'llamaindex'.")
        

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
            if not self.is_instrumented:
                self._instrumentor().instrument(tracer_provider=self._tracer_provider)
                self.is_instrumented = True
            print(f"Tracer started for project: {self.project_name}")
            return self
        elif self.tracer_type == "llamaindex":
            from ragaai_catalyst.tracers.llamaindex_callback import LlamaIndexTracer
            return LlamaIndexTracer(self._pass_user_data()).start()
        else:
            super().start()
            return self

    def stop(self):
        """Stop the tracer and initiate trace upload."""
        if self.tracer_type == "langchain":
            if not self.is_instrumented:
                logger.warning("Tracer was not started. No traces to upload.")
                return "No traces to upload"

            print("Stopping tracer and initiating trace upload...")
            self._cleanup()
            self._upload_task = self._run_async(self._upload_traces())
            return "Trace upload initiated. Use get_upload_status() to check the status."
        elif self.tracer_type == "llamaindex":
            from ragaai_catalyst.tracers.llamaindex_callback import LlamaIndexTracer
            return LlamaIndexTracer(self._pass_user_data()).stop()
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