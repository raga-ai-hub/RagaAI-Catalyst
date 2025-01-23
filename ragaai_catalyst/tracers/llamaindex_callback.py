from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
import uuid
import os
import requests
import tempfile

from ..ragaai_catalyst import RagaAICatalyst

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


class LlamaIndexTracer:
    def __init__(self, user_detail):
        self.trace_handler = None
        self.callback_manager = (
            CallbackManager()
        )  # Ensure callback manager is initialized
        self._original_inits = {}  # Store original __init__ methods
        self.project_name = user_detail["project_name"]
        self.project_id = user_detail["project_id"]
        self.dataset_name = user_detail["dataset_name"]
        self.user_detail = user_detail["trace_user_detail"]
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 10
        self.query_count = 0
        self._upload_task = None

    def start(self):
        """Start tracing - call this before your LlamaIndex operations"""
        outer_self = self  # Capture outer self reference for inner class

        class CustomTraceHandler(LlamaDebugHandler):
            def __init__(self):
                super().__init__()
                self.traces: List[Dict[str, Any]] = []
                self.current_query_traces: List[Dict[str, Any]] = []
                self.in_query = False
                self.query_event_id = None

            def on_event_start(
                self,
                event_type: Optional[str],
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                parent_id: str = "",
                **kwargs: Any
            ) -> None:
                trace = {
                    "event_type": event_type,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "payload": payload,
                    "status": "started",
                    "event_id": event_id,
                    "parent_id": parent_id,
                }
                if event_type == "query":
                    self.in_query = True
                    self.query_event_id = event_id
                    self.current_query_traces = []
                
                if self.in_query:
                    self.current_query_traces.append(trace)
                self.traces.append(trace)

            def on_event_end(
                self,
                event_type: Optional[str],
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                **kwargs: Any
            ) -> None:
                trace = {
                    "event_type": event_type,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "payload": payload,
                    "status": "completed",
                    "event_id": event_id,
                }
                if self.in_query:
                    self.current_query_traces.append(trace)
                self.traces.append(trace)
                
                # If this is the end of a query event, automatically save the traces
                if event_type == "query" and event_id == self.query_event_id:
                    self.in_query = False
                    outer_self._save_current_query_traces(self.current_query_traces)
                    self.current_query_traces = []
                

        self.trace_handler = CustomTraceHandler()
        self.callback_manager.add_handler(self.trace_handler)
        Settings.callback_manager = self.callback_manager


        # Monkey-patch LlamaIndex components
        self._monkey_patch()
        return self  # Return self to allow method chaining


    def _save_current_query_traces(self, query_traces):
        """Save traces for the current query"""
        self.query_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trace_query_{self.query_count}_{timestamp}.json"

        traces = self._add_traces_in_data(query_traces)

        # Write the tracer json files to a temporary directory
        temp_dir = tempfile.gettempdir()
        temp_file_path = f"{temp_dir}/{filename}"

        with open(temp_file_path, "w") as f:
            json.dump([traces], f, indent=2, cls=CustomEncoder)
        # print(f"Query traces saved to {temp_file_path}")

        # Upload the traces
        self._create_dataset_schema_with_trace()
        presignedUrl = self._get_presigned_url()
        self._put_presigned_url(presignedUrl, temp_file_path)
        self._insert_traces(presignedUrl)
        # print(f"Query {self.query_count} traces uploaded")


    def _monkey_patch(self):
        """Monkey-patch LlamaIndex components to automatically include the callback manager"""
        from llama_index.core import VectorStoreIndex, ServiceContext
        from llama_index.llms.openai import OpenAI

        # Import any other classes you need to patch here

        def make_new_init(original_init, callback_manager):
            def new_init(self, *args, **kwargs):
                # If 'callback_manager' is not provided, inject our tracer's callback manager
                if "callback_manager" not in kwargs:
                    kwargs["callback_manager"] = callback_manager
                original_init(self, *args, **kwargs)

            return new_init

        # Monkey-patch VectorStoreIndex
        self._original_inits["VectorStoreIndex"] = VectorStoreIndex.__init__
        VectorStoreIndex.__init__ = make_new_init(
            VectorStoreIndex.__init__, self.callback_manager
        )

        # Monkey-patch OpenAI LLM
        self._original_inits["OpenAI"] = OpenAI.__init__
        OpenAI.__init__ = make_new_init(OpenAI.__init__, self.callback_manager)

        # Monkey-patch ServiceContext
        self._original_inits["ServiceContext"] = ServiceContext.__init__
        ServiceContext.__init__ = make_new_init(
            ServiceContext.__init__, self.callback_manager
        )

        # To monkey-patch additional classes:
        # 1. Import the class you want to patch
        # from llama_index.some_module import SomeOtherClass

        # 2. Store the original __init__ method
        # self._original_inits['SomeOtherClass'] = SomeOtherClass.__init__

        # 3. Replace the __init__ method with the new one that injects the callback manager
        # SomeOtherClass.__init__ = make_new_init(SomeOtherClass.__init__, self.callback_manager)

        # Repeat steps 1-3 for each additional class you wish to monkey-patch

    def stop(self):
        """Stop tracing and restore original methods"""
        # self._upload_traces(save_json_to_pwd=True)
        self.callback_manager.remove_handler(self.trace_handler)
        self._restore_original_inits()
        print("Traces uploaded")
        self._upload_task = True

    def _restore_original_inits(self):
        """Restore the original __init__ methods of LlamaIndex components"""
        from llama_index.core import VectorStoreIndex, ServiceContext
        from llama_index.llms.openai import OpenAI

        # Import any other classes you patched

        # Restore VectorStoreIndex
        if "VectorStoreIndex" in self._original_inits:
            VectorStoreIndex.__init__ = self._original_inits["VectorStoreIndex"]

        # Restore OpenAI
        if "OpenAI" in self._original_inits:
            OpenAI.__init__ = self._original_inits["OpenAI"]

        # Restore ServiceContext
        if "ServiceContext" in self._original_inits:
            ServiceContext.__init__ = self._original_inits["ServiceContext"]

        # To restore additional classes:
        # Check if the class was patched, then restore the original __init__
        # if 'SomeOtherClass' in self._original_inits:
        #     SomeOtherClass.__init__ = self._original_inits['SomeOtherClass']

    def _generate_trace_id(self):
        """
        Generate a random trace ID using UUID4.
        Returns a string representation of the UUID with no hyphens.
        """
        return '0x'+str(uuid.uuid4()).replace('-', '')

    def _get_user_passed_detail(self):
        user_detail = self.user_detail
        user_detail["trace_id"] = self._generate_trace_id()
        metadata = user_detail["metadata"]
        metadata["log_source"] = "llamaindex_tracer"
        metadata["recorded_on"] = datetime.now().isoformat()
        user_detail["metadata"] = metadata
        return user_detail
    
    def _add_traces_in_data(self, traces=None):
        """Add traces to user detail"""
        user_detail = self._get_user_passed_detail()
        if traces is None:
            if not self.trace_handler:
                raise RuntimeError("No traces available. Did you call start()?")
            traces = self.trace_handler.traces
        user_detail["traces"] = traces
        return user_detail


    def _create_dataset_schema_with_trace(self):
        SCHEMA_MAPPING_NEW = {
            "trace_id": {"columnType": "traceId"},
            "trace_uri": {"columnType": "traceUri"},
            "prompt": {"columnType": "prompt"},
            "response":{"columnType": "response"},
            "context": {"columnType": "context"},
            "llm_model": {"columnType":"pipeline"},
            "recorded_on": {"columnType": "metadata"},
            "embed_model": {"columnType":"pipeline"},
            "log_source": {"columnType": "metadata"},
            "vector_store":{"columnType":"pipeline"},
            "feedback": {"columnType":"feedBack"}
        }
        def make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            payload = json.dumps({
                "datasetName": self.dataset_name,
                "schemaMapping": SCHEMA_MAPPING_NEW,
                "traceFolderUrl": None,
            })
            response = requests.request("POST",
                f"{self.base_url}/v1/llm/dataset/logs",
                headers=headers,
                data=payload,
                timeout=self.timeout
            )

            return response

        response = make_request()

        if response.status_code == 401:
            # get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return response.status_code
        return response.status_code
    
    def _get_presigned_url(self):
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "numFiles": 1,
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }

        response = requests.request("GET", 
                                    f"{self.base_url}/v1/llm/presigned-url", 
                                    headers=headers, 
                                    data=payload,
                                    timeout=self.timeout)
        if response.status_code == 200:
            presignedUrls = response.json()["data"]["presignedUrls"][0]
            return presignedUrls
        
    def _put_presigned_url(self, presignedUrl, filename):
        headers = {
                "Content-Type": "application/json",
            }

        if "blob.core.windows.net" in presignedUrl:  # Azure
            headers["x-ms-blob-type"] = "BlockBlob"
        print(f"Uploading traces...")
        with open(filename) as f:
            payload = f.read().replace("\n", "").replace("\r", "").encode()


        response = requests.request("PUT", 
                                    presignedUrl, 
                                    headers=headers, 
                                    data=payload,
                                    timeout=self.timeout)
        if response.status_code != 200 or response.status_code != 201:
            return response, response.status_code
    
    def _insert_traces(self, presignedUrl):
        headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
                "X-Project-Name": self.project_name,
            }
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "presignedUrl": presignedUrl,
            })
        response = requests.request("POST", 
                                    f"{self.base_url}/v1/llm/insert/trace", 
                                    headers=headers, 
                                    data=payload,
                                    timeout=self.timeout)
        

    def _upload_traces(self, save_json_to_pwd=None):
        """Save traces to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{timestamp}.json"

        traces = self._add_traces_in_data()

        if save_json_to_pwd:
            with open(filename, "w") as f:
                json.dump([traces], f, indent=2, cls=CustomEncoder)
        print(f"tracer is saved to {filename}")

        self._create_dataset_schema_with_trace()
        presignedUrl = self._get_presigned_url()
        self._put_presigned_url(presignedUrl, filename)
        self._insert_traces(presignedUrl)
        print("Traces uploaded")

    def get_upload_status(self):
        """Check the status of the trace upload."""
        if self._upload_task is None:
            return "No upload task in progress."
        if self._upload_task:
            return "Upload completed"
