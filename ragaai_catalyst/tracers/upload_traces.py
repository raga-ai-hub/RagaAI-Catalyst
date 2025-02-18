import requests
import json
import os
from datetime import datetime


class UploadTraces:
    def __init__(self, 
                 json_file_path,
                 project_name,
                 project_id,
                 dataset_name,
                 user_detail,
                 base_url):
        self.json_file_path = json_file_path
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_detail = user_detail
        self.base_url = base_url
        self.timeout = 10

    def _create_dataset_schema_with_trace(self, additional_metadata_keys=None, additional_pipeline_keys=None):
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

        if additional_metadata_keys:
            for key in additional_metadata_keys:
                if key == "model_name":
                    SCHEMA_MAPPING_NEW['response']["modelName"] = additional_metadata_keys[key]
                else:
                    SCHEMA_MAPPING_NEW[key] = {"columnType": key, "parentColumn": "response"}

        if additional_pipeline_keys:
            for key in additional_pipeline_keys:
                SCHEMA_MAPPING_NEW[key] = {"columnType": "pipeline"}
                
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

    def upload_traces(self, additional_metadata_keys=None, additional_pipeline_keys=None):
        try:
            self._create_dataset_schema_with_trace(additional_metadata_keys, additional_pipeline_keys)
            presignedUrl = self._get_presigned_url()
            if presignedUrl is None:
                return
            self._put_presigned_url(presignedUrl, self.json_file_path)
            self._insert_traces(presignedUrl)
            print("Traces uploaded")
        except Exception as e:
            print(f"Error while uploading agentic traces: {e}")