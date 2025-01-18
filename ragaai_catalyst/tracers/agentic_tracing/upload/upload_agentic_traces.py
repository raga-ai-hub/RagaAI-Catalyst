import requests
import json
import os
from datetime import datetime


class UploadAgenticTraces:
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
        self.timeout = 99999

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
                # "schemaMapping": SCHEMA_MAPPING_NEW,
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

        try:
            response = requests.request("GET", 
                                        f"{self.base_url}/v1/llm/presigned-url", 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code == 200:
                presignedUrls = response.json()["data"]["presignedUrls"][0]
                return presignedUrls
        except requests.exceptions.RequestException as e:
            print(f"Error while getting presigned url: {e}")
            return None

    def _put_presigned_url(self, presignedUrl, filename):
        headers = {
                "Content-Type": "application/json",
            }

        if "blob.core.windows.net" in presignedUrl:  # Azure
            headers["x-ms-blob-type"] = "BlockBlob"
        print(f"Uploading agentic traces...")
        try:
            with open(filename) as f:
                payload = f.read().replace("\n", "").replace("\r", "").encode()
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None
        try:
            response = requests.request("PUT", 
                                        presignedUrl, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code != 200 or response.status_code != 201:
                return response, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error while uploading to presigned url: {e}")
            return None

    def insert_traces(self, presignedUrl):
        headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
                "X-Project-Name": self.project_name,
            }
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "presignedUrl": presignedUrl,
                "datasetSpans": self._get_dataset_spans(), #Extra key for agentic traces
            })
        try:
            response = requests.request("POST", 
                                        f"{self.base_url}/v1/llm/insert/trace", 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code != 200:
                print(f"Error inserting traces: {response.json()['message']}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error while inserting traces: {e}")
            return None

    def _get_dataset_spans(self):
        try:
            with open(self.json_file_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None
        try:
            spans = data["data"][0]["spans"]
            datasetSpans = []
            for span in spans:
                if span["type"] != "agent":
                    existing_span = next((s for s in datasetSpans if s["spanHash"] == span["hash_id"]), None)
                    if existing_span is None:
                        datasetSpans.append({
                            "spanId": span["id"],
                            "spanName": span["name"],
                            "spanHash": span["hash_id"],
                            "spanType": span["type"],
                        })
                else:
                    datasetSpans.append({
                                "spanId": span["id"],
                                "spanName": span["name"],
                                "spanHash": span["hash_id"],
                                "spanType": span["type"],
                            })
                    children = span["data"]["children"]
                    for child in children:
                        existing_span = next((s for s in datasetSpans if s["spanHash"] == child["hash_id"]), None)
                        if existing_span is None:
                            datasetSpans.append({
                                "spanId": child["id"],
                                "spanName": child["name"],
                                "spanHash": child["hash_id"],
                                "spanType": child["type"],
                            })
            return datasetSpans
        except Exception as e:
            print(f"Error while reading dataset spans: {e}")
            return None
    
    def upload_agentic_traces(self):
        self._create_dataset_schema_with_trace()
        presignedUrl = self._get_presigned_url()
        if presignedUrl is None:
            return
        self._put_presigned_url(presignedUrl, self.json_file_path)
        self.insert_traces(presignedUrl)
        print("Agentic Traces uploaded")