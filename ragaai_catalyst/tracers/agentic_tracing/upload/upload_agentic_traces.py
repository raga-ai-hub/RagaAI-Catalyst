import requests
import json
import os
import time
import logging
from datetime import datetime
from urllib.parse import urlparse, urlunparse
import re

logger = logging.getLogger(__name__)


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
        self.timeout = 30


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
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/presigned-url"
            response = requests.request("GET", 
                                        endpoint, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code == 200:
                presignedURLs = response.json()["data"]["presignedUrls"][0]
                presignedurl = self.update_presigned_url(presignedURLs,self.base_url)
                return presignedurl
            
        except requests.exceptions.RequestException as e:
            print(f"Error while getting presigned url: {e}")
            return None
        
    def update_presigned_url(self, presigned_url, base_url):
        """Replaces the domain (and port, if applicable) of the presigned URL 
        with that of the base URL only if the base URL contains 'localhost' or an IP address."""
        #To Do: If Proxy URL has domain name how do we handle such cases
        
        presigned_parts = urlparse(presigned_url)
        base_parts = urlparse(base_url)
        # Check if base_url contains localhost or an IP address
        if re.match(r'^(localhost|\d{1,3}(\.\d{1,3}){3})$', base_parts.hostname):
            new_netloc = base_parts.hostname  # Extract domain from base_url
            if base_parts.port:  # Add port if present in base_url
                new_netloc += f":{base_parts.port}"
            updated_parts = presigned_parts._replace(netloc=new_netloc)
            return urlunparse(updated_parts)
        return presigned_url

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
            start_time = time.time()
            response = requests.request("PUT", 
                                        presignedUrl, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [PUT] {presignedUrl} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
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
            start_time = time.time()
            endpoint = f"{self.base_url}/v1/llm/insert/trace"
            response = requests.request("POST", 
                                        endpoint, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
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
                    datasetSpans.extend(self._get_agent_dataset_spans(span, datasetSpans))
            datasetSpans = [dict(t) for t in set(tuple(sorted(d.items())) for d in datasetSpans)]
            
            return datasetSpans
        except Exception as e:
            print(f"Error while reading dataset spans: {e}")
            return None

    def _get_agent_dataset_spans(self, span, datasetSpans):
        datasetSpans.append({
                                "spanId": span["id"],
                                "spanName": span["name"],
                                "spanHash": span["hash_id"],
                                "spanType": span["type"],
                            })
        children = span["data"]["children"]
        for child in children:
            if child["type"] != "agent":
                existing_span = next((s for s in datasetSpans if s["spanHash"] == child["hash_id"]), None)
                if existing_span is None:
                    datasetSpans.append({
                        "spanId": child["id"],
                        "spanName": child["name"],
                        "spanHash": child["hash_id"],
                        "spanType": child["type"],
                    })
            else:
                datasetSpans.append({
                            "spanId": child["id"],
                            "spanName": child["name"],
                            "spanHash": child["hash_id"],
                            "spanType": child["type"],
                        })
                self._get_agent_dataset_spans(child, datasetSpans)
        return datasetSpans
        

    def upload_agentic_traces(self):
        try:
            presignedUrl = self._get_presigned_url()
            if presignedUrl is None:
                return
            self._put_presigned_url(presignedUrl, self.json_file_path)
            self.insert_traces(presignedUrl)
        except Exception as e:
            print(f"Error while uploading agentic traces: {e}")
