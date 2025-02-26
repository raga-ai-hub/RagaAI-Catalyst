import os
import json
import re
import requests
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import RagaAICatalyst

def create_dataset_schema_with_trace(project_name, dataset_name, base_url=None):
    def make_request():
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        payload = json.dumps({
            "datasetName": dataset_name,
            "traceFolderUrl": None,
        })
        # Use provided base_url or fall back to default
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        response = requests.request("POST",
            f"{url_base}/v1/llm/dataset/logs",
            headers=headers,
            data=payload,
            timeout=10
        )
        return response
    response = make_request()
    return response