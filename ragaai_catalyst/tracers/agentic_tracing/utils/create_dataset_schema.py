import os
import json
import re
import requests
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import RagaAICatalyst

def create_dataset_schema_with_trace(project_name, dataset_name):
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
        response = requests.request("POST",
            f"{RagaAICatalyst.BASE_URL}/v1/llm/dataset/logs",
            headers=headers,
            data=payload,
            timeout=10
        )
        return response
    response = make_request()
    return response