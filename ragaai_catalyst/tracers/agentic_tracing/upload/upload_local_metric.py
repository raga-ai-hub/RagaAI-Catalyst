import os
from typing import Optional

import requests


def calculate_metric(project_id, metric_name, model, org_domain, provider, user_id,
                     prompt, response, context, expected_response:Optional[int] = None):
    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Id": str(project_id), #TODO to change it to project_id
        "Content-Type": "application/json"
    }

    payload = {
        "data": [
            {
                "metric_name": metric_name,
                "metric_config": {
                    "threshold": {
                        "isEditable": True,
                        "lte": 0.3
                    },
                    "model": model,
                    "orgDomain": org_domain,
                    "provider": provider,
                    "user_id": user_id,
                    "job_id": 1,
                    "metric_name": metric_name,
                    "request_id": 1
                },
                "trace_object": {
                    "Data": {
                        "DocId": "doc-1",
                        "Prompt": prompt,
                        "Response": response,
                        "Context": context,
                        "ExpectedResponse": "",
                        "ExpectedContext": expected_response,
                        "Chat": "",
                        "Instructions": "",
                        "SystemPrompt": "",
                        "Text": ""
                    },
                    "claims": {},
                    "last_computed_metrics": {
                        metric_name: {
                        }
                    }
                }
            }
        ]
    }

    try:
        BASE_URL = "http://74.225.145.109/api" #RagaAICatalyst.BASE_URL
        response = requests.post(f"{BASE_URL}/v1/llm/calculate-metric", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
