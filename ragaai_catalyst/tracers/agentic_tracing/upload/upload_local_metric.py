import logging
import os
import requests

from ragaai_catalyst import RagaAICatalyst

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


def calculate_metric(project_id, metric_name, model, provider, **kwargs):
    user_id = "1"
    org_domain = "raga"

    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Id": str(project_id),
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
                "variable_mapping": kwargs,
                "trace_object": {
                    "Data": {
                        "DocId": "doc-1",
                        "Prompt": kwargs.get("prompt"),
                        "Response": kwargs.get("response"),
                        "Context": kwargs.get("context"),
                        "ExpectedResponse": kwargs.get("expected_response"),
                        "ExpectedContext": kwargs.get("expected_context"),
                        "Chat": kwargs.get("chat"),
                        "Instructions": kwargs.get("instructions"),
                        "SystemPrompt": kwargs.get("system_prompt"),
                        "Text": kwargs.get("text")
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
        BASE_URL = RagaAICatalyst.BASE_URL
        response = requests.post(f"{BASE_URL}/v1/llm/calculate-metric", headers=headers, json=payload, timeout=30)
        logger.debug(f"Metric calculation response status {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error in calculate-metric api: {e}, payload: {payload}")
        raise Exception(f"Error in calculate-metric: {e}")
