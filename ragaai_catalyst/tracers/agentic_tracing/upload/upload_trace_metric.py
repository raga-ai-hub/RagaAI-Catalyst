import logging

import requests
import os
import json
import time
from ....ragaai_catalyst import RagaAICatalyst
from ..utils.get_user_trace_metrics import get_user_trace_metrics

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


def upload_trace_metric(json_file_path, dataset_name, project_name, base_url=None):
    try:
        with open(json_file_path, "r") as f:
            traces = json.load(f)

        metrics = get_trace_metrics_from_trace(traces)
        metrics = _change_metrics_format_for_payload(metrics)

        user_trace_metrics = get_user_trace_metrics(project_name, dataset_name)
        if user_trace_metrics:
            user_trace_metrics_list = [metric["displayName"] for metric in user_trace_metrics]

        if user_trace_metrics:
            for metric in metrics:
                if metric["displayName"] in user_trace_metrics_list:
                    metricConfig = next((user_metric["metricConfig"] for user_metric in user_trace_metrics if
                                         user_metric["displayName"] == metric["displayName"]), None)
                    if not metricConfig or metricConfig.get("Metric Source", {}).get("value") != "user":
                        raise ValueError(
                            f"Metrics {metric['displayName']} already exist in dataset {dataset_name} of project {project_name}.")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        payload = json.dumps({
            "datasetName": dataset_name,
            "metrics": metrics
        })
        url_base = base_url if base_url is not None else RagaAICatalyst.BASE_URL
        start_time = time.time()
        endpoint = f"{url_base}/v1/llm/trace/metrics"
        response = requests.request("POST",
                                    endpoint,
                                    headers=headers,
                                    data=payload,
                                    timeout=10)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
        if response.status_code != 200:
            raise ValueError(f"Error inserting agentic trace metrics")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error submitting traces: {e}")
        return None

    return response


def _get_children_metrics_of_agent(children_traces):
    metrics = []
    for span in children_traces:
        metrics.extend(span.get("metrics", []))
        if span["type"] != "agent":
            metric = span.get("metrics", [])
            if metric:
                metrics.extend(metric)
        else:
            metrics.extend(_get_children_metrics_of_agent(span["data"]["children"]))
    return metrics

def get_trace_metrics_from_trace(traces):
    metrics = []

    # get trace level metrics
    if "metrics" in traces.keys():
        if len(traces["metrics"]) > 0:
            metrics.extend(traces["metrics"])

    # get span level metrics
    for span in traces["data"][0]["spans"]:
        if span["type"] == "agent":
            # Add children metrics of agent
            children_metric = _get_children_metrics_of_agent(span["data"]["children"])
            if children_metric:
                metrics.extend(children_metric)

        metric = span.get("metrics", [])
        if metric:
            metrics.extend(metric)
    return metrics


def _change_metrics_format_for_payload(metrics):
    formatted_metrics = []
    for metric in metrics:
        if any(m["name"] == metric.get("displayName") or m['name'] == metric.get("name") for m in formatted_metrics):
            continue
        metric_display_name = metric["name"]
        if metric.get("displayName"):
            metric_display_name = metric['displayName']
        formatted_metrics.append({
            "name": metric_display_name,
            "displayName": metric_display_name,
            "config": {"source": "user"},
        })
    return formatted_metrics
