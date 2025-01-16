import json
from typing import List, Optional, Dict, Any
from datetime import datetime

def parse_timestamp(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)

def calculate_latency(start_time: str, end_time: str) -> Optional[float]:
    start = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    if start and end:
        return (end - start).total_seconds()
    else:
        return None

def extract_latency_data(trace_json: Dict[str, Any]) -> List[float]:
    latencies = []
    for call_type in ["llm_calls", "tool_calls"]:
        if call_type in trace_json:
            for call in trace_json[call_type]:
                if "start_time" in call and "end_time" in call:
                    latency = calculate_latency(call["start_time"], call["end_time"])
                    if latency is not None:
                        latencies.append(latency)
    return latencies

def execute_response_latency_metric(trace_json: Dict[str, Any], config: Dict[str, Any] = {}) -> Dict[str, Any]:
    try:
        # Extract latency data
        latencies = extract_latency_data(trace_json)

        if not latencies:
            return {
                "metric_name": "response_latency",
                "config": config,
                "result": {
                    "score": None,
                    "reason": "No response latencies found in the trace.",
                },
            }

        # Calculate primary statistics
        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        # Calculate additional statistics
        latencies.sort()
        median_latency = latencies[len(latencies) // 2]
        p90_latency = latencies[int(len(latencies) * 0.9)]
        std_dev_latency = (sum((x - average_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

        return {
            "metric_name": "response_latency",
            "config": config,
            "result": {
                "score": average_latency,
                "reason": "Response latency metrics successfully calculated.",
                "details": {
                    "average_latency": average_latency,
                    "min_latency": min_latency,
                    "max_latency": max_latency,
                    "median_latency": median_latency,
                    "p90_latency": p90_latency,
                    "std_dev_latency": std_dev_latency,
                },
            },
        }

    except Exception as e:
        return {
            "metric_name": "response_latency",
            "config": config,
            "result": {
                "score": None,
                "reason": f"An error occurred during evaluation: {str(e)}",
            },
        }