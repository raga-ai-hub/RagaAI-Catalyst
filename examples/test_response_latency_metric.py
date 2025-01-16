import pytest
from datetime import datetime
from agentneo.evaluation.metrics import execute_response_latency_metric

@pytest.fixture
def valid_trace_json():
    return {
        "llm_calls": [
            {"start_time": "2024-11-16T10:00:00", "end_time": "2024-11-16T10:00:02"},
            {"start_time": "2024-11-16T10:05:00", "end_time": "2024-11-16T10:05:03"},
        ],
        "tool_calls": [
            {"start_time": "2024-11-16T10:10:00", "end_time": "2024-11-16T10:10:01"},
            {"start_time": "2024-11-16T10:15:00", "end_time": "2024-11-16T10:15:04"},
        ],
    }


@pytest.fixture
def trace_json_with_missing_times():
    return {
        "llm_calls": [
            {"start_time": "2024-11-16T10:00:00", "end_time": "2024-11-16T10:00:02"},
            {"start_time": "2024-11-16T10:05:00"},  # Missing end_time
        ],
        "tool_calls": [
            {"start_time": "2024-11-16T10:10:00", "end_time": "2024-11-16T10:10:01"},
            {"end_time": "2024-11-16T10:15:04"},  # Missing start_time
        ],
    }


@pytest.fixture
def empty_trace_json():
    return {}


@pytest.fixture
def config():
    return {"metric_threshold": 2.0}


def test_valid_trace_json(valid_trace_json, config):
    result = execute_response_latency_metric(valid_trace_json, config)

    assert result["metric_name"] == "response_latency"
    assert result["result"]["score"] == 2.5  # Average latency: [2, 3, 1, 4]
    assert result["result"]["details"]["min_latency"] == 1
    assert result["result"]["details"]["max_latency"] == 4
    assert result["result"]["details"]["median_latency"] == 2.5  # Median of [1, 2, 3, 4]
    assert result["result"]["details"]["p90_latency"] == 4  # 90th percentile
    assert "std_dev_latency" in result["result"]["details"]


def test_trace_json_with_missing_times(trace_json_with_missing_times, config):
    result = execute_response_latency_metric(trace_json_with_missing_times, config)

    assert result["metric_name"] == "response_latency"
    assert result["result"]["score"] == 1.5  # Average latency: [2, 1]
    assert result["result"]["details"]["min_latency"] == 1
    assert result["result"]["details"]["max_latency"] == 2


def test_empty_trace_json(empty_trace_json, config):
    result = execute_response_latency_metric(empty_trace_json, config)

    assert result["metric_name"] == "response_latency"
    assert result["result"]["score"] is None
    assert "No response latencies found" in result["result"]["reason"]


def test_error_handling(config):
    def faulty_extract_latency_data(_):
        raise ValueError("Simulated error")

    with pytest.monkeypatch.context() as m:
        m.setattr("response_latency.extract_latency_data", faulty_extract_latency_data)

        result = execute_response_latency_metric({}, config)

        assert result["metric_name"] == "response_latency"
        assert result["result"]["score"] is None
        assert "An error occurred during evaluation" in result["result"]["reason"]