import pytest
from agentneo.evaluation.metrics import execute_error_detection_rate_metric

@pytest.fixture
def valid_trace_json():
    """
    Fixture providing a valid trace JSON with some tool and LLM errors.
    """
    return {
        "llm_calls": [
            {"output": "Response generated successfully."},
            {"output": "Error: Token limit exceeded."},
        ],
        "tool_calls": [
            {"output": "Success: Data retrieved."},
            {"output": "Error: Connection timeout."},
        ],
    }


@pytest.fixture
def trace_json_no_errors():
    """
    Fixture providing a trace JSON where no errors are present.
    """
    return {
        "llm_calls": [
            {"output": "Response generated successfully."},
            {"output": "Another successful response."},
        ],
        "tool_calls": [
            {"output": "Success: Data retrieved."},
            {"output": "Success: Calculation completed."},
        ],
    }


@pytest.fixture
def empty_trace_json():
    """
    Fixture providing an empty trace JSON with no calls.
    """
    return {}


@pytest.fixture
def config():
    """
    Fixture providing a sample configuration for the metric evaluation.
    """
    return {"error_threshold": 0.2}


def test_valid_trace_json(valid_trace_json, config):
    """
    Test the error detection metric with a valid trace JSON containing errors.
    """
    result = execute_error_detection_rate_metric(valid_trace_json, config)

    assert result["metric_name"] == "error_detection_rate"
    assert result["result"]["score"] == 0.5  # 2 errors out of 4 calls
    assert result["result"]["details"]["total_calls"] == 4
    assert result["result"]["details"]["error_count"] == 2
    assert "errors were detected successfully" in result["result"]["reason"]


def test_trace_json_no_errors(trace_json_no_errors, config):
    """
    Test the error detection metric with a trace JSON containing no errors.
    """
    result = execute_error_detection_rate_metric(trace_json_no_errors, config)

    assert result["metric_name"] == "error_detection_rate"
    assert result["result"]["score"] == 0.0  # No errors
    assert result["result"]["details"]["total_calls"] == 4
    assert result["result"]["details"]["error_count"] == 0
    assert "No errors were detected" in result["result"]["reason"]


def test_empty_trace_json(empty_trace_json, config):
    """
    Test the error detection metric with an empty trace JSON.
    """
    result = execute_error_detection_rate_metric(empty_trace_json, config)

    assert result["metric_name"] == "error_detection_rate"
    assert result["result"]["score"] is None
    assert "No calls found in the trace" in result["result"]["reason"]


def test_error_handling(config):
    """
    Test the error detection metric to handle exceptions gracefully.
    """
    def faulty_extract_error_data(_):
        raise ValueError("Simulated error during extraction")

    with pytest.monkeypatch.context() as m:
        m.setattr(
            "agentneo.evaluation.metrics.extract_error_data", faulty_extract_error_data
        )

        result = execute_error_detection_rate_metric({}, config)

        assert result["metric_name"] == "error_detection_rate"
        assert result["result"]["score"] is None
        assert "An error occurred during evaluation" in result["result"]["reason"]