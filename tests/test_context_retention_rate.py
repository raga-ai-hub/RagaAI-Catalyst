import pytest
from datetime import datetime
from agentneo.evaluation.metrics.context_retention_rate import  (
    execute_context_retention_metric,
    parse_timestamp,
    extract_conversations_from_json,
    analyze_context_usage,
    evaluate_context_coherence
)

@pytest.fixture
def valid_trace_json():
    return {
        "llm_calls": [
            {
                "start_time": "2024-11-16T10:00:00",
                "end_time": "2024-11-16T10:00:02",
                "input_prompt": [{"content": "What is AI?"}],
                "output": "AI stands for Artificial Intelligence."
            },
            {
                "start_time": "2024-11-16T10:05:00",
                "end_time": "2024-11-16T10:05:03",
                "input_prompt": [{"content": "Tell me more about AI."}],
                "output": "AI involves machine learning, deep learning, etc."
            },
        ],
        "tool_calls": [
            {
                "start_time": "2024-11-16T10:10:00",
                "end_time": "2024-11-16T10:10:01",
                "input_parameters": "Search AI research papers",
                "output": "Found 10 papers on AI."
            },
            {
                "start_time": "2024-11-16T10:15:00",
                "end_time": "2024-11-16T10:15:04",
                "input_parameters": "Summarize recent AI advancements",
                "output": "Recent advancements include GPT-4, BERT, etc."
            },
        ],
    }

@pytest.fixture
def trace_json_with_missing_times():
    return {
        "llm_calls": [
            {
                "start_time": "2024-11-16T10:00:00",
                "end_time": "2024-11-16T10:00:02",
                "input_prompt": [{"content": "What is AI?"}],
                "output": "AI stands for Artificial Intelligence."
            },
            {
                "start_time": "2024-11-16T10:05:00",  # Missing end_time
                "input_prompt": [{"content": "Tell me more about AI."}],
                "output": "AI involves machine learning, deep learning, etc."
            },
        ],
        "tool_calls": [
            {
                "start_time": "2024-11-16T10:10:00",
                "end_time": "2024-11-16T10:10:01",
                "input_parameters": "Search AI research papers",
                "output": "Found 10 papers on AI."
            },
            {
                "end_time": "2024-11-16T10:15:04",  # Missing start_time
                "input_parameters": "Summarize recent AI advancements",
                "output": "Recent advancements include GPT-4, BERT, etc."
            },
        ],
    }

@pytest.fixture
def empty_trace_json():
    return {}

@pytest.fixture
def config():
    return {
        "context_switch_threshold": 0.3,
        "context_maintenance_weight": 0.4,
        "coherence_weight": 0.6
    }

def test_valid_trace_json(valid_trace_json, config):
    result = execute_context_retention_metric(valid_trace_json, config)

    assert result["metric_name"] == "context_retention_rate"
    assert result["result"]["score"] > 0  # Ensure a positive score
    assert result["result"]["context_metrics"]["context_switches"] == 0
    assert result["result"]["context_metrics"]["total_key_points"] > 0
    assert result["result"]["coherence_analysis"]["coherence_score"] > 0
    assert "key_observations" in result["result"]["coherence_analysis"]

def test_trace_json_with_missing_times(trace_json_with_missing_times, config):
    result = execute_context_retention_metric(trace_json_with_missing_times, config)

    assert result["metric_name"] == "context_retention_rate"
    assert result["result"]["score"] >= 0  # Should handle missing times gracefully
    assert "key_observations" in result["result"]["coherence_analysis"]

def test_empty_trace_json(empty_trace_json, config):
    result = execute_context_retention_metric(empty_trace_json, config)

    assert result["metric_name"] == "context_retention_rate"
    assert result["result"]["score"] is None
    assert "No valid conversations found" in result["result"]["reason"]

def test_error_handling(config):
    def faulty_get_model_response(prompt, config):
        raise ValueError("Simulated error")

    with pytest.monkeypatch.context() as m:
        m.setattr("your_module.get_model_response", faulty_get_model_response)

        result = execute_context_retention_metric({}, config)

        assert result["metric_name"] == "context_retention_rate"
        assert result["result"]["score"] is None
        assert "An error occurred during evaluation" in result["result"]["reason"]
