import pytest
import json
from dataclasses import dataclass
from agentneo.utils import trace_utils
from unittest.mock import mock_open, patch


# Define a mock data class for testing log_event
@dataclass
class MockEvent:
    event_type: str
    description: str

# Test convert_usage_to_dict function
def test_convert_usage_to_dict():
    # Test with a dictionary input
    usage_dict = {"prompt_tokens": 5, "completion_tokens": 10, "reasoning_tokens": 2}
    expected_dict = {"input": 5, "completion": 10, "reasoning": 2}
    assert trace_utils.convert_usage_to_dict(usage_dict) == expected_dict

    # Test with an object with attributes
    class UsageObject:
        prompt_tokens = 8
        completion_tokens = 4
        reasoning_tokens = 1

    usage_object = UsageObject()
    expected_object_dict = {"input": 8, "completion": 4, "reasoning": 1}
    assert trace_utils.convert_usage_to_dict(usage_object) == expected_object_dict

    # Test with an unexpected type
    assert trace_utils.convert_usage_to_dict("invalid_type") == {"input": 0, "completion": 0, "reasoning": 0}

# Test calculate_cost function
def test_calculate_cost():
    token_usage = {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 2}
    input_cost_per_token = 0.01
    output_cost_per_token = 0.02
    reasoning_cost_per_token = 0.03

    expected_cost = {
        "input": 0.1,            # 10 * 0.01
        "completion": 0.1,       # 5 * 0.02
        "reasoning": 0.06,       # 2 * 0.03
        "total": 0.26            # sum of all costs
    }
    assert trace_utils.calculate_cost(token_usage, input_cost_per_token, output_cost_per_token, reasoning_cost_per_token) == expected_cost

# Test load_model_costs function
def test_load_model_costs():
    # Simulate file found case
    model_costs_content = {"modelA": {"input_cost": 0.01, "output_cost": 0.02}}
    with patch("builtins.open", mock_open(read_data=json.dumps(model_costs_content))):
        assert trace_utils.load_model_costs() == model_costs_content

    # Simulate file not found case, using a fallback file within resources
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("importlib.resources.open_text", mock_open(read_data=json.dumps(model_costs_content))):
            assert trace_utils.load_model_costs() == model_costs_content

# Test log_event function
def test_log_event(tmp_path):
    # Use a temporary file path for logging
    log_file_path = tmp_path / "log.txt"
    event = MockEvent(event_type="INFO", description="Test event")

    trace_utils.log_event(event, log_file_path)

    # Read the log file and verify its contents
    with open(log_file_path, "r") as f:
        log_entry = json.loads(f.readline())
        assert log_entry["event_type"] == "INFO"
        assert log_entry["description"] == "Test event"