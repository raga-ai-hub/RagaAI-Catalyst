from agentneo.evaluation.metrics.custom_evaluation_metric import validate_rubric, parse_timestamp, extract_conversations, get_model_response, extract_evaluation_criteria, execute_custom_evaluation_metric, create_evaluation_prompt, DEFAULT_RUBRIC
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json

# Fixtures
@pytest.fixture(scope="module")
def sample_trace_json():
    """Fixture providing sample trace JSON data."""
    return {
        "llm_calls": [
            {
                "name": "test_call",
                "start_time": "2024-01-01T10:00:00",
                "input_prompt": [{"content": "test prompt"}],
                "output": "test response"
            }
        ],
        "tool_calls": [
            {
                "name": "test_tool",
                "start_time": "2024-01-01T10:01:00",
                "input_parameters": "test params",
                "output": "test tool response"
            }
        ]
    }

@pytest.fixture(scope="module")
def valid_custom_rubric():
    """Fixture providing a valid evaluation rubric."""
    return {
        "criteria": {
            "technical_accuracy": {
                "description": "How technically accurate are the AI's responses?",
                "weight": 0.4
            },
            "code_quality": {
                "description": "How well-structured and maintainable is the suggested code?",
                "weight": 0.3
            },
            "security_awareness": {
                "description": "How well does the AI address security considerations?",
                "weight": 0.3
            }
        },
        "scoring_guidelines": {
            "0.0-0.2": "Critical issues present",
            "0.2-0.4": "Major improvements needed",
            "0.4-0.6": "Acceptable with improvements needed",
            "0.6-0.8": "Good quality with minor issues",
            "0.8-1.0": "Excellent technical implementation"
        }
    }

@pytest.fixture(scope="module")
def default_config():
    """Fixture providing default configuration."""
    return {
        "model": "gpt-4o-mini"
    }

# Test cases
def test_validate_rubric_valid(valid_custom_rubric):
    """Test rubric validation with valid input."""
    assert validate_rubric(valid_custom_rubric) is True

def test_validate_rubric_invalid_weights(valid_custom_rubric):
    """Test rubric validation with invalid weights."""
    invalid_rubric = valid_custom_rubric.copy()
    invalid_rubric["criteria"]["technical_accuracy"]["weight"] = 0.5
    assert validate_rubric(invalid_rubric) is False

def test_validate_rubric_missing_keys():
    """Test rubric validation with missing required keys."""
    invalid_rubric = {
        "criteria": {
            "test": {
                "weight": 1.0
            }
        }
    }
    assert validate_rubric(invalid_rubric) is False

def test_parse_timestamp():
    """Test timestamp parsing."""
    timestamp = "2024-01-01T10:00:00"
    parsed = parse_timestamp(timestamp)
    assert isinstance(parsed, datetime)
    assert parsed.year == 2024
    assert parsed.month == 1
    assert parsed.day == 1

def test_extract_conversations(sample_trace_json):
    """Test conversation extraction from trace JSON."""
    conversations = extract_conversations(sample_trace_json)
    assert len(conversations) == 2
    assert conversations[0]["function_name"] == "test_call"
    assert conversations[1]["function_name"] == "test_tool"

@patch('litellm.completion')
def test_get_model_response_success(mock_completion, default_config):
    """Test successful model response retrieval."""
    # Arrange
    expected_response = "test response"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=expected_response))]
    mock_completion.return_value = mock_response
    
    # Act
    response = get_model_response("test prompt", default_config)
    
    # Assert
    assert response == expected_response
    mock_completion.assert_called_once()

@patch('litellm.completion')
def test_get_model_response_error(mock_completion, default_config):
    """Test model response retrieval with error."""
    # Arrange
    mock_completion.side_effect = Exception("API Error")
    
    # Act
    response = get_model_response("test prompt", default_config)
    
    # Assert
    assert response is None

@patch('litellm.completion')
def test_extract_evaluation_criteria_custom(mock_completion):
    """Test extraction of custom evaluation criteria."""
    # Arrange
    expected_criterion = "test_criterion"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(
        message=MagicMock(
            content=json.dumps({
                "criteria": {
                    expected_criterion: {
                        "description": "test description",
                        "weight": 1.0
                    }
                },
                "scoring_guidelines": DEFAULT_RUBRIC["scoring_guidelines"]
            })
        )
    )]
    mock_completion.return_value = mock_response
    
    # Act
    result, is_custom = extract_evaluation_criteria("test prompt")
    
    # Assert
    assert is_custom is True
    assert expected_criterion in result["criteria"]

def test_create_evaluation_prompt():
    """Test creation of evaluation prompt."""
    # Arrange
    criteria = {
        "test_criterion": {
            "description": "test description",
            "weight": 1.0
        }
    }
    conversation = [{"role": "user", "content": "test"}]
    context = {"domain": "test"}
    
    # Act
    prompt = create_evaluation_prompt(criteria, conversation, context)
    
    # Assert
    assert "test_criterion" in prompt
    assert "test description" in prompt
    assert "domain" in prompt

@patch('litellm.completion')
def test_execute_custom_evaluation_metric_success(mock_completion, sample_trace_json, default_config, valid_custom_rubric):
    """Test successful execution of custom metric evaluation."""
    # Arrange
    expected_score = 0.8
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(
        message=MagicMock(
            content=json.dumps({
                "score": expected_score,
                "scores": {"test_criterion": expected_score},
                "reason": "test reason",
                "recommendations": ["test recommendation"]
            })
        )
    )]
    mock_completion.return_value = mock_response
    
    # Act
    result = execute_custom_evaluation_metric(
        sample_trace_json,
        default_config,
        valid_custom_rubric
    )
    
    # Assert
    assert result["metric_name"] == "custom_evaluation"
    assert result["result"]["score"] == expected_score
    assert "recommendations" in result["result"]

def test_execute_custom_evaluation_metric_error(sample_trace_json, default_config):
    """Test custom metric evaluation with error."""
    # Act
    result = execute_custom_evaluation_metric(
        sample_trace_json,
        default_config,
        "invalid_criteria"
    )
    
    # Assert
    assert result["metric_name"] == "custom_evaluation"
    assert "config" in result
    assert "result" in result

def test_validate_rubric_invalid_weight_type(valid_custom_rubric):
    """Test rubric validation with invalid weight type."""
    invalid_rubric = valid_custom_rubric.copy()
    invalid_rubric["criteria"]["technical_accuracy"]["weight"] = "0.4"  # string instead of float
    assert validate_rubric(invalid_rubric) is False

def test_validate_rubric_invalid_weight_range(valid_custom_rubric):
    """Test rubric validation with weight outside valid range."""
    invalid_rubric = valid_custom_rubric.copy()
    invalid_rubric["criteria"]["technical_accuracy"]["weight"] = 1.5  # > 1
    assert validate_rubric(invalid_rubric) is False

def test_validate_rubric_exception():
    """Test rubric validation with invalid input causing exception."""
    assert validate_rubric(None) is False
    assert validate_rubric("invalid") is False

@patch('litellm.completion')
def test_extract_evaluation_criteria_empty_weights(mock_completion):
    """Test extraction of evaluation criteria with missing weights."""
    # Arrange
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(
        message=MagicMock(
            content=json.dumps({
                "criteria": {
                    "criterion1": {
                        "description": "test description"
                    },
                    "criterion2": {
                        "description": "test description"
                    }
                }
            })
        )
    )]
    mock_completion.return_value = mock_response
    
    # Act
    result, is_custom = extract_evaluation_criteria("test prompt")
    
    # Assert
    assert all(c["weight"] == 0.5 for c in result["criteria"].values())  # Equal weights assigned
    assert is_custom is True

@patch('litellm.completion')
def test_execute_custom_evaluation_metric_json_decode_error(mock_completion, sample_trace_json, default_config, valid_custom_rubric):
    """Test custom metric execution with JSON decode error in model response."""
    # Arrange
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(
        message=MagicMock(content="invalid json")
    )]
    mock_completion.return_value = mock_response
    
    # Act
    result = execute_custom_evaluation_metric(
        sample_trace_json,
        default_config,
        valid_custom_rubric
    )
    
    # Assert
    assert "error" in result["result"]
    assert "Failed to parse model response as JSON" in result["result"]["error"]

def test_extract_conversations_empty():
    """Test conversation extraction with empty trace JSON."""
    empty_trace = {
        "llm_calls": [],
        "tool_calls": []
    }
    conversations = extract_conversations(empty_trace)
    assert len(conversations) == 0

def test_create_evaluation_prompt_no_context():
    """Test creation of evaluation prompt without context."""
    criteria = {
        "test_criterion": {
            "description": "test description",
            "weight": 1.0
        }
    }
    conversation = [{"role": "user", "content": "test"}]
    
    prompt = create_evaluation_prompt(criteria, conversation)
    
    assert "test_criterion" in prompt
    assert "Additional Context" not in prompt

def test_extract_evaluation_criteria_default_on_error():
    """Test that extract_evaluation_criteria returns default rubric on exception."""
    with patch('litellm.completion') as mock_completion:
        # Arrange
        mock_completion.side_effect = Exception("API Error")
        
        # Act
        result, is_custom = extract_evaluation_criteria("test prompt")
        
        # Assert
        assert result == DEFAULT_RUBRIC
        assert is_custom is False
        assert "problem_solving" in result["criteria"]
        assert "clarity" in result["criteria"]
        assert "efficiency" in result["criteria"]

def test_validate_rubric_scoring_guidelines():
    """Test rubric validation with missing or invalid scoring guidelines."""
    # Test with missing scoring guidelines
    invalid_rubric = {
        "criteria": {
            "test": {
                "description": "test",
                "weight": 1.0
            }
        }
    }
    assert validate_rubric(invalid_rubric) is False
    
    # Test with empty scoring guidelines
    invalid_rubric["scoring_guidelines"] = {}
    assert validate_rubric(invalid_rubric) is False

def test_get_model_response_with_response_format():
    """Test get_model_response with response_format parameter."""
    with patch('litellm.completion') as mock_completion:
        # Arrange
        expected_response = "test response"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=expected_response))]
        mock_completion.return_value = mock_response
        
        config = {"model": "test-model"}
        response_format = {"type": "json_object"}
        
        # Act
        response = get_model_response("test prompt", config, response_format)
        
        # Assert
        assert response == expected_response
        mock_completion.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )