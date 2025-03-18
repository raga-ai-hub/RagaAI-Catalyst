import pytest
from unittest.mock import patch, MagicMock
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import BaseTracer

@pytest.fixture
def sample_span_attributes():
    return {
        "test_metric": MagicMock(
            local_metrics=[
                {
                    "name": "accuracy",
                    "displayName": "Accuracy Score",
                    "model": "gpt-4",
                    "provider": "openai",
                    "mapping": {
                        "prompt": "test prompt",
                        "response": "test response"
                    }
                }
            ]
        )
    }

@pytest.fixture
def sample_metric_response():
    return {
        "data": {
            "data": [{
                "score": 0.95,
                "reason": "High accuracy",
                "cost": 0.01,
                "latency": 100,
                "metric_config": {
                    "job_id": "job123",
                    "displayName": "Accuracy Score",
                    "model": "gpt-4",
                    "orgDomain": "test.com",
                    "provider": "openai",
                    "reason": "test reason",
                    "request_id": "req123",
                    "user_id": "user123",
                    "threshold": {
                        "isEditable": True,
                        "lte": 0.8
                    }
                }
            }]
        }
    }

def test_get_formatted_metric_successful(sample_span_attributes, sample_metric_response):
    """Test successful metric calculation and formatting"""
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.calculate_metric') as mock_calculate:
        mock_calculate.return_value = sample_metric_response
        
        result = BaseTracer.get_formatted_metric(
            sample_span_attributes,
            project_id="test_project",
            name="test_metric"
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        metric = result[0]
        assert metric["name"] == "Accuracy Score"
        assert metric["displayName"] == "Accuracy Score"
        assert metric["score"] == 0.95
        assert metric["reason"] == "High accuracy"
        assert metric["source"] == "user"
        assert metric["cost"] == 0.01
        assert metric["latency"] == 100
        assert isinstance(metric["mappings"], list)
        
        config = metric["config"]
        assert config["job_id"] == "job123"
        assert config["metric_name"] == "Accuracy Score"
        assert config["model"] == "gpt-4"
        assert config["provider"] == "openai"
        assert config["threshold"]["is_editable"] is True
        assert config["threshold"]["lte"] == 0.8

def test_get_formatted_metric_missing_metric():
    """Test when the metric name is not in span attributes"""
    result = BaseTracer.get_formatted_metric(
        span_attributes_dict={},
        project_id="test_project",
        name="nonexistent_metric"
    )
    assert result is None

def test_get_formatted_metric_empty_local_metrics(sample_span_attributes):
    """Test when local_metrics is empty"""
    sample_span_attributes["test_metric"].local_metrics = []
    result = BaseTracer.get_formatted_metric(
        sample_span_attributes,
        project_id="test_project",
        name="test_metric"
    )
    assert result == []

def test_get_formatted_metric_calculation_error(sample_span_attributes):
    """Test error handling during metric calculation"""
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.calculate_metric') as mock_calculate:
        mock_calculate.side_effect = ValueError("Invalid metric parameters")
        
        result = BaseTracer.get_formatted_metric(
            sample_span_attributes,
            project_id="test_project",
            name="test_metric"
        )
        
        assert result == []

def test_get_formatted_metric_unexpected_error(sample_span_attributes):
    """Test handling of unexpected errors"""
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.calculate_metric') as mock_calculate:
        mock_calculate.side_effect = Exception("Unexpected error")
        
        result = BaseTracer.get_formatted_metric(
            sample_span_attributes,
            project_id="test_project",
            name="test_metric"
        )
        
        assert result == []
