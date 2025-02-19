import pytest
from unittest.mock import patch, MagicMock
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import BaseTracer

@pytest.fixture
def tracer():
    user_details = {
        "project_name": "test_project",
        "dataset_name": "test_dataset",
        "project_id": "test_id",
        "trace_name": "test_trace",
        "interval_time": 1
    }
    tracer = BaseTracer(user_details)
    tracer.trace_metrics = []
    tracer.visited_metrics = []
    return tracer

def test_add_metrics_individual_params(tracer):
    """Test adding metrics using individual parameters"""
    tracer.trace = {}  # Initialize trace
    tracer.add_metrics(
        name="test_metric",
        score=0.95,
        reasoning="Good performance",
        cost=0.01,
        latency=100,
        metadata={"key": "value"},
        config={"threshold": 0.8}
    )
    
    assert len(tracer.trace_metrics) == 1
    metric = tracer.trace_metrics[0]
    assert metric["name"] == "test_metric"
    assert metric["score"] == 0.95
    assert metric["reason"] == "Good performance"
    assert metric["source"] == "user"
    assert metric["cost"] == 0.01
    assert metric["latency"] == 100
    assert metric["metadata"] == {"key": "value"}
    assert metric["config"] == {"threshold": 0.8}

def test_add_metrics_dict_input(tracer):
    """Test adding metrics using dictionary input"""
    tracer.trace = {}
    metric_dict = {
        "name": "test_metric",
        "score": 0.95,
        "reasoning": "Good performance"
    }
    tracer.add_metrics(metric_dict)
    
    assert len(tracer.trace_metrics) == 1
    metric = tracer.trace_metrics[0]
    assert metric["name"] == "test_metric"
    assert metric["score"] == 0.95
    assert metric["reason"] == "Good performance"

def test_add_metrics_list_input(tracer):
    """Test adding multiple metrics using list input"""
    tracer.trace = {}
    metrics = [
        {"name": "metric1", "score": 0.95},
        {"name": "metric2", "score": 0.85}
    ]
    tracer.add_metrics(metrics)
    
    assert len(tracer.trace_metrics) == 2
    assert tracer.trace_metrics[0]["name"] == "metric1"
    assert tracer.trace_metrics[1]["name"] == "metric2"

def test_add_metrics_duplicate_names(tracer):
    """Test handling of duplicate metric names"""
    tracer.trace = {}
    metrics = [
        {"name": "metric1", "score": 0.95},
        {"name": "metric1", "score": 0.85}
    ]
    tracer.add_metrics(metrics)
    
    assert len(tracer.trace_metrics) == 2
    assert tracer.trace_metrics[0]["name"] == "metric1"
    assert tracer.trace_metrics[1]["name"] == "metric1_2"

def test_add_metrics_missing_required_fields(tracer):
    """Test validation of required fields"""
    tracer.trace = {}
    metrics = [{"name": "metric1"}]  # Missing score
    
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.logger') as mock_logger:
        tracer.add_metrics(metrics)
        mock_logger.error.assert_called_once_with(
            "Validation Error: Metric must contain 'name' and 'score' fields"
        )
    assert len(tracer.trace_metrics) == 0

def test_add_metrics_invalid_input_type(tracer):
    """Test handling of invalid input types"""
    tracer.trace = {}
    invalid_metrics = ["not_a_dict"]
    
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.logger') as mock_logger:
        tracer.add_metrics(invalid_metrics)
        mock_logger.error.assert_called_once_with(
            "Validation Error: Expected dict, got <class 'str'>"
        )
    assert len(tracer.trace_metrics) == 0

def test_add_metrics_before_trace_init(tracer):
    """Test adding metrics before trace initialization"""
    # Don't initialize trace
    with patch('ragaai_catalyst.tracers.agentic_tracing.tracers.base.logger') as mock_logger:
        tracer.add_metrics(name="test", score=0.95)
        mock_logger.warning.assert_called_with(
            "Cannot add metrics before trace is initialized. Call start() first."
        )
    assert not hasattr(tracer, 'trace_metrics') or not tracer.trace_metrics

def test_add_metrics_with_empty_optional_fields(tracer):
    """Test adding metrics with empty optional fields"""
    tracer.trace = {}
    tracer.add_metrics(name="test_metric", score=0.95)
    
    metric = tracer.trace_metrics[0]
    assert metric["reason"] == ""
    assert metric["metadata"] == {}
    assert metric["config"] == {}
    assert metric["mappings"] == []
    assert metric["cost"] is None
    assert metric["latency"] is None
