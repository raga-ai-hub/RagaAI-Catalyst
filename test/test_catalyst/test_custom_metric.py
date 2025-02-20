import pytest
import os
from unittest.mock import patch, MagicMock
from ragaai_catalyst import CustomMetric
from ragaai_catalyst.custom_metric import _get_extract_parameters


# Unit Test Fixtures
@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {'RAGAAI_CATALYST_TOKEN': 'test-token'}):
        yield

@pytest.fixture
def custom_metric(mock_env):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [
                    {"name": "test-project", "id": "123"}
                ]
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        return CustomMetric("test-project")

# Unit Tests
def test_init_with_invalid_project(mock_env):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [
                    {"name": "other-project", "id": "123"}
                ]
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        with pytest.raises(ValueError, match="Project not found"):
            CustomMetric("non-existent-project")

def test_list_custom_metrics(custom_metric):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [
                    {"name": "metric1", "id": "1"},
                    {"name": "metric2", "id": "2"}
                ]
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        metrics = custom_metric.list_custom_metrics()
        assert metrics == [("metric1", "1"), ("metric2", "2")]

def test_get_grading_criteria(custom_metric):
    with patch('requests.get') as mock_get:
        expected_criteria = ['Float (0 to 1)', 'Boolean (0 or 1)']
        mock_get.return_value.json.return_value = {
            "data": {
                "gradingCriteriaList": expected_criteria
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        criteria = custom_metric.get_grading_criteria()
        assert criteria == expected_criteria

def test_create_custom_metrics(custom_metric):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "data": {
                "metricId": "new-metric-id"
            }
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        metric_id = custom_metric.create_custom_metrics("new-metric", "test description")
        assert metric_id == "new-metric-id"

def test_get_model_parameters(custom_metric):
    with patch('requests.post') as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        custom_metric.get_model_parameters("gpt-4", "openai")
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json'] == {
            "providerName": "openai",
            "modelName": "gpt-4"
        }

def test_verify_grading_criteria(custom_metric):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "message": "Verification successful"
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        steps = [{"output": {"response": "5.0"}}]
        result = custom_metric.verify_grading_criteria("metric-id", "criteria", steps)
        assert result == "Verification successful"

def test_get_extract_parameters():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"name": "temp", "value": "0.7", "type": "float"},
            {"name": "max_tokens", "value": "100", "type": "int"},
            {"name": "stream", "value": "true", "type": "bool"},
            {"name": "model", "value": "gpt-4", "type": "string"}
        ]
    }
    
    params = _get_extract_parameters(mock_response)
    assert len(params) == 4
    assert params[0]["value"] == 0.7  # float
    assert params[1]["value"] == 100  # int
    assert params[2]["value"] == True  # bool
    assert params[3]["value"] == "gpt-4"  # string

def test_error_handling(custom_metric):
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Test error")
        
        result = custom_metric.list_custom_metrics()
        assert result == []  # Should return empty list on error

def test_get_custom_metric_versions(custom_metric):
    with patch('requests.get') as mock_get:
        expected_versions = [
            {"id": "1", "name": "v1"},
            {"id": "2", "name": "v2"}
        ]
        mock_get.return_value.json.return_value = {
            "data": expected_versions
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        versions = custom_metric.get_custom_metric_versions("metric-id")
        assert versions == expected_versions

def test_deploy_custom_metric(custom_metric):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "message": "Deployment successful"
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        result = custom_metric.deploy_custom_metric("metric-id", "v1.0")
        assert result == "Deployment successful"

def test_commit_custom_metric(custom_metric):
    with patch('requests.post') as mock_post, \
         patch.object(custom_metric, 'get_model_parameters') as mock_get_params:
        
        # Mock get_model_parameters response
        mock_params_response = MagicMock()
        mock_params_response.json.return_value = {
            "data": [
                {"name": "temp", "value": "0.7", "type": "float"}
            ]
        }
        mock_get_params.return_value = mock_params_response
        
        # Mock commit response
        mock_post.return_value.json.return_value = {
            "data": "Commit successful"
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        steps = {
            "steps": [{"step": 1}],
            "variables": [{"name": "var1", "type": "string", "value": "test"}]
        }
        output_steps = [{"output": {"response": "5.0"}}]
        
        result = custom_metric.commit_custom_metric(
            "metric-id",
            steps,
            "gpt-4",
            "openai",
            output_steps,
            "Final reason",
            "Test commit"
        )
        
        assert result == "Commit successful"
