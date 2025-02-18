import pytest
import os
import requests
from unittest.mock import patch, MagicMock
from ragaai_catalyst.evaluation import Evaluation

@pytest.fixture
def evaluation():
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        # Mock project list response
        mock_get.return_value.json.return_value = {
            "data": {
                "content": [{
                    "id": "test_project_id",
                    "name": "test_project"
                }]
            }
        }
        mock_get.return_value.status_code = 200
        
        # Mock dataset list response
        mock_post.return_value.json.return_value = {
            "data": {
                "content": [{
                    "id": "test_dataset_id",
                    "name": "test_dataset"
                }]
            }
        }
        mock_post.return_value.status_code = 200
        
        return Evaluation(project_name="test_project", dataset_name="test_dataset")

@pytest.fixture
def valid_metrics():
    return [{
        "name": "accuracy",
        "config": {"threshold": 0.8},
        "column_name": "accuracy_col",
        "schema_mapping": {"input": "test_input"}
    }]

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "success": True,
        "message": "Metrics added successfully",
        "data": {"jobId": "test_job_123"}
    }
    return mock

def test_add_metrics_success(evaluation, valid_metrics, mock_response):
    """Test successful addition of metrics"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.return_value = mock_response
        evaluation.add_metrics(valid_metrics)
        
        # Verify the request was made with correct project_id
        assert mock_post.call_args[1]['headers']['X-Project-Id'] == str(evaluation.project_id)
        assert evaluation.jobId == "test_job_123"

def test_add_metrics_missing_required_keys(evaluation):
    """Test validation of required keys"""
    invalid_metrics = [{
        "name": "accuracy",
        "config": {"threshold": 0.8}
        # missing column_name and schema_mapping
    }]
    
    with pytest.raises(ValueError) as exc_info:
        evaluation.add_metrics(invalid_metrics)
    
    assert "required for each metric evaluation" in str(exc_info.value)

def test_add_metrics_invalid_metric_name(evaluation, valid_metrics):
    """Test validation of metric names"""
    with patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["different_metric"]):
        
        with pytest.raises(ValueError) as exc_info:
            evaluation.add_metrics(valid_metrics)
        
        assert "Enter a valid metric name" in str(exc_info.value)

def test_add_metrics_duplicate_column_name(evaluation, valid_metrics):
    """Test validation of duplicate column names"""
    with patch.object(evaluation, '_get_executed_metrics_list', 
                     return_value=["accuracy_col"]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]):
        
        with pytest.raises(ValueError) as exc_info:
            evaluation.add_metrics(valid_metrics)
        
        assert "Column name 'accuracy_col' already exists" in str(exc_info.value)

def test_add_metrics_http_error(evaluation, valid_metrics):
    """Test handling of HTTP errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.HTTPError("HTTP Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception

def test_add_metrics_connection_error(evaluation, valid_metrics):
    """Test handling of connection errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception

def test_add_metrics_timeout_error(evaluation, valid_metrics):
    """Test handling of timeout errors"""
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}):
        
        mock_post.side_effect = requests.exceptions.Timeout("Timeout Error")
        evaluation.add_metrics(valid_metrics)
        # Should log error but not raise exception

def test_add_metrics_bad_request(evaluation, valid_metrics):
    """Test handling of 400 bad request"""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"message": "Bad request error"}
    
    with patch('requests.post') as mock_post, \
         patch.object(evaluation, '_get_executed_metrics_list', return_value=[]), \
         patch.object(evaluation, 'list_metrics', return_value=["accuracy"]), \
         patch.object(evaluation, '_update_base_json', return_value={}), \
         patch('ragaai_catalyst.evaluation.logger') as mock_logger:
        
        mock_post.return_value = mock_response
        evaluation.add_metrics(valid_metrics)
        
        # Verify error is logged
        mock_logger.error.assert_called_with(
            "An unexpected error occurred: Bad request error"
        )
        assert evaluation.jobId is None
