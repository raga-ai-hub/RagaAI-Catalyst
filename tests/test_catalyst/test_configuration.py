import pytest
import os
import requests
from unittest.mock import patch, MagicMock
import dotenv
dotenv.load_dotenv()
import os

from ragaai_catalyst import RagaAICatalyst


# Mock environment variables for testing
@pytest.fixture
def mock_env_vars():
    original_environ = os.environ.copy()
    RAGAAI_CATALYST_ACCESS_KEY = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
    RAGAAI_CATALYST_SECRET_KEY = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    RAGAAI_CATALYST_BASE_URL = os.getenv("RAGAAI_CATALYST_BASE_URL")
    
    yield
    
    os.environ.clear()
    os.environ.update(original_environ)

@pytest.fixture
def raga_catalyst(mock_env_vars):
    with patch('ragaai_catalyst.RagaAICatalyst.get_token', return_value='test_token'):
        catalyst = RagaAICatalyst(
            os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),  
            os.getenv("RAGAAI_CATALYST_SECRET_KEY")  
        )
    return catalyst



def test_project_use_cases():
        catalyst = RagaAICatalyst(
            access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
            secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
            base_url=os.getenv("RAGAAI_CATALYST_BASE_URL")
        )
        use_case = catalyst.project_use_cases()
        assert use_case ==['Chatbot', 'Text2SQL', 'Q/A', 'Code Generation', 'Others']


def test_list_project():
        catalyst = RagaAICatalyst(
            access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
            secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
            base_url=os.getenv("RAGAAI_CATALYST_BASE_URL")
        )
        use_case = catalyst.list_projects()
        assert use_case is not None  # Check if the result is not None


def test_existing_projectname():
        with pytest.raises(ValueError, match="already exists. Please choose a different name."):
            catalyst = RagaAICatalyst(
                access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
                secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
                base_url=os.getenv("RAGAAI_CATALYST_BASE_URL")
            )
            project = catalyst.create_project(
                project_name="prompt_metric_dataset3",
                usecase="Chatbot"
            )

def test_initialization_missing_credentials():
    """Test initialization with missing credentials"""
    with pytest.raises(ValueError, match="RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set"):
        RagaAICatalyst('', '')

@patch('requests.post')
def test_get_token_success(mock_post, mock_env_vars):
    """Test token retrieval success"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'success': True,
        'data': {'token': 'test_token'}
    }
    mock_post.return_value = mock_response

    token = RagaAICatalyst.get_token()
    assert token == 'test_token'
    assert os.getenv('RAGAAI_CATALYST_TOKEN') == 'test_token'

@patch('requests.post')
def test_get_token_failure(mock_post, mock_env_vars):
    """Test token retrieval failure"""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        'message': 'Please enter valid credentials'
    }
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="Authentication failed"):
        RagaAICatalyst.get_token()

@patch('requests.get')
def test_project_use_cases_success(mock_get, raga_catalyst):
    """Test retrieving project use cases"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': {'usecase': ['Q/A', 'Chatbot', 'Summarization']}
    }
    mock_get.return_value = mock_response

    use_cases = raga_catalyst.project_use_cases()
    assert use_cases == ['Q/A', 'Chatbot', 'Summarization']

@patch('requests.get')
def test_project_use_cases_failure(mock_get, raga_catalyst):
    """Test project use cases retrieval failure"""
    mock_get.side_effect = requests.exceptions.RequestException("Network Error")

    use_cases = raga_catalyst.project_use_cases()
    assert use_cases == []

@patch('requests.post')
@patch('ragaai_catalyst.RagaAICatalyst.list_projects')
def test_create_project_success(mock_list_projects, mock_post, raga_catalyst):
    """Test successful project creation"""
    mock_list_projects.return_value = []  # No existing projects
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        'data': {'name': 'TestProject'}
    }
    mock_post.return_value = mock_post_response

    with patch('ragaai_catalyst.RagaAICatalyst.project_use_cases', return_value=['Q/A']):
        result = raga_catalyst.create_project('TestProject')
        assert 'Project Created Successfully' in result

@patch('requests.post')
@patch('ragaai_catalyst.RagaAICatalyst.list_projects')
def test_create_project_duplicate(mock_list_projects, mock_post, raga_catalyst):
    """Test project creation with duplicate name"""
    mock_list_projects.return_value = ['TestProject']

    with pytest.raises(ValueError, match="Project name 'TestProject' already exists"):
        raga_catalyst.create_project('TestProject')

@patch('requests.get')
def test_list_projects_success(mock_get, raga_catalyst):
    """Test successful project listing"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': {
            'content': [
                {'name': 'Project1'},
                {'name': 'Project2'}
            ]
        }
    }
    mock_get.return_value = mock_response

    projects = raga_catalyst.list_projects()
    assert projects == ['Project1', 'Project2']

@patch('requests.get')
def test_list_metrics_success(mock_get):
    """Test successful metrics listing"""
    with patch.dict(os.environ, {'RAGAAI_CATALYST_TOKEN': 'test_token'}):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'metrics': [
                    {'name': 'hallucination', 'category': 'quality'},
                    {'name': 'toxicity', 'category': 'safety'}
                ]
            }
        }
        mock_get.return_value = mock_response

        metrics = RagaAICatalyst.list_metrics()
        assert metrics == ['hallucination', 'toxicity']

def test_initialization_invalid_credentials():
    """Test initialization with invalid credentials"""
    with pytest.raises(Exception, match="Authentication failed. Invalid credentials provided."):
        RagaAICatalyst(
            access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY")+"a",
            secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
            base_url=os.getenv("RAGAAI_CATALYST_BASE_URL")
        )

def test_initialization_invalid_base_url():
    with pytest.raises(ConnectionError, match="The provided base_url is not accessible. Please re-check the base_url."):
        RagaAICatalyst(
            access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
            secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
            base_url=os.getenv("RAGAAI_CATALYST_BASE_URL") +"a",
        )
