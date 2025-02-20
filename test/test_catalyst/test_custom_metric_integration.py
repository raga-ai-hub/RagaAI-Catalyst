import pytest
import os
from ragaai_catalyst import RagaAICatalyst, CustomMetric
import time
import json

@pytest.fixture(scope="module")
def access_keys():
    """Provide access keys for RagaAI Catalyst API"""
    return {
        "access_key": os.environ["RAGAAI_CATALYST_ACCESS_KEY"],
        "secret_key": os.environ["RAGAAI_CATALYST_SECRET_KEY"]
    }

@pytest.fixture(scope="module")
def base_url():
    """Provide base URL for RagaAI Catalyst API"""
    return os.environ.setdefault("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")

@pytest.fixture(scope="module")
def catalyst_session(base_url, access_keys):
    """Create a RagaAI Catalyst session for integration tests"""
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"],
        base_url=base_url
    )
    yield catalyst

@pytest.fixture(scope="module")
def custom_metric_manager(catalyst_session):
    """Create a CustomMetric instance for integration tests"""
    project_name = "jsonl-dataset-upload"
    try:
        # Try to create the project if it doesn't exist
        catalyst_session.create_project(project_name=project_name)
    except Exception as e:
        print(f"Project creation failed or already exists: {e}")
    
    custom_metric_manager = CustomMetric(project_name=project_name)
    yield custom_metric_manager

@pytest.fixture(scope="module")
def test_steps():
    """Test steps fixture with example prompt and Python script"""
    return {
        "steps": [
            {
                "type": "PROMPT",
                "name": "step1",
                "order": 1,
                "textFields": [
                    {
                        "role": "system",
                        "content": "You are the master of the science"
                    },
                    {
                        "role": "user",
                        "content": "Who is the father of the {{topic}}"
                    }
                ]
            },
            {
                "type": "PYTHON",
                "name": "step2",
                "order": 2,
                "pythonScript": "def main():\n  a = \"This is the tezt on1\"\n  b = \"{{entertext}}\"\n  c = len(b)/len(a)\n  return c\n"
            }
        ],
        "variables": [
            {
                "name": "topic",
                "value": "Chemistry"
            },
            {
                "name": "entertext",
                "value": "India is a country"
            }
        ]
    }

@pytest.mark.integration
def test_list_custom_metrics(custom_metric_manager):
    """Test listing custom metrics"""
    metrics = custom_metric_manager.list_custom_metrics()
    assert isinstance(metrics, list)
    # Each metric should be a tuple of (name, id)
    for metric in metrics:
        assert isinstance(metric, tuple)
        assert len(metric) == 2

@pytest.mark.integration
def test_create_custom_metric(custom_metric_manager):
    """Test creating a new custom metric"""
    # Define test data
    current_time = time.time()
    microseconds = int((current_time - int(current_time)) * 1000000)
    metric_name = f"Test_Metric_{int(current_time)}_{microseconds}"  # Unix timestamp and microseconds for uniqueness
    description = "Test Description"

    # Create custom metric
    metric_id = custom_metric_manager.create_custom_metrics(
        metric_name=metric_name,
        description=description
    )
    assert metric_id is not None
    assert isinstance(metric_id, int)
    return metric_id

@pytest.mark.integration
def test_run_step(custom_metric_manager, test_steps):
    """Test running custom metric steps"""
    # First create a custom metric
    metric_id = test_create_custom_metric(custom_metric_manager)
    
    # Add models and providers
    model = "gpt-4o-mini"
    provider = "openai"

    # Run custom metric steps
    output_steps = custom_metric_manager.run_step(
        custom_metric_id=metric_id,
        steps=test_steps,
        model=model,
        provider=provider
    )
    print("\n=== test_run_step output ===")
    print("metric_id:", metric_id)
    print("output_steps:", json.dumps(output_steps, indent=2))
    return output_steps, metric_id

@pytest.mark.integration
def test_get_grading_criteria(custom_metric_manager):
    """Test getting grading criteria"""
    grading_criteria = custom_metric_manager.get_grading_criteria()
    assert isinstance(grading_criteria, list)
    assert len(grading_criteria) > 0
    assert 'Float (0 to 1)' in grading_criteria
    assert 'Boolean (0 or 1)' in grading_criteria

@pytest.mark.integration
def test_verify_grading_criteria(custom_metric_manager, test_steps):
    """Test verifying grading criteria"""
    output_steps, metric_id = test_run_step(custom_metric_manager, test_steps)
    # Get grading criteria first
    grading_criteria = custom_metric_manager.get_grading_criteria()
    
    # Verify grading criteria
    grading_criteria_result = custom_metric_manager.verify_grading_criteria(
        custom_metric_id=metric_id,
        grading_criteria="Float (0 to 1)",
        steps=output_steps
    )
    assert grading_criteria_result is not None
    return grading_criteria_result

@pytest.mark.integration
def test_commit_custom_metric(custom_metric_manager, test_steps):
    """Test committing a custom metric"""
    output_steps, metric_id = test_run_step(custom_metric_manager, test_steps)
    
    # Commit metric
    model = "gpt-4"
    provider = "openai"
    final_reason = None
    commit_message = "Test commit"
    metric_commit_output = custom_metric_manager.commit_custom_metric(
        custom_metric_id=metric_id,
        steps=test_steps,
        model=model,
        provider=provider,
        output_steps=output_steps,
        final_reason=final_reason,
        commit_message=commit_message
    )
    assert metric_commit_output is not None
    assert isinstance(metric_commit_output, dict)
    assert 'metricId' in metric_commit_output
    assert 'versionName' in metric_commit_output
    return metric_commit_output

@pytest.mark.integration
def test_deploy_custom_metric(custom_metric_manager, test_steps):
    """Test deploying a custom metric"""
    metric_output = test_commit_custom_metric(custom_metric_manager, test_steps)
    
    # Deploy metric
    deploy_output = custom_metric_manager.deploy_custom_metric(
        custom_metric_id=metric_output['metricId'],
        version_name=metric_output['versionName']
    )
    assert deploy_output is not None

@pytest.mark.integration
def test_get_custom_metric_versions(custom_metric_manager):
    """Test getting custom metric versions"""
    metric_id = test_create_custom_metric(custom_metric_manager)
    versions = custom_metric_manager.get_custom_metric_versions(metric_id)
    assert isinstance(versions, list)
    for version in versions:
        assert isinstance(version, dict)
        assert "version_name" in version
        assert "status" in version
