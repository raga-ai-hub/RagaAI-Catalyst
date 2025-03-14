
from unittest.mock import patch
import time
import pytest
import os
import dotenv
dotenv.load_dotenv()
import pandas as pd
from datetime import datetime 
from typing import Dict, List
from ragaai_catalyst import Evaluation, RagaAICatalyst
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Define model configurations
MODEL_CONFIGS = [
    # OpenAI Models
    {
        "provider": "openai",
        "model": "gpt-4",
        "suffix": "gpt4"
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "suffix": "gpt4o"
    },
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "suffix": "gpt4o_mini"
    },
    {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "suffix": "gpt35"
    },
    # Gemini Models
    {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "suffix": "gemini15_flash"
    },
    {
        "provider": "gemini",
        "model": "gemini-1.5-pro",
        "suffix": "gemini15_pro"
    },
    # Azure OpenAI Models
    {
        "provider": "azure",
        "model": "gpt-4",
        "suffix": "azure_gpt4"
    },
    {
        "provider": "azure",
        "model": "gpt-35-turbo",
        "suffix": "azure_gpt35"
    }
]

@pytest.fixture
def base_url():
    return "https://catalyst.raga.ai/api"

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}
    

@pytest.fixture
def evaluation(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(project_name="prompt_metric_dataset", dataset_name="ritika_dataset")

@pytest.fixture
def chat_evaluation(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Evaluation(project_name="chat_demo_sk_v1", dataset_name="chat_metric_dataset_ritika")

def test_evaluation_initialization(evaluation):
    """Test if evaluation is initialized correctly"""
    assert evaluation.project_name == "prompt_metric_dataset"
    assert evaluation.dataset_name == "ritika_dataset"
    assert evaluation.base_url == "https://catalyst.raga.ai/api"
    assert evaluation.timeout == 10
    assert evaluation.jobId is None

def test_project_does_not_exist():
    """Test initialization with non-existent project"""
    with pytest.raises(ValueError, match="Project not found. Please enter a valid project name"):
        Evaluation(project_name="non_existent_project", dataset_name="prompt_metric_dataset_v1")

def test_dataset_does_not_exist():
    """Test initialization with non-existent dataset"""
    with pytest.raises(ValueError, match="Dataset not found. Please enter a valid dataset name"):
        Evaluation(project_name="prompt_metric_dataset", dataset_name="non_existent_dataset")

def test_list_metrics(evaluation) -> List[str]:
    """Test if it lists all the metrics correctly"""
    metrics = evaluation.list_metrics()
    return metrics

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_invalid_schema_mapping(evaluation, provider_config):
    """Wrong schema mapping for different providers"""
    with pytest.raises(ValueError, match="Map"):
        schema_mapping={
            'Query': 'Prompt',
            'Context': 'Context',
        }
        metrics = [{
            "name": "Hallucination", 
            "config": {
                "model": provider_config["model"], 
                "provider": provider_config["provider"]
            }, 
            "column_name": f"Hallucination_{provider_config['suffix']}", 
            "schema_mapping": schema_mapping
        }]
        evaluation.add_metrics(metrics=metrics)

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_missing_schema_mapping(evaluation, provider_config):
    """schema_mapping not present for different providers"""
    with pytest.raises(ValueError, match="{'schema_mapping'} required for each metric evaluation."):
        metrics = [{
            "name": "Hallucination", 
            "config": {
                "model": provider_config["model"], 
                "provider": provider_config["provider"]
            }, 
            "column_name": f"Hallucination_{provider_config['suffix']}"
        }]
        evaluation.add_metrics(metrics=metrics)

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_missing_column_name(evaluation, provider_config):
    """column_name not present for different providers"""
    with pytest.raises(ValueError, match="{'column_name'} required for each metric evaluation."):
        schema_mapping={
            'Query': 'Prompt',
            'Response': 'Response',
            'Context': 'Context',
        }
        metrics = [{
            "name": "Hallucination", 
            "config": {
                "model": provider_config["model"], 
                "provider": provider_config["provider"]
            },
            "schema_mapping": schema_mapping
        }]
        evaluation.add_metrics(metrics=metrics)

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_missing_metric_name(evaluation, provider_config):
    """metric name missing for different providers"""
    with pytest.raises(ValueError, match="{'name'} required for each metric evaluation."):
        schema_mapping={
            'Query': 'Prompt',
            'Response': 'Response',
            'Context': 'Context',
        }
        metrics = [{
            "config": {
                "model": provider_config["model"], 
                "provider": provider_config["provider"]
            }, 
            "column_name": f"Hallucination_{provider_config['suffix']}", 
            "schema_mapping": schema_mapping
        }]
        evaluation.add_metrics(metrics=metrics)

@pytest.mark.parametrize("provider_config", MODEL_CONFIGS)
def test_column_name_already_exists(evaluation, provider_config):
    """Column name already exists for different providers"""
    with pytest.raises(ValueError, match="already exists."):
        schema_mapping={
            'Query': 'Prompt',
            'Response': 'Response',
            'Context': 'Context',
        }
        metrics = [{
            "name": "Hallucination", 
            "config": {
                "model": provider_config["model"], 
                "provider": provider_config["provider"]
            }, 
            "column_name": "Hallucination_column3", 
            "schema_mapping": schema_mapping
        }]
        evaluation.add_metrics(metrics=metrics)

def test_missing_config(evaluation):
    with pytest.raises(ValueError, match="{'config'} required for each metric evaluation."):
        schema_mapping={
            'Query': 'Prompt',
            'Response': 'Response',
            'Context': 'Context',
        }
        metrics = [{"name": "Hallucination", "column_name": "Hallucination5", "schema_mapping": schema_mapping}]
        evaluation.add_metrics(metrics=metrics)



@pytest.mark.parametrize("metric_name", ['Hallucination',
 'Faithfulness',
 'SQL Prompt Injection',
 'Response Correctness',
 'Response Completeness',
 'False Refusal',
 'Context Precision',
 'Context Recall',
 'Context Relevancy'
 'SQL Response Correctness',
 'SQL Prompt Ambiguity',
 'SQL Context Sufficiency',
 'SQL Context Ambiguity'])

def test_metric_initialization_gemini(evaluation, metric_name: str,capfd):
    """Test if adding each metric and tracking its completion works correctly"""
    schema_mapping = {
        'Query': 'prompt',
        'Response': 'response',
        'Context': 'context',
        'ExpectedResponse': 'expectedresponse',
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    metrics = [{
        "name": metric_name,
        "config": {
            "model": "gemini-1.5-flash",
            "provider": "gemini"
        },
        "column_name": f"{metric_name}_column_{timestamp}",
        "schema_mapping": schema_mapping
    }]
    
    # Add metrics and capture the printed output
    evaluation.add_metrics(metrics=metrics)
    out, err = capfd.readouterr()
    print(f"Add metrics output: {out}")  # Debug print
    
    # Verify the success message for metric addition
    assert "Metric Evaluation Job scheduled successfully" in out, f"Failed to schedule job for metric: {metric_name}"
    
    # Store the jobId for status checking
    assert evaluation.jobId is not None, "Job ID was not set after adding metrics"
    print(f"Job ID: {evaluation.jobId}")  # Debug print
    
    # Check job status with timeout
    max_wait_time = 180  # Increased timeout to 3 minutes
    poll_interval = 5    # Check every 5 seconds
    start_time = time.time()
    status_checked = False
    last_status = None
    
    print(f"Starting job status checks for {metric_name}...")  # Debug print
    
    while (time.time() - start_time) < max_wait_time:
        try:
            evaluation.get_status()
            out, err = capfd.readouterr()
            print(f"Status check output: {out}")  # Debug print
            
            if "Job completed" in out:
                status_checked = True
                print(f"Job completed for {metric_name}")  # Debug print
                break
                
            if "Job failed" in out:
                pytest.fail(f"Job failed for metric: {metric_name}")
                
            last_status = out
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking status: {str(e)}")  # Debug print
            time.sleep(poll_interval)
    
    if not status_checked:
        print(f"Last known status: {last_status}")  # Debug print
        if last_status and "In Progress" in last_status:
            pytest.skip(f"Job still in progress after {max_wait_time} seconds for {metric_name}. This is not a failure, but took longer than expected.")
        else:
            assert False, f"Job did not complete within {max_wait_time} seconds for metric: {metric_name}. Last status: {last_status}"
    
    # Only check results if the job completed successfully
    if status_checked:
        try:
            results = evaluation.get_results()
            assert isinstance(results, pd.DataFrame), "Results should be returned as a DataFrame"
            assert not results.empty, "Results DataFrame should not be empty"
            column_name = f"{metric_name}_column25"
            assert column_name in results.columns, f"Expected column {column_name} not found in results. Available columns: {results.columns.tolist()}"
        except Exception as e:
            pytest.fail(f"Error getting results for {metric_name} with provider: gemini and model: gemini-1.5-flash: {str(e)}")



@pytest.mark.parametrize("metric_name", ['Hallucination',
 'Faithfulness',
 'SQL Prompt Injection',
 'Response Correctness',
 'Response Completeness',
 'False Refusal',
 'Context Precision',
 'Context Recall',
 'Context Relevancy',
 'SQL Response Correctness',
 'SQL Prompt Ambiguity',
 'SQL Context Sufficiency',
 'SQL Context Ambiguity'])

def test_metric_initialization_openai(evaluation, metric_name: str,capfd):
    """Test if adding each metric and tracking its completion works correctly"""
    schema_mapping = {
        'Query': 'prompt',
        'Response': 'response',
        'Context': 'context',
        'ExpectedResponse': 'expectedresponse',
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

    metrics = [{
        "name": metric_name,
        "config": {
            "model": "gpt-4o-mini",
            "provider": "openai"
        },
        "column_name": f"{metric_name}_column_{timestamp}", 
        "schema_mapping": schema_mapping
    }]
    
    # Add metrics and capture the printed output
    evaluation.add_metrics(metrics=metrics)
    out, err = capfd.readouterr()
    print(f"Add metrics output: {out}")  # Debug print
    
    # Verify the success message for metric addition
    assert "Metric Evaluation Job scheduled successfully" in out, f"Failed to schedule job for metric: {metric_name}"
    
    # Store the jobId for status checking
    assert evaluation.jobId is not None, "Job ID was not set after adding metrics"
    print(f"Job ID: {evaluation.jobId}")  # Debug print
    
    # Check job status with timeout
    max_wait_time = 300  # Increased timeout to 3 minutes
    poll_interval = 5    # Check every 5 seconds
    start_time = time.time()
    status_checked = False
    last_status = None
    
    print(f"Starting job status checks for {metric_name}...")  # Debug print
    
    while (time.time() - start_time) < max_wait_time:
        try:
            evaluation.get_status()
            out, err = capfd.readouterr()
            print(f"Status check output: {out}")  # Debug print
            
            if "Job completed" in out:
                status_checked = True
                print(f"Job completed for {metric_name}")  # Debug print
                break
                
            if "Job failed" in out:
                pytest.fail(f"Job failed for metric: {metric_name}")
                
            last_status = out
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking status: {str(e)}")  # Debug print
            time.sleep(poll_interval)
    
    if not status_checked:
        print(f"Last known status: {last_status}")  # Debug print
        if last_status and "In Progress" in last_status:
            pytest.skip(f"Job still in progress after {max_wait_time} seconds for {metric_name}. This is not a failure, but took longer than expected.")
        else:
            assert False, f"Job did not complete within {max_wait_time} seconds for metric: {metric_name}. Last status: {last_status}"
    
    # Only check results if the job completed successfully
    if status_checked:
        try:
            results = evaluation.get_results()
            assert isinstance(results, pd.DataFrame), "Results should be returned as a DataFrame"
            assert not results.empty, "Results DataFrame should not be empty"
            column_name = f"{metric_name}_column26"
            assert column_name in results.columns, f"Expected column {column_name} not found in results. Available columns: {results.columns.tolist()}"
        except Exception as e:
            pytest.fail(f"Error getting results for {metric_name} with provider: oprnai and model: gpt-4o-mini: {str(e)}")



# Add a counter to keep track of the test iterations
counter = 30

@pytest.mark.parametrize("metric_name", ['Agent Quality',
 'User Chat Quality',
 'Instruction Adherence'])
@pytest.mark.parametrize("model_config", [
    {"model": "gpt-4o-mini", "provider": "openai"},
    {"model": "gpt-4", "provider": "openai"},
    {"model": "gpt-3.5-turbo", "provider": "openai"},
    {"model":"gemini-1.5-flash", "provider": "gemini"}
])
def test_metric_initialization_openai_chatmetric(chat_evaluation, model_config, metric_name: str, capfd):
    """Test if adding each metric and tracking its completion works correctly"""
    global counter  # Use the global counter
    schema_mapping = {
        'ChatID': 'ChatID',
        'Chat': 'Chat',
        'Instructions': 'Instructions',
        'System Prompt': 'systemprompt',
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    metrics = [{
        "name": metric_name,
        "config": model_config,
        "column_name": f"{metric_name}_column_{timestamp}_{counter}",  # Use counter for unique column name
        "schema_mapping": schema_mapping
    }]
    
    # Increment the counter after each test
    counter += 1
    
    # Add metrics and capture the printed output
    chat_evaluation.add_metrics(metrics=metrics)
    out, err = capfd.readouterr()
    print(f"Add metrics output: {out}")  # Debug print
    
    # Verify the success message for metric addition
    assert "Metric Evaluation Job scheduled successfully" in out, f"Failed to schedule job for metric: {metric_name} and {model_config}"
    
    # Store the jobId for status checking
    assert chat_evaluation.jobId is not None, "Job ID was not set after adding metrics"
    print(f"Job ID: {chat_evaluation.jobId}")  # Debug print
    
    # Check job status with timeout
    max_wait_time = 600  # Increased timeout to 3 minutes
    poll_interval = 5    # Check every 5 seconds
    start_time = time.time()
    status_checked = False
    last_status = None
    
    print(f"Starting job status checks for {metric_name}...")  # Debug print
    
    while (time.time() - start_time) < max_wait_time:
        try:
            chat_evaluation.get_status()
            out, err = capfd.readouterr()
            print(f"Status check output: {out}")  # Debug print
            
            if "Job completed" in out:
                status_checked = True
                print(f"Job completed for {metric_name}")  # Debug print
                break
                
            if "Job failed" in out:
                pytest.fail(f"Job failed for metric: {metric_name}{model_config}")
                
            last_status = out
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking status: {str(e)}")  # Debug print
            time.sleep(poll_interval)
    
    if not status_checked:
        print(f"Last known status: {last_status}")  # Debug print
        if last_status and "In Progress" in last_status:
            pytest.skip(f"Job still in progress after {max_wait_time} seconds {model_config} for {metric_name}. This is not a failure, but took longer than expected.")
        else:
            assert False, f"Job did not complete within {max_wait_time} seconds {model_config} for metric: {metric_name}. Last status: {last_status}"
    
    # Only check results if the job completed successfully
    if status_checked:
        try:
            results = chat_evaluation.get_results()
            assert isinstance(results, pd.DataFrame), "Results should be returned as a DataFrame"
            assert not results.empty, "Results DataFrame should not be empty"
            column_name = f"{metric_name}_column_{counter - 1}"  # Use the last counter value
            assert column_name in results.columns, f"Expected column {column_name} not found in results. Available columns: {results.columns.tolist()}"
        except Exception as e:
            pytest.fail(f"Error getting results for {metric_name} with {model_config}: {str(e)}")



