import pytest
import os
import dotenv
dotenv.load_dotenv()
import pandas as pd
from datetime import datetime
from typing import Dict, List
from unittest.mock import patch, Mock
import requests
from ragaai_catalyst import Dataset,RagaAICatalyst


@pytest.fixture
def base_url():
    return "https://catalyst.raga.ai/api"

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}

@pytest.fixture
def dataset(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return Dataset(project_name="prompt_metric_dataset")

def test_list_dataset(dataset) -> List[str]:
    datasets = dataset.list_datasets()
    return datasets


def test_get_dataset_columns(dataset)  -> List[str]:
    dataset_column = dataset.get_dataset_columns(dataset_name="ritika_dataset")
    return dataset_column

def test_incorrect_dataset(dataset):
    with pytest.raises(ValueError, match="Please enter a valid dataset name"):
        dataset.get_dataset_columns(dataset_name="ritika_datset")

def test_get_schema_mapping(dataset):
    schema_mapping_columns= dataset.get_schema_mapping()
    return schema_mapping_columns


def test_upload_csv(dataset):
    project_name = 'prompt_metric_dataset'

    schema_mapping = {
        'Query': 'prompt',
        'Response': 'response',
        'Context': 'context',
        'ExpectedResponse': 'expected_response',
    }

    csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    dataset_name = f"schema_metric_dataset_ritika_{timestamp}"  

    

    dataset.create_from_csv(
        csv_path=csv_path,
        dataset_name=dataset_name,
        schema_mapping=schema_mapping
    )

def test_upload_csv_repeat_dataset(dataset):
    with pytest.raises(ValueError, match="already exists"):
        project_name = 'prompt_metric_dataset'

        schema_mapping = {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expected_response',
        }

        csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

        dataset.create_from_csv(
            csv_path=csv_path,
            dataset_name="schema_metric_dataset_ritika_3",
            schema_mapping=schema_mapping
        )


def test_upload_csv_no_schema_mapping(dataset):
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        project_name = 'prompt_metric_dataset'

        schema_mapping = {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expected_response',
        }

        csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

        dataset.create_from_csv(
            csv_path=csv_path,
            dataset_name="schema_metric_dataset_ritika_3",
        )

def test_upload_csv_empty_csv_path(dataset):
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        project_name = 'prompt_metric_dataset'

        schema_mapping = {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expected_response',
        }

        csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

        dataset.create_from_csv(
            csv_path="",
            dataset_name="schema_metric_dataset_ritika_12",
            schema_mapping=schema_mapping

        )

def test_upload_csv_empty_schema_mapping(dataset):
    with pytest.raises(AttributeError):
        project_name = 'prompt_metric_dataset'

        schema_mapping = {
            'Query': 'prompt',
            'Response': 'response',
            'Context': 'context',
            'ExpectedResponse': 'expected_response',
        }

        csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

        dataset.create_from_csv(
            csv_path=csv_path,
            dataset_name="schema_metric_dataset_ritika_12",
            schema_mapping=""

        )


def test_upload_csv_invalid_schema(dataset):
    with pytest.raises(ValueError, match="Invalid schema mapping provided"):

        project_name = 'prompt_metric_dataset'

        schema_mapping={
            'prompt': 'prompt',
            'response': 'response',
            'chatId': 'chatId',
            'chatSequence': 'chatSequence'
        }

        csv_path= "/Users/siddharthakosti/Downloads/catalyst_error_handling/catalyst_v2/catalyst_v2_new_1/data/prompt_metric_dataset_v1.csv"

        dataset.create_from_csv(
            csv_path=csv_path,
            dataset_name="schema_metric_dataset_ritika_12",
            schema_mapping=schema_mapping)
