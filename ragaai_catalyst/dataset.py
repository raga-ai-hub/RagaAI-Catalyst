import os
import requests
from .utils import response_checker
from typing import Union
import logging
from .ragaai_catalyst import RagaAICatalyst
import pandas as pd
logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token


class Dataset:
    BASE_URL = None
    TIMEOUT = 30

    def __init__(self, project_name):
        self.project_name = project_name
        self.num_projects = 99999
        Dataset.BASE_URL = RagaAICatalyst.BASE_URL
        headers = {
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.get(
                f"{Dataset.BASE_URL}/v2/llm/projects?size={self.num_projects}",
                headers=headers,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            logger.debug("Projects list retrieved successfully")
            
            project_list = [
                project["name"] for project in response.json()["data"]["content"]
            ]

            if project_name not in project_list:
                raise ValueError("Project not found. Please enter a valid project name")

            self.project_id = [
                project["id"] for project in response.json()["data"]["content"] if project["name"] == project_name
            ][0]

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve projects list: {e}")
            raise

    def list_datasets(self):
        """
        Retrieves a list of datasets for a given project.

        Returns:
            list: A list of dataset names.

        Raises:
            None.
        """

        def make_request():
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
            try:
                response = requests.post(
                    f"{Dataset.BASE_URL}/v2/llm/dataset",
                    headers=headers,
                    json=json_data,
                    timeout=Dataset.TIMEOUT,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to list datasets: {e}")
                raise

        try:
            response = make_request()
            response_checker(response, "Dataset.list_datasets")
            if response.status_code == 401:
                get_token()  # Fetch a new token and set it in the environment
                response = make_request()  # Retry the request
            if response.status_code != 200:
                return {
                    "status_code": response.status_code,
                    "message": response.json(),
                }
            datasets = response.json()["data"]["content"]
            dataset_list = [dataset["name"] for dataset in datasets]
            return dataset_list
        except Exception as e:
            logger.error(f"Error in list_datasets: {e}")
            raise

    def get_schema_mapping(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }
        try:
            response = requests.get(
                f"{Dataset.BASE_URL}/v1/llm/schema-elements",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            response.raise_for_status()
            response_data = response.json()["data"]["schemaElements"]
            if not response.json()['success']:
                raise ValueError('Unable to fetch Schema Elements for the CSV')
            return response_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get CSV schema: {e}")
            raise

    ###################### CSV Upload APIs ###################

    def get_dataset_columns(self, dataset_name):
        list_dataset = self.list_datasets()
        if dataset_name not in list_dataset:
            raise ValueError(f"Dataset {dataset_name} does not exists. Please enter a valid dataset name")

        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }
        headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
        json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
        try:
            response = requests.post(
                f"{Dataset.BASE_URL}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=Dataset.TIMEOUT,
            )
            response.raise_for_status()
            datasets = response.json()["data"]["content"]
            dataset_id = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name][0]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list datasets: {e}")
            raise

        try:
            response = requests.get(
                f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            response.raise_for_status()
            dataset_columns = response.json()["data"]["datasetColumnsResponses"]
            dataset_columns = [item["displayName"] for item in dataset_columns]
            dataset_columns = [data for data in dataset_columns if not data.startswith('_')]
            if not response.json()['success']:
                raise ValueError('Unable to fetch details of for the CSV')
            return dataset_columns
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get CSV columns: {e}")
            raise

    def create_from_csv(self, csv_path, dataset_name, schema_mapping):
        list_dataset = self.list_datasets()
        if dataset_name in list_dataset:
            raise ValueError(f"Dataset name {dataset_name} already exists. Please enter a unique dataset name")

        #### get presigned URL
        def get_presignedUrl():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            try:
                response = requests.get(
                    f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url",
                    headers=headers,
                    timeout=Dataset.TIMEOUT,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get presigned URL: {e}")
                raise

        try:
            presignedUrl = get_presignedUrl()
            if presignedUrl['success']:
                url = presignedUrl['data']['presignedUrl']
                filename = presignedUrl['data']['fileName']
            else:
                raise ValueError('Unable to fetch presignedUrl')
        except Exception as e:
            logger.error(f"Error in get_presignedUrl: {e}")
            raise

        #### put csv to presigned URL
        def put_csv_to_presignedUrl(url):
            headers = {
                'Content-Type': 'text/csv',
                'x-ms-blob-type': 'BlockBlob',
            }
            try:
                with open(csv_path, 'rb') as file:
                    response = requests.put(
                        url,
                        headers=headers,
                        data=file,
                        timeout=Dataset.TIMEOUT,
                    )
                    response.raise_for_status()
                    return response
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to put CSV to presigned URL: {e}")
                raise

        try:

            put_csv_response = put_csv_to_presignedUrl(url)
            print(put_csv_response)
            if put_csv_response.status_code not in (200, 201):
                raise ValueError('Unable to put csv to the presignedUrl')
        except Exception as e:
            logger.error(f"Error in put_csv_to_presignedUrl: {e}")
            raise

        ## Upload csv to elastic
        def upload_csv_to_elastic(data):
            header = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id)
            }
            try:
                response = requests.post(
                    f"{Dataset.BASE_URL}/v2/llm/dataset/csv",
                    headers=header,
                    json=data,
                    timeout=Dataset.TIMEOUT,
                )
                if response.status_code==400:
                    raise ValueError(response.json()["message"])
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to upload CSV to elastic: {e}")
                raise

        def generate_schema(mapping):
            result = {}
            for column, schema_element in mapping.items():
                result[column] = {"columnType": schema_element}
            return result

        try:
            schema_mapping = generate_schema(schema_mapping)
            data = {
                "projectId": str(self.project_id),
                "datasetName": dataset_name,
                "fileName": filename,
                "schemaMapping": schema_mapping,
                "opType": "insert",
                "description": ""
            }
            upload_csv_response = upload_csv_to_elastic(data)
            if not upload_csv_response['success']:
                raise ValueError('Unable to upload csv')
            else:
                print(upload_csv_response['message'])
        except Exception as e:
            logger.error(f"Error in create_from_csv: {e}")
            raise
