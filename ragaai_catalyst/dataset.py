import os
import csv
import json
import tempfile
import requests
from .utils import response_checker
from typing import Union
import logging
from .ragaai_catalyst import RagaAICatalyst
import pandas as pd
logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token

# Job status constants
JOB_STATUS_FAILED = "failed"
JOB_STATUS_IN_PROGRESS = "in_progress"
JOB_STATUS_COMPLETED = "success"

class Dataset:
    BASE_URL = None
    TIMEOUT = 30

    def __init__(self, project_name):
        self.project_name = project_name
        self.num_projects = 99999
        Dataset.BASE_URL = RagaAICatalyst.BASE_URL
        self.jobId = None
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
                self.jobId = upload_csv_response['data']['jobId']
        except Exception as e:
            logger.error(f"Error in create_from_csv: {e}")
            raise

    def add_rows(self, csv_path, dataset_name):
        """
        Add rows to an existing dataset from a CSV file.

        Args:
            csv_path (str): Path to the CSV file to be added
            dataset_name (str): Name of the existing dataset to add rows to

        Raises:
            ValueError: If dataset does not exist or columns are incompatible
        """
        # Get existing dataset columns
        existing_columns = self.get_dataset_columns(dataset_name)

        # Read the CSV file to check columns
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_columns = df.columns.tolist()
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise ValueError(f"Unable to read CSV file: {e}")

        # Check column compatibility
        for column in existing_columns:
            if column not in csv_columns:
                df[column] = None  

        # Get presigned URL for the CSV
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

        # Upload CSV to presigned URL
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
            if put_csv_response.status_code not in (200, 201):
                raise ValueError('Unable to put csv to the presignedUrl')
        except Exception as e:
            logger.error(f"Error in put_csv_to_presignedUrl: {e}")
            raise

        # Prepare schema mapping (assuming same mapping as original dataset)
        def generate_schema_mapping(dataset_name):
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {
                "size": 12, 
                "page": "0", 
                "projectId": str(self.project_id), 
                "search": ""
            }
            try:
                # First get dataset details
                response = requests.post(
                    f"{Dataset.BASE_URL}/v2/llm/dataset",
                    headers=headers,
                    json=json_data,
                    timeout=Dataset.TIMEOUT,
                )
                response.raise_for_status()
                datasets = response.json()["data"]["content"]
                dataset_id = [dataset["id"] for dataset in datasets if dataset["name"]==dataset_name][0]

                # Get dataset details to extract schema mapping
                response = requests.get(
                    f"{Dataset.BASE_URL}/v2/llm/dataset/{dataset_id}?initialCols=0",
                    headers=headers,
                    timeout=Dataset.TIMEOUT,
                )
                response.raise_for_status()
                
                # Extract schema mapping
                schema_mapping = {}
                for col in response.json()["data"]["datasetColumnsResponses"]:
                    schema_mapping[col["displayName"]] = {"columnType": col["columnType"]}
                
                return schema_mapping
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get schema mapping: {e}")
                raise

        # Upload CSV to elastic
        try:
            schema_mapping = generate_schema_mapping(dataset_name)
            
            data = {
                "projectId": str(self.project_id),
                "datasetName": dataset_name,
                "fileName": filename,
                "schemaMapping": schema_mapping,
                "opType": "update",  # Use update for adding rows
                "description": "Adding new rows to dataset"
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id)
            }
            
            response = requests.post(
                f"{Dataset.BASE_URL}/v2/llm/dataset/csv",
                headers=headers,
                json=data,
                timeout=Dataset.TIMEOUT,
            )
            
            if response.status_code == 400:
                raise ValueError(response.json().get("message", "Failed to add rows"))
            
            response.raise_for_status()
            
            # Check response
            response_data = response.json()
            if response_data.get('success', False):
                print(f"{response_data['message']}")
                self.jobId = response_data['data']['jobId']
            else:
                raise ValueError(response_data.get('message', 'Failed to add rows'))
        
        except Exception as e:
            logger.error(f"Error in add_rows_to_dataset: {e}")
            raise

    def add_columns(self, text_fields, dataset_name, column_name, provider, model, variables={}):
        """
        Add a column to a dataset with dynamically fetched model parameters
        
        Args:
            project_id (int): Project ID
            dataset_id (int): Dataset ID
            column_name (str): Name of the new column
            provider (str): Name of the model provider
            model (str): Name of the model
        """
        # First, get model parameters

        # Validate text_fields input
        if not isinstance(text_fields, list):
            raise ValueError("text_fields must be a list of dictionaries")
        
        for field in text_fields:
            if not isinstance(field, dict) or 'role' not in field or 'content' not in field:
                raise ValueError("Each text field must be a dictionary with 'role' and 'content' keys")
            
        # First, get the dataset ID
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Id": str(self.project_id),
        }
        json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
        
        try:
            # Get dataset list
            response = requests.post(
                f"{Dataset.BASE_URL}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=Dataset.TIMEOUT,
            )
            response.raise_for_status()
            datasets = response.json()["data"]["content"]
            
            # Find dataset ID
            dataset_id = next((dataset["id"] for dataset in datasets if dataset["name"] == dataset_name), None)
            
            if dataset_id is None:
                raise ValueError(f"Dataset {dataset_name} not found")



            parameters_url= f"{Dataset.BASE_URL}/playground/providers/models/parameters/list"
            
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            
            # Fetch model parameters
            parameters_payload = {
                "providerName": provider,
                "modelName": model
            }
        
            # Get model parameters
            params_response = requests.post(
                parameters_url, 
                headers=headers, 
                json=parameters_payload, 
                timeout=30
            )
            params_response.raise_for_status()
            
            # Extract parameters
            all_parameters = params_response.json().get('data', [])
            
            # Filter and transform parameters for add-column API
            formatted_parameters = []
            for param in all_parameters:
                value = param.get('value')
                param_type = param.get('type')

                if value is None:
                    formatted_param = {
                        "name": param.get('name'),
                        "value": None,  # Pass None if the value is null
                        "type": param.get('type')
                    }
                else:
                    # Improved type handling
                    if param_type == "float":
                        value = float(value)  # Ensure value is converted to float
                    elif param_type == "int":
                        value = int(value)  # Ensure value is converted to int
                    elif param_type == "bool":
                        value = bool(value)  # Ensure value is converted to bool
                    elif param_type == "string":
                        value = str(value)  # Ensure value is converted to string
                    else:
                        raise ValueError(f"Unsupported parameter type: {param_type}")  # Handle unsupported types

                    formatted_param = {
                        "name": param.get('name'),
                        "value": value,
                        "type": param.get('type')
                    }
                formatted_parameters.append(formatted_param)
            dataset_id = next((dataset["id"] for dataset in datasets if dataset["name"] == dataset_name), None)

            # Prepare payload for add column API
            add_column_payload = {
                "rowFilterList": [],
                "columnName": column_name,
                "datasetId": dataset_id,
                "variables": variables,
                "promptTemplate": {
                    "textFields": text_fields,
                    "modelSpecs": {
                        "model": f"{provider}/{model}",
                        "parameters": formatted_parameters
                    }
                }
            }
            if variables:
                variable_specs = []
                for key, values in variables.items():
                    variable_specs.append({
                        "name": key,
                        "type": "string",
                        "schema": "query"
                    })
                add_column_payload["promptTemplate"]["variableSpecs"] = variable_specs
            
            # Make API call to add column
            add_column_url = f"{Dataset.BASE_URL}/v2/llm/dataset/add-column"
            
            response = requests.post(
                add_column_url, 
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                    "X-Project-Id": str(self.project_id)
                }, 
                json=add_column_payload,
                timeout=30
            )
            
            # Check response
            response.raise_for_status()
            response_data = response.json()
            
            if response_data.get('success', False):
                print(f"Column '{column_name}' added successfully to dataset '{dataset_name}'")
                self.jobId = response_data['data']['jobId']
            else:
                raise ValueError(response_data.get('message', 'Failed to add column'))
        
        except requests.exceptions.RequestException as e:
            print(f"Error adding column: {e}")
            raise

    def get_status(self):
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{Dataset.BASE_URL}/job/status', 
                headers=headers, 
                timeout=30)
            response.raise_for_status()
            if response.json()["success"]:

                status_json = [item["status"] for item in response.json()["data"]["content"] if item["id"]==self.jobId]
                status_json = status_json[0]
                if status_json == "Failed":
                    print("Job failed. No results to fetch.")
                    return JOB_STATUS_FAILED
                elif status_json == "In Progress":
                    print(f"Job in progress. Please wait while the job completes.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to track")
                    return JOB_STATUS_IN_PROGRESS
                elif status_json == "Completed":
                    print(f"Job completed. Fetching results.\nVisit Job Status: {Dataset.BASE_URL.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to check")
                    return JOB_STATUS_COMPLETED
                else:
                    logger.error(f"Unknown status received: {status_json}")
                    return JOB_STATUS_FAILED
            else:
                logger.error("Request was not successful")
                return JOB_STATUS_FAILED
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return JOB_STATUS_FAILED
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
            return JOB_STATUS_FAILED
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
            return JOB_STATUS_FAILED
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
            return JOB_STATUS_FAILED
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return JOB_STATUS_FAILED

    def _jsonl_to_csv(self, jsonl_file, csv_file):
        """Convert a JSONL file to a CSV file."""
        with open(jsonl_file, 'r', encoding='utf-8') as infile:
            data = [json.loads(line) for line in infile]
        
        if not data:
            print("Empty JSONL file.")
            return
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Converted {jsonl_file} to {csv_file}")

    def create_from_jsonl(self, jsonl_path, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting JSONL to CSV: {e}")
            raise
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def add_rows_from_jsonl(self, jsonl_path, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.add_rows(tmp_csv_path, dataset_name)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting JSONL to CSV: {e}")
            raise
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def create_from_df(self, df, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting DataFrame to CSV: {e}")
            raise
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")

    def add_rows_from_df(self, df, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.add_rows(tmp_csv_path, dataset_name)
        except (IOError, UnicodeError) as e:
            logger.error(f"Error converting DataFrame to CSV: {e}")
            raise
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception as e:
                    logger.error(f"Error removing temporary CSV file: {e}")