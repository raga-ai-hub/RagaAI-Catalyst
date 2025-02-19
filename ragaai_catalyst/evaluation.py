import os
import requests
import pandas as pd
import io
from .ragaai_catalyst import RagaAICatalyst
import logging
import json

logger = logging.getLogger(__name__)

# Job status constants
JOB_STATUS_FAILED = "failed"
JOB_STATUS_IN_PROGRESS = "in_progress"
JOB_STATUS_COMPLETED = "success"

class Evaluation:

    def __init__(self, project_name, dataset_name):
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 20
        self.jobId = None
        self.num_projects=99999

        try:
            response = requests.get(
                f"{self.base_url}/v2/llm/projects?size={self.num_projects}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
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

        try:

            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
            response = requests.post(
                f"{self.base_url}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )
            
            response.raise_for_status()
            datasets_content = response.json()["data"]["content"]
            dataset_list = [dataset["name"] for dataset in datasets_content]

            if dataset_name not in dataset_list:
                raise ValueError("Dataset not found. Please enter a valid dataset name")
                
            self.dataset_id = [dataset["id"] for dataset in datasets_content if dataset["name"]==dataset_name][0]

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve dataset list: {e}")
            raise

    
    def list_metrics(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{self.base_url}/v1/llm/llm-metrics', 
                headers=headers,
                timeout=self.timeout)
            response.raise_for_status()
            metric_names = [metric["name"] for metric in response.json()["data"]["metrics"]]
            return metric_names
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []

    def _get_dataset_id_based_on_dataset_type(self, metric_to_evaluate):
        try:
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
            response = requests.post(
                f"{self.base_url}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )
            
            response.raise_for_status()
            datasets_content = response.json()["data"]["content"]
            dataset = [dataset for dataset in datasets_content if dataset["name"]==self.dataset_name][0]
            if (dataset["datasetType"]=="prompt" and metric_to_evaluate=="prompt") or (dataset["datasetType"]=="chat" and metric_to_evaluate=="chat") or dataset["datasetType"]==None:
                return dataset["id"]
            else:
                return dataset["derivedDatasetId"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve dataset list: {e}")
            raise


    def _get_dataset_schema(self, metric_to_evaluate=None):
        #this dataset_id is based on which type of metric_to_evaluate  
        data_set_id=self._get_dataset_id_based_on_dataset_type(metric_to_evaluate)
        self.dataset_id=data_set_id

        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'Content-Type': 'application/json',
            'X-Project-Id': str(self.project_id),
        }
        data = {
            "datasetId": str(data_set_id),
            "fields": [],
            "rowFilterList": []
        }
        try:
            response = requests.post(
                f'{self.base_url}/v1/llm/docs', 
                headers=headers,
                json=data,
                timeout=self.timeout)
            response.raise_for_status()
            if response.status_code == 200:
                return response.json()["data"]["columns"]
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return {}


    def _get_variablename_from_user_schema_mapping(self, schemaName, metric_name, schema_mapping, metric_to_evaluate):
        user_dataset_schema = self._get_dataset_schema(metric_to_evaluate)
        user_dataset_columns = [item["displayName"] for item in user_dataset_schema]
        variableName = None
        for key, val in schema_mapping.items():
            if "".join(val.split("_")).lower()==schemaName:
                if key in user_dataset_columns:
                    variableName=key
                else:
                    raise ValueError(f"Column '{key}' is not present in '{self.dataset_name}' dataset")
        if variableName:
            return variableName
        else:
            raise ValueError(f"Map '{schemaName}' column in schema_mapping for {metric_name} metric evaluation")


    def _get_mapping(self, metric_name, metrics_schema, schema_mapping):
        
        mapping = []
        for schema in metrics_schema:
            if schema["name"]==metric_name:
                requiredFields = schema["config"]["requiredFields"]

                #this is added to check if "Chat" column is required for metric evaluation
                required_variables = [_["name"].lower() for _ in requiredFields]
                if "chat" in required_variables:
                    metric_to_evaluate = "chat"
                else:
                    metric_to_evaluate = "prompt"

                for field in requiredFields:
                    schemaName = field["name"]
                    variableName = self._get_variablename_from_user_schema_mapping(schemaName.lower(), metric_name, schema_mapping, metric_to_evaluate)
                    mapping.append({"schemaName": schemaName, "variableName": variableName})
        return mapping

    def _get_metricParams(self):
        return {
                "metricSpec": {
                    "name": "metric_to_evaluate",
                    "config": {
                        "model": "null",
                        "params": {
                            "model": {
                                "value": ""
                            }
                        },
                        "mappings": "mappings"
                    },
                    "displayName": "displayName"
                },
                "rowFilterList": []
            }
    
    def _get_metrics_schema_response(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{self.base_url}/v1/llm/llm-metrics', 
                headers=headers,
                timeout=self.timeout)
            response.raise_for_status()
            metrics_schema = [metric for metric in response.json()["data"]["metrics"]]
            return metrics_schema
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []

    def _update_base_json(self, metrics):
        metrics_schema_response = self._get_metrics_schema_response()
        sub_providers = ["openai","azure","gemini","groq","anthropic","bedrock"]
        metricParams = []
        for metric in metrics:
            base_json = self._get_metricParams()
            base_json["metricSpec"]["name"] = metric["name"]
            
            #pasing model configuration
            for key, value in metric["config"].items():
                #checking if provider is one of the allowed providers
                if key.lower()=="provider" and value.lower() not in sub_providers:
                    raise ValueError("Enter a valid provider name. The following Provider names are supported: openai, azure, gemini, groq, anthropic, bedrock")
    
                if key.lower()=="threshold":
                    if len(value)>1:
                        raise ValueError("'threshold' can only take one argument gte/lte/eq")
                    else:
                        for key_thres, value_thres in value.items():
                            base_json["metricSpec"]["config"]["params"][key] = {f"{key_thres}":value_thres}
                else:
                    base_json["metricSpec"]["config"]["params"][key] = {"value": value}


            # if metric["config"]["model"]:
            #     base_json["metricSpec"]["config"]["params"]["model"]["value"] = metric["config"]["model"]
            base_json["metricSpec"]["displayName"] = metric["column_name"]
            mappings = self._get_mapping(metric["name"], metrics_schema_response, metric["schema_mapping"])
            base_json["metricSpec"]["config"]["mappings"] = mappings
            metricParams.append(base_json)
        metric_schema_mapping = {"datasetId":self.dataset_id}
        metric_schema_mapping["metricParams"] = metricParams
        return metric_schema_mapping

    def _get_executed_metrics_list(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f"{self.base_url}/v2/llm/dataset/{str(self.dataset_id)}?initialCols=0",
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            dataset_columns = response.json()["data"]["datasetColumnsResponses"]
            dataset_columns = [item["displayName"] for item in dataset_columns]
            executed_metric_list = [data for data in dataset_columns if not data.startswith('_')]

            return executed_metric_list
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []

    def add_metrics(self, metrics):
        #Handle required key if missing
        required_keys = {"name", "config", "column_name", "schema_mapping"}
        for metric in metrics:
            missing_keys = required_keys - metric.keys()
            if missing_keys:
                raise ValueError(f"{missing_keys} required for each metric evaluation.")

        executed_metric_list = self._get_executed_metrics_list()
        metrics_name = self.list_metrics()
        user_metric_names = [metric["name"] for metric in metrics]
        for user_metric in user_metric_names:
            if user_metric not in metrics_name:
                raise ValueError("Enter a valid metric name")
        column_names = [metric["column_name"] for metric in metrics]
        for column_name in column_names:
            if column_name in executed_metric_list:
                raise ValueError(f"Column name '{column_name}' already exists.")

        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        metric_schema_mapping = self._update_base_json(metrics)
        try:
            response = requests.post(
                f'{self.base_url}/v2/llm/metric-evaluation',
                headers=headers, 
                json=metric_schema_mapping,
                timeout=self.timeout
                )
            if response.status_code == 400:
                raise ValueError(response.json()["message"])
            response.raise_for_status()
            if response.json()["success"]:
                print(response.json()["message"])
                self.jobId = response.json()["data"]["jobId"]

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def append_metrics(self, display_name):
        if not isinstance(display_name, str):
            raise ValueError("display_name should be a string")
        
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
            'Content-Type': 'application/json',
        }

        payload = json.dumps({
        "datasetId": self.dataset_id,
        "metricParams": [
            {
            "metricSpec": {
                "displayName": display_name
            }
            }
        ]
        })
        
        try:
            response = requests.request(
                "POST", 
                f'{self.base_url}/v2/llm/metric-evaluation-rerun', 
                headers=headers, 
                data=payload, 
                timeout=self.timeout)
            if response.status_code == 400:
                raise ValueError(response.json()["message"])
            response.raise_for_status()
            if response.json()["success"]:
                print(response.json()["message"])
                self.jobId = response.json()["data"]["jobId"]

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def get_status(self):
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{self.base_url}/job/status', 
                headers=headers, 
                timeout=self.timeout)
            response.raise_for_status()
            if response.json()["success"]:
                status_json = [item["status"] for item in response.json()["data"]["content"] if item["id"]==self.jobId][0]
                if status_json == "Failed":
                    print("Job failed. No results to fetch.")
                    return JOB_STATUS_FAILED
                elif status_json == "In Progress":
                    print(f"Job in progress. Please wait while the job completes.\nVisit Job Status: {self.base_url.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to track")
                    return JOB_STATUS_IN_PROGRESS
                elif status_json == "Completed":
                    print(f"Job completed. Fetching results.\nVisit Job Status: {self.base_url.removesuffix('/api')}/projects/job-status?projectId={self.project_id} to check")
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

    def get_results(self):

        def get_presignedUrl():
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                'X-Project-Id': str(self.project_id),
                }
            
            data = {
                "fields": [
                    "*"
                ],
                "datasetId": str(self.dataset_id),
                "rowFilterList": [],
                "export": True
                }
            try:    
                response = requests.post(
                    f'{self.base_url}/v1/llm/docs', 
                    headers=headers, 
                    json=data,
                    timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error occurred: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Timeout error occurred: {timeout_err}")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"An error occurred: {req_err}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                return {}

        def parse_response():
            try:
                response = get_presignedUrl()
                preSignedURL = response["data"]["preSignedURL"]
                response = requests.get(preSignedURL, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error occurred: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Timeout error occurred: {timeout_err}")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"An error occurred: {req_err}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                return ""

        response_text = parse_response()
        if response_text:
            df = pd.read_csv(io.StringIO(response_text))

            column_list = df.columns.to_list()
            # Remove unwanted columns
            column_list = [col for col in column_list if not col.startswith('_')]
            column_list = [col for col in column_list if '.' not in col]
            # Remove _claims_ columns
            column_list = [col for col in column_list if '_claims_' not in col]
            return df[column_list]
        else:
            return pd.DataFrame()
