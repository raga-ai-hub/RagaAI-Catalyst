import json
import os
import requests
import logging
from .ragaai_catalyst import RagaAICatalyst

logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token





class CustomMetric:
    BASE_URL = None
    TIMEOUT = 30

    def __init__(self, project_name):
        self.project_name = project_name
        self.num_projects = 99999
        self.timeout = 10
        CustomMetric.BASE_URL = RagaAICatalyst.BASE_URL

        headers = {
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.get(
                f"{CustomMetric.BASE_URL}/v2/llm/projects?size={self.num_projects}",
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

    def list_custom_metrics(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }

        try:
            response = requests.get(
                f'{CustomMetric.BASE_URL}/custom-metric?size=100',
                headers=headers,
                timeout=self.timeout)
            response.raise_for_status()
            custom_metrics = [(metric["name"], metric["id"]) for metric in response.json()["data"]["content"]]
            return custom_metrics
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

    def create_custom_metrics(self, metric_name, description):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            json_data = {
                "name": metric_name,
                "description": description
            }
            response = requests.post(
                f'{CustomMetric.BASE_URL}/custom-metric',
                headers=headers,
                json=json_data,
                timeout=self.timeout)
            response.raise_for_status()
            custom_metric_id = response.json()["data"]["metricId"]
            return custom_metric_id
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

    def get_grading_criteria(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{CustomMetric.BASE_URL}/custom-metric/configurations',
                headers=headers,
                timeout=self.timeout)
            response.raise_for_status()
            custom_metrics = response.json()["data"]["gradingCriteriaList"]
            return custom_metrics
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

    def run_step(self, custom_metric_id, steps, model, provider):
        params_response = self.get_model_parameters(model, provider)
        formatted_parameters = _get_extract_parameters(params_response)

        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        custom_metric_template = {}

        model_specs = {
            "model": f"{provider}/{model}",
            "parameters": formatted_parameters
        }
        custom_metric_template["steps"] = steps["steps"]
        custom_metric_template["modelSpecs"] = model_specs
        input_variables = steps["variables"]
        variable_specs = []
        variables = []
        for item in input_variables:
            variable_spec = {"name": item["name"], "type": "string", "query": "query"}
            variable = {"name": item["name"], "type": "string", "query": "query", "value": item["value"]}
            variable_specs.append(variable_spec)
            variables.append(variable)
        custom_metric_template["variableSpecs"] = variable_specs
        try:
            custom_metric_payload = {
                "customMetricTemplate": custom_metric_template,
                "variables": variables
            }
            logger.info(headers)
            response = requests.post(
                f'{CustomMetric.BASE_URL}/custom-metric/{custom_metric_id}/run',
                headers=headers,
                json=custom_metric_payload,
                timeout=self.timeout)
            response.raise_for_status()
            return response.json()['data']["customMetricTemplate"]["steps"]
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

    def get_model_parameters(self, model, provider):
        # Get all the model parameters
        parameters_url = f"{CustomMetric.BASE_URL}/playground/providers/models/parameters/list"
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Id": str(self.project_id)
        }
        # Model parameters payload
        parameters_payload = {
            "providerName": provider,
            "modelName": model
        }
        # Get model parameters
        params_response = requests.post(
            parameters_url,
            headers=headers,
            json=parameters_payload,
            timeout=self.timeout
        )
        params_response.raise_for_status()
        return params_response

    def verify_grading_criteria(self, custom_metric_id, grading_criteria, steps):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        score = float(steps[(len(steps) - 1)]["output"]["response"])
        json_data = {
            "gradingCriteria": grading_criteria,
            "score": score
        }
        try:
            response = requests.post(
                f'{CustomMetric.BASE_URL}/custom-metric/{custom_metric_id}/verify',
                headers=headers,
                json=json_data,
                timeout=self.timeout)
            response.raise_for_status()
            return response.json()['message']
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

    def commit_custom_metric(self, custom_metric_id, steps, model, provider, output_steps, final_reason,
                             commit_message):
        project_id = str(self.project_id)
        params_response = self.get_model_parameters(model, provider)
        formatted_parameters = _get_extract_parameters(params_response)

        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': project_id,
        }
        custom_metric_template = {}

        model_specs = {
            "model": f"{provider}/{model}",
            "parameters": formatted_parameters
        }
        custom_metric_template["steps"] = steps["steps"]
        input_variables = steps["variables"]
        custom_metric_template["modelSpecs"] = model_specs
        custom_metric_template["finalScore"] = output_steps[len(output_steps)-1]["output"]["response"]
        custom_metric_template["finalReason"] = final_reason
        variable_specs = []
        for item in input_variables:
            variable_spec = {"name": item["name"], "type": "string", "query": "query"}
            variable_specs.append(variable_spec)
        custom_metric_template["variableSpecs"] = variable_specs
        try:
            custom_metric_payload = {
                "customMetricTemplate": custom_metric_template,
                "commitMessage": commit_message
            }
            response = requests.post(
                f'{CustomMetric.BASE_URL}/custom-metric/{custom_metric_id}/commit',
                headers=headers,
                json=custom_metric_payload,
                timeout=self.timeout)
            response.raise_for_status()
            return response.json()['data']
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

    def get_custom_metric_versions(self, custom_metric_id):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{CustomMetric.BASE_URL}/custom-metric/{custom_metric_id}/versions',
                headers=headers,
                timeout=self.timeout)
            response.raise_for_status()
            version_responses = response.json()["data"]
            version_list = []
            for version in version_responses:
                version_info = {"id": version["id"], "name": version["name"]}
                version_list.append(version_info)
            return version_list
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

    def deploy_custom_metric(self, custom_metric_id, version_name):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        version_json = {"versionName": version_name}
        try:
            response = requests.post(
                f'{CustomMetric.BASE_URL}/custom-metric/{custom_metric_id}/deploy',
                headers=headers,
                json=version_json,
                timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]
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
def _get_extract_parameters(params_response):
    # Extract parameters
    all_parameters = params_response.json().get('data', [])
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
    return formatted_parameters
