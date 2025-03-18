import requests
import json
import os
import logging
logger = logging.getLogger(__name__)
from .utils import response_checker
from .ragaai_catalyst import RagaAICatalyst


class GuardrailsManager:
    def __init__(self, project_name):
        """
        Initialize the GuardrailsManager with the given project name.
        
        :param project_name: The name of the project to manage guardrails for.
        """
        self.project_name = project_name
        self.timeout = 10
        self.num_projects = 99999
        self.deployment_name = "NA"
        self.deployment_id = "NA"
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        list_projects, project_name_with_id = self._get_project_list()
        if project_name not in list_projects:
            raise ValueError(f"Project '{self.project_name}' does not exists")
        
        self.project_id = [_["id"] for _ in project_name_with_id if _["name"]==self.project_name][0]


    def _get_project_list(self):
        """
        Retrieve the list of projects and their IDs from the API.
        
        :return: A tuple containing a list of project names and a list of dictionaries with project IDs and names.
        """
        headers = {'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'}
        response = requests.request("GET", f"{self.base_url}/v2/llm/projects?size={self.num_projects}", headers=headers, timeout=self.timeout)
        project_content = response.json()["data"]["content"]
        list_project = [_["name"] for _ in project_content]
        project_name_with_id = [{"id": _["id"], "name": _["name"]} for _ in project_content]
        return list_project, project_name_with_id


    def list_deployment_ids(self):
        """
        List all deployment IDs and their names for the current project.
        
        :return: A list of dictionaries containing deployment IDs and names.
        """
        payload = {}
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("GET", f"{self.base_url}/guardrail/deployment?size={self.num_projects}&page=0&sort=lastUsedAt,desc", headers=headers, data=payload, timeout=self.timeout)
        deployment_ids_content = response.json()["data"]["content"]
        deployment_ids_content = [{"id": _["id"], "name": _["name"]} for _ in deployment_ids_content]
        return deployment_ids_content


    def get_deployment(self, deployment_id):
        """
        Get details of a specific deployment ID, including its name and guardrails.
        
        :param deployment_id: The ID of the deployment to retrieve details for.
        :return: A dictionary containing the deployment name and a list of guardrails.
        """
        payload = {}
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("GET", f"{self.base_url}/guardrail/deployment/{deployment_id}", headers=headers, data=payload, timeout=self.timeout)
        if response.json()['success']:
            return response.json()
        else:
            print('Error in retrieving deployment details:',response.json()['message'])
            return None


    def list_guardrails(self):
        """
        List all available guardrails for the current project.
        
        :return: A list of guardrail names.
        """
        payload = {}
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("GET", f"{self.base_url}/v1/llm/llm-metrics?category=Guardrail", headers=headers, data=payload, timeout=self.timeout)
        list_guardrails_content = response.json()["data"]["metrics"]
        list_guardrails = [_["name"] for _ in list_guardrails_content]
        return list_guardrails


    def list_fail_condition(self):
        """
        List all fail conditions for the current project's deployments.
        
        :return: A list of fail conditions.
        """
        payload = {}
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("GET", f"{self.base_url}/guardrail/deployment/configurations", headers=headers, data=payload, timeout=self.timeout)
        return response.json()["data"]

    
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
                    f"{self.base_url}/v2/llm/dataset",
                    headers=headers,
                    json=json_data,
                    timeout=30,
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


    def create_deployment(self, deployment_name, deployment_dataset_name):
        """
        Create a new deployment ID with the given name.
        
        :param deployment_name: The name of the new deployment.
        :param deployment_dataset_name: The name of the tracking dataset.
        :raises ValueError: If a deployment with the given name already exists.
        """
        self.deployment_name = deployment_name
        self.deployment_dataset_name = deployment_dataset_name
        list_deployment_ids = self.list_deployment_ids()
        list_deployment_names = [_["name"] for _ in list_deployment_ids]
        if deployment_name in list_deployment_names:
            raise ValueError(f"Deployment with '{deployment_name}' already exists, choose a unique name")
        
        # Check if dataset name exists
        list_datasets = self.list_datasets()
        # Assuming this method exists to get list of datasets
        is_new_dataset = deployment_dataset_name not in list_datasets
        
        payload = json.dumps({
            "name": str(deployment_name),
            "trackingDataset": {
                "addNew": is_new_dataset,
                "name": str(deployment_dataset_name)
            }
        })
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'Content-Type': 'application/json',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("POST", f"{self.base_url}/guardrail/deployment", headers=headers, data=payload, timeout=self.timeout)
        if response.status_code == 409:
            raise ValueError(f"Data with '{deployment_name}' already exists, choose a unique name")
        if response.json()["success"]:
            print(response.json()["message"])
            deployment_ids = self.list_deployment_ids()
            self.deployment_id = [_["id"] for _ in deployment_ids if _["name"]==self.deployment_name][0]
            return self.deployment_id
        else:
            print(response)
            

    def add_guardrails(self, deployment_id, guardrails, guardrails_config={}):
        """
        Add guardrails to the current deployment.
        
        :param guardrails: A list of guardrails to add.
        :param guardrails_config: Configuration settings for the guardrails.
        :raises ValueError: If a guardrail name or type is invalid.
        """
        # Checking if guardrails names given already exist or not
        self.deployment_id = deployment_id
        deployment_details = self.get_deployment(self.deployment_id)
        if not deployment_details:
            return None
        deployment_id_name = deployment_details["data"]["name"]
        deployment_id_guardrails = deployment_details["data"]["guardrailsResponse"]
        guardrails_type_name_exists = [{_['metricSpec']["name"]:_['metricSpec']["displayName"]} for _ in deployment_id_guardrails]
        guardrails_type_name_exists = [list(d.values())[0] for d in guardrails_type_name_exists]
        user_guardrails_name_list = [_["name"] for _ in guardrails]
        for g_name in user_guardrails_name_list:
            if g_name in guardrails_type_name_exists:
                raise ValueError(f"Guardrail with '{g_name} already exists, choose a unique name'")
        # Checking if guardrails type is correct or not
        available_guardrails_list = self.list_guardrails()
        user_guardrails_type_list = [_["name"] for _ in guardrails]
        for g_type in user_guardrails_type_list:
            if g_type not in available_guardrails_list:
                raise ValueError(f"Guardrail type '{g_type} does not exists, choose a correct type'")

        payload = self._get_guardrail_config_payload(guardrails_config)
        payload["guardrails"] = self._get_guardrail_list_payload(guardrails)
        payload = json.dumps(payload)
        headers = {
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'Content-Type': 'application/json',
                'X-Project-Id': str(self.project_id)
                }
        response = requests.request("POST", f"{self.base_url}/guardrail/deployment/{str(self.deployment_id)}/configure", headers=headers, data=payload)
        if response.json()["success"]:
            print(response.json()["message"])
        else:
            print('Error updating guardrail ',response.json()['message'])

    def _get_guardrail_config_payload(self, guardrails_config):
        """
        Construct the payload for guardrail configuration.
        
        :param guardrails_config: Configuration settings for the guardrails.
        :return: A dictionary representing the guardrail configuration payload.
        """
        data = {
            "isActive": guardrails_config.get("isActive",False),
            "guardrailFailConditions": guardrails_config.get("guardrailFailConditions",["FAIL"]),
            "deploymentFailCondition": guardrails_config.get("deploymentFailCondition","ONE_FAIL"),
            "failAction": {
                "action": "ALTERNATE_RESPONSE",
                "args": f'{{\"alternateResponse\": \"{guardrails_config.get("alternateResponse","This is the Alternate Response")}\"}}'
                },
            "guardrails" : []
            }
        return data

    def _get_guardrail_list_payload(self, guardrails):
        """
        Construct the payload for a list of guardrails.
        
        :param guardrails: A list of guardrails to include in the payload.
        :return: A list of dictionaries representing each guardrail's data.
        """
        guardrails_list_payload = []
        for guardrail in guardrails:
            guardrails_list_payload.append(self._get_one_guardrail_data(guardrail))
        return guardrails_list_payload

    def _get_one_guardrail_data(self, guardrail):
        """
        Construct the data for a single guardrail.
        
        :param guardrail: A dictionary containing the guardrail's attributes.
        :return: A dictionary representing the guardrail's data.
        """
        if 'config' in guardrail:
            if 'mappings' in guardrail.get('config'):
                for mapping in guardrail.get('config',{}).get('mappings',{}):
                    if mapping['schemaName'] not in ['Text','Prompt','Context','Response']:
                        raise(ValueError('Invalid schemaName in guardrail mapping schema'))
                    if mapping['variableName'] not in ['Instruction','Prompt','Context','Response']:
                        raise(ValueError('Invalid variableName in guardrail mapping schema'))
            if 'model' in guardrail.get('config'):
                if guardrail.get('config',{}).get('model','') not in ['gpt-4o-mini','gpt-4o','gpt-4-turbo']:
                    raise(ValueError('Invalid model name in guardrail model schema'))
            if 'params' not in guardrail.get('config'):
                guardrail['config']['params'] = {
                    "isActive": {"value": False},
                    "isHighRisk": {"value": False},
                    "threshold": {"lt": 1}
                }


        data = {
            "displayName": guardrail["displayName"],
            "name": guardrail["name"],
            "config": guardrail.get("config", {})
        }
        '''
        if "lte" in guardrail["threshold"]:
            data["threshold"]["lte"] = guardrail["threshold"]["lte"]
        elif "gte" in guardrail["threshold"]:
            data["threshold"]["gte"] = guardrail["threshold"]["gte"]
        elif "eq" in guardrail["threshold"]:
            data["threshold"]["eq"] = guardrail["threshold"]["eq"]
        else:
            data["threshold"]["gte"] = 0.0'''
        return data


    def _run(self, **kwargs):
        """
        Execute the guardrail checks with the provided variables.
        """
