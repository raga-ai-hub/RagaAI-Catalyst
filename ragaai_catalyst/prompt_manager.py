import os
import requests
import json
import re
from .ragaai_catalyst import RagaAICatalyst
import copy

class PromptManager:
    NUM_PROJECTS = 100
    TIMEOUT = 10

    def __init__(self, project_name):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name (str): The name of the project.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If the project is not found.
        """
        self.project_name = project_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10
        self.size = 99999 #Number of projects to fetch

        try:
            response = requests.get(
                f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={self.size}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            # logger.debug("Projects list retrieved successfully")

            project_list = [
                project["name"] for project in response.json()["data"]["content"]
            ]
            self.project_id = [
            project["id"] for project in response.json()["data"]["content"] if project["name"]==project_name
            ][0]

        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing project list: {str(e)}")

        if self.project_name not in project_list:
            raise ValueError("Project not found. Please enter a valid project name")


        self.headers = {
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": str(self.project_id)
            }


    def list_prompts(self):
        """
        List all available prompts.

        Returns:
            list: A list of prompt names.

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        prompt = Prompt()
        try:
            prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
    
    def get_prompt(self, prompt_name, version=None):
        """
        Get a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt.

        Raises:
            ValueError: If the prompt or version is not found.
            requests.RequestException: If there's an error with the API request.
        """
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")

        try:
            prompt_versions = self.list_prompt_versions(prompt_name)
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")

        if version and version not in prompt_versions.keys():
            raise ValueError("Version not found. Please enter a valid version name")

        prompt = Prompt()
        try:
            prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
            return prompt_object
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt: {str(e)}")

    def list_prompt_versions(self, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts.

        Raises:
            ValueError: If the prompt is not found.
            requests.RequestException: If there's an error with the API request.
        """
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")
        
        prompt = Prompt()
        try:
            prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")


class Prompt:
    def __init__(self):
        """
        Initialize the Prompt class.
        """
        pass

    def list_prompts(self, url, headers, timeout):
        """
        List all available prompts.

        Args:
            url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.

        Returns:
            list: A list of prompt names.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If there's an error parsing the prompt list.
        """
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            prompt_list = [prompt["name"] for prompt in response.json()["data"]]                        
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt list: {str(e)}")

    def _get_response_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt.

        Returns:
            response: The response object containing the prompt version data.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If there's an error parsing the prompt version.
        """
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}?version={version}",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

    def _get_response(self, base_url, headers, timeout, prompt_name):
        """
        Get the latest version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.

        Returns:
            response: The response object containing the latest prompt version data.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If there's an error parsing the prompt version.
        """
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}",
                                headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

    def _get_prompt_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt.

        Returns:
            str: The text of the prompt.

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        prompt_text = response.json()["data"]["docs"][0]["textFields"]
        return prompt_text

    def get_prompt(self, base_url, headers, timeout, prompt_name, version=None):
        """
        Get a prompt, optionally specifying a version.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt.

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        if version:
            response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        else:
            response = self._get_response(base_url, headers, timeout, prompt_name)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        return PromptObject(prompt_text, prompt_parameters, model)


    def list_prompt_versions(self, base_url, headers, timeout, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If there's an error parsing the prompt versions.
        """
        try:
            response = requests.get(f"{base_url}/{prompt_name}/version",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
            version_names = [version["name"] for version in response.json()["data"]]
            prompt_versions = {}
            for version in version_names:
                prompt_versions[version] = self._get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompt versions: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt versions: {str(e)}")


class PromptObject:
    def __init__(self, text, parameters, model):
        """
        Initialize a PromptObject with the given text.

        Args:
            text (str): The text of the prompt.
            parameters (dict): The parameters of the prompt.
            model (str): The model of the prompt.
        """
        self.text = text
        self.parameters = parameters
        self.model = model
    
    def _extract_variable_from_content(self, content):
        """
        Extract variables from the content.

        Args:
            content (str): The content containing variables.

        Returns:
            list: A list of variable names found in the content.
        """
        pattern = r'\{\{(.*?)\}\}'
        matches = re.findall(pattern, content)
        variables = [match.strip() for match in matches if '"' not in match]
        return variables

    def _add_variable_value_to_content(self, content, user_variables):
        """
        Add variable values to the content.

        Args:
            content (str): The content containing variables.
            user_variables (dict): A dictionary of user-provided variable values.

        Returns:
            str: The content with variables replaced by their values.
        """
        variables = self._extract_variable_from_content(content)
        for key, value in user_variables.items():
            if not isinstance(value, str):
                raise ValueError(f"Value for variable '{key}' must be a string, not {type(value).__name__}")
            if key in variables:
                content = content.replace(f"{{{{{key}}}}}", value)
        return content

    def compile(self, **kwargs):
        """
        Compile the prompt by replacing variables with provided values.

        Args:
            **kwargs: Keyword arguments where keys are variable names and values are their replacements.

        Returns:
            str: The compiled prompt with variables replaced.

        Raises:
            ValueError: If there are missing or extra variables, or if a value is not a string.
        """
        required_variables = self.get_variables()
        provided_variables = set(kwargs.keys())

        missing_variables = [item for item in required_variables if item not in provided_variables]
        extra_variables = [item for item in provided_variables if item not in required_variables]

        if missing_variables:
            raise ValueError(f"Missing variable(s): {', '.join(missing_variables)}")
        if extra_variables:
            raise ValueError(f"Extra variable(s) provided: {', '.join(extra_variables)}")

        updated_text = copy.deepcopy(self.text)

        for item in updated_text:
            item["content"] = self._add_variable_value_to_content(item["content"], kwargs)

        return updated_text
    
    def get_variables(self):
        """
        Get all variables in the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        variables = set()
        for item in self.text:
            content = item["content"]
            for var in self._extract_variable_from_content(content):
                variables.add(var)
        if variables:
            return list(variables)
        else:
            return []
    
    def _convert_value(self, value, type_):
        """
        Convert value based on type.

        Args:
            value: The value to be converted.
            type_ (str): The type to convert the value to.

        Returns:
            The converted value.
        """
        if type_ == "float":
            return float(value)
        elif type_ == "int":
            return int(value)
        return value  # Default case, return as is

    def get_model_parameters(self):
        """
        Get all parameters in the prompt text.

        Returns:
            dict: A dictionary of parameters found in the prompt text.
        """
        parameters = {}
        for param in self.parameters:
            if "value" in param:
                parameters[param["name"]] = self._convert_value(param["value"], param["type"])
            else:
                parameters[param["name"]] = ""
        parameters["model"] = self.model
        return parameters    
    
    def get_prompt_content(self):
        return self.text
