import os
import requests
import logging
import pandas as pd
from .utils import response_checker
from .ragaai_catalyst import RagaAICatalyst

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

get_token = RagaAICatalyst.get_token


class Experiment:
    BASE_URL = None
    TIMEOUT = 10
    NUM_PROJECTS = 100

    def __init__(
        self, project_name, experiment_name, experiment_description, dataset_name
    ):
        """
        Initializes the Experiment object with the provided project details and initializes various attributes.

        Parameters:
            project_name (str): The name of the project.
            experiment_name (str): The name of the experiment.
            experiment_description (str): The description of the experiment.
            dataset_name (str): The name of the dataset.

        Returns:
            None
        """
        Experiment.BASE_URL = RagaAICatalyst.BASE_URL
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.dataset_name = dataset_name
        self.experiment_id = None
        self.job_id = None

        params = {
            "size": str(self.NUM_PROJECTS),
            "page": "0",
            "type": "llm",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        response = requests.get(
            f"{RagaAICatalyst.BASE_URL}/projects",
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        # logger.debug("Projects list retrieved successfully")
        experiment_list = [exp["name"] for project in response.json()["data"]["content"] if project["name"] == self.project_name for exp in project["experiments"]]
        # print(experiment_list)
        if self.experiment_name in experiment_list:
            raise ValueError("The experiment name already exists in the project. Enter a unique experiment name.")

        self.access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        self.secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")

        self.token = (
            os.getenv("RAGAAI_CATALYST_TOKEN")
            if os.getenv("RAGAAI_CATALYST_TOKEN") is not None
            else get_token()
        )
        
        if not self._check_if_project_exists(project_name=project_name):
            raise ValueError(f"Project '{project_name}' not found. Please enter a valid project name")
        
        if not self._check_if_dataset_exists(project_name=project_name,dataset_name=dataset_name):
            raise ValueError(f"dataset '{dataset_name}' not found. Please enter a valid dataset name")


        self.metrics = []
    def _check_if_dataset_exists(self,project_name,dataset_name):
        headers = {
            "X-Project-Name":project_name,
            # "accept":"application/json, text/plain, */*",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        response = requests.get(
            f"{RagaAICatalyst.BASE_URL}/v1/llm/sub-datasets?projectName={project_name}",
            headers=headers,
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        logger.debug("dataset list retrieved successfully")
        dataset_list = [
            item['name'] for item in response.json()['data']['content']
        ]
        exists = dataset_name in dataset_list
        if exists:
            logger.info(f"dataset '{dataset_name}' exists.")
        else:
            logger.info(f"dataset '{dataset_name}' does not exist.")
        return exists




    def _check_if_project_exists(self,project_name,num_projects=100):
        # TODO: 1. List All projects
        params = {
            "size": str(num_projects),
            "page": "0",
            "type": "llm",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        response = requests.get(
            f"{RagaAICatalyst.BASE_URL}/projects",
            params=params,
            headers=headers,
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        logger.debug("Projects list retrieved successfully")
        project_list = [
            project["name"] for project in response.json()["data"]["content"]
        ]
        
        # TODO: 2. Check if the given project_name exists
        # TODO: 3. Return bool (True / False output)
        exists = project_name in project_list
        if exists:
            logger.info(f"Project '{project_name}' exists.")
        else:
            logger.info(f"Project '{project_name}' does not exist.")
        return exists
        
    def list_experiments(self):
        """
        Retrieves a list of experiments associated with the current project.

        Returns:
            list: A list of experiment names.

        Raises:
            requests.exceptions.RequestException: If the request fails.

        """

        def make_request():
            headers = {
                "authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            params = {
                "name": self.project_name,
            }
            response = requests.get(
                f"{Experiment.BASE_URL}/project",
                headers=headers,
                params=params,
                timeout=Experiment.TIMEOUT,
            )
            return response

        response = make_request()
        response_checker(response, "Experiment.list_experiments")
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return {
                "status_code": response.status_code,
                "message": response.json(),
            }
        experiments = response.json()["data"]["experiments"]
        return [experiment["name"] for experiment in experiments]

    def add_metrics(self, metrics):
        """
        Adds metrics to the experiment and handles different status codes in the response.

        Parameters:
            metrics: The metrics to be added to the experiment. It can be a single metric or a list of metrics.

        Returns:
            If the status code is 200, returns a success message with the added metric names.
            If the status code is 401, retries the request, updates the job and experiment IDs, and returns the test response.
            If the status code is not 200 or 401, logs an error, and returns an error message with the response check.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Name": self.project_name,
        }

        if not isinstance(metrics, list):
            metrics = [metrics]
        else:
            metrics_list = metrics
        sub_providers = ["openai","azure","gemini","groq"]
        sub_metrics = RagaAICatalyst.list_metrics()  
        for metric in metrics_list:
            provider = metric.get('config', {}).get('provider', '').lower()
            if provider and provider not in sub_providers:
                raise ValueError("Enter a valid provider name. The following Provider names are supported: OpenAI, Azure, Gemini, Groq")

            if metric['name'] not in sub_metrics:
                raise ValueError("Enter a valid metric name. Refer to RagaAI Metric Library to select a valid metric")

        json_data = {
            "projectName": self.project_name,
            "datasetName": self.dataset_name,
            "experimentName": self.experiment_name,
            "metrics": metrics_list,
        }
        logger.debug(
            f"Preparing to add metrics for '{self.experiment_name}': {metrics}"
        )
        response = requests.post(
            f"{Experiment.BASE_URL}/v1/llm/experiment",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )

        status_code = response.status_code
        if status_code == 200:
            test_response = response.json()
            self.job_id = test_response.get("data").get("jobId")
            self.experiment_id = test_response.get("data").get("experiment").get("id")
            self.project_id = (
                test_response.get("data").get("experiment").get("projectId")
            )
            print(f"Metrics added successfully. Job ID: {self.job_id}")
            metric_names = [
                execution["metricName"]
                for execution in test_response["data"]["experiment"]["executions"]
            ]
            return f"Metrics {metric_names} added successfully"
        elif status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name,
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/v1/llm/experiment",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            status_code = response.status_code
            if status_code == 200:
                test_response = response.json()
                self.job_id = test_response.get("data").get("jobId")
                self.experiment_id = (
                    test_response.get("data").get("experiment").get("id")
                )
                self.project_id = (
                    test_response.get("data").get("experiment").get("projectId")
                )

                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.run_tests")
        else:
            logger.error(f"Failed to add metrics: HTTP {status_code}")
            return (
                "Error in running tests",
                response_checker(response, "Experiment.run_tests"),
            )

    def get_status(self, job_id=None):
        """
        Retrieves the status of a job based on the provided job ID.

        Returns the status of the job if the status code is 200, otherwise handles different status codes.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Name": self.project_name,
        }
        if job_id is not None:
            job_id_to_check = job_id
        else:
            job_id_to_check = self.job_id

        if job_id_to_check is None:
            logger.warning("Attempt to fetch status without a valid job ID.")
            return "Please run an experiment test first"
        json_data = {
            "jobId": job_id_to_check,
        }
        logger.debug(f"Fetching status for Job ID: {job_id_to_check}")
        response = requests.get(
            f"{Experiment.BASE_URL}/job/status",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )
        status_code = response_checker(response, "Experiment.get_status")
        if status_code == 200:
            test_response = response.json()
            jobs = test_response["data"]["content"]
            for job in jobs:
                if job["id"] == job_id_to_check:
                    return job["status"]
        elif status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name,
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/job/status",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            status_code = response_checker(response, "Experiment.get_status")
            if status_code == 200:
                test_response = response.json()
                self.experiment_id = (
                    test_response.get("data").get("experiment").get("id")
                )
                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.get_status")
        else:
            return (
                "Error in running tests",
                response_checker(response, "Experiment.get_status"),
            )

    def get_results(self, job_id=None):
        """
        A function that retrieves results based on the experiment ID.
        It makes a POST request to the BASE_URL to fetch results using the provided JSON data.
        If the request is successful (status code 200), it returns the retrieved test response.
        If the status code is 401, it retries the request and returns the test response if successful.
        If the status is neither 200 nor 401, it logs an error and returns the response checker result.
        """
        if job_id is not None:
            job_id_to_use = job_id
        else:
            job_id_to_use = self.job_id

        if job_id_to_use is None:
            logger.warning("Results fetch attempted without prior job execution.")
            return "Please run an experiment test first"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Id": str(self.project_id),
        }

        json_data = {
            "fields": [],
            "experimentId": self.experiment_id,
            "numRecords": 4,
            "projectId": self.project_id,
            "filterList": [],
        }
        base_url_without_api = Experiment.BASE_URL.removesuffix('/api')

        status_json = self.get_status(job_id_to_use)
        if status_json == "Failed":
            return print("Job failed. No results to fetch.")
        elif status_json == "In Progress":
            return print(f"Job in progress. Please wait while the job completes.\n Visit Job Status: {base_url_without_api}/home/job-status to track")
        elif status_json == "Completed":
            print(f"Job completed. fetching results.\n Visit Job Status: {base_url_without_api}/home/job-status to track")

        response = requests.post(
            f"{Experiment.BASE_URL}/v1/llm/docs",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )
        if response.status_code == 200:
            print("Results successfully retrieved.")
            test_response = response.json()

            if test_response["success"]:
                parse_success, parsed_response = self.parse_response(test_response)
                if parse_success:
                    return parsed_response
                else:
                    logger.error(f"Failed to parse response: {test_response}")
                    raise FailedToRetrieveResults(
                        f"Failed to parse response: {test_response}"
                    )

            else:
                logger.error(f"Failed to retrieve results for job: {job_id_to_use}")
                raise FailedToRetrieveResults(
                    f"Failed to retrieve results for job: {job_id_to_use}"
                )

            return parsed_response
        elif response.status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": str(self.project_id),
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/v1/llm/docs",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            if response.status_code == 200:
                test_response = response.json()
                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.get_test_results")
        else:
            return (
                "Error in running tests",
                response_checker(response, "Experiment.get_test_results"),
            )

    def parse_response(self, response):
        """
        Parse the response to get the results
        """
        try:
            x = pd.DataFrame(response["data"]["docs"])

            column_names_to_replace = [
                {item["columnName"]: item["displayName"]}
                for item in response["data"]["columns"]
            ]

            if column_names_to_replace:
                for item in column_names_to_replace:
                    x = x.rename(columns=item)

                dict_cols = [
                    col
                    for col in x.columns
                    if x[col].dtype == "object"
                    and x[col].apply(lambda y: isinstance(y, dict)).any()
                ]

                for dict_col in dict_cols:
                    x[f"{dict_col}_reason"] = x[dict_col].apply(
                        lambda y: y.get("reason") if isinstance(y, dict) else None
                    )
                    x[f"{dict_col}_metric_config"] = x[dict_col].apply(
                        lambda y: (
                            y.get("metric_config") if isinstance(y, dict) else None
                        )
                    )
                    x[f"{dict_col}_status"] = x[dict_col].apply(
                        lambda y: y.get("status") if isinstance(y, dict) else None
                    )

                    x = x.drop(columns=[dict_col])

            x.columns = x.columns.str.replace("_reason_reason", "_reason")
            x.columns = x.columns.str.replace("_reason_metric_config", "_metric_config")
            x.columns = x.columns.str.replace("_reason_status", "_status")

            columns_list = x.columns.tolist()
            #remove trace_uri from columns_list if it exists
            columns_list = list(set(columns_list) - {"trace_uri"})
            x = x[columns_list]

            return True, x

        except Exception as e:
            logger.error(f"Failed to parse response: {e}", exc_info=True)
            return False, pd.DataFrame()


class FailedToRetrieveResults(Exception):
    pass
