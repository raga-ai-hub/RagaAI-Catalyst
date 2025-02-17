import requests
import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger("RagaAICatalyst")
logging_level = logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") else logger.setLevel(logging.INFO)

class RagaAICatalyst:
    BASE_URL = None
    TIMEOUT = 10

    def __init__(self, access_key, secret_key, api_keys: Optional[Dict[str, str]] = None, base_url: Optional[str] = None):
        if not access_key or not secret_key:
            raise ValueError("RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY must be set")

        self.access_key = access_key
        self.secret_key = secret_key
        self.api_keys = api_keys or {}
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key

        RagaAICatalyst.BASE_URL = base_url or "https://catalyst.raga.ai/api"
        self.get_token()

    def get_token(self):
        headers = {"Content-Type": "application/json"}
        json_data = {"accessKey": self.access_key, "secretKey": self.secret_key}
        
        response = requests.post(f"{RagaAICatalyst.BASE_URL}/token", headers=headers, json=json_data, timeout=self.TIMEOUT)
        
        if response.status_code == 400:
            token_response = response.json()
            if token_response.get("message") == "Please enter valid credentials":
                raise Exception("Authentication failed. Invalid credentials provided.")

        response.raise_for_status()
        token_response = response.json()
        token = token_response.get("data", {}).get("token")
        
        if token:
            os.environ["RAGAAI_CATALYST_TOKEN"] = token
            print("Token(s) set successfully")
            return token
        return None

    def list_projects(self, num_projects=99999):
        headers = {"Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'}
        response = requests.get(
            f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={num_projects}",
            headers=headers,
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        project_list = [project["name"] for project in response.json()["data"]["content"]]
        project_ids = [project["id"] for project in response.json()["data"]["content"]]
        project_id = project_ids[0] if project_ids else None
        return project_list, project_id
    
    def project_use_cases(self):
        try:
            headers = {
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            }
            response = requests.get(
                f"{RagaAICatalyst.BASE_URL}/v2/llm/usecase",
                headers=headers,
                timeout=self.TIMEOUT
            )
            response.raise_for_status()  # Use raise_for_status to handle HTTP errors
            usecase = response.json()["data"]["usecase"]
            return usecase
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve project use cases: {e}")
            return []
    
    def create_project(self, project_name, usecase="Q/A", type="llm"):
        """
        Creates a project with the given project_name, type, and description.

        Parameters:
            project_name (str): The name of the project to be created.
            type (str, optional): The type of the project. Defaults to "llm".
            description (str, optional): Description of the project. Defaults to "".

        Returns:
            str: A message indicating the success or failure of the project creation.
        """
        # Check if the project already exists
        project_list, project_id = self.list_projects()
        if project_name in project_list:
            raise ValueError(f"Project name '{project_name}' already exists. Please choose a different name.")

        usecase_list = self.project_use_cases()
        if usecase not in usecase_list:
            raise ValueError(f"Select a valid usecase from {usecase_list}")
        
        json_data = {"name": project_name, "type": type, "usecase": usecase}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.post(
                f"{RagaAICatalyst.BASE_URL}/v2/llm/project",
                headers=headers,
                json=json_data,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            print(
                f"Project Created Successfully with name {response.json()['data']['name']} & usecase {usecase}"
            )
            return f'Project Created Successfully with name {response.json()["data"]["name"]} & usecase {usecase}'

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                self.get_token()
                headers["Authorization"] = (
                    f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
                )
                try:
                    response = requests.post(
                        f"{RagaAICatalyst.BASE_URL}/v2/llm/project",
                        headers=headers,
                        json=json_data,
                        timeout=self.TIMEOUT,
                    )
                    response.raise_for_status()
                    print(
                        "Project Created Successfully with name %s after token refresh",
                        response.json()["data"]["name"],
                    )
                    return f'Project Created Successfully with name {response.json()["data"]["name"]}'
                except requests.exceptions.HTTPError as refresh_http_err:
                    logger.error(
                        "Failed to create project after token refresh: %s",
                        str(refresh_http_err),
                    )
                    return f"Failed to create project: {response.json().get('message', 'Authentication error after token refresh')}"
            else:
                logger.error("Failed to create project: %s", str(http_err))
                return f"Failed to create project: {response.json().get('message', 'Unknown error')}"
        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                "Request timed out while creating project: %s", str(timeout_err)
            )
            return "Failed to create project: Request timed out"
        except Exception as general_err1:
            logger.error(
                "Unexpected error while creating project: %s", str(general_err1)
            )
            return "An unexpected error occurred while creating the project"

class UploadAgenticTraces:
    def __init__(self, 
                 json_file_path,
                 project_name,
                 project_id,
                 dataset_name,
                 user_detail,
                 base_url):
        self.json_file_path = json_file_path
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_detail = user_detail
        self.base_url = base_url
        self.timeout = 30


    def _get_presigned_url(self):
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "numFiles": 1,
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }

        try:
            response = requests.request("GET", 
                                        f"{self.base_url}/v1/llm/presigned-url", 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code == 200:
                presignedUrls = response.json()["data"]["presignedUrls"][0]
                return presignedUrls
        except requests.exceptions.RequestException as e:
            print(f"Error while getting presigned url: {e}")
            return None

    def _put_presigned_url(self, presignedUrl, filename):
        headers = {
                "Content-Type": "application/json",
            }

        if "blob.core.windows.net" in presignedUrl:  # Azure
            headers["x-ms-blob-type"] = "BlockBlob"
        print("Uploading agentic traces...")
        try:
            with open(filename) as f:
                payload = f.read().replace("\n", "").replace("\r", "").encode()
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None
        try:
            response = requests.request("PUT", 
                                        presignedUrl, 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code not in [200, 201]:
                return response, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error while uploading to presigned url: {e}")
            return None

    def insert_traces(self, presignedUrl):
        headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
                "X-Project-Name": self.project_name,
            }
        payload = json.dumps({
                "datasetName": self.dataset_name,
                "presignedUrl": presignedUrl,
                "datasetSpans": self._get_dataset_spans(), #Extra key for agentic traces
            })
        try:
            response = requests.request("POST", 
                                        f"{self.base_url}/v1/llm/insert/trace", 
                                        headers=headers, 
                                        data=payload,
                                        timeout=self.timeout)
            if response.status_code != 200:
                print(f"Error inserting traces: {response.json()['message']}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error while inserting traces: {e}")
            return None

    def _get_dataset_spans(self):
        try:
            with open(self.json_file_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None
        try:
            spans = data["data"][0]["spans"]
            datasetSpans = []
            for span in spans:
                if span["type"] != "agent":
                    existing_span = next((s for s in datasetSpans if s["spanHash"] == span["hash_id"]), None)
                    if existing_span is None:
                        datasetSpans.append({
                            "spanId": span["id"],
                            "spanName": span["name"],
                            "spanHash": span["hash_id"],
                            "spanType": span["type"],
                        })
                else:
                    datasetSpans.append({
                                "spanId": span["id"],
                                "spanName": span["name"],
                                "spanHash": span["hash_id"],
                                "spanType": span["type"],
                            })
                    children = span["data"]["children"]
                    for child in children:
                        existing_span = next((s for s in datasetSpans if s["spanHash"] == child["hash_id"]), None)
                        if existing_span is None:
                            datasetSpans.append({
                                "spanId": child["id"],
                                "spanName": child["name"],
                                "spanHash": child["hash_id"],
                                "spanType": child["type"],
                            })
            return datasetSpans
        except Exception as e:
            print(f"Error while reading dataset spans: {e}")
            return None
    
    def upload_agentic_traces(self):
        try:
            presignedUrl = self._get_presigned_url()
            if presignedUrl is None:
                return
            self._put_presigned_url(presignedUrl, self.json_file_path)
            self.insert_traces(presignedUrl)
        except Exception as e:
            print(f"Error while uploading agentic traces: {e}")

def upload_code(hash_id, zip_path, project_name, dataset_name):
    code_hashes_list = _fetch_dataset_code_hashes(project_name, dataset_name)

    if hash_id not in code_hashes_list:
        presigned_url = _fetch_presigned_url(project_name, dataset_name)
        _put_zip_presigned_url(project_name, presigned_url, zip_path)
        response = _insert_code(dataset_name, hash_id, presigned_url, project_name)
        return response
    else:
        return "Code already exists"

def _fetch_dataset_code_hashes(project_name, dataset_name):
    payload = {}
    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }

    try:
        response = requests.request("GET", 
                                    f"{RagaAICatalyst.BASE_URL}/v2/llm/dataset/code?datasetName={dataset_name}", 
                                    headers=headers, 
                                    data=payload,
                                    timeout=300)
        if response.status_code == 200:
            return response.json()["data"]["codeHashes"]
        else:
            raise Exception(f"Failed to fetch code hashes: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list datasets: {e}")
        raise 

def _fetch_presigned_url(project_name, dataset_name):
    payload = json.dumps({
            "datasetName": dataset_name,
            "numFiles": 1,
            "contentType": "application/zip"
            })

    headers = {
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "Content-Type": "application/json",
        "X-Project-Name": project_name,
    }

    try:
        response = requests.request("GET", 
                                    f"{RagaAICatalyst.BASE_URL}/v1/llm/presigned-url", 
                                    headers=headers, 
                                    data=payload,
                                    timeout=300)

        if response.status_code == 200:
            return response.json()["data"]["presignedUrls"][0]
        else:
            raise Exception(f"Failed to fetch code hashes: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list datasets: {e}")
        raise

def _put_zip_presigned_url(project_name, presignedUrl, filename):
    headers = {
            "X-Project-Name": project_name,
            "Content-Type": "application/zip",
        }
    if "blob.core.windows.net" in presignedUrl:  # Azure
        headers["x-ms-blob-type"] = "BlockBlob"


    with open(filename, 'rb') as f:
        payload = f.read()
    
    response = requests.request("PUT", 
                                presignedUrl, 
                                headers=headers, 
                                data=payload,
                                timeout=300)
    if response.status_code not in [200, 201]:
        return response, response.status_code

def _insert_code(dataset_name, hash_id, presigned_url, project_name):
    payload = json.dumps({
        "datasetName": dataset_name,
        "codeHash": hash_id,
        "presignedUrl": presigned_url
        })
    
    headers = {
        'X-Project-Name': project_name,
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
        }
    try:
        response = requests.request("POST", 
                                    f"{RagaAICatalyst.BASE_URL}/v2/llm/dataset/code", 
                                    headers=headers, 
                                    data=payload,
                                    timeout=300)
        if response.status_code == 200:
            return response.json()["message"]
        else:
            raise Exception(f"Failed to insert code: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to insert code: {e}")
        raise

def create_dataset_schema_with_trace(project_name, dataset_name):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        "X-Project-Name": project_name,
    }
    payload = json.dumps({
        "datasetName": dataset_name,
        "traceFolderUrl": None,
    })
    response = requests.request(
        "POST",
        f"{RagaAICatalyst.BASE_URL}/v1/llm/dataset/logs",
        headers=headers,
        data=payload,
        timeout=10
    )
    return response

def upload_trace_metric(json_file_path, dataset_name, project_name):
    try:
        with open(json_file_path, "r") as f:
            traces = json.load(f)

        metrics = get_trace_metrics_from_trace(traces)
        metrics = _change_metrics_format_for_payload(metrics)

        user_trace_metrics = get_user_trace_metrics(project_name, dataset_name)
        if user_trace_metrics:
            user_trace_metrics_list = [metric["displayName"] for metric in user_trace_metrics]

        if user_trace_metrics:
            for metric in metrics:
                if metric["displayName"] in user_trace_metrics_list:
                    metricConfig = next((user_metric["metricConfig"] for user_metric in user_trace_metrics if
                                         user_metric["displayName"] == metric["displayName"]), None)
                    if not metricConfig or metricConfig.get("Metric Source", {}).get("value") != "user":
                        raise ValueError(
                            f"Metrics {metric['displayName']} already exist in dataset {dataset_name} of project {project_name}.")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        payload = json.dumps({
            "datasetName": dataset_name,
            "metrics": metrics
        })
        response = requests.request("POST",
                                    f"{RagaAICatalyst.BASE_URL}/v1/llm/trace/metrics",
                                    headers=headers,
                                    data=payload,
                                    timeout=10)
        if response.status_code != 200:
            raise ValueError(f"Error inserting agentic trace metrics")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error submitting traces: {e}")
        return None

    return response

def _get_children_metrics_of_agent(children_traces):
    metrics = []
    for span in children_traces:
        metrics.extend(span.get("metrics", []))
        if span["type"] != "agent":
            metric = span.get("metrics", [])
            if metric:
                metrics.extend(metric)
        else:
            metrics.extend(_get_children_metrics_of_agent(span["data"]["children"]))
    return metrics

def get_trace_metrics_from_trace(traces):
    metrics = []

    # get trace level metrics
    if "metrics" in traces.keys():
        if len(traces["metrics"]) > 0:
            metrics.extend(traces["metrics"])

    # get span level metrics
    for span in traces["data"][0]["spans"]:
        if span["type"] == "agent":
            # Add children metrics of agent
            children_metric = _get_children_metrics_of_agent(span["data"]["children"])
            if children_metric:
                metrics.extend(children_metric)

        metric = span.get("metrics", [])
        if metric:
            metrics.extend(metric)
    return metrics

def _change_metrics_format_for_payload(metrics):
    formatted_metrics = []
    for metric in metrics:
        if any(m["name"] == metric.get("displayName") or m['name'] == metric.get("name") for m in formatted_metrics):
            continue
        metric_display_name = metric["name"]
        if metric.get("displayName"):
            metric_display_name = metric['displayName']
        formatted_metrics.append({
            "name": metric_display_name,
            "displayName": metric_display_name,
            "config": {"source": "user"},
        })
    return formatted_metrics

def get_user_trace_metrics(project_name, dataset_name):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": project_name,
        }
        response = requests.request("GET", 
                                    f"{RagaAICatalyst.BASE_URL}/v1/llm/trace/metrics?datasetName={dataset_name}", 
                                    headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching traces metrics: {response.json()['message']}")
            return None
        
        return response.json()["data"]["columns"]
    except Exception as e:
        print(f"Error fetching traces metrics: {e}")
        return None

# Main execution
if __name__ == "__main__":
    access_key = os.getenv('RAGAAI_ACCESS_KEY')
    secret_key = os.getenv('RAGAAI_SECRET_KEY')
    base_url = os.getenv('RAGAAI_BASE_URL')
    json_file_path = os.getenv('RAGAAI_JSON_FILE_PATH')

    project_name = 'project_name'
    usecase = "Agentic Application"
    dataset_name = "dataset_name"
    zip_path = "zip_path"

    auth = RagaAICatalyst(access_key=access_key, secret_key=secret_key, base_url=base_url, api_keys=None)
    #if want to create project
    # auth.create_project(
    #     project_name=project_name,
    #     usecase=usecase,
    #     type="llm",
    # )
    
    project_list, project_id = auth.list_projects()


    response = create_dataset_schema_with_trace(project_name=project_name, dataset_name=dataset_name)

    response = upload_trace_metric(
        json_file_path=json_file_path,
        dataset_name=dataset_name,
        project_name=project_name,
    )

    upload_traces = UploadAgenticTraces(
        json_file_path=json_file_path,
        project_name=project_name,
        project_id=project_id,
        dataset_name=dataset_name,
        user_detail=None,
        base_url=base_url,
    ).upload_agentic_traces()

    response = upload_code(
        hash_id=os.path.basename(zip_path).split('.')[0],
        zip_path=zip_path,
        project_name=project_name,
        dataset_name=dataset_name,
    )
    print(response)



