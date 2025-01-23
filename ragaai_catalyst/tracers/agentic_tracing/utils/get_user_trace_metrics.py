import requests
import os
from ....ragaai_catalyst import RagaAICatalyst
from ....dataset import Dataset

def get_user_trace_metrics(project_name, dataset_name):
    try:
        list_datasets = Dataset(project_name=project_name).list_datasets()
        if not list_datasets:
            return []
        elif dataset_name not in list_datasets:
            return []
        else:
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