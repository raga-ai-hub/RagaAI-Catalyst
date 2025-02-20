from ragaai_catalyst import Dataset

class UploadResult:
    def __init__(self, project_name):
        self.project_name = project_name
        self.dataset_manager = Dataset(self.project_name)


    def list_datasets(self):
        list_datasets = self.dataset_manager.list_datasets()
        print("List of datasets: ", list_datasets)
        return list_datasets


    def upload_result(self, csv_path, dataset_name):
        
        schema_mapping={
            'detector':'metadata',
            'scenario':'metadata',
            'user_message': 'prompt',
            'app_response': 'response',
            'evaluation_score': 'metadata',
            'evaluation_reason': 'metadata'
        }
        self.dataset_manager.create_from_csv(
            csv_path=csv_path,
            dataset_name=dataset_name,
            schema_mapping=schema_mapping
        )





        
        


