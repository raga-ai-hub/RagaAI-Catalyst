## Dataset Management

Create and manage datasets easily for your projects using the `ragaai_catalyst` library. This guide provides steps to list, create, and manage datasets efficiently.

#### - Initialize Dataset Management

To start managing datasets for a specific project, initialize the `Dataset` class with your project name.

```python
from ragaai_catalyst import Dataset

# Initialize Dataset management for a specific project
dataset_manager = Dataset(project_name="project_name")

# List existing datasets
datasets = dataset_manager.list_datasets()
print("Existing Datasets:", datasets)
```

#### 1. Create a New Dataset from Trace

Create a dataset by applying filters to trace data. Below is an example of creating a dataset with specific criteria.

```python
dataset_manager.create_from_trace(
    dataset_name='Test-dataset-1',
    filter_list=[
        {
            "name": "llm_model",
            "values": ["gpt-3.5-turbo", "gpt-4"]
        },
        {
            "name": "prompt_length",
            "lte": 27,
            "gte": 23
        }
    ]
)
```

#### 2. Create a New Dataset from CSV

You can create a new dataset by uploading a CSV file and mapping its columns to the required schema elements.

##### a. Retrieve CSV Schema Elements with `get_csv_schema()`

This function retrieves the valid schema elements that the CSV column names must map to. It helps ensure that your CSV column names align correctly with the expected schema.

###### Returns

- A dictionary containing schema information:
  - `success`: A Boolean indicating whether the schema elements were fetched successfully.
  - `data['schemaElements']`: A list of valid schema column names.

```python
schemaElements = dataset_manager.get_csv_schema()['data']['schemaElements']
print('Supported column names: ', schemaElements)
```

##### b. Create a Dataset from CSV with `create_from_csv()`

Uploads the CSV file to the server, performs schema mapping, and creates a new dataset.

###### Parameters

- `csv_path` (str): Path to the CSV file.
- `dataset_name` (str): The name you want to assign to the new dataset created from the CSV.
- `schema_mapping` (dict): A dictionary that maps CSV columns to schema elements in the format `{csv_column: schema_element}`.

Example usage:

```python
dataset_manager.create_from_csv(
    csv_path='path/to/your.csv',
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)
```

#### Understanding `schema_mapping`

The `schema_mapping` parameter is crucial when creating datasets from a CSV file. It ensures that the data in your CSV file correctly maps to the expected schema format required by the system.

##### Explanation of `schema_mapping`

- **Keys**: The keys in the `schema_mapping` dictionary represent the column names in your CSV file.
- **Values**: The values correspond to the expected schema elements that the columns should map to. These schema elements define how the data is stored and interpreted in the dataset.

##### Example of `schema_mapping`

Suppose your CSV file has columns `user_id` and `response_time`. If the valid schema elements for these are `user_identifier` and `response_duration`, your `schema_mapping` would look like this:

```python
schema_mapping = {
    'user_id': 'user_identifier',
    'response_time': 'response_duration'
}
```

This mapping ensures that when the CSV is uploaded, the data in `user_id` is understood as `user_identifier`, and `response_time` is understood as `response_duration`, aligning the data with the system's expectations.
