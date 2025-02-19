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

#### 1. Create a New Dataset from CSV

You can create a new dataset by uploading a CSV file and mapping its columns to the required schema elements.

##### a. Retrieve CSV Schema Elements with `get_schema_mapping()`

This function retrieves the valid schema elements that the CSV column names must map to. It helps ensure that your CSV column names align correctly with the expected schema.

###### Returns

- A list containing schema information

```python
schemaElements = dataset_manager.get_schema_mapping()
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


##### c. Add rows in the existing dataset from CSV

```python
add_rows_csv_path = "path to dataset"
dataset_manager.add_rows(csv_path=add_rows_csv_path, dataset_name=dataset_name)
```

##### d. Add columns in the existing dataset from CSV

```python
text_fields = [
      {
        "role": "system",
        "content": "you are an evaluator, which answers only in yes or no."
      },
      {
        "role": "user",
        "content": "are any of the {{context1}} {{feedback1}} related to broken hand"
      }
    ]
column_name = "column_name"
provider = "openai"
model = "gpt-4o-mini"

variables={
    "context1": "context",
    "feedback1": "feedback"
}
```

```python
dataset_manager.add_columns(
    text_fields=text_fields,
    dataset_name=dataset_name,
    column_name=column_name,
    provider=provider,
    model=model,
    variables=variables
)
```

#### 2. Create a New Dataset from JSONl

##### a. Create a Dataset from JSONl with `create_from_jsonl()`

```python
dataset_manager.create_from_jsonl(
    jsonl_path='jsonl_path',
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)
```

##### b. Add rows from JSONl with `add_rows_from_jsonl()`

```python
dataset_manager.add_rows_from_jsonl(
    jsonl_path='jsonl_path',
    dataset_name='MyDataset',
)
```

#### 3. Create a New Dataset from DataFrame

##### a. Create a Dataset from DataFrame with `create_from_df()`

```python
dataset_manager.create_from_df(
    df=df,
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)
```

##### b. Add rows from DataFrame with `add_rows_from_df()`

```python
dataset_manager.add_rows_from_df(
    df=df.tail(2),
    dataset_name='MyDataset',
)
```