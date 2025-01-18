# Quick Start Guide

Get up and running with AgentNeo in minutes!

## Basic Setup

```python
from agentneo import AgentNeo, Tracer, Evaluation, launch_dashboard

# Create session and project
neo_session = AgentNeo(session_name="my_session")
neo_session.create_project(project_name="my_project")

# Initialize tracer
tracer = Tracer(session=neo_session)
tracer.start()
```

## Adding Traces

## Initialize Pinecone client
```python
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pc = Pinecone(api_key=api_key)
index_name = "pineconetesting"
dimension = 1024
metric = "cosine"
namespace = "ns1"
```

## Create Pinecone index if it doesn't exist
```python
existing_indexes = pc.list_indexes()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
```

## Wait for the index to be ready
```python
while True:
    index_status = pc.describe_index(index_name).status
    if index_status['ready']:
        break
    time.sleep(1)

index = pc.Index(index_name)
```

## Define your functions
```python
@tracer.trace_tool("data_processor")
def process_data(data):
    return {"processed": data}

@tracer.trace_agent("main_agent")
def main_agent(input_data):
    processed = process_data(input_data)
    return processed

@tracer.trace_pinecone_upsert
def upsert_data(data):
    # Upsert data into Pinecone index
    pass

@tracer.trace_pinecone_create_query_vector
def create_query_vector(query_text):
    # Create a query vector for Pinecone
    pass

@tracer.trace_pinecone_similarity_search
def similarity_search(query_vector, top_k=3):
    # Perform a similarity search in Pinecone
    pass
```

## Run your application
```python
data = [
    {"id": "vec1", "text": "Example text 1"},
    {"id": "vec2", "text": "Example text 2"},
    # More data...
]

upsert_data(data)

query_text = "Example query"
query_vector = create_query_vector(query_text)

results = similarity_search(query_vector)
print(results)
```

### Trace LLM Calls
```python
@tracer.trace_llm("my_llm_call")
async def get_llm_response(prompt):
    # Your LLM call here
    pass
```

### Trace Tools
```python
@tracer.trace_tool("my_tool")
def calculate_metrics(data):
    # Your tool logic here
    pass
```

### Trace Agents
```python
@tracer.trace_agent("my_agent")
def process_task(input_data):
    # Your agent logic here
    pass
```

## Viewing Results

```python
# Stop tracing
tracer.stop()

# Launch dashboard
launch_dashboard(port=3000)
```

Visit `http://localhost:3000` to view your traces and metrics.

## Complete Example

```python
from agentneo import AgentNeo, Tracer, Evaluation, launch_dashboard

# Setup
neo_session = AgentNeo(session_name="example_session")
neo_session.create_project(project_name="example_project")
tracer = Tracer(session=neo_session)
tracer.start()

# Define your functions
@tracer.trace_tool("data_processor")
def process_data(data):
    return {"processed": data}

@tracer.trace_agent("main_agent")
def main_agent(input_data):
    processed = process_data(input_data)
    return processed

# Run your application
result = main_agent({"data": "example"})

# Stop and view
tracer.stop()
launch_dashboard(port=3000)
```

## Next Steps
- [Basic Usage](../core-concepts/basic-usage.md)
- [Core Concepts](../core-concepts/key-features.md)
- [Metrics and Evaluation](../metrics/overview.md)