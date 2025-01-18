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