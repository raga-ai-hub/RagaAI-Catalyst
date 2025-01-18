# Basic Usage

## Setting Up AgentNeo

```python
from agentneo import AgentNeo, Tracer

# Initialize
neo_session = AgentNeo(session_name="my_session")
neo_session.create_project(project_name="my_project")

# Start tracing
tracer = Tracer(session=neo_session)
tracer.start()
```

## Using Decorators

### LLM Tracing
```python
@tracer.trace_llm("text_generation")
async def generate_text(prompt):
    response = await llm.generate(prompt)
    return response

# Usage
text = await generate_text("Tell me a story")
```

### Tool Tracing
```python
@tracer.trace_tool("data_processor")
def process_data(data):
    # Processing logic
    return processed_data

# Usage
result = process_data(raw_data)
```

### Agent Tracing
```python
@tracer.trace_agent("task_agent")
def execute_task(input_data):
    # Agent logic
    return result

# Usage
output = execute_task(data)
```

## Dashboard Usage

### Launching
```python
from agentneo import launch_dashboard

# After your traces are complete
tracer.stop()
launch_dashboard(port=3000)
```

### Viewing Data
1. Open `http://localhost:3000`
2. Navigate through different views:
   - Execution flow
   - Metrics
   - Timeline
   - Analytics