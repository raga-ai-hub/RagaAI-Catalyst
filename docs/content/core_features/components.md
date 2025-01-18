# Main Components

## Tracer
The core component for instrumenting and monitoring code.

### Features
- Function decoration
- Event tracking
- Metric collection
- Data buffering

### Usage
```python
from agentneo import Tracer

tracer = Tracer(session=neo_session)
tracer.start()

@tracer.trace_llm("my_llm")
def llm_function(): pass

@tracer.trace_tool("my_tool")
def tool_function(): pass

tracer.stop()
```

## Dashboard
Interactive web interface for visualization and analysis.

### Features
- Real-time monitoring
- Interactive graphs
- Performance metrics
- Detailed Trace views

### Access
```python
from agentneo import launch_dashboard
launch_dashboard(port=3000)
```

## Storage
Data persistence layer for traces and metrics.

### Features
- SQLite database
- JSON log files
- Query interface
- Data export

### Configuration
```python
AgentNeo(
    session_name="my_session",
    storage_path="./data",
    storage_type="sqlite"
)
```