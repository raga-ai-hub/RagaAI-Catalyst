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

### Pinecone DB Tracing
```python
@tracer.trace_pinecone_upsert
def upsert_data(data):
    # Upsert data into Pinecone index
    pass

@tracer.trace_pinecone_create_query_vector
def create_query_vector(query_text):
    # Create a query vector for Pinecone
    pass

@tracer.trace_pinecone_similarity_search
def similarity_search(query_vector):
    # Perform a similarity search in Pinecone
    pass

# Usage
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

### Theme Configuration
The dashboard now supports theme customization! Access theme settings via the dropdown menu:

1. Theme Options

- Light: ☀️ Light theme
- Dark: 🌙 Dark theme for reduced eye strain
- System: 💻 Default - automatically matches your system preferences

2. Theme Persistence

- Your theme preference is automatically saved in browser localStorage
- Settings persist across browser sessions

### Viewing Data
1. Open `http://localhost:3000`
2. Navigate through different views:
   - Execution flow
   - Metrics
   - Timeline
   - Analytics