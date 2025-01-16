# Architecture

## System Overview

AgentNeo follows a modular architecture designed for flexibility and extensibility:

![AgentNeo Architecture](../../assets/architecture.svg)

## Key Components

### 1. Tracer Module
- Decorators for instrumentation
- Decorators for vector db (Pinecone)
- Event collection
- Metric computation
- Data buffering

### 2. Storage Layer
- SQLite database
- JSON log files
- Data persistence
- Query interface

### 3. Dashboard Service
- Web interface
- Real-time updates
- Data visualization
- Analysis tools

### 4. Evaluation
- Metric computation
- Performance analysis

## Data Flow

1. **Instrumentation**
   ```python
   @tracer.trace_llm()
   def your_function():
       pass
   ```

2. **Trace Collection**
   - Capture function calls
   - Capture vector database interaction (pinecone)
   - Record timestamps
   - Collect metrics

3. **Data Processing**
   - Compute metrics
   - Generate summaries
   - Update storage

4. **Visualization**
   - Display in dashboard