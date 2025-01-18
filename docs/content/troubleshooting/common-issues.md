# Common Issues and Solutions

## Installation Issues

### Problem: Import Errors
```python
ImportError: No module named 'agentneo'
```

**Solution:**
1. Verify installation:
```bash
pip list | grep agentneo
```
2. Reinstall package:
```bash
pip uninstall agentneo
pip install agentneo
```

### Problem: Version Conflicts
**Solution:**
```bash
pip install --upgrade agentneo
pip install --upgrade "agentneo[all]"
```

## Tracing Issues

### Problem: Decorators Not Working
```python
@tracer.trace_llm()
def my_function():
    # Function not being traced
```

**Solution:**
1. Verify tracer initialization
```python
# Correct initialization
neo_session = AgentNeo(session_name="my_session")
neo_session.create_project("my_project")
tracer = Tracer(session=neo_session)
tracer.start()  # Don't forget this!
```

### Problem: Missing Data in Dashboard
**Solution:**
1. Check tracer status
2. Verify data storage
3. Ensure proper stopping
```python
# Correct order
tracer.stop()  # Stop before launching dashboard
launch_dashboard(port=3000)
```

## Dashboard Issues

### Problem: Connection Refused
**Solution:**
1. Check port availability
2. Verify server status
3. Try alternative port
```python
launch_dashboard(port=3001)  # Try different port
```

## Evaluation Issues

### Problem: Metrics Not Computing
**Solution:**
1. Verify trace data exists
2. Check metric configuration
```python
# Correct usage
exe = Evaluation(
    session=neo_session,
    trace_id=tracer.trace_id  # Verify trace_id exists
)
```