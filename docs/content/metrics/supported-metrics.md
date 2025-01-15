# Supported Metrics

## Core Metrics

### 1. Goal Decomposition Efficiency
Measures how effectively an agent breaks down complex tasks.

```python
exe.evaluate(metric_list=['goal_decomposition_efficiency'])
```

**Configuration Options:**
```python
config = {
    "model": "gpt-4o-mini",
}
```

### 2. Goal Fulfillment Rate
Assesses the success rate of achieving defined objectives.

```python
exe.evaluate(metric_list=['goal_fulfillment_rate'])
```

**Configuration Options:**
```python
config = {
    "model": "gpt-4o-mini",
}
```

### 3. Tool Call Correctness Rate
Evaluates the accuracy of tool usage.

```python
exe.evaluate(metric_list=['tool_call_correctness_rate'])
```

**Metadata Example:**
```python
metadata = {
    "tools": [
        {
            "name": "calculator",
            "expected_usage": ["multiplication", "division"]
        }
    ]
}
```

### 4. Tool Call Success Rate
Measures the reliability of tool executions.
# Supported Metrics

## Core Metrics

### 5. Learning Adaptability Rate
Evaluates how well an agent adapts to changes in tools and workflows.

**Description:**
The Learning Adaptability Rate metric measures the agent's ability to adjust its strategies and tool usage in response to changes in its environment or task requirements. A high score indicates that the agent can effectively adapt to new tools or workflows, maintaining consistent performance.




```python
exe.evaluate(metric_list=['tool_call_success_rate'])
```

## Using Multiple Metrics
```python
# Evaluate multiple metrics together
exe.evaluate(
    metric_list=[
        'goal_decomposition_efficiency',
        'goal_fulfillment_rate',
        'tool_call_correctness_rate',
        'learning_adaptability_rate'
        'tool_call_success_rate'
    ]
)
```