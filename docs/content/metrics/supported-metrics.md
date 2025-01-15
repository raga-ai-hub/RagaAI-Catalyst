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

```python
exe.evaluate(metric_list=['tool_call_success_rate'])
```

### 5. Custom User Defined Metrics
Enables users to define there own personlized metrics system

##### Default metrics

```python
exe.evaluate(metric_list=['custom_evaluation_metric',],
                 )
```

##### User defined metrics
```python
prompt1 = """
    Evaluate the conversation based on:
    1. Technical accuracy (40%) - How accurate is the technical information
    2. Response time (30%) - How quickly does the AI respond
    3. Completeness (30%) - How thorough are the responses
    
    Use very strict scoring criteria where excellent means perfect execution.
    """

exe.evaluate(metric_list=['custom_evaluation_metric',],
                 custom_criteria=prompt1)
```

**Custom Metric Exampls**
```python
DEFAULT_RUBRIC = {
    "criteria": {
        "problem_solving": {
            "description": "How effectively does the AI solve the user's problem?",
            "weight": 0.4
        },
        "clarity": {
            "description": "How clear and understandable are the AI's responses?",
            "weight": 0.3
        },
        "efficiency": {
            "description": "How efficient is the conversation in reaching the goal?",
            "weight": 0.3
        }
    },
    "scoring_guidelines": {
        "0.0-0.2": "Poor performance, significant improvements needed",
        "0.2-0.4": "Below average, notable issues present",
        "0.4-0.6": "Average performance, some improvements possible",
        "0.6-0.8": "Above average, minor improvements possible",
        "0.8-1.0": "Excellent performance, meets or exceeds expectations"
    }
}


CUSTOM_RUBRIC = {
    "criteria": {
        "problem_solving": {
            "description": "How effectively does the AI solve the user's problem?",
            "weight": 0.45
        },
        "clarity": {
            "description": "How clear and understandable are the AI's responses?",
            "weight": 0.45
        },
        "useful":{
            "description":"Is the response from AI useful to the user",
            "weight":0.1
        }
    },
    "scoring_guidelines": {
        "0.0-0.5":"Poor performance, significant improvements needed",
        "0.5-1.0": "Excellent performance, meets or exceeds expectations"
        }
    }
```

## Using Multiple Metrics
```python
# Evaluate multiple metrics together
exe.evaluate(
    metric_list=[
        'goal_decomposition_efficiency',
        'goal_fulfillment_rate',
        'tool_call_correctness_rate',
        'tool_call_success_rate'
    ]
)
```