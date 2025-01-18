# Metric Configuration

## Basic Configuration

```python
# Standard configuration
config = {
    "model": "gpt-4-turbo",
    "threshold": 0.8,
    "detailed_output": True
}

# Standard metadata
metadata = {
    "tools": [
        {
            "name": "tool_name",
            "description": "tool_description"
        }
    ]
}

# Evaluate with config
exe.evaluate(
    metric_list=['metric_name'],
    config=config,
    metadata=metadata
)
```

## Advanced Configuration Options

### Global Settings
```python
global_config = {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "evaluation_mode": "strict",
    "logging_level": "detailed"
}
```

### Metric-Specific Settings
```python
metric_config = {
    "goal_decomposition_efficiency": {
        "min_steps": 2,
        "max_steps": 10,
        "complexity_threshold": 0.7
    },
    "tool_call_correctness_rate": {
        "strict_mode": True,
        "partial_credit": False
    }
}
```

### Custom Metadata
```python
custom_metadata = {
    "application_context": "financial_analysis",
    "expected_behaviors": ["data_analysis", "prediction"],
    "tools": [
        {
            "name": "market_analyzer",
            "expected_usage": ["analysis", "prediction"],
            "critical_operations": ["risk_assessment"]
        }
    ]
}
```