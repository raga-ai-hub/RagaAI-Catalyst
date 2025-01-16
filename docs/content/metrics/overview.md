# Metrics and Evaluation Overview

## Introduction
AgentNeo provides comprehensive metrics and evaluation capabilities to assess and optimize your AI applications' performance. The evaluation framework helps you understand your agents' behavior, efficiency, and effectiveness.

## Key Capabilities
- Performance Assessment
- Cost Analysis
- Quality Metrics
- Behavioral Analysis
- Custom Metric Support

## Getting Started with Metrics
```python
from agentneo import Evaluation

# Initialize evaluation
exe = Evaluation(session=neo_session, trace_id=tracer.trace_id)

# Run evaluation
exe.evaluate(metric_list=['goal_decomposition_efficiency'], max_workers = 4 ,max_evaluations_per_thread=2 )

# Get results
results = exe.get_results()
```

## Evaluation Process
1. Collect trace data
2. Compute metrics
3. Generate insights
4. Provide recommendations