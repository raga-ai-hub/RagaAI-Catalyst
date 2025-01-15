import json
from typing import Dict, Any, List
import litellm

def extract_tool_changes(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract instances where tools were added, removed, or the tool name changed.
    """
    changes = []
    previous_tools = set()

    for index, call in enumerate(tool_calls):
        current_tools = set([call.get("name")])

        if previous_tools and current_tools != previous_tools:
            changes.append({
                "timestamp": call.get("start_time"),
                "added_tools": list(current_tools - previous_tools),
                "removed_tools": list(previous_tools - current_tools),
                "call_index": index,
            })

        previous_tools = current_tools

    return changes

def analyze_post_change_performance(
    llm_calls: List[Dict[str, Any]], change_points: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Analyze performance after each change point.
    """
    performance_data = []

    for change in change_points:
        change_index = change["call_index"]
        post_change_calls = llm_calls[change_index + 1 : change_index + 6]  # Analyze next 5 calls

        successful_adaptations = 0
        total_attempts = len(post_change_calls)

        for call in post_change_calls:
            if not call.get("error") and call.get("output"):
                successful_adaptations += 1

        performance_data.append({
            "change": change,
            "successful_adaptations": successful_adaptations,
            "total_attempts": total_attempts,
        })

    return performance_data

def calculate_overall_lar(performance_data: List[Dict[str, Any]]) -> float:
    """
    Calculate the Learning Adaptability Rate (LAR) based on the detailed evaluation score.
    """
    if not performance_data:
        return 0.0

    total_successful = sum(data["successful_adaptations"] for data in performance_data)
    total_attempts = sum(data["total_attempts"] for data in performance_data)

    if total_attempts == 0:
        return 0.0

    # Apply exponential decay to weigh recent changes more heavily
    weighted_sum = 0
    weighted_total = 0

    for i, data in enumerate(performance_data):
        weight = 1.0 / (2 ** i)  # Recent changes have higher weights
        weighted_sum += data["successful_adaptations"] * weight
        weighted_total += data["total_attempts"] * weight

    # Calculate the base score
    score = weighted_sum / weighted_total if weighted_total > 0 else 0.0

    # Introduce a penalty for lack of attempts
    no_attempt_penalty = sum(1 for data in performance_data if data["total_attempts"] == 0) * 0.1
    score -= no_attempt_penalty

    # Normalize the score to ensure it doesn't exceed 1.0 and is not negative
    return max(min(score, 1.0), 0.0)

def evaluate_learning_adaptability(
    performance_data: List[Dict[str, Any]], config: Dict[str, Any], overall_lar: float
) -> Dict[str, Any]:
    """
    Evaluate the learning adaptability based on performance data.
    """
    prompt = f"""
You are an expert AI system evaluator tasked with assessing the Learning Adaptability Rate of an AI agent.
Your task is to analyze how effectively the agent adapts to changes in its available tools and workflows.

Context:
The agent has undergone several changes in its available tools and workflows. After each change:
{json.dumps(performance_data, indent=2)}

Please evaluate:
1. How quickly the agent adapted to changes
2. The success rate of tasks immediately following changes
3. The consistency of adaptation across different changes
4. The overall learning adaptability pattern

Provide your evaluation in JSON format with:
- A score between 0.0 and 1.0 (use the provided overall LAR score: {overall_lar})
- A detailed explanation of the score
- Key observations about the adaptation pattern
- Recommendations for improvement

Remember that a score of 1.0 indicates perfect adaptation, while 0.0 indicates complete failure to adapt.
"""

    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

def generate_reason(result_info: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Generate a brief reason for the result using a language model.
    """
    prompt = f"""
Based on the following result information, provide a brief reason for the score:
{json.dumps(result_info, indent=2)}

The reason should be concise and explain the key factors contributing to the score.
"""

    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "text"},
    )

    return response.choices[0].message.content.strip()

def execute_learning_adaptability_rate_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the Learning Adaptability Rate metric evaluation.
    """
    try:
        # Extract LLM calls from trace
        tool_calls = trace_json.get("tool_calls", [])

        if not tool_calls:
            return {
                "metric_name": "learning_adaptability_rate",
                "config": config,
                "result": {
                    "score": 0.0,
                    "reason": "No LLM calls found in the trace.",
                },
            }

        # Identify points where tools or workflows changed
        change_points = extract_tool_changes(tool_calls)

        if not change_points:
            return {
                "metric_name": "learning_adaptability_rate",
                "config": config,
                "result": {
                    "score": 1.0,
                    "reason": "No tool changes detected - agent maintained consistent performance.",
                },
            }

        # Analyze performance after each change
        performance_data = analyze_post_change_performance(tool_calls, change_points)

        # Calculate overall LAR
        overall_lar = calculate_overall_lar(performance_data)

        # Get detailed evaluation
        evaluation = evaluate_learning_adaptability(performance_data, config, overall_lar)

        # Generate reason for the result
        result_info = {
            "score": overall_lar,
            "detailed_evaluation": evaluation,
            "change_points": change_points,
            "performance_data": performance_data,
        }
        reason = generate_reason(result_info, config)

        return {
            "metric_name": "learning_adaptability_rate",
            "config": config,
            "result": {
                "score": overall_lar,
                "detailed_evaluation": evaluation,
                "change_points": change_points,
                "performance_data": performance_data,
                "reason": reason,
            },
        }

    except Exception as e:
        return {
            "metric_name": "learning_adaptability_rate",
            "config": config,
            "result": {
                "score": 0.0,
                "reason": f"Error executing metric: {str(e)}",
            },
        }