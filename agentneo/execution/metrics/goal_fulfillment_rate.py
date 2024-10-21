import json
from typing import Dict, Any
import litellm
import ast
import os


def evaluate_goal_fulfillment(
    query: str, response: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""
    Evaluate the goal fulfillment rate for the following query and response:

    User Query: {query}

    System Response: {response}

    Please provide:
    1. A score between 0.0 and 1.0
    2. A detailed explanation for the score

    Format your response as a JSON object:
    {{
        "score": <float>,
        "explanation": "<detailed explanation>"
    }}
    """

    try:
        evaluation = litellm.completion(
            model=config.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(evaluation.choices[0].message.content)

    except json.JSONDecodeError:
        # If the output is not valid JSON, return a default error response
        result = {
            "score": 0.0,
            "explanation": "Error: The model did not return a valid JSON response.",
        }

    return result


def execute_goal_fulfillment_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:

    # Extract the first and last LLM/tool call based on timestamp
    llm_calls = trace_json.get("llm_calls", [])
    tool_calls = trace_json.get("tool_calls", [])

    # Combine and sort all calls by start_time for the first call
    all_calls = llm_calls + tool_calls
    all_calls.sort(key=lambda x: x["start_time"])

    # Extract query from the first call
    if all_calls:
        first_call = all_calls[0]
        query = first_call["input_prompt"][0]["content"] if "input_prompt" in first_call else first_call["input_parameters"]
    else:
        query = "No query found"

    # Sort all calls by end_time for the last call
    all_calls.sort(key=lambda x: x["end_time"])

    # Extract response from the last call
    if all_calls:
        last_call = all_calls[-1]
        response = last_call["output"]
    else:
        response = "No response found"

    # Evaluate goal fulfillment
    result = evaluate_goal_fulfillment(query, response, config)

    return {
        "metric_name": "goal_fulfillment_rate",
        "config": config,
        "result": {"inputGoal":query, "finalResponse":response, "score": result["score"], "reason": result["explanation"]},
    }


# Example usage
# json_path = "/Users/vijay/Desktop/AgentNeo_folder/AgentNeo/travel.json"
# with open(json_path) as j:
#     trace_json = json.load(j)

# config = {"model": "gpt-4o-mini"}
# result = execute_goal_fulfillment_metric(trace_json, config)
# print(json.dumps(result, indent=2))
