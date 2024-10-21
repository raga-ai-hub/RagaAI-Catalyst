import json
from typing import Dict, Any
import litellm
import os
import ast


def determine_intended_tool(query: str, tools: list, config: Dict[str, Any]) -> str:
    prompt = f"""
    Given the following user query and list of available tools, determine which tool would be most appropriate to use.
    
    User query: {query}
    
    Available tools: {', '.join(tools)}
    
    Respond with only the name of the most appropriate tool.
    """

    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()

def execute_tool_correctness_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:

    # Extract query from the trace
    query = trace_json["llm_calls"][0]["input_prompt"][0]["content"]

    # Find tool calls
    tool_calls = trace_json["tool_calls"]
    available_tools = list(set(call["name"] for call in tool_calls))

    intended_tool = determine_intended_tool(query, available_tools, config)

    correct_calls = sum(1 for call in tool_calls if call["name"] == intended_tool)
    total_calls = len(tool_calls)

    correctness_rate = correct_calls / total_calls if total_calls > 0 else 0

    return {
        "metric_name": "tool_correctness",
        "config": config,
        "result": {
            "score": correctness_rate,
            "reason": json.dumps({
                "correct_calls": correct_calls,
                "total_calls": total_calls,
                "intended_tool": intended_tool,
                "available_tools": available_tools,
            }),
        },
    }

def execute_tool_call_success_rate_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    # Find tool calls
    tool_calls = trace_json["tool_calls"]

    successful_calls = 0
    total_calls = len(tool_calls)

    for call in tool_calls:
        output = call["output"]
        # use llm to determine the error scenario
        if not output:
            successful_calls += 1
        else:
            if "error" not in str(output).lower() and "exception" not in str(output).lower():
                successful_calls += 1

    success_rate = successful_calls / total_calls if total_calls > 0 else 0

    return {
        "metric_name": "tool_call_success_rate",
        "config": config,
        "result": {
            "score": success_rate,
            "reason": json.dumps({
                "successful_calls": successful_calls,
                "total_calls": total_calls,
            }),
        },
    }

# Example usage
# json_path = "/Users/vijay/Desktop/AgentNeo_folder/AgentNeo/travel.json"
# with open(json_path) as j:
#     trace_json = json.load(j)

# config = {"model": "gpt-4o-mini"}
# correctness_metric = execute_tool_correctness_metric(trace_json, config)
# success_rate_metric = execute_tool_call_success_rate_metric(trace_json, config)