import json
from typing import Dict, Any
import litellm
import os


def execute_tool_call_success_rate(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    # Find tool calls
    tool_calls = trace_json["tool_calls"]

    successful_calls = 0
    total_calls = len(tool_calls)
    call_results = []

    for call in tool_calls:
        output = call["output"]
        success, reason = judge_tool_call_success(output)
        if success:
            successful_calls += 1
        call_results.append({"success": success, "reason": reason})

    success_rate = successful_calls / total_calls if total_calls > 0 else 1

    final_reason = create_final_reason(success_rate, call_results)

    return {
        "metric_name": "tool_call_success_rate",
        "config": config,
        "result": {
            "score": success_rate,
            "reason": final_reason,
        },
    }


def judge_tool_call_success(output: str) -> tuple[bool, str]:
    prompt = f"""
    Analyze the following tool call output and determine if it was successful or if it contains an error:

    Tool call output:
    {output}

    Respond with a JSON object containing two fields:
    1. "success": A boolean indicating whether the tool call was successful (true) or contained an error (false).
    2. "reason": A brief explanation of your decision.

    JSON response:
    """

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )

    result = json.loads(response.choices[0].message.content)
    return result["success"], result["reason"]


def create_final_reason(success_rate: float, call_results: list) -> str:
    prompt = f"""
    Analyze the following tool call results and create a final reason summarizing the overall performance:

    Success rate: {success_rate:.2f}
    Tool call results:
    {json.dumps(call_results, indent=2)}

    Provide a concise summary of the tool call performance, highlighting any patterns or issues observed.

    Final reason:
    """

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()
