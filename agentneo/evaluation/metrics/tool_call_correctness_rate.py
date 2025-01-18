import json
from typing import Dict, Any
import litellm
import json

def determine_intended_tools(query: str, tools: list, config: Dict[str, Any]) -> list:
    prompt = f"""
    Given the following user query and list of available tools, determine which tools would be appropriate to use.
    
    User query: {query}
    
    Available tools: {', '.join(tools)}
    
    Respond with a JSON object containing a list of appropriate tools.
    If no tools are appropriate, respond with an empty list.
    Only include tools from the provided list.
    
    Your response should be in this exact JSON format:
    {{"tools": ["tool1", "tool2", "tool3"]}}
    
    Or if no tools are appropriate:
    {{"tools": []}}
    """

    try:
        response = litellm.completion(
            model=config.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(content)
            intended_tools = result.get("tools", [])
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {content}")
            return []

        # Validate that returned tools exist in the provided list
        valid_tools = [tool for tool in intended_tools if tool in tools]
        
        return valid_tools

    except Exception as e:
        print(f"Error in determine_intended_tools: {e}")
        return []

def execute_tool_call_correctness_rate(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    
    # Extract query 
    try:
        query = trace_json["llm_calls"][0]["input_prompt"][0]["content"]
    except (IndexError, KeyError) as e:
        print(f"Error extracting query: {e}")
        return {
            "metric_name": "tool_correctness",
            "config": config,
            "result": {
                "score": 0,
                "reason": f"Unable to extract query from trace_json: {e}",
            },
        }

    # Find tool calls
    tool_calls = trace_json["tool_calls"]
    available_tools = list(set(call["name"] for call in tool_calls))

    intended_tools = determine_intended_tools(query, available_tools, config)

    correct_calls = sum(1 for call in tool_calls if call["name"] in intended_tools)
    total_calls = len(tool_calls)

    correctness_rate = correct_calls / total_calls if total_calls > 0 else 1

    # Use LLM to generate the reason
    reason_prompt = f"""
    Analyze the tool usage in this interaction and provide a concise explanation for the correctness rate.

    Query: {query}
    Available tools: {', '.join(available_tools)}
    Intended tools: {', '.join(intended_tools)}
    Correct calls: {correct_calls}
    Total calls: {total_calls}
    Correctness rate: {correctness_rate:.2f}

    Provide a brief explanation for this correctness rate, considering the query, intended tools, and actual tool usage.
    """

    reason_response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": reason_prompt}],
        temperature=0.0,
    )

    return {
        "metric_name": "tool_correctness",
        "config": config,
        "result": {
            "score": correctness_rate,
            "reason": reason_response.choices[0].message.content.strip(),
            "details": {
                "correct_calls": correct_calls,
                "total_calls": total_calls,
                "intended_tools": intended_tools,
                "available_tools": available_tools,
            },
        },
    }