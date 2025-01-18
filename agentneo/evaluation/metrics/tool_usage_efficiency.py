import json
from typing import Dict, Any, List, Union, Optional
import litellm
import os
import ast


def tool_calls_with_path(
    traces: Union[Dict[str, Any], List[Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Extracts tool calls with their full path from the given traces.

    Args:
        traces (Dict[str, Any] or List[Dict[str, Any]]): The JSON traces to analyze.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the name and full path of a tool call.
    """
    if metadata is not None:
        accepted_tools = [tool.get("name") for tool in metadata.get("tools", [])]

    def extract_tool_calls(
        node: Union[Dict[str, Any], List[Dict[str, Any]]], path: List[str] = []
    ) -> List[Dict[str, str]]:
        tool_calls = []
        if isinstance(node, list):
            for index, item in enumerate(node):
                if (
                    "type" in item
                    and item["type"] != "llm_call"
                    and item["type"] != "root"
                ):
                    if (metadata is not None) and (item["name"] in accepted_tools):
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                    elif metadata is None:
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                    elif "children" in item and item["children"]:
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                elif "children" in item and item["children"]:
                    tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
        elif isinstance(node, dict):
            if "type" in node and node["type"] != "llm_call" and node["type"] != "root":
                if (metadata is not None) and (node["name"] in accepted_tools):
                    tool_calls.append(
                        {
                            "name": node["name"],
                            "inputs": node["inputs"],
                            "outputs": node["outputs"],
                            "start": node["start"],
                            "end": node["end"],
                            "path": ".".join(path),
                        }
                    )
                elif metadata is None:
                    tool_calls.append(
                        {
                            "name": node["name"],
                            "inputs": node["inputs"],
                            "outputs": node["outputs"],
                            "start": node["start"],
                            "end": node["end"],
                            "path": ".".join(path),
                        }
                    )
            if "children" in node:
                tool_calls.extend(
                    extract_tool_calls(node["children"], path + [node.get("name", "")])
                )
        return tool_calls

    return extract_tool_calls(traces)


def llm_calls(
    traces: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """
    Extracts tool calls with their full path from the given traces.

    Args:
        traces (Dict[str, Any] or List[Dict[str, Any]]): The JSON traces to analyze.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the name and full path of a tool call.
    """

    def extract_llm_calls(
        node: Union[Dict[str, Any], List[Dict[str, Any]]], path: List[str] = []
    ) -> List[Dict[str, str]]:
        llm_calls = []
        if isinstance(node, list):
            for index, item in enumerate(node):
                if "type" in item and item["type"] == "llm_call":
                    llm_calls.extend(extract_llm_calls(item, path + [str(index)]))
                elif "children" in item and item["children"]:
                    llm_calls.extend(extract_llm_calls(item, path + [str(index)]))
        elif isinstance(node, dict):
            if "type" in node and node["type"] == "llm_call":
                llm_calls.append(
                    {
                        "name": node["name"],
                        "inputs": node["inputs"],
                        "outputs": node["outputs"],
                        "path": ".".join(path),
                    }
                )
            if "children" in node:
                llm_calls.extend(
                    extract_llm_calls(node["children"], path + [node.get("name", "")])
                )
        return llm_calls

    return extract_llm_calls(traces)


def get_nested_value(nested_dict, path):
    full_path = []
    for key in path:
        if isinstance(key, int):
            nested_dict = nested_dict[key]
        elif isinstance(nested_dict, dict):
            nested_dict = nested_dict.get(key, {})
        else:
            return None
    return nested_dict


def get_all_prior_trace(
    traces: Dict[str, Any], path: List[Union[str, int]]
) -> Dict[str, Any]:
    """
    Get all prior traces including parents and immediate siblings with smaller indices.

    Args:
        traces (Dict[str, Any]): The full trace dictionary.
        path (List[Union[str, int]]): The path to the target element.

    Returns:
        Dict[str, Any]: A new trace dictionary containing only the relevant elements.
    """

    def recursive_build(
        current: Union[Dict[str, Any], List[Any]],
        current_path: List[Union[str, int]],
        target_path: List[Union[str, int]],
    ) -> Union[Dict[str, Any], List[Any]]:
        if not target_path:
            return current

        if isinstance(current, dict):
            new_dict = {}
            for key, value in current.items():
                if key in ["name", "inputs", "outputs"]:
                    new_dict[key] = value
                elif key == target_path[0] or key == "children":
                    new_dict[key] = recursive_build(
                        value, current_path + [key], target_path[1:]
                    )
            return new_dict
        elif isinstance(current, list):
            new_list = []
            for index, item in enumerate(current):
                if index == target_path[0]:
                    new_list.append(
                        recursive_build(item, current_path + [index], target_path[1:])
                    )
                    break
                else:
                    new_list.append(
                        {
                            k: v
                            for k, v in item.items()
                            if k in ["name", "inputs", "outputs"]
                        }
                    )
            return new_list
        else:
            return current

    return recursive_build(traces, [], path)


def extract_user_intent_task(
    traces: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""
You are tasked with extracting the user's intent and context summary from traces of an LLM agentic application. The traces are provided in JSON format as a Python dictionary. Your goal is to analyze the user queries and outputs from different components in the application to create a concise summary of the user's intent and the context of their interaction.

Here are the traces you will be working with:

**Traces JSON:**
{json.dumps(traces, indent=2)}

----------------------------------------------------------------------------------------------------
Follow these steps to extract the user's intent and context summary:

1. Identify all user queries in the traces. These are typically found in the 'input' field of components labelled as 'user' or similar.
2. Analyze the outputs of various components, especially those that directly respond to user queries or process user input.
3. Look for any context-setting information, such as initial prompts or system messages that might provide background for the user's intent.
4. Examine the flow of the conversation and the sequence of components to understand the overall direction of the interaction.
5. Pay attention to any clarifications or refinements of the user's request throughout the trace.
6. Identify the main task or goal the user is trying to accomplish.
7. Note any constraints, preferences, or specific requirements mentioned by the user.

When summarizing the user's intent and context:
- Be concise but comprehensive. Capture the essence of what the user wants to achieve.
- Include relevant details that provide important context.
- Avoid including unnecessary technical details from the trace structure.
- Use clear, straightforward language.

Your output should be a JSON object with a single key 'intent' and a string value containing the summarized intent and context. For example:

{{
  "intent": "User wants to book a flight from New York to London for next week, preferring a direct flight with a reputable airline. They are flexible on the exact day but want to keep the cost under $1000."
}}

Here are examples of good and bad summaries:

Good: "User is researching the impact of climate change on polar bear populations, focusing on data from the last decade and seeking peer-reviewed sources."

Bad: "User asked about polar bears and climate change. The agent searched for information and provided some results."

For complex traces, use a scratchpad to organize your thoughts before formulating the final summary. You can use the following format:

<scratchpad>
Key points:
1. [List main points from the trace]
2. ...

User's primary goal: [Identify the overarching objective]

Important context: [Note any crucial background information]

Refinements or constraints: [List any specific requirements or limitations]
</scratchpad>

After analyzing the traces and organizing your thoughts, provide your final output in the required JSON format with the 'intent' key and the summarized string as its value.
"""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result


def extract_agent_thought_process(
    llm_traces: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""
You are tasked with extracting planning information from LLM (Large Language Model) outputs in an agentic application. You will be given a JSON string representing traces of LLM calls and responses, with everything else removed. Your goal is to identify and summarize any planning steps the LLM outlines to complete a task.

Here is the JSON string containing the LLM traces:

**LLM Traces:**
{json.dumps(llm_traces, indent=2)}


----------------------------------------------------------------------------------------------------
To extract the planning information, follow these steps:

1. Parse the JSON string into a Python dictionary.
2. Iterate through the LLM outputs in the traces.
3. Look for sections or phrases that indicate planning, such as:
   - "First, I will..."
   - "The steps to complete this task are..."
   - "My plan is to..."
   - Numbered or bulleted lists of actions
   - Phrases like "step 1", "step 2", etc.
4. Collect and summarize these planning steps.
5. If no clear planning is found, return an empty string.

Your output should be a JSON object with a single key "plan" containing a string value. This string should be either:
- A stepwise summary of the plan extracted from the LLM outputs
- An empty string if no clear planning is found

Use the following best practices:
- Be concise in your summary, focusing on key actions and their order
- Maintain the original ordering of steps if present
- Combine similar or repetitive steps
- Use clear, action-oriented language

Here are examples of good and bad outputs:

Good output:
{{
  "plan": "1. Analyze the given data\n2. Identify key trends\n3. Create visualizations\n4. Write a summary report\n5. Present findings to stakeholders"
}}

Bad output:
{{
  "plan": "The LLM talked about analyzing data and then it mentioned something about trends. It also said something about a report but I'm not sure if that's part of the plan."
}}

Another good output (when no plan is found):
{{
  "plan": ""
}}

Remember:
- Focus only on extracting planning information, not on other aspects of the LLM's output
- If you're unsure whether something constitutes a plan, err on the side of inclusion
- Ensure your output is valid JSON with proper escaping of special characters

Provide your output in the following format:
<output>
{{
  "plan": "Your extracted plan here"
}}
</output>
"""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result


def evaluate_tool_usage_efficiency(
    all_tool_calls_w_parameters: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""
You are an expert AI system evaluator tasked with assessing the Tool Usage Efficiency of an Agentic AI system. Your objective is to analyze how effectively and efficiently the AI uses the tools at its disposal to accomplish tasks. To accomplish the task of evaluating the Tool Usage Efficiency of an Agentic AI system, follow these steps:

### Steps to Accomplish the Task:

1. **Understand the Task Context**:
   - Review the task or query presented to the AI.
   - Familiarize yourself with the AI’s thought process, if available.

2. **Identify the Tools Used**:
   - Extract the list of tools used by the AI from the tool usage events.
   - Pay attention to the order in which the tools were invoked.

3. **Evaluate the Tool Usage**:
   - **Necessity**: Were all tool usages necessary, or were some redundant?
   - **Sequence Optimization**: Was the order of tool usage logical and efficient to accomplish the task?
   - **Parameter Optimization**: Were the input parameters chosen appropriately and efficiently for each tool call?
   - **Result Utilization**: How effectively did the AI make use of the outputs or results from each tool?
   - **Resource Management**: Did the AI account for computational or time costs when using the tools?
   - **Adaptability**: How well did the AI adjust its tool usage based on intermediate results?
   - **Goal Alignment**: Did the tool usage align with the AI’s overall objective for the task?

4. **Assign a Score**:
   - Based on your analysis, assign a score between 0 and 1 (to two decimal places).
     - **0**: Completely inefficient or ineffective tool usages.
     - **1**: Optimal and highly efficient tool usages.

5. **Justify the Score**:
   - Provide a brief explanation (2-3 sentences) justifying the score you gave, considering the efficiency and effectiveness factors mentioned above.

6. **Identify Inefficiencies (if any)**:
   - Highlight any specific inefficiency or missed optimization opportunities you observed (e.g., unnecessary tool calls, poor sequence of operations).

### Output Format:
The output should follow this JSON format:
{{
    "tools_used": ["<List of tools used, in order>"],
    "score": <Your score from 0 to 1, to two decimal places>,
    "justification": "<2-3 sentences explaining your score>",
    "inefficiency_identified": "<Brief description of any notable inefficiency, if applicable>"
}}

### Example of Expected Output:
{{
    "tools_used": ["browser", "python"],
    "score": 0.85,
    "justification": "The AI effectively used both tools to accomplish the task. The browser was used to gather information, and python was used for computation. However, an unnecessary second call to the browser slightly decreased efficiency.",
    "inefficiency_identified": "An extra browser call was made when the initial result was sufficient."
}}
---

Make sure that the evaluations for all tool calls adhere to this format, and that the explanations are concise yet sufficiently detailed to explain the reasoning behind the score. Consistently apply these criteria across all traces provided.

### Input:
Tool Usage Data:
{json.dumps(all_tool_calls_w_parameters, indent=2)}
"""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    return result


def execute_tool_usage_efficiency_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any], metadata: dict
) -> Dict[str, Any]:
    # Set the environment variables (unchanged)
    keys_to_remove = []
    for key in config.keys():
        if "key" in key.lower():
            os.environ[key] = config[key]
            keys_to_remove.append(key)

    tool_calls = tool_calls_with_path(trace_json, metadata)

    tool_usage_efficiency_input_parameters = []
    for tool_call in tool_calls:
        input_parameters = {}
        input_parameters["tool_name"] = tool_call["name"]
        path = tool_call["path"].split(".")
        path = [int(element) if element.isdigit() else "children" for element in path]
        tool_call_trace = get_all_prior_trace(trace_json, path)
        intent_task = extract_user_intent_task(tool_call_trace, config)
        input_parameters["task"] = intent_task["intent"]
        llm_call_outputs = llm_calls(tool_call_trace)
        agent_thought_process = extract_agent_thought_process(llm_call_outputs, config)
        input_parameters["agent_planning"] = agent_thought_process["plan"]

        if metadata:
            tool_call_outcome = get_nested_value(trace_json, path)
            input_parameters["tool_call_outcome"] = tool_call_outcome.get("outputs")
            input_parameters["tool_call_inputs"] = tool_call_outcome.get("inputs")
            input_parameters["tool_call_start"] = tool_call_outcome.get("start")
            input_parameters["tool_call_end"] = tool_call_outcome.get("end")
            input_parameters["all_available_tools"] = metadata["tools"]

        tool_usage_efficiency_input_parameters.append(input_parameters)

    all_tool_usage_efficiency_input_parameters = json.dumps(
        tool_usage_efficiency_input_parameters, indent=2
    )

    tool_usage_efficiency = evaluate_tool_usage_efficiency(
        tool_usage_efficiency_input_parameters, config
    )
    return {
        "metric_name": "tool_usage_efficiency",
        "config": config,
        "result": {
            "score": tool_usage_efficiency["score"],
            "tools_used": tool_usage_efficiency["tools_used"],
            "justification": tool_usage_efficiency["justification"],
            "inefficiency_identified": tool_usage_efficiency["inefficiency_identified"]
        },
    }
