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
                    tool_calls.append({"name": node["name"], "path": ".".join(path)})
                elif metadata is None:
                    tool_calls.append({"name": node["name"], "path": ".".join(path)})
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


def evaluate_tool_selection_accuracy(
    all_tool_selection_input_parameters: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
You are an expert AI system evaluator tasked with assessing the Tool Selection Accuracy of an Agentic AI system. Your goal is to analyze the AI's choices in selecting tools for various tasks and determine how appropriate and effective these selections are.
Context
You will be provided with a series of langraph tool calls, each containing:

1. Tasks: The task or query presented to the AI till before the tool call
2. Thought Process: The AI's thought process (if available) till before the tool call
3. Selected Tool: The tool selected by the AI
4. Available Tools: A list of available tools and their descriptions
5. Tool Outcomes: The outcome or result of using the selected tool till before the tool call

**Your Task**
For each tool call, you must:

1. Evaluate the appropriateness of the tool selections
2. Provide a score from 0 to 1, where:

  - 0 represents a completely inappropriate tool selection
  - 1 represents the most optimal tool selection


3. Justify your score with a brief explanation

**Evaluation Criteria**
Consider the following factors in your evaluation:

1. Relevance: How well does the selected tool align with the given task?
2. Efficiency: Is the chosen tool the most efficient option for accomplishing the task?
3. Alternatives: Were there better tools available that the AI overlooked?
4. Context Understanding: Did the AI demonstrate a clear understanding of the task requirements in its selection?
5. Potential Impact: How significant is the tool selection in the overall success of the task?

**Output Format**
For each selected tool, provide your evaluation in the following JSON format:
{{
"evaluation": [
        {{
            "selected_tool": [Name of the tool selected by the AI],
            "score": [Your score from 0 to 1, to two decimal places],
            "justification": [2-3 sentences explaining your score],
            "improvement_suggestion": [If applicable, a brief suggestion for better tool selection]
        }}, 
        ...
    ]
}}

**Additional Guidelines**

1. Be objective and consistent in your evaluations across different tool calls.
2. Consider the nuances of each task and the specific capabilities of the available tools.
3. If the AI's thought process is available, use it to inform your understanding of the decision-making behind the tool selection.
4. If you notice any patterns or trends across multiple tool calls, mention them in a summary after evaluating all individual tool calls.

Remember, your goal is to provide an accurate and fair assessment of the AI's Tool Selection Accuracy to help improve its performance and decision-making capabilities.

Tool Selection Data:
{json.dumps(all_tool_selection_input_parameters, indent=2)}
"""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    return result


def execute_tool_selection_accuracy_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any], metadata: dict
) -> Dict[str, Any]:
    # Set the environment variables (unchanged)
    keys_to_remove = []
    for key in config.keys():
        if "key" in key.lower():
            os.environ[key] = config[key]
            keys_to_remove.append(key)

    tool_calls = tool_calls_with_path(trace_json, metadata)

    tool_selection_input_parameters = []
    for tool_call in tool_calls:
        input_parameters = {}
        input_parameters["selected_tool"] = tool_call["name"]
        path = tool_call["path"].split(".")
        path = [int(element) if element.isdigit() else "children" for element in path]
        tool_call_trace = get_all_prior_trace(trace_json, path)
        intent_task = extract_user_intent_task(tool_call_trace, config)
        input_parameters["intent_task"] = intent_task["intent"]
        llm_call_outputs = llm_calls(tool_call_trace)
        agent_thought_process = extract_agent_thought_process(llm_call_outputs, config)
        input_parameters["agent_planning"] = agent_thought_process["plan"]

        if metadata is not None:
            tool_call_outcome = get_nested_value(trace_json, path)
            input_parameters["tool_call_outcome"] = tool_call_outcome.get("outputs")
            input_parameters["tool_call_inputs"] = tool_call_outcome.get("inputs")
            input_parameters["all_available_tools"] = metadata["tools"]
        tool_selection_input_parameters.append(input_parameters)

    all_tool_selection_input_parameters = json.dumps(
        tool_selection_input_parameters, indent=2
    )

    tool_selection_accuracy = evaluate_tool_selection_accuracy(
        tool_selection_input_parameters, config
    )["evaluation"]
    overall_score = sum(tool["score"] for tool in tool_selection_accuracy) / (
        len(tool_selection_accuracy) + 1e-8
    )
    selected_tools = [tool["selected_tool"] for tool in tool_selection_accuracy]
    reasoning = [tool["justification"] for tool in tool_selection_accuracy]

    return {
        "metric_name": "tool_selection_accuracy",
        "config": config,
        "result": {
            "score": overall_score,
            "selected_tools": json.dumps(selected_tools),
            "reasoning": json.dumps(reasoning),
        },
    }
