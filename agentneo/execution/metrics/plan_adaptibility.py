import json
from typing import Dict, Any, List, Union, Optional
import litellm
import os
import ast

def tool_calls_with_path(traces: Union[Dict[str, Any], List[Dict[str, Any]]], metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, str]]:
    """
    Extracts tool calls with their full path from the given traces.

    Args:
        traces (Dict[str, Any] or List[Dict[str, Any]]): The JSON traces to analyze.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the name and full path of a tool call.
    """
    if metadata is not None:
        accepted_tools = [tool.get("name") for tool in metadata.get("tools", [])]
    def extract_tool_calls(node: Union[Dict[str, Any], List[Dict[str, Any]]], path: List[str] = []) -> List[Dict[str, str]]:
        tool_calls = []
        if isinstance(node, list):
            for index, item in enumerate(node):
                if 'type' in item and item['type'] != 'llm_call' and item['type'] != 'root':
                    if (metadata is not None) and (item['name'] in accepted_tools):
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                    elif metadata is None:
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                    elif 'children' in item and item['children']:
                        tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
                elif 'children' in item and item['children']:
                    tool_calls.extend(extract_tool_calls(item, path + [str(index)]))
        elif isinstance(node, dict):
            if 'type' in node and node['type'] != 'llm_call' and node['type'] != 'root':
                if (metadata is not None) and (node['name'] in accepted_tools):
                    tool_calls.append({
                        'name': node['name'],
                        'inputs': node['inputs'],
                        'outputs': node['outputs'],
                        'start': node['start'],
                        'end': node['end'],
                        'path': '.'.join(path)
                    })
                elif metadata is None:
                    tool_calls.append({
                        'name': node['name'],
                        'inputs': node['inputs'],
                        'outputs': node['outputs'],
                        'start': node['start'],
                        'end': node['end'],
                        'path': '.'.join(path)
                    })
            if 'children' in node:
                tool_calls.extend(extract_tool_calls(node['children'], path + [node.get('name', '')]))
        return tool_calls

    return extract_tool_calls(traces)



def get_llm_calls(traces: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Extracts tool calls with their full path from the given traces.

    Args:
        traces (Dict[str, Any] or List[Dict[str, Any]]): The JSON traces to analyze.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the name and full path of a tool call.
    """
    def extract_llm_calls(node: Union[Dict[str, Any], List[Dict[str, Any]]], path: List[str] = []) -> List[Dict[str, str]]:
        llm_calls = []
        if isinstance(node, list):
            for index, item in enumerate(node):
                if 'type' in item and item['type'] == 'llm_call':
                    llm_calls.extend(extract_llm_calls(item, path + [str(index)]))
                elif 'children' in item and item['children']:
                    llm_calls.extend(extract_llm_calls(item, path + [str(index)]))
        elif isinstance(node, dict):
            if 'type' in node and node['type'] == 'llm_call':
                llm_calls.append({
                    'name': node['name'],
                    'inputs': node['inputs'],
                    'outputs': node['outputs'],
                    'path': '.'.join(path)
                })
            if 'children' in node:
                llm_calls.extend(extract_llm_calls(node['children'], path + [node.get('name', '')]))
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


def get_all_prior_trace(traces: Dict[str, Any], path: List[Union[str, int]]) -> Dict[str, Any]:
    """
    Get all prior traces including parents and immediate siblings with smaller indices.

    Args:
        traces (Dict[str, Any]): The full trace dictionary.
        path (List[Union[str, int]]): The path to the target element.

    Returns:
        Dict[str, Any]: A new trace dictionary containing only the relevant elements.
    """
    def recursive_build(current: Union[Dict[str, Any], List[Any]], current_path: List[Union[str, int]], target_path: List[Union[str, int]]) -> Union[Dict[str, Any], List[Any]]:
        if not target_path:
            return current

        if isinstance(current, dict):
            new_dict = {}
            for key, value in current.items():
                if key in ['name', 'inputs', 'outputs']:
                    new_dict[key] = value
                elif key == target_path[0] or key == 'children':
                    new_dict[key] = recursive_build(value, current_path + [key], target_path[1:])
            return new_dict
        elif isinstance(current, list):
            new_list = []
            for index, item in enumerate(current):
                if index == target_path[0]:
                    new_list.append(recursive_build(item, current_path + [index], target_path[1:]))
                    break
                else:
                    new_list.append({k: v for k, v in item.items() if k in ['name', 'inputs', 'outputs']})
            return new_list
        else:
            return current

    return recursive_build(traces, [], path)


def extract_user_intent_task(traces: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any] :
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


def extract_agent_thought_process(llm_traces: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
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

def extract_plan_altering_info(traces: List[Dict[str, Any]], initial_plan: str, config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an AI assistant tasked with analyzing traces from an LLM agentic application and determining if any events in these traces require alterations to an initial plan. Your goal is to extract relevant information and present it in a structured JSON format.

First, review the initial plan:
**Initial Plan:**
{initial_plan}

Now, you will be presented with traces in JSON format (as a Python dictionary). These traces contain information about the execution of the LLM agentic application, including user interactions, tool usage, and potential errors.

**Traces:**
{json.dumps(traces, indent=2)}

To complete this task, follow these steps:

1. Parse the JSON traces carefully, ensuring you understand the structure and content.

2. Identify key events in the traces that might require plan alteration. These may include:
   - New information provided by the user
   - Errors encountered during tool execution
   - Denial of Service (DoS) issues with tools
   - Unexpected results from tool usage
   - Changes in the environment or context

3. For each identified event, determine if it necessitates a change in the initial plan.

4. Prepare a JSON output with the following structure:
   {{
     "plan_alterations_required": boolean,
     "events_requiring_alteration": [
       {{
         "event_type": string,
         "description": string,
         "reason_for_alteration": string
       }},
       ...
     ]
   }}

5. If no events require plan alteration, set "plan_alterations_required" to false and provide an empty list for "events_requiring_alteration".

Here's an example of a good response:
<example_good_response>
{{
  "plan_alterations_required": true,
  "events_requiring_alteration": [
    {{
      "event_type": "user_input",
      "description": "User provided new deadline: 2023-07-01",
      "reason_for_alteration": "The initial plan did not account for this specific deadline, requiring adjustment of task priorities and timelines."
    }},
    {{
      "event_type": "tool_error",
      "description": "Database query tool returned 'Connection Timeout' error",
      "reason_for_alteration": "The plan assumed database availability. An alternative data source or retry mechanism needs to be incorporated."
    }}
  ]
}}
</example_good_response>

Here's an example of a bad response:
<example_bad_response>
{{
  "plan_alterations_required": true,
  "events_requiring_alteration": [
    {{
      "event_type": "user_input",
      "description": "User said hello",
      "reason_for_alteration": "We should be polite and say hello back"
    }}
  ]
}}
</example_bad_response>

The bad response identifies an irrelevant event that doesn't actually require plan alteration.

Remember to:
- Be thorough in your analysis of the traces
- Only identify events that genuinely require plan alteration
- Provide clear and concise descriptions and reasons
- Ensure your output is in valid JSON format

Present your final output within <output> tags.
"""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result


def get_goal_achievement(overall_intent: str, task_outcome: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are tasked with evaluating how well a final outcome accomplishes a user's goal in an LLM agent chatbot interaction. Your evaluation should be thorough, impartial, and based solely on the information provided. Follow these steps carefully to complete your analysis:

1. First, you will be presented with an overall intent summary of the user's goal:

**Overall Intent Summary:**
{overall_intent}

2. Next, you will be given the final outcome from the application:

**Final Outcome:**
{task_outcome}


3. To determine how well the outcome accomplishes the user's goal, follow these steps:

   a. Carefully read and understand the user's goal from the overall intent summary.
   b. Analyze the final outcome in detail, noting all relevant aspects.
   c. Compare the final outcome to the user's original goal, identifying areas where the outcome meets, exceeds, or falls short of the goal.
   d. Determine the level of accomplishment, considering factors such as completeness, accuracy, and relevance.
   e. Provide clear, concise reasoning for your evaluation, highlighting specific points that influenced your decision.
   f. Based on your analysis, assign a score between 0.0 and 1.0, where 0.0 represents no accomplishment of the goal, and 1.0 represents perfect accomplishment.

4. Present your evaluation in JSON format, using the following structure:

**Output:**
{{
  "score": [Your score between 0.0 and 1.0],
  "reason": "[Your concise reasoning for the score]"
}}

Remember to be objective and thorough in your analysis. Consider all aspects of the user's goal and how well the final outcome addresses each of them. Provide specific examples from both the intent summary and the final outcome to support your reasoning.

Ensure that your score accurately reflects the degree to which the outcome accomplishes the user's goal. A score of 1.0 should only be given if the outcome perfectly and completely fulfills the user's intent, while a score of 0.0 should be reserved for outcomes that entirely fail to address the user's goal.

Now, proceed with your analysis and provide your evaluation in the specified JSON format."""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result

def evaluate_plan_adaptibility(llm_calls: List[Dict[str, Any]], initial_plan: str, overall_intent: str, task_outcome: str, goal_achievement: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an expert AI system evaluator tasked with assessing the Plan Adaptability of an Agentic AI system. Your objective is to analyze how effectively the AI adjusts its plans in response to changing circumstances, unexpected outcomes, or new information during task execution.
Context
You will be provided with a series of langraph traces, each containing:

1. The original task or goal presented to the AI
2. The AI's initial plan
3. A sequence of events, including:
  - Unexpected outcomes or new information
  - The AI's thought process (if available)
  - Plan adjustments made by the AI
4. The final outcome or result of the task execution
5. The rubric for achieving the goal

**Your Task**
For each trace, you must:
1. Evaluate the adaptability and effectiveness of the AI's plan adjustments
2. Provide a score from 0 to 1, where:
  - 0 represents complete failure to adapt or inappropriate adaptations
  - 1 represents optimal and highly effective plan adaptations
3. Justify your score with a brief explanation

**Evaluation Criteria**
Consider the following factors in your evaluation:

1. Responsiveness: How quickly and appropriately did the AI react to changes or new information?
2. Relevance: Were the plan adjustments relevant to the changes encountered?
3. Effectiveness: Did the adaptations improve the likelihood of task success?
4. Efficiency: Did the AI minimize unnecessary changes while making needed adjustments?
5. Consistency: Did the adaptations maintain alignment with the original goal?
6. Creativity: Did the AI demonstrate novel or innovative approaches in its adaptations when necessary?
7. Proportionality: Were the magnitude of adjustments proportional to the changes encountered?

**Output Format**
For each trace, provide your evaluation in the following JSON format:
{{
  "original_task": "<Brief description of the original task or goal>",
  "major_changes_encountered": "<List of significant unexpected events or new information>",
  "number_of_plan_adjustments": <Total number of distinct plan adjustments made>,
  "score": "<Your score from 0 to 1, to two decimal places>",
  "justification": "<2-3 sentences explaining your score>",
  "most_effective_adaptation": "<Brief description of the most successful plan adjustment>",
  "missed_opportunity": "<If applicable, a situation where better adaptation could have been made>",
  "improvement_suggestion": "<A concise suggestion for enhancing the AI's plan adaptability>"
}}

Additional Guidelines

Be objective and consistent in your evaluations across different traces.
Consider the complexity of the task and the significance of the changes encountered when assessing the appropriateness of adaptations.
If the AI's thought process is available, use it to understand the reasoning behind plan adjustments.
Pay attention to how well the AI balances persistence in its original plan with necessary adaptations.
Evaluate how well the AI anticipates potential future changes in its adaptations.
Consider whether the AI learns from its adaptations and applies this learning to subsequent changes within the same task.
If you notice any patterns or trends across multiple traces, mention them in a summary after evaluating all individual traces.


**Overall Intent:**
{overall_intent}

**Initial Plan:**
{initial_plan}

**Task Outcome:**
{task_outcome}

**Goal Achievement:**
{json.dumps(goal_achievement, indent=2)}

**Event Sequence:**
{llm_calls}



Remember, your goal is to provide an accurate and fair assessment of the AI's Plan Adaptability to help improve its performance in dynamic and uncertain environments."""
    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result

def execute_plan_adaptibility_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    # Set the environment variables (unchanged)
    keys_to_remove = []
    for key in config.keys():
        if "key" in key.lower():
            os.environ[key] = config[key]
            keys_to_remove.append(key)

    llm_calls = get_llm_calls(trace_json)

    llm_calls_ = []
    # agent_thought_processes = []
    llm_traces = []
    for llm_call in llm_calls:
        path = llm_call['path'].split('.')
        path = [int(element) if element.isdigit() else 'children' for element in path]
        llm_calls_.append({k: v for k, v in llm_call.items() if k != 'path'})
        llm_call_trace = get_all_prior_trace(trace_json, path)
        llm_traces.append(llm_call_trace)
        agent_thought_process = extract_agent_thought_process(llm_calls_, config)
        intent_task = extract_user_intent_task(llm_call_trace, config)
        # agent_thought_processes.append(agent_thought_process)
        llm_call['task'] = intent_task['intent']
        llm_call['plan'] = agent_thought_process['plan']
    
    initial_plan = None
    for llm_call in llm_calls:
        if llm_call['plan'] != '':
            initial_plan = llm_call['plan']
            break

    if initial_plan is None:
        return {
            "metric_name": "plan_adaptibility",
            "config": config,
            "result": {
                "score": 0,
                "initial_plan": None, 
                "justification": "No plan found in the traces"
            },
        }
    
    for trace, llm_call in zip(llm_traces, llm_calls):
        plan_alterations_required = extract_plan_altering_info(trace, initial_plan, config)
        llm_call['plan_alterations_required'] = plan_alterations_required['plan_alterations_required']
        llm_call['events_requiring_alteration'] = plan_alterations_required['events_requiring_alteration']
    
    if not any(llm_call['plan_alterations_required'] for llm_call in llm_calls):
        return {
            "metric_name": "plan_adaptibility",
            "config": config,
            "result": { 
                 
                "score": 0,
                "initial_plan": initial_plan, 
                "justification": "No plan alterations required"
            },
        }
    
    overall_intent = extract_user_intent_task(trace_json, config)
    task_outcome = trace_json['children'][-1]['children'][-1]['outputs']

    goal_achievement = get_goal_achievement(overall_intent, task_outcome, config)

    
    