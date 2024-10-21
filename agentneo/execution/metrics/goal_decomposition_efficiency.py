import json
from typing import Dict, Any
from dotenv import load_dotenv
import litellm
import ast
import os
from typing import List, Optional
import re
from datetime import datetime

load_dotenv()

def get_model_response(prompt, config):
    evaluation = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = evaluation.choices[0].message.content
    return result

def extract_info(data, results=None):
    if results is None:
        results = []

    # Base case: when there are no more children to process
    if "children" not in data or not data["children"]:
        return results

    for item in data["children"]:
        # Extract basic information
        function_name = item.get("name", "Unknown")
        query = item.get("inputs", {})
        response = item.get("outputs", {})

        # Store the current node's data
        current_data = {
            "function_name": function_name,
            "input": query,
            "output": response,
        }

        # Add the current node's data to the results list
        results.append(current_data)

        # Recursive call to process any children of the current node
        extract_info(item, results)

    return results

def intent_identification(conversation, config):
    prompt = f"""
    You are an AI system designed to analyze conversations between users and language models (LLMs) to identify the primary user intent. Your task is to process the conversation history and provide a clear, concise statement of the user's main goal or purpose.

    ## Input
    You will receive a conversation log containing:
    1. Function names
    2. User inputs or system prompts
    3. LLM or system responses

    ## Output
    Provide a clear description of the main goal driving the user's interaction, expressed in 3-4 sentences.

    ## Analysis Guidelines

    1. Clarity is Key: Provide a clear, unambiguous statement of intent. Avoid vague or overly broad interpretations.
    2. Context Matters: Consider the full context of the conversation, including any background information provided by the user.
    3. Evolution of Intent: If the user's intent changes during the conversation, identify the most recent or prominent intent.
    4. Objectivity: Base your assessment solely on the conversation content, avoiding unsupported assumptions or inferences.
    5. Conciseness: Express the intent in 3-4 well-formulated sentences.
    6. User-Centric: Focus on what the user wants to achieve, not on the system's actions or responses.
    7. Action-Oriented: When possible, frame the intent in terms of an action or goal the user wants to accomplish.
    8. Completeness: Use the 3-4 sentences to capture any nuances or complexities in the user's intent.

    ## Output Format
    Your output should be 3-4 sentences

    ## Example
    Input Conversation:
    [
        {{
            "function_name": "get_travel_recommendations",
            "input": "I'm planning a trip to Japan next month. Can you suggest some must-visit places in Tokyo?",
            "response": "Certainly! Tokyo has many exciting attractions. Some must-visit places include the historic Senso-ji Temple, the bustling Shibuya Crossing, and the serene Meiji Shrine. The Tokyo Skytree offers panoramic views of the city, while the Tsukiji Outer Market is perfect for food lovers."
        }},
        {{
            "function_name": "get_cultural_experiences",
            "input": "Those sound great! I'm particularly interested in experiencing traditional Japanese culture. Any specific recommendations for that?",
            "response": "For traditional Japanese culture in Tokyo, I'd highly recommend visiting..."
        }}
    ]

    Output:
    The user intends to plan a culturally enriching trip to Tokyo, Japan. They are seeking recommendations for must-visit places, with a particular emphasis on experiences that showcase traditional Japanese culture. The user's intent has evolved from a general interest in Tokyo's attractions to a more focused desire for authentic cultural experiences, indicating a preference for immersive and historically significant sites over modern or purely touristic destinations.

    Analyse the given conversation:
    {conversation}

    Remember, your goal is to provide a clear, high-quality identification of the user's primary intent in 3-4 sentences. Prioritize accuracy and clarity in your analysis while capturing the full scope of the user's goals.
    """

    return get_model_response(prompt, config)

def extract_thought_process(conversation_flow, intent, config):
    prompt = f"""
        Extract a concise AI-generated plan from the given conversation flow, considering the user's intent:

    1. Review the user's intent and the list of function calls and responses.

    2. Identify AI-generated plans by looking for:
    - Responses explicitly mentioning "plan," "steps," or "approach"
    - Structured lists of actions in responses
    - Outputs from planning-related functions (e.g., "plan_task," "create_strategy")
    - Responses outlining a series of actions to achieve the user's goal
    - A "plan" refers specifically to the sequence of steps or actions taken by the AI to get the solution. It does not refer to the structure of the AI's response, but rather to the concrete actions proposed in AI's territory.

    3. If a clear AI-generated plan is present:
    - Extract only the main steps or components that represent actions to be performed
    - Preserve the original order and structure of the steps
    - Omit explanatory text or details not directly part of the actionable steps

    4. If no explicit plan is found, state: "No coherent AI-generated plan detected."

    6. Guidelines:
    - Focus solely on AI-generated plans in responses that outline steps to solve the user's problem
    - Do not include information from user inputs or function names
    - Do not invent steps not supported by AI responses
    - Exclude any response information not directly related to actionable steps

    7. Format:
    - Use a concise numbered or bulleted list
    - Limit each step to one brief sentence describing an action to be taken
    - Omit explanations or references to specific responses

    8. Keep the extracted plan as short and to the point as possible while accurately representing the AI's proposed actions to address the user's intent.

    conversation_flow:
    {conversation_flow}

    intent:
    {intent}
    """
    return get_model_response(prompt, config)

def evaluate_goal_decomposition_efficiency(
    task, subtasks, tools_executed, tool_descriptions, config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""You are an expert AI system evaluator tasked with assessing the Goal Decomposition Efficiency of an Agentic AI system. 
    Your objective is to analyze how effectively the AI breaks down complex goals into manageable sub-tasks and evaluate the quality of this decomposition process.

    Context
    You will be provided with a series of langraph traces, each containing:

    The original complex goal or task presented to the AI
    {task}
    The list of sub-tasks or sub-goals generated by the AI
    {subtasks}
    A list of dictionaries representing the tools executed, each containing:
    {tools_executed}
    [
        {{
            "input": "[Input provided to the tool]",
            "response": "[Response received from the tool]",
            "tool_name": "[Name of the tool used]"
        }},
        ...
    ]
    The tool descriptions
    {tool_descriptions}

    Your Task
    For each trace, you must:

    1. Associate all tools executed with generated sub-goals based on the task description from the metadata. In case task description is not available, assign according to tool name.
    2. If the subgoals are not available, return score 0, with reason "No subtasks detected"
    2. Evaluate the efficiency and effectiveness of the goal decomposition
    3. Provide a score from 0 to 1, where:
    - 0 represents a completely ineffective or inefficient goal decomposition
    - 1 represents an optimal and highly efficient goal decomposition
    4. Justify your score with a brief explanation

    Evaluation Criteria
    Consider the following factors in your evaluation:

    - Tool Assignment: Are the tools appropriately assigned to each sub-task?
    - Goal Achievement: Has the user goal been achieved? Have all the sub-tasks achieved their respective goals?
    - Completeness: Does the set of sub-tasks cover all aspects of the original goal?
    - Granularity: Are the sub-tasks broken down to an appropriate level of detail?
    - Logical Sequence: Is there a clear and logical order to the sub-tasks?
    - Independence: Can sub-tasks be executed independently where appropriate?
    - Clarity: Are the sub-tasks clearly defined and unambiguous?
    - Efficiency: Does the decomposition avoid unnecessary redundancy or overlap?
    - Scalability: Would this decomposition approach work well for similar but more complex goals?

    Output Format
    {{
        "originalGoal": "[Brief description of the original goal]",
        "subtasks": "[list of sub-tasks generated]",
        "score": 0.00,
        "reason": "[2-3 sentences explaining your score]",
    }}

    Additional Guidelines:
    - Be objective and consistent in your evaluations across different traces.
    - Consider the nature and complexity of the original goal when assessing the appropriateness of the decomposition.
    - If the AI's thought process is available, use it to understand the reasoning behind the goal decomposition.
    - Pay attention to how well the sub-tasks align with the original goal's objectives.
    - Consider how well the decomposition would facilitate parallel processing or task delegation, if applicable.
    - If you notice any patterns or trends across multiple traces, mention them in a summary after evaluating all individual traces.
    - Ensure that the tool assignments are logical and align with the sub-task objectives.
    - Evaluate whether the overall user goal has been achieved and if all sub-tasks have achieved their respective goals.
    - Analyze the input and response for each tool execution to assess its relevance and effectiveness in achieving the sub-task.

    Remember, your goal is to provide an accurate and fair assessment of the AI's Goal Decomposition Efficiency to help improve its strategic planning and task management capabilities.
    json"""

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
            "reson": "Error: The model did not return a valid JSON response.",
        }

    return result

def parse_timestamp(timestamp):
    return datetime.fromisoformat(timestamp)

def extract_conversations_from_json(json_data):
    # Extract and sort LLM calls and tool calls by timestamp
    llm_calls = sorted(json_data['llm_calls'], key=lambda x: parse_timestamp(x['start_time']))
    tool_calls = sorted(json_data['tool_calls'], key=lambda x: parse_timestamp(x['start_time']))

    # Combine and sort all calls
    all_calls = llm_calls + tool_calls
    all_calls_sorted = sorted(all_calls, key=lambda x: parse_timestamp(x['start_time']))

    # Prepare the conversation data
    conversations = []
    for call in all_calls_sorted:
        if 'input_prompt' in call:
            input_content = call['input_prompt'][0]['content']
        else:
            input_content = call['input_parameters']
        
        conversation_entry = {
            "function_name": call.get('name', 'Unknown'),
            "input": input_content,
            "response": call.get('output', 'No response')
        }
        conversations.append(conversation_entry)

    return conversations

def execute_goal_decomposition_efficiency_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]={}, metadata: Dict={},
) -> Dict[str, Any]:
   
    # Extract conversations from trace_json
    conversations = extract_conversations_from_json(trace_json)

    user_intent = intent_identification(conversations, config)
    thought_process = extract_thought_process(conversations, user_intent, config)
    tool_descriptions = metadata

    # Evaluate goal fulfillment
    result = evaluate_goal_decomposition_efficiency(
        user_intent, thought_process, conversations, tool_descriptions, config
    )

    return {
        "metric_name": "goal_fulfillment_rate",
        "config": config,
        "result": result,
    }

# Example usage
# json_path = "/Users/vijay/Desktop/AgentNeo_folder/AgentNeo/travel.json"
# with open(json_path) as j:
#     trace_json = json.load(j)

# config = {"model": "gpt-4o-mini"}
# metadata = {
#     "tools": [
#       {
#         "name": "flight_price_estimator_tool",
#         "description": "flight_price_estimator_tool"
#       },
#       {
#         "name": "currency_converter_tool",
#         "description": "currency_converter_tool"
#       },
#     ]
#   }

# result = execute_goal_decomposition_efficiency_metric(trace_json, config, metadata)
# print(json.dumps(result, indent=2))