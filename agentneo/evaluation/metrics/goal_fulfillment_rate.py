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

def get_model_response(prompt, config):
    evaluation = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = evaluation.choices[0].message.content
    return result

from datetime import datetime
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


def execute_goal_fulfillment_metric(
    trace_json: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:

    #### Get Initial Goal ####
    conversations = extract_conversations_from_json(trace_json)
    user_intent = intent_identification(conversations, config)

    #### Get Relevant Responses ####
    # Instead of just the final response, we'll consider all relevant responses
    relevant_responses = extract_relevant_responses(conversations)

    # Evaluate goal fulfillment based on all relevant responses
    result = evaluate_goal_fulfillment(user_intent, relevant_responses, config)

    return {
        "metric_name": "goal_fulfillment_rate",
        "config": config,
        "result": {"inputGoal": user_intent, "relevantResponses": relevant_responses, "score": result["score"], "reason": result["explanation"]},
    }

def extract_relevant_responses(conversations):
    # This function will extract and combine relevant responses from the conversation
    relevant_responses = []
    for entry in conversations:
        if entry["function_name"] != "Unknown" and entry["response"] != "No response":
            relevant_responses.append(f"{entry['function_name']}: {entry['response']}")
    return "\n\n".join(relevant_responses)

# Update the evaluate_goal_fulfillment function to handle multiple responses
def evaluate_goal_fulfillment(
    query: str, responses: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = f"""
    Evaluate the goal fulfillment rate for the following user intent and system responses:

    User Intent: {query}

    System Responses:
    {responses}

    Please provide:
    1. A score between 0.0 and 1.0
    2. A detailed explanation for the score

    Consider all the responses provided and how well they collectively address the user's intent.

    Format your response as a JSON object:
    {{
        "score": <float>,
        "explanation": "<detailed explanation>"
    }}
    """

    try:
        response = get_model_response(prompt, config)
        
        # Try to parse the JSON from the response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the string
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("Unable to extract valid JSON from the response")

        # Ensure the result has the expected structure
        if not isinstance(result, dict) or 'score' not in result or 'explanation' not in result:
            raise ValueError("The parsed result does not have the expected structure")

        return result

    except Exception as e:
        # If any error occurs, return a default error response
        return {
            "score": 0.0,
            "explanation": f"Error: Unable to process the model's response. {str(e)}",
        }


# Example usage
# json_path = "/Users/vijay/Desktop/AgentNeo_folder/AgentNeo/travel.json"
# with open(json_path) as j:
#     trace_json = json.load(j)

# config = {"model": "gpt-4o-mini"}
# result = execute_goal_fulfillment_metric(trace_json, config)
# print(json.dumps(result, indent=2))
