import json
from typing import Dict, Any, List
from datetime import datetime
import litellm
import time


def get_model_response(prompt, config):
    evaluation = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = evaluation.choices[0].message.content
    return result

def analyze_context_usage(conversations: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes how effectively context is maintained and used across interactions.
    """
    context_metrics = {
        "referenced_info": [],
        "context_switches": 0,
        "context_maintenance": 0.0,
        "information_reuse": []
    }

    # Get the context switch threshold from config or use default
    context_switch_threshold = config.get('context_switch_threshold', 0.3)

    for i, conv in enumerate(conversations):
        # Extract key information from current conversation
        current_info = extract_key_information(conv['input'], conv['response'], config)

        # Track referenced information
        if i > 0:
            prev_info = extract_key_information(conversations[i-1]['input'], conversations[i-1]['response'], config)
            context_overlap = analyze_information_overlap(current_info, prev_info)
            context_metrics['information_reuse'].append(context_overlap)

            # Detect context switches using the threshold from config
            if context_overlap < context_switch_threshold:
                context_metrics['context_switches'] += 1

        context_metrics['referenced_info'].extend(current_info)

    # Calculate context maintenance score
    if len(conversations) > 1:
        context_metrics['context_maintenance'] = calculate_context_maintenance(
            context_metrics['information_reuse']
        )

    return context_metrics


def extract_key_information(input_text: str, response_text: str, config: Dict[str, Any]) -> List[str]:
    """
    Extracts key information points from input and response.
    """
    prompt = f"""
    Extract key information points from this conversation segment:
    
    Input: {input_text}
    Response: {response_text}
    
    Return only the core information points, one per line.
    """
    
    result = get_model_response(prompt, config)
    return [point.strip() for point in result.split('\n') if point.strip()]

def analyze_information_overlap(current_info: List[str], previous_info: List[str]) -> float:
    """
    Analyzes the overlap between current and previous information.
    """
    current_set = set(current_info)
    previous_set = set(previous_info)
    
    if not previous_set:
        return 1.0
        
    overlap = len(current_set.intersection(previous_set))
    total = len(previous_set)
    
    return overlap / total

def calculate_context_maintenance(reuse_scores: List[float]) -> float:
    """
    Calculates overall context maintenance score.
    """
    if not reuse_scores:
        return 0.0
    return sum(reuse_scores) / len(reuse_scores)

def evaluate_context_coherence(conversations: List[Dict[str, Any]], context_metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the coherence of context usage throughout the conversation.
    """
    prompt = f"""
    Evaluate the coherence and effectiveness of context retention in this conversation:
    
    Conversation Flow: {json.dumps(conversations, indent=2)}
    Context Metrics: {json.dumps(context_metrics, indent=2)}
    
    Consider:
    1. How well information is maintained across interactions
    2. Appropriateness of context switches
    3. Relevance of referenced information
    4. Consistency in information usage
    
    Provide a score between 0.0 and 1.0 and explain your reasoning.
    
    Format as JSON:
    {{
        "score": <float>,
        "explanation": "<explanation>",
        "key_observations": [<list of observations>]
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
    except Exception as e:
        result = {
            "score": 0.0,
            "explanation": f"Error in evaluation: {str(e)}",
            "key_observations": []
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

def execute_context_retention_metric(
    trace_json: Dict[str, Any],
    config: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Executes the Context Retention Rate metric evaluation.

    Parameters:
    - trace_json: Dictionary containing the agent's execution trace.
    - config: Configuration parameters.

    Returns:
    - Dictionary containing the evaluation results.
    """
    # Extract conversations
    conversations = extract_conversations_from_json(trace_json)
    
    # Analyze context usage
    context_metrics = analyze_context_usage(conversations, config)
    
    # Evaluate context coherence
    coherence_result = evaluate_context_coherence(
        conversations, context_metrics, config
    )
    
    # Calculate final score with configurable weights
    context_maintenance_weight = config.get("context_maintenance_weight", 0.4)
    coherence_weight = config.get("coherence_weight", 0.6)
    
    final_score = round(
        context_metrics['context_maintenance'] * context_maintenance_weight +
        coherence_result['score'] * coherence_weight,
        2
    )
    
    # Prepare the reason/explanation
    reason = coherence_result['explanation']
    
    # Construct the result dictionary with 'score' and 'reason' at the top level of 'result'
    return {
        "metric_name": "context_retention_rate",
        "config": config,
        "result": {
            "score": final_score,
            "reason": reason,
            "context_metrics": {
                "context_switches": context_metrics['context_switches'],
                "context_maintenance_score": round(context_metrics['context_maintenance'], 2),
                "total_key_points": len(set(context_metrics['referenced_info']))
            },
            "coherence_analysis": {
                "coherence_score": coherence_result['score'],
                "key_observations": coherence_result.get('key_observations', [])
            },
            "execution_stats": {
                "total_interactions": len(conversations),
                "average_info_reuse": round(
                    sum(context_metrics['information_reuse']) / 
                    len(context_metrics['information_reuse'])
                    if context_metrics['information_reuse'] else 0,
                    2
                )
            }
        }
    }