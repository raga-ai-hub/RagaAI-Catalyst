import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import litellm

# Default evaluation rubric
DEFAULT_RUBRIC = {
    "criteria": {
        "problem_solving": {
            "description": "How effectively does the AI solve the user's problem?",
            "weight": 0.4
        },
        "clarity": {
            "description": "How clear and understandable are the AI's responses?",
            "weight": 0.3
        },
        "efficiency": {
            "description": "How efficient is the conversation in reaching the goal?",
            "weight": 0.3
        }
    },
    "scoring_guidelines": {
        "0.0-0.2": "Poor performance, significant improvements needed",
        "0.2-0.4": "Below average, notable issues present",
        "0.4-0.6": "Average performance, some improvements possible",
        "0.6-0.8": "Above average, minor improvements possible",
        "0.8-1.0": "Excellent performance, meets or exceeds expectations"
    }
}

def get_model_response(prompt: str, config: Dict[str, Any], response_format: Optional[Dict] = None) -> Any:
    """Get response from the language model."""
    try:
        kwargs = {
            "model": config.get("model", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
            
        evaluation = litellm.completion(**kwargs)
        return evaluation.choices[0].message.content
        
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None

def extract_evaluation_criteria(user_prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Extract evaluation criteria from a user prompt and format it according to the required structure.
    
    Args:
        user_prompt: The user's input describing their evaluation criteria
        model: The LLM model to use for extraction
        
    Returns:
        Dictionary containing the structured evaluation criteria
    """
    
    extraction_prompt = f"""
    Given the following user prompt describing evaluation criteria, extract and structure the information into a standardized evaluation rubric.

    User Prompt:
    {user_prompt}

    Please extract:
    1. Evaluation criteria (including descriptions and weights if specified)
    2. Scoring guidelines if provided

    Rules for processing:
    - If weights are not specified for criteria, assign equal weights that sum to 1
    - If scoring guidelines are not provided, use the default guidelines
    - Ensure all weights sum to 1 (rounded to 2 decimal places)
    - Create clear, concise descriptions for each criterion

    Return the result in the following JSON format:
    {{
        "criteria": {{
            "criterion_name": {{
                "description": "criterion description",
                "weight": numeric_weight
            }},
            ...
        }},
        "scoring_guidelines": {{
            "range": "description",
            ...
        }}
    }}

    For scoring guidelines, if not specified in the prompt, use these defaults:
    {json.dumps(DEFAULT_RUBRIC["scoring_guidelines"], indent=2)}

    Ensure the response:
    1. Is valid JSON
    2. Has weights that sum to 1
    3. Includes clear descriptions for each criterion
    4. Uses the default scoring guidelines if none are provided
"""

    try:
        # Get LLM response
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        try:
            extracted_content = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "metric_name": "custom_evaluation",
                "result": {
                    "error": "Failed to parse model response as JSON",
                    "raw_response": response
                }
            }

        # Validate and process criteria weights
        criteria = extracted_content.get("criteria", {})
        if criteria:
            # If weights not provided or invalid, assign equal weights
            weights = [c.get("weight", 0) for c in criteria.values()]
            if not weights or abs(sum(weights) - 1) > 0.01:
                equal_weight = round(1.0 / len(criteria), 2)
                for criterion in criteria.values():
                    criterion["weight"] = equal_weight
        
        # Use default scoring guidelines if none provided
        if "scoring_guidelines" not in extracted_content:
            extracted_content["scoring_guidelines"] = DEFAULT_RUBRIC["scoring_guidelines"]
        
        return {
            "criteria": criteria,
            "scoring_guidelines": extracted_content["scoring_guidelines"]
        }, True
        
    except Exception as e:
        print(f"Error extracting criteria: {str(e)}")
        return DEFAULT_RUBRIC, False

def validate_rubric(rubric: Dict[str, Any]) -> bool:
    """
    Validate the structure and content of a evaluation rubric.
    
    Args:
        rubric: Dictionary containing the rubric definition
        
    Returns:
        bool: True if rubric is valid, False otherwise
    """
    try:
        # Check for required top-level keys
        if not all(key in rubric for key in ['criteria', 'scoring_guidelines']):
            return False
            
        # Validate criteria
        criteria = rubric['criteria']
        if not isinstance(criteria, dict):
            return False
            
        # Check each criterion
        total_weight = 0
        for criterion in criteria.values():
            if not all(key in criterion for key in ['description', 'weight']):
                return False
            if not isinstance(criterion['weight'], (int, float)):
                return False
            if not 0 <= criterion['weight'] <= 1:
                return False
            total_weight += criterion['weight']
            
        # Validate total weights sum to approximately 1
        if not 0.99 <= total_weight <= 1.01:
            return False

        # Validate scoring guidelines
        scoring_guidelines = rubric['scoring_guidelines']
        if not isinstance(scoring_guidelines, dict) or not scoring_guidelines:
            return False

        # Validate scoring guideline ranges
        for range_str in scoring_guidelines.keys():
            try:
                # Check if range string follows the format "X.X-X.X"
                if not isinstance(range_str, str):
                    return False
                
                lower, upper = map(float, range_str.split('-'))
                # Validate range values are between 0 and 1
                if not (0 <= lower <= 1 and 0 <= upper <= 1 and lower < upper):
                    return False
                    
                # Validate range description
                if not isinstance(scoring_guidelines[range_str], str):
                    return False
            except (ValueError, AttributeError):
                return False
               
        return True
        
    except Exception:
        return False


def create_evaluation_prompt(criteria: Dict[str, Any], conversation: List[Dict], context: Optional[Dict] = None) -> str:
    """Create evaluation prompt based on user criteria."""
    prompt = f"""
    You are an expert evaluator analyzing a conversation between users and an AI system.
    
    Evaluation Criteria:
    {json.dumps(criteria, indent=2)}
    
    Conversation to Evaluate:
    {json.dumps(conversation, indent=2)}
    """
    
    if context:
        prompt += f"\nAdditional Context:\n{json.dumps(context, indent=2)}"
        
    prompt += """
    
    Please evaluate the conversation based on the provided criteria.
    
    Provide your evaluation in the following JSON format:
    {
        "score": <float between 0 and 1>,
        "scores": {
            "criterion_1": <float between 0 and 1>,
            "criterion_2": <float between 0 and 1>,
            ...
        },
        "reason": "<detailed reason of scores>",
        "recommendations": ["<recommendation_1>", "<recommendation_2>", ...]
    }
    
    Ensure that:
    1. The overall score is between 0 and 1
    2. Each criterion score is between 0 and 1
    3. The explanation justifies both the overall score and individual criterion scores
    4. Recommendations are actionable and specific to improving performance
    """
    
    return prompt

def parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO format timestamp."""
    return datetime.fromisoformat(timestamp)

def extract_conversations(trace_json: Dict[str, Any]) -> List[Dict]:
    """Extract conversations from trace JSON."""
    llm_calls = sorted(trace_json['llm_calls'], 
                     key=lambda x: parse_timestamp(x['start_time']))
    tool_calls = sorted(trace_json['tool_calls'], 
                      key=lambda x: parse_timestamp(x['start_time']))

    all_calls = sorted(llm_calls + tool_calls,
                     key=lambda x: parse_timestamp(x['start_time']))

    conversations = []
    for call in all_calls:
        input_content = (call['input_prompt'][0]['content'] 
                       if 'input_prompt' in call 
                       else call['input_parameters'])
        
        conversation_entry = {
            "function_name": call.get('name', 'Unknown'),
            "input": input_content,
            "response": call.get('output', 'No response')
        }
        conversations.append(conversation_entry)

    return conversations

def execute_custom_evaluation_metric(
    trace_json: Dict[str, Any], 
    config: Dict[str, Any],
    custom_criteria: Optional[Union[Dict[str, Any], str]] = None,
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute metric evaluation based on either custom or default criteria.
    
    Args:
        trace_json: The trace JSON to evaluate
        config: Configuration dictionary containing model settings
        custom_criteria: Optional user-defined evaluation criteria
        context: Optional additional context for evaluation
    
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Validate and determine criteria
        if custom_criteria and validate_rubric(custom_criteria):
            criteria = custom_criteria
            used_criteria = "custom"
        elif custom_criteria:
            criteria, custom_bool = extract_evaluation_criteria(custom_criteria)
            used_criteria = "custom" if custom_bool else "default"
        else:
            criteria = DEFAULT_RUBRIC
            used_criteria = "default"

        # Extract conversations
        conversations = extract_conversations(trace_json)
        
        # Create evaluation prompt
        prompt = create_evaluation_prompt(criteria, conversations, context)
        
        # Get model evaluation
        response = get_model_response(
            prompt, 
            config,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            return {
                "metric_name": "custom_evaluation",
                "config": config,
                "result": {
                    "error": "Failed to parse model response as JSON",
                    "raw_response": response
                }
            }
        
        return {
            "metric_name": "custom_evaluation",
            "config": config,
            "result": {
                "criteria": criteria,
                "used_criteria": used_criteria,
                "scores": result.get("scores", {}),
                "score": result.get("score", 0.0),
                "reason": result.get("reason", ""),
                "recommendations": result.get("recommendations", [])
            }
        }
        
    except Exception as e:
        return {
            "metric_name": "custom_evaluation",
            "config": config,
            "result": {
                "error": f"Evaluation failed: {str(e)}",
                "criteria": DEFAULT_RUBRIC  # Default rubric as fallback
            }
        }


# # Example usage
# config = {
#     "model": "gpt-4o-mini"
# }

# # Sample custom rubric
# custom_criteria = {
#     "criteria": {
#         "technical_accuracy": {
#             "description": "How technically accurate are the AI's responses?",
#             "weight": 0.4
#         },
#         "code_quality": {
#             "description": "How well-structured and maintainable is the suggested code?",
#             "weight": 0.3
#         },
#         "security_awareness": {
#             "description": "How well does the AI address security considerations?",
#             "weight": 0.3
#         }
#     },
#     "scoring_guidelines": {
#         "0.0-0.2": "Critical issues present",
#         "0.2-0.4": "Major improvements needed",
#         "0.4-0.6": "Acceptable with improvements needed",
#         "0.6-0.8": "Good quality with minor issues",
#         "0.8-1.0": "Excellent technical implementation"
#     }
# }

# json_path = "path/to/your/trace.json"
# context = {
#     "user_expertise": "beginner",
#     "conversation_goal": "technical support",
#     "domain": "software development"
# }
# with open(json_path) as f:
#     trace_json = json.load(f)
    
# result_custom = execute_custom_evaluation_metric(
#     trace_json, 
#     config, 
#     custom_criteria=custom_rubric, 
#     context=context
# )
# print("Custom Rubric Evaluation:")
# print(json.dumps(result_custom, indent=2))

# # Example 2: Using default rubric
# result_default = execute_custom_evaluation_metric(
#     trace_json, 
#     config, 
#     custom_criteria=None, 
#     context=context
# )
# print("\nDefault Rubric Evaluation:")
# print(json.dumps(result_default, indent=2))