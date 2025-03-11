from ..data.data_structure import LLMCall
from .trace_utils import (
    calculate_cost,
    convert_usage_to_dict,
)
from importlib import resources
#from litellm import model_cost
import json
import os
import asyncio
import psutil
import tiktoken
import logging

logger = logging.getLogger(__name__)

def get_model_cost():
    """Load model costs from a JSON file. 
    Note: This file should be updated periodically or whenever a new package is created to ensure accurate cost calculations.
    To Do: Implement to do this automatically.
    """
    file="model_prices_and_context_window_backup.json"
    d={}
    with resources.open_text("ragaai_catalyst.tracers.utils", file) as f:
        d= json.load(f)
    return d 

model_cost = get_model_cost()

def extract_model_name(args, kwargs, result):
    """Extract model name from kwargs or result"""
    # First try direct model parameter
    model = kwargs.get("model", "")
    
    if not model:
        # Try to get from instance
        instance = kwargs.get("self", None)
        if instance:
            # Try model_name first (Google format)
            if hasattr(instance, "model_name"):
                model = instance.model_name
            # Try model attribute
            elif hasattr(instance, "model"):
                model = instance.model

    # Handle vertex ai case
    if not model:
        manager = kwargs.get("run_manager", None)
        if manager:
            if hasattr(manager, 'metadata'):
                metadata = manager.metadata
                model_name = metadata.get('ls_model_name', None)
                if model_name:
                    model = model_name       
                    
    if not model:
        if 'to_dict' in dir(result):
            result = result.to_dict()
            if 'model_version' in result:
                model = result['model_version']  
    try:
        if not model:
            model = result.raw.model
    except Exception as e:
        pass
    
    
    # Normalize Google model names
    if model and isinstance(model, str):
        model = model.lower()
        if "gemini-1.5-flash" in model:
            return "gemini-1.5-flash"
        if "gemini-1.5-pro" in model:
            return "gemini-1.5-pro"
        if "gemini-pro" in model:
            return "gemini-pro"

    if 'response_metadata' in dir(result):
        if 'model_name' in result.response_metadata:
            model = result.response_metadata['model_name']
    
    return model or "default"


def extract_parameters(kwargs):
    """Extract all non-null parameters from kwargs"""
    parameters = {k: v for k, v in kwargs.items() if v is not None}

    # Remove contents key in parameters (Google LLM Response)
    if 'contents' in parameters:
        del parameters['contents']

    # Remove messages key in parameters (OpenAI message)
    if 'messages' in parameters:
        del parameters['messages']
        
    if 'run_manager' in parameters:
        del parameters['run_manager']

    if 'generation_config' in parameters:
        generation_config = parameters['generation_config']
        # If generation_config is already a dict, use it directly
        if isinstance(generation_config, dict):
            config_dict = generation_config
        else:
            # Convert GenerationConfig to dictionary if it has a to_dict method, otherwise try to get its __dict__
            config_dict = getattr(generation_config, 'to_dict', lambda: generation_config.__dict__)()
        parameters.update(config_dict)
        del parameters['generation_config']
        
    return parameters


def extract_token_usage(result):
    """Extract token usage from result"""
    # Handle coroutines
    if asyncio.iscoroutine(result):
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Run the coroutine in the current event loop
        result = loop.run_until_complete(result)

    # Handle text attribute responses (JSON string for Vertex AI)
    if hasattr(result, "text") and isinstance(result.text, (str, bytes, bytearray)):
        # First try parsing as JSON for OpenAI responses
        try:
            import json
            json_data = json.loads(result.text)
            if isinstance(json_data, dict) and "usage" in json_data:
                usage = json_data["usage"]
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        # If JSON parsing fails, try Vertex AI format
        total_tokens = getattr(result, "token_count", 0)
        if not total_tokens and hasattr(result, "_raw_response"):
            total_tokens = getattr(result._raw_response, "token_count", 0)
        if total_tokens:  # Only return if we actually found tokens
            return {
                "prompt_tokens": 0,  # Vertex AI doesn't provide this breakdown
                "completion_tokens": total_tokens,
                "total_tokens": total_tokens
            }

    # Handle Claude 3 message format
    if hasattr(result, "usage"):
        usage = result.usage
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens
            }
        # Handle standard OpenAI/Anthropic format
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }
    
    # Handle Google GenerativeAI format with usage_metadata
    if hasattr(result, "usage_metadata"):
        metadata = result.usage_metadata
        if hasattr(metadata, "prompt_token_count"):
            return {
                "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0)
            }
        elif hasattr(metadata, "input_tokens"):
            return {
                "prompt_tokens": getattr(metadata, "input_tokens", 0),
                "completion_tokens": getattr(metadata, "output_tokens", 0),
                "total_tokens": getattr(metadata, "total_tokens", 0)
            }
        elif "input_tokens" in metadata:
            return {
                "prompt_tokens": metadata["input_tokens"],
                "completion_tokens": metadata["output_tokens"],
                "total_tokens": metadata["total_tokens"]
            }


    
    # Handle ChatResponse format with raw usuage
    if hasattr(result, "raw") and hasattr(result.raw, "usage"):
        usage = result.raw.usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }
    
    # Handle ChatResult format with generations
    if hasattr(result, "generations") and result.generations:
        # Get the first generation
        generation = result.generations[0]
        
        # Try to get usage from generation_info
        if hasattr(generation, "generation_info"):
            metadata = generation.generation_info.get("usage_metadata", {})
            if metadata:
                return {
                    "prompt_tokens": metadata.get("prompt_token_count", 0),
                    "completion_tokens": metadata.get("candidates_token_count", 0),
                    "total_tokens": metadata.get("total_token_count", 0)
                }
        
        # Try to get usage from message's usage_metadata
        if hasattr(generation, "message") and hasattr(generation.message, "usage_metadata"):
            metadata = generation.message.usage_metadata
            return {
                "prompt_tokens": metadata.get("input_tokens", 0),
                "completion_tokens": metadata.get("output_tokens", 0),
                "total_tokens": metadata.get("total_tokens", 0)
            }
    
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

def num_tokens_from_messages(model="gpt-4o-mini-2024-07-18", prompt_messages=None, response_message=None):
    """Calculate the number of tokens used by messages.
    
    Args:
        messages: Optional list of messages (deprecated, use prompt_messages and response_message instead)
        model: The model name to use for token calculation
        prompt_messages: List of prompt messages
        response_message: Response message from the assistant
    
    Returns:
        dict: A dictionary containing:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total number of tokens
    """
    #import pdb; pdb.set_trace()
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        logging.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(model="gpt-3.5-turbo-0125", 
                                     prompt_messages=prompt_messages, response_message=response_message)
    elif "gpt-4o-mini" in model:
        logging.warning("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(model="gpt-4o-mini-2024-07-18",
                                     prompt_messages=prompt_messages, response_message=response_message)
    elif "gpt-4o" in model:
        logging.warning("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(model="gpt-4o-2024-08-06",
                                     prompt_messages=prompt_messages, response_message=response_message)
    elif "gpt-4" in model:
        logging.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(model="gpt-4-0613",
                                     prompt_messages=prompt_messages, response_message=response_message)
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    
    all_messages = []
    if prompt_messages:
        all_messages.extend(prompt_messages)
    if response_message:
        if isinstance(response_message, dict):
            all_messages.append(response_message)
        else:
            all_messages.append({"role": "assistant", "content": response_message})
    
    prompt_tokens = 0
    completion_tokens = 0
    
    for message in all_messages:
        num_tokens = tokens_per_message
        for key, value in message.items():
            token_count = len(encoding.encode(str(value)))  # Convert value to string for safety
            num_tokens += token_count
            if key == "name":
                num_tokens += tokens_per_name
        
        # Add tokens to prompt or completion based on role
        if message.get("role") == "assistant":
            completion_tokens += num_tokens
        else:
            prompt_tokens += num_tokens
    
    # Add the assistant message prefix tokens to completion tokens if we have a response
    if completion_tokens > 0:
        completion_tokens += 3  # <|start|>assistant<|message|>
    
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

def extract_input_data(args, kwargs, result):
    """Sanitize and format input data, including handling of nested lists and dictionaries."""

    def sanitize_value(value):
        if isinstance(value, (int, float, bool, str)):
            return value
        elif isinstance(value, list):
            return [sanitize_value(item) for item in value]
        elif isinstance(value, dict):
            return {key: sanitize_value(val) for key, val in value.items()}
        else:
            return str(value)  # Convert non-standard types to string

    return {
        "args": [sanitize_value(arg) for arg in args],
        "kwargs": {key: sanitize_value(val) for key, val in kwargs.items()},
    }


def calculate_llm_cost(token_usage, model_name, model_costs, model_custom_cost=None):
    """Calculate cost based on token usage and model"""
    if model_custom_cost is None:
        model_custom_cost = {}
    model_costs.update(model_custom_cost)
    if not isinstance(token_usage, dict):
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": token_usage if isinstance(token_usage, (int, float)) else 0
        }
    
    # Get model costs, defaulting to default costs if unknown
    model_cost = model_cost = model_costs.get(model_name, {
        "input_cost_per_token": 0.0,   
        "output_cost_per_token": 0.0   
    })
    if model_cost['input_cost_per_token'] == 0.0 and model_cost['output_cost_per_token'] == 0.0:
        provide_name = model_name.split('-')[0]
        if provide_name == 'azure':
            model_name = os.path.join('azure', '-'.join(model_name.split('-')[1:]))

            model_cost = model_costs.get(model_name, {
                "input_cost_per_token": 0.0,   
                "output_cost_per_token": 0.0   
            })

    input_cost = (token_usage.get("prompt_tokens", 0)) * model_cost.get("input_cost_per_token", 0.0)
    output_cost = (token_usage.get("completion_tokens", 0)) * model_cost.get("output_cost_per_token", 0.0)
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 10),
        "output_cost": round(output_cost, 10),
        "total_cost": round(total_cost, 10)
    }


def sanitize_api_keys(data):
    """Remove sensitive information from data"""
    if isinstance(data, dict):
        return {k: sanitize_api_keys(v) for k, v in data.items() 
                if not any(sensitive in k.lower() for sensitive in ['key', 'token', 'secret', 'password'])}
    elif isinstance(data, list):
        return [sanitize_api_keys(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_api_keys(item) for item in data)
    return data


def sanitize_input(args, kwargs):
    """Convert input arguments to text format.
    
    Args:
        args: Input arguments that may contain nested dictionaries
        
    Returns:
        str: Text representation of the input arguments
    """
    if isinstance(args, dict):
        return str({k: sanitize_input(v, {}) for k, v in args.items()})
    elif isinstance(args, (list, tuple)):
        return str([sanitize_input(item, {}) for item in args])
    return str(args)


def extract_llm_output(result):
    """Extract output from LLM response"""
    class OutputResponse:
        def __init__(self, output_response):
            self.output_response = output_response

    # Handle coroutines
    if asyncio.iscoroutine(result):
        # For sync context, run the coroutine
        if not asyncio.get_event_loop().is_running():
            result = asyncio.run(result)
        else:
            # We're in an async context, but this function is called synchronously
            # Return a placeholder and let the caller handle the coroutine
            return OutputResponse([{'content': "Coroutine result pending", "role": "assistant"}])

    # Handle Google GenerativeAI format
    if hasattr(result, "result"):
        candidates = getattr(result.result, "candidates", [])
        output = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    if hasattr(part, "text"):
                        output.append({
                            "content": part.text,
                            "role": getattr(content, "role", "assistant"),
                            "finish_reason": getattr(candidate, "finish_reason", None)
                        })
        return OutputResponse(output)
    
    # Handle AIMessage Format
    if hasattr(result, "content"):
        return OutputResponse([{
            "content": result.content,
            "role": getattr(result, "role", "assistant")
        }])
    
    # Handle Vertex AI format
    # format1
    if hasattr(result, "text"):
        return OutputResponse([{
            "content": result.text,
            "role": "assistant"
        }])


    # format2
    if hasattr(result, "generations"):
        output = []
        for generation in result.generations:
            output.append({
                "content": generation.text,
                "role": "assistant"
            })
        return OutputResponse(output)
    
    # Handle OpenAI format
    if hasattr(result, "choices"):
        return OutputResponse([{
            "content": choice.message.content,
            "role": choice.message.role
        } for choice in result.choices])


    # Handle Anthropic format
    if hasattr(result, "content"):
        return OutputResponse([{
            "content": result.content[0].text,
            "role": "assistant"
        }])
    
    # Default case
    return OutputResponse([{'content': result, 'role': 'assistant'}])


def extract_llm_data(args, kwargs, result):
    # Initialize variables
    model_name = None
    output_response = ""
    function_call = None
    tool_call = None
    token_usage = {}
    cost = {}

    # Try to get model_name from result or result.content
    model_name = extract_model_name(args, kwargs, result)

    # Try to get choices from result or result.content
    choices = None
    if hasattr(result, "choices"):
        choices = result.choices
    elif hasattr(result, "content"):
        try:
            content_dict = json.loads(result.content)
            choices = content_dict.get("choices", None)
        except (json.JSONDecodeError, TypeError):
            choices = None

    if choices and len(choices) > 0:
        first_choice = choices[0]

        # Get message or text
        message = None
        if hasattr(first_choice, "message"):
            message = first_choice.message
        elif isinstance(first_choice, dict) and "message" in first_choice:
            message = first_choice["message"]

        if message:
            # For chat completion
            # Get output_response
            if hasattr(message, "content"):
                output_response = message.content
            elif isinstance(message, dict) and "content" in message:
                output_response = message["content"]

            # Get function_call
            if hasattr(message, "function_call"):
                function_call = message.function_call
            elif isinstance(message, dict) and "function_call" in message:
                function_call = message["function_call"]

            # Get tool_calls (if any)
            if hasattr(message, "tool_calls"):
                tool_call = message.tool_calls
            elif isinstance(message, dict) and "tool_calls" in message:
                tool_call = message["tool_calls"]
        else:
            # For completion
            # Get output_response
            if hasattr(first_choice, "text"):
                output_response = first_choice.text
            elif isinstance(first_choice, dict) and "text" in first_choice:
                output_response = first_choice["text"]
            else:
                output_response = ""

            # No message, so no function_call or tool_call
            function_call = None
            tool_call = None
    else:
        output_response = ""
        function_call = None
        tool_call = None

    # Set tool_call to function_call if tool_call is None
    if not tool_call:
        tool_call = function_call

    # Parse tool_call
    parsed_tool_call = None
    if tool_call:
        if isinstance(tool_call, dict):
            arguments = tool_call.get("arguments", "{}")
            name = tool_call.get("name", "")
        else:
            # Maybe it's an object with attributes
            arguments = getattr(tool_call, "arguments", "{}")
            name = getattr(tool_call, "name", "")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            else:
                arguments = arguments  # If already a dict
        except json.JSONDecodeError:
            arguments = {}
        parsed_tool_call = {"arguments": arguments, "name": name}

    # Try to get token_usage from result.usage or result.content
    usage = None
    if hasattr(result, "usage"):
        usage = result.usage
    elif hasattr(result, "content"):
        try:
            content_dict = json.loads(result.content)
            usage = content_dict.get("usage", {})
        except (json.JSONDecodeError, TypeError):
            usage = {}
    else:
        usage = {}

    token_usage = extract_token_usage(result)

    # Load model costs
    model_costs = model_cost

    # Calculate cost
    cost = calculate_llm_cost(token_usage, model_name, model_costs)

    llm_data = LLMCall(
        name="",
        model_name=model_name,
        input_prompt="",  # Not available here
        output_response=output_response,
        token_usage=token_usage,
        cost=cost,
        tool_call=parsed_tool_call,
    )
    return llm_data
