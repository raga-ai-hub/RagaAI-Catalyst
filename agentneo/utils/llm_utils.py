from agentneo.data import LLMCall
from .trace_utils import (
    calculate_cost,
    convert_usage_to_dict,
    load_model_costs,
)
from importlib import resources
import json


# Load the Json configuration
try:
    with open("agentneo/configs/model_costs.json", "r") as file:
        config = json.load(file)
except FileNotFoundError:
    # with resources.open_text("agentneo", "configs/model_costs.json") as file:
    #     config = json.load(file)
    from importlib.resources import files
    with (files("agentneo.configs") / "model_costs.json").open("r") as file:
        config = json.load(file)



def extract_llm_output(result):

    # import pdb

    # pdb.set_trace()
    # Initialize variables
    model_name = None
    output_response = ""
    function_call = None
    tool_call = None
    token_usage = {}
    cost = {}

    # Try to get model_name from result or result.content
    model_name = None
    if hasattr(result, "model"):
        model_name = result.model
    elif hasattr(result, "content"):
        try:
            content_dict = json.loads(result.content)
            model_name = content_dict.get("model", None)
        except (json.JSONDecodeError, TypeError):
            model_name = None

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

    token_usage = convert_usage_to_dict(usage)

    # Load model costs
    model_costs = load_model_costs()

    # Calculate cost
    if model_name in model_costs:
        model_config = model_costs[model_name]
        input_cost_per_token = model_config.get("input_cost_per_token", 0.0)
        output_cost_per_token = model_config.get("output_cost_per_token", 0.0)
        reasoning_cost_per_token = model_config.get(
            "reasoning_cost_per_token", output_cost_per_token
        )
    else:
        # Default costs or log a warning
        print(
            f"Warning: Model '{model_name}' not found in config. Using default costs."
        )
        input_cost_per_token = 0.0
        output_cost_per_token = 0.0
        reasoning_cost_per_token = 0.0

    cost = calculate_cost(
        token_usage,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        reasoning_cost_per_token=reasoning_cost_per_token,
    )

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
