import json
import os
from importlib import resources
from dataclasses import asdict


def convert_usage_to_dict(usage):
    # Initialize the token_usage dictionary with default values
    token_usage = {
        "input": 0,
        "completion": 0,
        "reasoning": 0,  # Default reasoning tokens to 0 unless specified
    }

    if usage:
        if isinstance(usage, dict):
            # Access usage data as dictionary keys
            token_usage["input"] = usage.get("prompt_tokens", 0)
            token_usage["completion"] = usage.get("completion_tokens", 0)
            # If reasoning tokens are provided, adjust accordingly
            token_usage["reasoning"] = usage.get("reasoning_tokens", 0)
        else:
            # Handle the case where usage is not a dictionary
            # This could be an object with attributes, or something else
            try:
                token_usage["input"] = getattr(usage, "prompt_tokens", 0)
                token_usage["completion"] = getattr(usage, "completion_tokens", 0)
                token_usage["reasoning"] = getattr(usage, "reasoning_tokens", 0)
            except AttributeError:
                # If attributes are not found, log or handle the error as needed
                print(f"Warning: Unexpected usage type: {type(usage)}")

    return token_usage


def calculate_cost(
    token_usage,
    input_cost_per_token=0.0,
    output_cost_per_token=0.0,
    reasoning_cost_per_token=0.0,
):
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    reasoning_tokens = token_usage.get("reasoning_tokens", 0)

    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token
    reasoning_cost = reasoning_tokens * reasoning_cost_per_token

    total_cost = input_cost + output_cost + reasoning_cost

    return {
        "input": input_cost,
        "completion": output_cost,
        "reasoning": reasoning_cost,
        "total": total_cost,
    }


def load_model_costs():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_costs_path = os.path.join(current_dir, "model_costs.json")
        with open(model_costs_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        with resources.open_text("utils", "model_costs.json") as file:
            return json.load(file)


def log_event(event_data, log_file_path):
    event_data = asdict(event_data)
    with open(log_file_path, "a") as f:
        f.write(json.dumps(event_data) + "\n")
