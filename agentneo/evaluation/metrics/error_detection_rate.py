import json
from typing import Dict, Any, List
import litellm


def execute_error_detection_rate_metric(trace_json: Dict[str, Any], config: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Main function to calculate the Error Detection Rate metric.
    """
    # Step 1: Extract errors from trace
    logged_errors = extract_logged_errors(trace_json)
    tool_call_errors = detect_tool_call_errors(trace_json)

    total_logged_errors = len(logged_errors)
    total_detected_errors = len(tool_call_errors)

    # Step 2: Calculate error detection rate
    error_detection_rate = calculate_detection_rate(total_logged_errors, total_detected_errors)

    # Step 3: Generate a detailed explanation
    explanation = generate_error_detection_reason(
        logged_errors, tool_call_errors, total_logged_errors, total_detected_errors, error_detection_rate, config
    )

    return {
        "metric_name": "error_detection_rate",
        "config": config,
        "result": {
            "score": error_detection_rate,
            "reason": explanation,
            "details": {
                "total_logged_errors": total_logged_errors,
                "total_detected_errors": total_detected_errors,
                "detected_errors": tool_call_errors,
            },
        },
    }


def extract_logged_errors(trace_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract logged errors from the trace JSON.
    """
    return trace_json.get("errors", [])


def detect_tool_call_errors(trace_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect errors in tool calls based on missing or invalid outputs.
    """
    tool_calls = trace_json.get("tool_calls", [])
    detected_errors = []

    for call in tool_calls:
        output = call.get("output")
        if is_tool_call_error(output):
            detected_errors.append({
                "tool_name": call.get("name", "Unknown"),
                "input": call.get("input_parameters", {}),
                "reason": "Invalid or missing output",
                "timestamp": call.get("end_time", "Unknown"),
            })

    return detected_errors


def is_tool_call_error(output: Any) -> bool:
    """
    Check if a tool call output indicates an error.
    """
    return not output or output == "error"


def calculate_detection_rate(total_logged_errors: int, total_detected_errors: int) -> float:
    """
    Calculate the error detection rate.
    """
    total_errors = total_logged_errors + total_detected_errors
    return total_detected_errors / total_errors if total_errors > 0 else 1.0


def generate_error_detection_reason(
    logged_errors: List[Dict[str, Any]],
    detected_errors: List[Dict[str, Any]],
    total_logged_errors: int,
    total_detected_errors: int,
    detection_rate: float,
    config: Dict[str, Any],
) -> str:
    """
    Generate an explanation for the error detection rate.
    """
    prompt = f"""
    Analyze the error detection rate for the given interaction trace.

    Total errors logged in the system: {total_logged_errors}
    Total errors detected in tool calls: {total_detected_errors}
    Overall error detection rate: {detection_rate:.2f}

    Logged errors:
    {json.dumps(logged_errors, indent=2)}

    Detected tool call errors:
    {json.dumps(detected_errors, indent=2)}

    Provide a concise explanation of the error detection rate and suggest areas for improvement.
    """

    response = litellm.completion(
        model=config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()
