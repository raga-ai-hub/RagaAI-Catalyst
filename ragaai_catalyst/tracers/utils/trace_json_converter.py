import json
import sys
from datetime import datetime
from typing import final, List, Dict, Any, Optional
import pytz
import uuid
from ragaai_catalyst.tracers.agentic_tracing.utils.llm_utils import calculate_llm_cost, get_model_cost

def convert_time_format(original_time_str, target_timezone_str="Asia/Kolkata"):
    """
    Converts a UTC time string to a specified timezone format.

    Args:
        original_time_str (str): The original time string in UTC format (e.g., "2025-02-28T22:05:57.945146Z").
        target_timezone_str (str): The target timezone to convert the time to (default is "Asia/Kolkata").

    Returns:
        str: The converted time string in the specified timezone format.
    """
    # Parse the original time string into a datetime object
    utc_time = datetime.strptime(original_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Set the timezone to UTC
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    # Convert the UTC time to the target timezone
    target_timezone = pytz.timezone(target_timezone_str)
    target_time = utc_time.astimezone(target_timezone)
    # Format the datetime object to the desired string format
    formatted_time = target_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    # Add a colon in the timezone offset for better readability
    formatted_time = formatted_time[:-2] + ':' + formatted_time[-2:]
    return formatted_time


def get_uuid(name):
    """Generate a random UUID (not based on name)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def get_ordered_family(parent_children_mapping: Dict[str, Any]) -> List[str]:
    def ordering_function(parent_id: str, ordered_family: List[str]):
        children = parent_children_mapping.get(parent_id, [])
        parent_child_ids =[child['id'] for child in children if child['id'] in parent_children_mapping]
        for child_id in parent_child_ids:
            if child_id not in ordered_family:
                ordered_family.append(child_id)
                ordering_function(child_id, ordered_family)
    ordered_family = [None]
    ordering_function(None, ordered_family)
    return reversed(ordered_family)

def get_spans(input_trace, custom_model_cost):
    span_map = {}
    parent_children_mapping = {}
    span_type_mapping={"AGENT":"agent","LLM":"llm","TOOL":"tool"}
    span_name_occurrence = {}
    for span in input_trace:
        final_span = {}
        span_type=span_type_mapping.get(span["attributes"]["openinference.span.kind"],"custom")
        span_id = span["context"]["span_id"]
        parent_id = span["parent_id"]
        final_span["id"] = span_id
        if span["name"] not in span_name_occurrence:
            span_name_occurrence[span['name']]=0
        else:
            span_name_occurrence[span['name']]+=1
        final_span["name"] = span["name"]+"."+str(span_name_occurrence[span['name']])
        final_span["hash_id"] = get_uuid(final_span["name"])
        final_span["source_hash_id"] = None
        final_span["type"] = span_type
        final_span["start_time"] = convert_time_format(span['start_time'])
        final_span["end_time"] = convert_time_format(span['end_time'])
        final_span["parent_id"] = parent_id
        final_span["extra_info"] = None
        '''Handle Error if any'''
        if span["status"]["status_code"].lower() == "error":
            final_span["error"] = span["status"]
        else:
            final_span["error"] = None
        # ToDo: Find final trace format for sending error description
        final_span["metrics"] = []
        final_span["feedback"] = None
        final_span["data"]={}
        final_span["info"]={}
        final_span["metrics"] =[]
        final_span["extra_info"]={}
        if span_type=="agent":
            if "input.value" in span["attributes"]:
                try:
                    final_span["data"]["input"] = json.loads(span["attributes"]["input.value"])
                except Exception as e:
                    final_span["data"]["input"] = span["attributes"]["input.value"]
            else:
                final_span["data"]["input"] = ""
            if "output.value" in span["attributes"]:
                try:
                    final_span["data"]["output"] = json.loads(span["attributes"]["output.value"])
                except Exception as e:
                    final_span["data"]["output"] = span["attributes"]["output.value"]
            else:
                final_span["data"]["output"] = ""
            final_span["data"]['children'] = []
        
        elif span_type=="tool":
            available_fields = list(span['attributes'].keys())
            tool_fields = [key for key in available_fields if 'tool' in key]
            if "input.value" in span["attributes"]:
                try:
                    final_span["data"]["input"] = json.loads(span["attributes"]["input.value"])
                except Exception as e:
                    final_span["data"]["input"] = span["attributes"]["input.value"]
            else:
                final_span["data"]["input"] = ""
            if "output.value" in span["attributes"]:
                try:
                    final_span["data"]["output"] = json.loads(span["attributes"]["output.value"])
                except Exception as e:
                    final_span["data"]["output"] = span["attributes"]["output.value"]
            else:
                final_span["data"]["output"] = ""
            input_data={}
            for key in tool_fields:
                input_data[key] = span['attributes'].get(key, None)
            final_span["info"].update(input_data)

        elif span_type=="llm":
            available_fields = list(span['attributes'].keys())
            input_fields = [key for key in available_fields if 'input' in key]
            input_data = {}
            for key in input_fields:
                if 'mime_type' not in key:
                    try:
                        input_data[key] = json.loads(span['attributes'][key])
                    except json.JSONDecodeError as e:
                        input_data[key] = span['attributes'].get(key, None)
            final_span["data"]["input"] = input_data
            
            output_fields = [key for key in available_fields if 'output' in key]
            output_data = {}
            output_data['content'] = {}
            for key in output_fields:
                if 'mime_type' not in key:
                    try:
                        output_data['content'][key] = json.loads(span['attributes'][key])
                    except json.JSONDecodeError as e:
                        output_data['content'][key] = span['attributes'].get(key, None)
            final_span["data"]["output"] = [output_data]

            if "llm.model_name" in span["attributes"]:
                final_span["info"]["model"] = span["attributes"]["llm.model_name"]
            else:
                final_span["info"]["model"] = None
            if "llm.invocation_parameters" in span["attributes"]:
                try:
                    final_span["info"].update(**json.loads(span["attributes"]["llm.invocation_parameters"]))
                except json.JSONDecodeError as e:
                    print(f"Error in parsing: {e}")
                    
                try:
                    final_span["extra_info"]["llm_parameters"] = json.loads(span["attributes"]["llm.invocation_parameters"])
                except json.JSONDecodeError as e:
                    final_span["extra_info"]["llm_parameters"] = span["attributes"]["llm.invocation_parameters"]
            else:
                final_span["extra_info"]["llm_parameters"] = None

        else:
            if "input.value" in span["attributes"]:
                try:
                    final_span["data"]["input"] = json.loads(span["attributes"]["input.value"])
                except Exception as e:
                    final_span["data"]["input"] = span["attributes"]["input.value"]
            if "output.value" in span["attributes"]:
                try:
                    final_span["data"]["output"] = json.loads(span["attributes"]["output.value"])
                except Exception as e:
                    final_span["data"]["output"] = span["attributes"]["output.value"]

        final_span["info"]["cost"] = {}
        final_span["info"]["tokens"] = {}

        if "model" in final_span["info"]:
            model_name = final_span["info"]["model"] 
        
        model_costs = {
                "default": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
            }
        try:
            model_costs = get_model_cost()
        except Exception as e:
           pass 
       
        if "resource" in span:
            final_span["info"].update(span["resource"])
        if "llm.token_count.prompt" in span['attributes']:
            final_span["info"]["tokens"]["prompt_tokens"] = span['attributes']['llm.token_count.prompt']
        if "llm.token_count.completion" in span['attributes']:
            final_span["info"]["tokens"]["completion_tokens"] = span['attributes']['llm.token_count.completion']
        if "llm.token_count.total" in span['attributes']:
            final_span["info"]["tokens"]["total_tokens"] = span['attributes']['llm.token_count.total']
        
        if "info" in final_span:
            if "tokens" in final_span["info"]:
                if "prompt_tokens" in final_span["info"]["tokens"]:
                    token_usage = {
                        "prompt_tokens": final_span["info"]["tokens"]["prompt_tokens"],
                        "completion_tokens": final_span["info"]["tokens"]["completion_tokens"],
                        "total_tokens": final_span["info"]["tokens"]["total_tokens"]
                    }
                    final_span["info"]["cost"] = calculate_llm_cost(token_usage=token_usage, model_name=model_name, model_costs=model_costs, model_custom_cost=custom_model_cost) 
        span_map[span_id] = final_span
        if parent_id not in parent_children_mapping:
            parent_children_mapping[parent_id] = []
        parent_children_mapping[parent_id].append(final_span)
    ordered_family = get_ordered_family(parent_children_mapping)
    data = []
    for parent_id in ordered_family:
        children = parent_children_mapping[parent_id]
        if parent_id in span_map:
            parent_type = span_map[parent_id]["type"]
            if parent_type == 'agent':
                span_map[parent_id]['data']["children"] = children
            else:
                grand_parent_id = span_map[parent_id]["parent_id"]
                parent_children_mapping[grand_parent_id].extend(children)
        else:
            data = children
    return data

def convert_json_format(input_trace, custom_model_cost):
    """
    Converts a JSON from one format to UI format.

    Args:
        input_trace (str): The input JSON string.

    Returns:
        final_trace: The converted JSON, or None if an error occurs.
    """
    final_trace = {
        "id": input_trace[0]["context"]["trace_id"],
        "trace_name": "",  
        "project_name": "",  
        "start_time": convert_time_format(min(item["start_time"] for item in input_trace)),  # Find the minimum start_time of all spans
        "end_time": convert_time_format(max(item["end_time"] for item in input_trace))  # Find the maximum end_time of all spans
    }
    final_trace["metadata"] = {
        "tokens": {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0
        },
        "cost": {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0
        }    
    }
    final_trace["replays"]={"source":None}
    final_trace["data"]=[{}]
    try:
        final_trace["data"][0]["spans"] = get_spans(input_trace, custom_model_cost)
    except Exception as e:
        raise Exception(f"Error in get_spans function: {e}")
    final_trace["network_calls"] =[]
    final_trace["interactions"] = []

    for itr in final_trace["data"][0]["spans"]:
        if itr["type"]=="llm":
            if "tokens" in itr["info"]:
                if "prompt_tokens" in itr["info"]["tokens"]:
                    final_trace["metadata"]["tokens"]["prompt_tokens"] += itr["info"]["tokens"].get('prompt_tokens', 0.0)
                    final_trace["metadata"]["cost"]["input_cost"] += itr["info"]["cost"].get('input_cost', 0.0) 
                if "completion_tokens" in itr["info"]["tokens"]:
                    final_trace["metadata"]["tokens"]["completion_tokens"] += itr["info"]["tokens"].get('completion_tokens', 0.0)
                    final_trace["metadata"]["cost"]["output_cost"] += itr["info"]["cost"].get('output_cost', 0.0) 
            if "tokens" in itr["info"]:
                if "total_tokens" in itr["info"]["tokens"]:
                    final_trace["metadata"]["tokens"]["total_tokens"] += itr["info"]["tokens"].get('total_tokens', 0.0)
                    final_trace["metadata"]["cost"]["total_cost"] += itr["info"]["cost"].get('total_cost', 0.0) 

    # get the total tokens, cost
    final_trace["metadata"]["total_cost"] = final_trace["metadata"]["cost"]["total_cost"] 
    final_trace["metadata"]["total_tokens"] = final_trace["metadata"]["tokens"]["total_tokens"]

    return final_trace
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_openinference_trace_path> <output_trace_path>")
        print("Example: python convert.py sample_openinference_trace/test.json output.json")
        sys.exit(1)
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    with open(input_file_path,'r') as fin:
        input_trace=[]
        for line in fin:
            data=json.loads(line)
            input_trace.append(data)
        payload = convert_json_format(input_trace)
        print(payload)
        with open(output_file_path,"w") as fout:
            json.dump(payload,fout)
            fout.write("\n")
