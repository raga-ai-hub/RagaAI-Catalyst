import json
import sys
from datetime import datetime
import pytz
import uuid

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

def get_spans(input_trace):
    data=[]
    span_type_mapping={"AGENT":"agent","LLM":"llm","TOOL":"tool"}
    span_name_occurrence = {}
    for span in input_trace:
        final_span = {}
        span_type=span_type_mapping.get(span["attributes"]["openinference.span.kind"],"custom")
        final_span["id"] = span["context"]["span_id"]
        if span["name"] not in span_name_occurrence:
            span_name_occurrence[span['name']]=0
        else:
            span_name_occurrence[span['name']]+=1
        final_span["name"] = span["name"]+"."+str(span_name_occurrence[span['name']])
        final_span["hash_id"] = get_uuid(span["name"])
        final_span["source_hash_id"] = None
        final_span["type"] = span_type
        final_span["start_time"] = convert_time_format(span['start_time'])
        final_span["end_time"] = convert_time_format(span['end_time'])
        final_span["parent_id"] = span["parent_id"]
        final_span["extra_info"] = None
        '''Handle Error if any'''
        if span["status"]["status_code"].lower() == "error":
            final_span["error"] = span["status"]["status_code"]
        else:
            final_span["error"] = None
        # ToDo: Find final trace format for sending error description
        final_span["metadata"] = {}
        final_span["metrics"] = []
        final_span["feedback"] = None
        final_span["network_calls"] =[]
        final_span["interactions"] = []
        final_span["data"]={}
        final_span["info"]={}
        final_span["replays"]={"source":None}
        final_span["metrics"] =[]
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

        elif span_type=="llm":
            try:
                span_input_value = json.loads(span["attributes"]["input.value"])
            except json.JSONDecodeError:
                span_input_value = span["attributes"]["input.value"]
            role = span["attributes"]["llm.input_messages.0.message.role"]
            content = span["attributes"]["llm.input_messages.0.message.content"]
            final_span["data"]["input"] = {"value":span_input_value,"role":role,"content":content}
            try:
                span_output_value = json.loads(span["attributes"]["output.value"])
            except json.JSONDecodeError:
                span_output_value = span["attributes"]["output.value"]
            role = span["attributes"]["llm.output_messages.0.message.role"]
            content = span["attributes"]["llm.output_messages.0.message.content"]
            final_span["data"]["output"] = {"value":span_output_value,"role":role,"content":content}
            final_span["info"]["model_name"] = span["attributes"]["llm.model_name"]
            final_span["info"]["llm_parameters"] = span["attributes"]["llm.invocation_parameters"]
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
        if "resource" in span:
            final_span["info"].update(span["resource"])
        #ToDo: Add tool span specific information
        #ToDo: Add prompt usage information
        #ToDo: Add available Trace metadata information 
        #ToDo Code for Workflow goes here
        final_span["workflow"]=[]
        data.append(final_span)
    return data

def convert_json_format(input_trace):
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
    final_trace["metadata"] ={}
    final_trace["data"]=[{}]
    final_trace["data"][0]["spans"] = get_spans(input_trace)
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