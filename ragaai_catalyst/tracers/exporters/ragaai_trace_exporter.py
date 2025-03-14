import os
import json
import tempfile
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
import logging
from dataclasses import asdict
from ragaai_catalyst.tracers.utils.trace_json_converter import convert_json_format
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import TracerJSONEncoder
from ragaai_catalyst.tracers.agentic_tracing.utils.system_monitor import SystemMonitor
from ragaai_catalyst.tracers.agentic_tracing.upload.trace_uploader import submit_upload_task
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import zip_list_of_unique_files


logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class RAGATraceExporter(SpanExporter):
    def __init__(self, files_to_zip, project_name, project_id, dataset_name, user_details, base_url, custom_model_cost):
        self.trace_spans = dict()
        self.tmp_dir = tempfile.gettempdir()
        self.files_to_zip = files_to_zip
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_details = user_details
        self.base_url = base_url
        self.custom_model_cost = custom_model_cost
        self.system_monitor = SystemMonitor(dataset_name)

    def export(self, spans):
        for span in spans:
            span_json = json.loads(span.to_json())
            trace_id = span_json.get("context").get("trace_id")
            if trace_id is None:
                raise Exception("Trace ID is None")

            if trace_id not in self.trace_spans:
                self.trace_spans[trace_id] = list()

            self.trace_spans[trace_id].append(span_json)

            if span_json["parent_id"] is None:
                trace = self.trace_spans[trace_id]
                try:
                    self.process_complete_trace(trace, trace_id)
                except Exception as e:
                    raise Exception(f"Error processing complete trace: {e}")
                try:
                    del self.trace_spans[trace_id]
                except Exception as e:
                    raise Exception(f"Error deleting trace: {e}")

        return SpanExportResult.SUCCESS

    def shutdown(self):
        # Process any remaining traces during shutdown
        for trace_id, spans in self.trace_spans.items():
            self.process_complete_trace(spans, trace_id)
        self.trace_spans.clear()

    def process_complete_trace(self, spans, trace_id):
        # Convert the trace to ragaai trace format
        try:
            ragaai_trace_details = self.prepare_trace(spans, trace_id)
        except Exception as e:
            print(f"Error converting trace {trace_id}: {e}")
        
        # Upload the trace if upload_trace function is provided
        try:
            self.upload_trace(ragaai_trace_details, trace_id)
        except Exception as e:
            print(f"Error uploading trace {trace_id}: {e}")

    def prepare_trace(self, spans, trace_id):
        try:
            ragaai_trace = convert_json_format(spans, self.custom_model_cost)   
            interactions = self.format_interactions(ragaai_trace)         
            ragaai_trace["workflow"] = interactions['workflow']

            # Add source code hash
            hash_id, zip_path = zip_list_of_unique_files(
                self.files_to_zip, output_dir=self.tmp_dir
            )

            ragaai_trace["metadata"]["system_info"] = asdict(self.system_monitor.get_system_info())
            ragaai_trace["metadata"]["resources"] = asdict(self.system_monitor.get_resources())
            ragaai_trace["metadata"]["system_info"]["source_code"] = hash_id

            ragaai_trace["data"][0]["start_time"] = ragaai_trace["start_time"]
            ragaai_trace["data"][0]["end_time"] = ragaai_trace["end_time"]

            ragaai_trace["project_name"] = self.project_name
            
            # Save the trace_json 
            trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
            with open(trace_file_path, "w") as file:
                json.dump(ragaai_trace, file, cls=TracerJSONEncoder, indent=2)

            return {
                'trace_file_path': trace_file_path,
                'code_zip_path': zip_path,
                'hash_id': hash_id
            }
        except Exception as e:
            logger.error(f"Error converting trace {trace_id}: {str(e)}")
            return None

    def _process_child_interactions(self, child, interaction_id, interactions):
        """
        Helper method to process child interactions recursively.
        
        Args:
            child (dict): The child span to process
            interaction_id (int): Current interaction ID
            interactions (list): List of interactions to append to
            
        Returns:
            int: Next interaction ID to use
        """
        child_type = child.get("type")

        if child_type == "tool":
            # Tool call start
            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "tool_call_start",
                    "name": child.get("name"),
                    "content": {
                        "parameters": [
                            child.get("data", {}).get("input", {}).get("args"),
                            child.get("data", {}).get("input", {}).get("kwargs"),
                        ]
                    },
                    "timestamp": child.get("start_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

            # Tool call end
            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "tool_call_end",
                    "name": child.get("name"),
                    "content": {
                        "returns": child.get("data", {}).get("output"),
                    },
                    "timestamp": child.get("end_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

        elif child_type == "llm":
            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "llm_call_start",
                    "name": child.get("name"),
                    "content": {
                        "prompt": child.get("data", {}).get("input"),
                    },
                    "timestamp": child.get("start_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "llm_call_end",
                    "name": child.get("name"),
                    "content": {"response": child.get("data", {}).get("output")},
                    "timestamp": child.get("end_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

        elif child_type == "agent":
            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "agent_call_start",
                    "name": child.get("name"),
                    "content": None,
                    "timestamp": child.get("start_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

            # Process nested children recursively
            if "children" in child.get("data", {}):
                for nested_child in child["data"]["children"]:
                    interaction_id = self._process_child_interactions(
                        nested_child, interaction_id, interactions
                    )

            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": "agent_call_end",
                    "name": child.get("name"),
                    "content": child.get("data", {}).get("output"),
                    "timestamp": child.get("end_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

        else:
            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": f"{child_type}_call_start",
                    "name": child.get("name"),
                    "content": child.get("data", {}),
                    "timestamp": child.get("start_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

            interactions.append(
                {
                    "id": str(interaction_id),
                    "span_id": child.get("id"),
                    "interaction_type": f"{child_type}_call_end",
                    "name": child.get("name"),
                    "content": child.get("data", {}),
                    "timestamp": child.get("end_time"),
                    "error": child.get("error"),
                }
            )
            interaction_id += 1

        # Process additional interactions and network calls
        if "interactions" in child:
            for interaction in child["interactions"]:
                interaction["id"] = str(interaction_id)
                interaction["span_id"] = child.get("id")
                interaction["error"] = None
                interactions.append(interaction)
                interaction_id += 1

        if "network_calls" in child:
            for child_network_call in child["network_calls"]:
                network_call = {}
                network_call["id"] = str(interaction_id)
                network_call["span_id"] = child.get("id")
                network_call["interaction_type"] = "network_call"
                network_call["name"] = None
                network_call["content"] = {
                    "request": {
                        "url": child_network_call.get("url"),
                        "method": child_network_call.get("method"),
                        "headers": child_network_call.get("headers"),
                    },
                    "response": {
                        "status_code": child_network_call.get("status_code"),
                        "headers": child_network_call.get("response_headers"),
                        "body": child_network_call.get("response_body"),
                    },
                }
                network_call["timestamp"] = child_network_call.get("start_time")
                network_call["error"] = child_network_call.get("error")
                interactions.append(network_call)
                interaction_id += 1

        return interaction_id

    def format_interactions(self, trace) -> dict:
        """
        Format interactions from trace data into a standardized format.
        Returns a dictionary containing formatted interactions based on trace data.

        The function processes spans from self.trace and formats them into interactions
        of various types including: agent_start, agent_end, input, output, tool_call_start,
        tool_call_end, llm_call, file_read, file_write, network_call.

        Returns:
            dict: A dictionary with "workflow" key containing a list of interactions
                  sorted by timestamp.
        """
        interactions = []
        interaction_id = 1

        if 'data' not in trace or not trace['data'][0]["spans"]:
            return {"workflow": []}

        for span in trace['data'][0]["spans"]:
            # Process agent spans
            if span['type'] == "agent":
                # Add agent_start interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "agent_call_start",
                        "name": span['name'],
                        "content": None,
                        "timestamp": span['start_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

                # Process children of agent recursively
                if "children" in span['data']:
                    for child in span['data']["children"]:
                        interaction_id = self._process_child_interactions(
                            child, interaction_id, interactions
                        )

                # Add agent_end interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "agent_call_end",
                        "name": span['name'],
                        "content": span['data'].get("output"),
                        "timestamp": span['end_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

            elif span['type'] == "tool":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "tool_call_start",
                        "name": span['name'],
                        "content": {
                            "prompt": span['data'].get("input"),
                            "response": span['data'].get("output"),
                        },
                        "timestamp": span['start_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "tool_call_end",
                        "name": span['name'],
                        "content": {
                            "prompt": span['data'].get("input"),
                            "response": span['data'].get("output"),
                        },
                        "timestamp": span['end_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

            elif span['type'] == "llm":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "llm_call_start",
                        "name": span['name'],
                        "content": {
                            "prompt": span['data'].get("input"),
                        },
                        "timestamp": span['start_time'],
                        "error": span['error']
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": "llm_call_end",
                        "name": span['name'],
                        "content": {"response": span['data'].get("output")},
                        "timestamp": span['end_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

            else:
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": f"{span['type']}_call_start",
                        "name": span['name'],
                        "content": span['data'],
                        "timestamp": span['start_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span['id'],
                        "interaction_type": f"{span['type']}_call_end",
                        "name": span['name'],
                        "content": span['data'],
                        "timestamp": span['end_time'],
                        "error": span['error'],
                    }
                )
                interaction_id += 1

            # Process interactions from span.data if they exist
            if 'interactions' in span:
                for span_interaction in span['interactions']:
                    interaction = {}
                    interaction["id"] = str(interaction_id)
                    interaction["span_id"] = span['id']
                    interaction["interaction_type"] = span_interaction['type']
                    interaction["content"] = span_interaction['content']
                    interaction["timestamp"] = span_interaction['timestamp']
                    interaction["error"] = span['error']
                    interactions.append(interaction)
                    interaction_id += 1

            if 'network_calls' in span:
                for span_network_call in span['network_calls']:
                    network_call = {}
                    network_call["id"] = str(interaction_id)
                    network_call["span_id"] = span['id']
                    network_call["interaction_type"] = "network_call"
                    network_call["name"] = None
                    network_call["content"] = {
                        "request": {
                            "url": span_network_call.get("url"),
                            "method": span_network_call.get("method"),
                            "headers": span_network_call.get("headers"),
                        },
                        "response": {
                            "status_code": span_network_call.get("status_code"),
                            "headers": span_network_call.get("response_headers"),
                            "body": span_network_call.get("response_body"),
                        },
                    }
                    network_call["timestamp"] = span_network_call.get("timestamp")
                    network_call["error"] = span_network_call.get("error")
                    interactions.append(network_call)
                    interaction_id += 1

        # Sort interactions by timestamp
        sorted_interactions = sorted(
            interactions, key=lambda x: x["timestamp"] if x["timestamp"] else ""
        )

        # Reassign IDs to maintain sequential order after sorting
        for idx, interaction in enumerate(sorted_interactions, 1):
            interaction["id"] = str(idx)

        return {"workflow": sorted_interactions}

    def upload_trace(self, ragaai_trace_details, trace_id):
        filepath = ragaai_trace_details['trace_file_path']
        hash_id = ragaai_trace_details['hash_id']
        zip_path = ragaai_trace_details['code_zip_path']

        

        self.upload_task_id = submit_upload_task(
                filepath=filepath,
                hash_id=hash_id,
                zip_path=zip_path,
                project_name=self.project_name,
                project_id=self.project_id,
                dataset_name=self.dataset_name,
                user_details=self.user_details,
                base_url=self.base_url
            )

        logger.info(f"Submitted upload task with ID: {self.upload_task_id}")