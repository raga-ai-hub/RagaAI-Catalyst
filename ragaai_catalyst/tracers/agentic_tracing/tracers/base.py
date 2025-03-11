import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Optional
import uuid
import sys
import tempfile
import threading
import time

from ragaai_catalyst.tracers.agentic_tracing.upload.upload_local_metric import calculate_metric
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing.data.data_structure import (
    Trace,
    Metadata,
    SystemInfo,
    Resources,
    Component,
)
from ragaai_catalyst.tracers.agentic_tracing.utils.file_name_tracker import TrackName
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import zip_list_of_unique_files
from ragaai_catalyst.tracers.agentic_tracing.utils.span_attributes import SpanAttributes
from ragaai_catalyst.tracers.agentic_tracing.utils.system_monitor import SystemMonitor
from ragaai_catalyst.tracers.agentic_tracing.upload.trace_uploader import submit_upload_task, get_task_status, ensure_uploader_running

import logging

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class TracerJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return str(obj)  # Fallback to string representation
        if hasattr(obj, "to_dict"):  # Handle objects with to_dict method
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            # Filter out None values and handle nested serialization
            return {
                k: v
                for k, v in obj.__dict__.items()
                if v is not None and not k.startswith("_")
            }
        try:
            # Try to convert to a basic type
            return str(obj)
        except:
            return None  # Last resort: return None instead of failing


class BaseTracer:
    def __init__(self, user_details):
        self.user_details = user_details
        self.project_name = self.user_details["project_name"]
        self.dataset_name = self.user_details["dataset_name"]
        self.project_id = self.user_details["project_id"]
        self.trace_name = self.user_details["trace_name"]
        self.base_url = self.user_details.get("base_url", RagaAICatalyst.BASE_URL)  # Get base_url from user_details or fallback to default
        self.visited_metrics = []
        self.trace_metrics = []

        # Initialize trace data
        self.trace_id = None
        self.start_time = None
        self.components: List[Component] = []
        self.file_tracker = TrackName()
        self.span_attributes_dict = {}

        self.interval_time = self.user_details['interval_time']
        self.memory_usage_list = []
        self.cpu_usage_list = []
        self.disk_usage_list = []
        self.network_usage_list = []
        self.tracking_thread = None
        self.tracking = False
        self.system_monitor = None
        self.gt = None

        # For upload tracking
        self.upload_task_id = None
        
        # For backward compatibility
        self._upload_tasks = []
        self._is_uploading = False
        self._upload_completed_callback = None
        
        ensure_uploader_running()

    def _get_system_info(self) -> SystemInfo:
        return self.system_monitor.get_system_info()

    def _get_resources(self) -> Resources:
        return self.system_monitor.get_resources()

    def _track_memory_usage(self):
        self.memory_usage_list = []
        while self.tracking:
            usage = self.system_monitor.track_memory_usage()
            self.memory_usage_list.append(usage)
            try:
                time.sleep(self.interval_time)
            except Exception as e:
                logger.warning(f"Sleep interrupted in memory tracking: {str(e)}")

    def _track_cpu_usage(self):
        self.cpu_usage_list = []
        while self.tracking:
            usage = self.system_monitor.track_cpu_usage(self.interval_time)
            self.cpu_usage_list.append(usage)
            try:
                time.sleep(self.interval_time)
            except Exception as e:
                logger.warning(f"Sleep interrupted in CPU tracking: {str(e)}")

    def _track_disk_usage(self):
        self.disk_usage_list = []
        while self.tracking:
            usage = self.system_monitor.track_disk_usage()
            self.disk_usage_list.append(usage)
            try:
                time.sleep(self.interval_time)
            except Exception as e:
                logger.warning(f"Sleep interrupted in disk tracking: {str(e)}")

    def _track_network_usage(self):
        self.network_usage_list = []
        while self.tracking:
            usage = self.system_monitor.track_network_usage()
            self.network_usage_list.append(usage)
            try:
                time.sleep(self.interval_time)
            except Exception as e:
                logger.warning(f"Sleep interrupted in network tracking: {str(e)}")

    def start(self):
        """Initialize a new trace"""
        self.tracking = True
        self.trace_id = str(uuid.uuid4())
        self.file_tracker.trace_main_file()
        self.system_monitor = SystemMonitor(self.trace_id)
        threading.Thread(target=self._track_memory_usage).start()
        threading.Thread(target=self._track_cpu_usage).start()
        threading.Thread(target=self._track_disk_usage).start()
        threading.Thread(target=self._track_network_usage).start()

        # Reset metrics
        self.visited_metrics = []
        self.trace_metrics = []

        metadata = Metadata(
            cost={},
            tokens={},
            system_info=self._get_system_info(),
            resources=self._get_resources(),
        )

        # Get the start time
        self.start_time = datetime.now().astimezone().isoformat()

        self.data_key = [
            {
                "start_time": datetime.now().astimezone().isoformat(),
                "end_time": "",
                "spans": self.components,
            }
        ]

        self.trace = Trace(
            id=self.trace_id,
            trace_name=self.trace_name,
            project_name=self.project_name,
            start_time=datetime.now().astimezone().isoformat(),
            end_time="",  # Will be set when trace is stopped
            metadata=metadata,
            data=self.data_key,
            replays={"source": None},
            metrics=[]  # Initialize empty metrics list
        )

    def on_upload_completed(self, callback_fn):
        """
        Register a callback function to be called when all uploads are completed.
        For backward compatibility - simulates the old callback mechanism.
        
        Args:
            callback_fn: A function that takes a single argument (the tracer instance)
        """
        self._upload_completed_callback = callback_fn
        
        # Check for status periodically and call callback when complete
        def check_status_and_callback():
            if self.upload_task_id:
                status = self.get_upload_status()
                if status.get("status") in ["completed", "failed"]:
                    self._is_uploading = False
                    # Execute callback
                    try:
                        if self._upload_completed_callback:
                            self._upload_completed_callback(self)
                    except Exception as e:
                        logger.error(f"Error in upload completion callback: {e}")
                    return
                
                # Schedule next check
                threading.Timer(5.0, check_status_and_callback).start()
        
        # Start status checking if we already have a task
        if self.upload_task_id:
            threading.Timer(5.0, check_status_and_callback).start()
            
        return self
        
    def wait_for_uploads(self, timeout=None):
        """
        Wait for all async uploads to complete.
        This provides backward compatibility with the old API.
        
        Args:
            timeout: Maximum time to wait in seconds (None means wait indefinitely)
            
        Returns:
            True if all uploads completed successfully, False otherwise
        """
        if not self.upload_task_id:
            return True
            
        start_time = time.time()
        while True:
            # Check if timeout expired
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning(f"Upload wait timed out after {timeout} seconds")
                return False
                
            # Get current status
            status = self.get_upload_status()
            if status.get("status") == "completed":
                return True
            elif status.get("status") == "failed":
                logger.error(f"Upload failed: {status.get('error')}")
                return False
            elif status.get("status") == "unknown":
                logger.warning("Upload task not found, assuming completed")
                return True
                
            # Sleep before checking again
            time.sleep(1.0)

    def stop(self):
        """Stop the trace and save to JSON file, then submit to background uploader"""
        if hasattr(self, "trace"):
            # Set end times
            self.trace.data[0]["end_time"] = datetime.now().astimezone().isoformat()
            self.trace.end_time = datetime.now().astimezone().isoformat()

            # Stop tracking metrics
            self.tracking = False
            
            # Process and aggregate metrics
            self._process_resource_metrics()
            
            # Process trace spans
            self.trace = self._change_span_ids_to_int(self.trace)
            self.trace = self._change_agent_input_output(self.trace)
            self.trace = self._extract_cost_tokens(self.trace)

            # Create traces directory and prepare file paths
            self.traces_dir = tempfile.gettempdir()
            filename = self.trace.id + ".json"
            filepath = f"{self.traces_dir}/{filename}"

            # Process source files
            list_of_unique_files = self.file_tracker.get_unique_files()
            hash_id, zip_path = zip_list_of_unique_files(
                list_of_unique_files, output_dir=self.traces_dir
            )
            self.trace.metadata.system_info.source_code = hash_id

            # Prepare trace data for saving
            trace_data = self.trace.to_dict()
            trace_data["metrics"] = self.trace_metrics
            cleaned_trace_data = self._clean_trace(trace_data)
            
            # Add interactions
            interactions = self.format_interactions()
            cleaned_trace_data["workflow"] = interactions["workflow"]

            # Save trace data to file
            with open(filepath, "w") as f:
                json.dump(cleaned_trace_data, f, cls=TracerJSONEncoder, indent=2)

            logger.info("Traces saved successfully.")
            logger.debug(f"Trace saved to {filepath}")
            
            # Make sure uploader process is available
            ensure_uploader_running()

            logger.debug("Base URL used for uploading: {}".format(self.base_url))
            
            # Submit to background process for uploading using futures
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
            
            # For backward compatibility
            self._is_uploading = True
            
            # Start checking for completion if a callback is registered
            if self._upload_completed_callback:
                # Start a thread to check status and call callback when complete
                def check_status_and_callback():
                    status = self.get_upload_status()
                    if status.get("status") in ["completed", "failed"]:
                        self._is_uploading = False
                        # Execute callback
                        try:
                            self._upload_completed_callback(self)
                        except Exception as e:
                            logger.error(f"Error in upload completion callback: {e}")
                        return
                    
                    # Check again after a delay
                    threading.Timer(5.0, check_status_and_callback).start()
                
                # Start checking
                threading.Timer(5.0, check_status_and_callback).start()
            
            logger.info(f"Submitted upload task with ID: {self.upload_task_id}")

        # Cleanup local resources
        self.components = []
        self.file_tracker.reset()
        
    def get_upload_status(self):
        """
        Get the status of the upload task.
        
        Returns:
            dict: Status information
        """
        if not self.upload_task_id:
            return {"status": "not_started", "message": "No upload has been initiated"}
            
        return get_task_status(self.upload_task_id)

    def _process_resource_metrics(self):
        """Process and aggregate all resource metrics"""
        # Process memory metrics
        self.trace.metadata.resources.memory.values = self.memory_usage_list
        
        # Process CPU metrics
        self.trace.metadata.resources.cpu.values = self.cpu_usage_list
        
        # Process network and disk metrics
        network_uploads, network_downloads = 0, 0
        disk_read, disk_write = 0, 0

        # Handle cases where lists might have different lengths
        min_len = min(len(self.network_usage_list), len(self.disk_usage_list)) if self.network_usage_list and self.disk_usage_list else 0
        for i in range(min_len):
            network_usage = self.network_usage_list[i]
            disk_usage = self.disk_usage_list[i]

            # Safely get network usage values with defaults of 0
            network_uploads += network_usage.get('uploads', 0) or 0
            network_downloads += network_usage.get('downloads', 0) or 0

            # Safely get disk usage values with defaults of 0
            disk_read += disk_usage.get('disk_read', 0) or 0
            disk_write += disk_usage.get('disk_write', 0) or 0

        # Set aggregate values
        disk_list_len = len(self.disk_usage_list)
        self.trace.metadata.resources.disk.read = [disk_read / disk_list_len if disk_list_len > 0 else 0]
        self.trace.metadata.resources.disk.write = [disk_write / disk_list_len if disk_list_len > 0 else 0]

        network_list_len = len(self.network_usage_list)
        self.trace.metadata.resources.network.uploads = [
            network_uploads / network_list_len if network_list_len > 0 else 0]
        self.trace.metadata.resources.network.downloads = [
            network_downloads / network_list_len if network_list_len > 0 else 0]

        # Set interval times
        self.trace.metadata.resources.cpu.interval = float(self.interval_time)
        self.trace.metadata.resources.memory.interval = float(self.interval_time)
        self.trace.metadata.resources.disk.interval = float(self.interval_time)
        self.trace.metadata.resources.network.interval = float(self.interval_time)
    
    def add_component(self, component: Component):
        """Add a component to the trace"""
        self.components.append(component)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _process_children(self, children_list, parent_id, current_id):
        """Helper function to process children recursively."""
        for child in children_list:
            child["id"] = current_id
            child["parent_id"] = parent_id
            current_id += 1
            # Recursively process nested children if they exist
            if "children" in child["data"]:
                current_id = self._process_children(child["data"]["children"], child["id"], current_id)
        return current_id

    def _change_span_ids_to_int(self, trace):
        id, parent_id = 1, 0
        for span in trace.data[0]["spans"]:
            span.id = id
            span.parent_id = parent_id
            id += 1
            if span.type == "agent" and "children" in span.data:
                id = self._process_children(span.data["children"], span.id, id)
        return trace

    def _change_agent_input_output(self, trace):
        for span in trace.data[0]["spans"]:
            if span.type == "agent":
                childrens = span.data["children"]
                span.data["input"] = None
                span.data["output"] = None
                if childrens:
                    # Find first non-null input going forward
                    for child in childrens:
                        if "data" not in child:
                            continue
                        input_data = child["data"].get("input")

                        if input_data:
                            span.data["input"] = (
                                input_data["args"]
                                if hasattr(input_data, "args")
                                else input_data
                            )
                            break

                    # Find first non-null output going backward
                    for child in reversed(childrens):
                        if "data" not in child:
                            continue
                        output_data = child["data"].get("output")

                        if output_data and output_data != "" and output_data != "None":
                            span.data["output"] = output_data
                            break
        return trace

    def _extract_cost_tokens(self, trace):
        cost = {}
        tokens = {}

        def process_span_info(info):
            if not isinstance(info, dict):
                return
            cost_info = info.get("cost", {})
            for key, value in cost_info.items():
                if key not in cost:
                    cost[key] = 0
                cost[key] += value
            token_info = info.get("tokens", {})
            for key, value in token_info.items():
                if key not in tokens:
                    tokens[key] = 0
                tokens[key] += value

        def process_spans(spans):
            for span in spans:
                # Get span type, handling both span objects and dictionaries
                span_type = span.type if hasattr(span, 'type') else span.get('type')
                span_info = span.info if hasattr(span, 'info') else span.get('info', {})
                span_data = span.data if hasattr(span, 'data') else span.get('data', {})

                # Process direct LLM spans
                if span_type == "llm":
                    process_span_info(span_info)
                # Process agent spans recursively
                elif span_type == "agent":
                    # Process LLM children in the current agent span
                    children = span_data.get("children", [])
                    for child in children:
                        child_type = child.get("type")
                        if child_type == "llm":
                            process_span_info(child.get("info", {}))
                        # Recursively process nested agent spans
                        elif child_type == "agent":
                            process_spans([child])

        process_spans(trace.data[0]["spans"])
        trace.metadata.cost = cost
        trace.metadata.tokens = tokens
        return trace

    def _clean_trace(self, trace):
        # Convert span to dict if it has to_dict method
        def _to_dict_if_needed(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return obj

        def deduplicate_spans(spans):
            seen_llm_spans = {}  # Dictionary to track unique LLM spans
            unique_spans = []

            for span in spans:
                # Convert span to dictionary if needed
                span_dict = _to_dict_if_needed(span)

                # Skip spans without hash_id
                if "hash_id" not in span_dict:
                    continue

                if span_dict.get("type") == "llm":
                    # Create a unique key based on hash_id, input, and output
                    span_key = (
                        span_dict.get("hash_id"),
                        str(span_dict.get("data", {}).get("input")),
                        str(span_dict.get("data", {}).get("output")),
                    )

                    # Check if we've seen this span before
                    if span_key not in seen_llm_spans:
                        seen_llm_spans[span_key] = True
                        unique_spans.append(span)
                    else:
                        # If we have interactions in the current span, replace the existing one
                        current_interactions = span_dict.get("interactions", [])
                        if current_interactions:
                            # Find and replace the existing span with this one that has interactions
                            for i, existing_span in enumerate(unique_spans):
                                existing_dict = (
                                    existing_span
                                    if isinstance(existing_span, dict)
                                    else existing_span.__dict__
                                )
                                if (
                                        existing_dict.get("hash_id")
                                        == span_dict.get("hash_id")
                                        and str(existing_dict.get("data", {}).get("input"))
                                        == str(span_dict.get("data", {}).get("input"))
                                        and str(existing_dict.get("data", {}).get("output"))
                                        == str(span_dict.get("data", {}).get("output"))
                                ):
                                    unique_spans[i] = span
                                    break
                
                else:
                    # For non-LLM spans, process their children if they exist
                    if "data" in span_dict and "children" in span_dict["data"]:
                        children = span_dict["data"]["children"]
                        # Filter and deduplicate children
                        filtered_children = deduplicate_spans(children)
                        if isinstance(span, dict):
                            span["data"]["children"] = filtered_children
                        else:
                            span.data["children"] = filtered_children
                    unique_spans.append(span)                    

            # Process spans to update model information for LLM spans with same name
            llm_spans_by_name = {}
            for i, span in enumerate(unique_spans):
                span_dict = span if isinstance(span, dict) else span.__dict__
                
                if span_dict.get('type') == 'llm':
                    span_name = span_dict.get('name')
                    if span_name:
                        if span_name not in llm_spans_by_name:
                            llm_spans_by_name[span_name] = []
                        llm_spans_by_name[span_name].append((i, span_dict))
            
            # Update model information for spans with same name
            for spans_with_same_name in llm_spans_by_name.values():
                if len(spans_with_same_name) > 1:
                    # Check if any span has non-default model
                    has_custom_model = any(
                        span[1].get('info', {}).get('model') != 'default'
                        for span in spans_with_same_name
                    )
                    
                    # If we have a custom model, update all default models to 'custom'
                    if has_custom_model:
                        for idx, span_dict in spans_with_same_name:
                            if span_dict.get('info', {}).get('model') == 'default':
                                if isinstance(unique_spans[idx], dict):
                                    if 'info' not in unique_spans[idx]:
                                        unique_spans[idx]['info'] = {}
                                    # unique_spans[idx]['info']['model'] = 'custom'
                                    unique_spans[idx]['type'] = 'custom'
                                else:
                                    if not hasattr(unique_spans[idx], 'info'):
                                        unique_spans[idx].info = {}
                                    # unique_spans[idx].info['model'] = 'custom'
                                    unique_spans[idx].type = 'custom'
            
            return unique_spans

        # Remove any spans without hash ids
        for data in trace.get("data", []):
            if "spans" in data:
                # First filter out spans without hash_ids, then deduplicate
                data["spans"] = deduplicate_spans(data["spans"])

        return trace

    def add_tags(self, tags: List[str]):
        raise NotImplementedError

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

    def format_interactions(self) -> dict:
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

        if not hasattr(self, "trace") or not self.trace.data:
            return {"workflow": []}

        for span in self.trace.data[0]["spans"]:
            # Process agent spans
            if span.type == "agent":
                # Add agent_start interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "agent_call_start",
                        "name": span.name,
                        "content": None,
                        "timestamp": span.start_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

                # Process children of agent recursively
                if "children" in span.data:
                    for child in span.data["children"]:
                        interaction_id = self._process_child_interactions(
                            child, interaction_id, interactions
                        )

                # Add agent_end interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "agent_call_end",
                        "name": span.name,
                        "content": span.data.get("output"),
                        "timestamp": span.end_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

            elif span.type == "tool":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "tool_call_start",
                        "name": span.name,
                        "content": {
                            "prompt": span.data.get("input"),
                            "response": span.data.get("output"),
                        },
                        "timestamp": span.start_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "tool_call_end",
                        "name": span.name,
                        "content": {
                            "prompt": span.data.get("input"),
                            "response": span.data.get("output"),
                        },
                        "timestamp": span.end_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

            elif span.type == "llm":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "llm_call_start",
                        "name": span.name,
                        "content": {
                            "prompt": span.data.get("input"),
                        },
                        "timestamp": span.start_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": "llm_call_end",
                        "name": span.name,
                        "content": {"response": span.data.get("output")},
                        "timestamp": span.end_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

            else:
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": f"{span.type}_call_start",
                        "name": span.name,
                        "content": span.data,
                        "timestamp": span.start_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.id,
                        "interaction_type": f"{span.type}_call_end",
                        "name": span.name,
                        "content": span.data,
                        "timestamp": span.end_time,
                        "error": span.error,
                    }
                )
                interaction_id += 1

            # Process interactions from span.data if they exist
            if span.interactions:
                for span_interaction in span.interactions:
                    interaction = {}
                    interaction["id"] = str(interaction_id)
                    interaction["span_id"] = span.id
                    interaction["interaction_type"] = span_interaction.type
                    interaction["content"] = span_interaction.content
                    interaction["timestamp"] = span_interaction.timestamp
                    interaction["error"] = span.error
                    interactions.append(interaction)
                    interaction_id += 1

            if span.network_calls:
                for span_network_call in span.network_calls:
                    network_call = {}
                    network_call["id"] = str(interaction_id)
                    network_call["span_id"] = span.id
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

    # TODO: Add support for execute metrics. Maintain list of all metrics to be added for this span

    def execute_metrics(self,
                        name: str,
                        model: str,
                        provider: str,
                        prompt: str,
                        context: str,
                        response: str
                        ):
        if not hasattr(self, 'trace'):
            logger.warning("Cannot add metrics before trace is initialized. Call start() first.")
            return

        # Convert individual parameters to metric dict if needed
        if isinstance(name, str):
            metrics = [{
                "name": name
            }]
        else:
            # Handle dict or list input
            metrics = name if isinstance(name, list) else [name] if isinstance(name, dict) else []

        try:
            for metric in metrics:
                if not isinstance(metric, dict):
                    raise ValueError(f"Expected dict, got {type(metric)}")

                if "name" not in metric :
                    raise ValueError("Metric must contain 'name'") #score was written not required here

                # Handle duplicate metric names on executing metric
                metric_name = metric["name"]
                if metric_name in self.visited_metrics:
                    count = sum(1 for m in self.visited_metrics if m.startswith(metric_name))
                    metric_name = f"{metric_name}_{count + 1}"
                self.visited_metrics.append(metric_name)

                result = calculate_metric(project_id=self.project_id,
                                          metric_name=metric_name,
                                          model=model,
                                          org_domain="raga",
                                          provider=provider,
                                          user_id="1",  # self.user_details['id'],
                                          prompt=prompt,
                                          context=context,
                                          response=response
                                          )

                result = result['data']
                formatted_metric = {
                    "name": metric_name,
                    "score": result.get("score"),
                    "reason": result.get("reason", ""),
                    "source": "user",
                    "cost": result.get("cost"),
                    "latency": result.get("latency"),
                    "mappings": [],
                    "config": result.get("metric_config", {})
                }

                logger.debug(f"Executed metric: {formatted_metric}")

        except ValueError as e:
            logger.error(f"Validation Error: {e}")
        except Exception as e:
            logger.error(f"Error adding metric: {e}")

    def add_metrics(
            self,
            name: str | List[Dict[str, Any]] | Dict[str, Any] = None,
            score: float | int = None,
            reasoning: str = "",
            cost: float = None,
            latency: float = None,
            metadata: Dict[str, Any] = None,
            config: Dict[str, Any] = None,
    ):
        """Add metrics at the trace level.

        Can be called in two ways:
        1. With individual parameters:
           tracer.add_metrics(name="metric_name", score=0.9, reasoning="Good performance")
           
        2. With a dictionary or list of dictionaries:
           tracer.add_metrics({"name": "metric_name", "score": 0.9})
           tracer.add_metrics([{"name": "metric1", "score": 0.9}, {"name": "metric2", "score": 0.8}])

        Args:
            name: Either the metric name (str) or a metric dictionary/list of dictionaries
            score: Score value (float or int) when using individual parameters
            reasoning: Optional explanation for the score
            cost: Optional cost associated with the metric
            latency: Optional latency measurement
            metadata: Optional additional metadata as key-value pairs
            config: Optional configuration parameters
        """
        if not hasattr(self, 'trace'):
            logger.warning("Cannot add metrics before trace is initialized. Call start() first.")
            return

        # Convert individual parameters to metric dict if needed
        if isinstance(name, str):
            metrics = [{
                "name": name,
                "score": score,
                "reasoning": reasoning,
                "cost": cost,
                "latency": latency,
                "metadata": metadata or {},
                "config": config or {}
            }]
        else:
            # Handle dict or list input
            metrics = name if isinstance(name, list) else [name] if isinstance(name, dict) else []

        try:
            for metric in metrics:
                if not isinstance(metric, dict):
                    raise ValueError(f"Expected dict, got {type(metric)}")

                if "name" not in metric or "score" not in metric:
                    raise ValueError("Metric must contain 'name' and 'score' fields")

                # Handle duplicate metric names
                metric_name = metric["name"]
                if metric_name in self.visited_metrics:
                    count = sum(1 for m in self.visited_metrics if m.startswith(metric_name))
                    metric_name = f"{metric_name}_{count + 1}"
                self.visited_metrics.append(metric_name)

                formatted_metric = {
                    "name": metric_name,
                    "score": metric["score"],
                    "reason": metric.get("reasoning", ""),
                    "source": "user",
                    "cost": metric.get("cost"),
                    "latency": metric.get("latency"),
                    "metadata": metric.get("metadata", {}),
                    "mappings": [],
                    "config": metric.get("config", {})
                }

                self.trace_metrics.append(formatted_metric)
                logger.debug(f"Added trace-level metric: {formatted_metric}")

        except ValueError as e:
            logger.error(f"Validation Error: {e}")
        except Exception as e:
            logger.error(f"Error adding metric: {e}")

    def span(self, span_name):
        if span_name not in self.span_attributes_dict:
            self.span_attributes_dict[span_name] = SpanAttributes(span_name, self.project_id)
        return self.span_attributes_dict[span_name]

    @staticmethod
    def get_formatted_metric(span_attributes_dict, project_id, name):
        if name in span_attributes_dict:
            local_metrics = span_attributes_dict[name].local_metrics or []
            local_metrics_results = []
            for metric in local_metrics:
                try:
                    logger.info("calculating the metric, please wait....")

                    mapping = metric.get("mapping", {})
                    result = calculate_metric(project_id=project_id,
                                              metric_name=metric.get("name"),
                                              model=metric.get("model"),
                                              provider=metric.get("provider"),
                                              **mapping
                                              )

                    result = result['data']['data'][0]
                    config = result['metric_config']
                    metric_config = {
                        "job_id": config.get("job_id"),
                        "metric_name": config.get("displayName"),
                        "model": config.get("model"),
                        "org_domain": config.get("orgDomain"),
                        "provider": config.get("provider"),
                        "reason": config.get("reason"),
                        "request_id": config.get("request_id"),
                        "user_id": config.get("user_id"),
                        "threshold": {
                            "is_editable": config.get("threshold").get("isEditable"),
                            "lte": config.get("threshold").get("lte")
                        }
                    }
                    formatted_metric = {
                        "name": metric.get("displayName"),
                        "displayName": metric.get("displayName"),
                        "score": result.get("score"),
                        "reason": result.get("reason", ""),
                        "source": "user",
                        "cost": result.get("cost"),
                        "latency": result.get("latency"),
                        "mappings": [],
                        "config": metric_config
                    }
                    local_metrics_results.append(formatted_metric)
                except ValueError as e:
                    logger.error(f"Validation Error: {e}")
                except Exception as e:
                    logger.error(f"Error executing metric: {e}")

            return local_metrics_results


    def upload_directly(self):
        """Upload trace directly without using the background process"""
        # Check if we have necessary details
        if not hasattr(self, 'trace') or not self.trace_id:
            print("No trace to upload")
            return False
            
        # Get the filepath from the last trace
        trace_dir = tempfile.gettempdir()
        trace_file = os.path.join(trace_dir, f"{self.trace_id}.json")
        
        # If filepath wasn't saved from previous stop() call, try to find it
        if not os.path.exists(trace_file):
            print(f"Looking for trace file for {self.trace_id}")
            # Try to find the trace file by pattern
            for file in os.listdir(trace_dir):
                if file.endswith(".json") and self.trace_id in file:
                    trace_file = os.path.join(trace_dir, file)
                    print(f"Found trace file: {trace_file}")
                    break
        
        if not os.path.exists(trace_file):
            print(f"Trace file not found for ID {self.trace_id}")
            return False
            
        print(f"Starting direct upload of {trace_file}")
        
        try:
            # 1. Create the dataset schema
            print("Creating dataset schema...")
            from ragaai_catalyst.tracers.agentic_tracing.utils.create_dataset_schema import create_dataset_schema_with_trace
            response = create_dataset_schema_with_trace(
                dataset_name=self.dataset_name,
                project_name=self.project_name
            )
            print(f"Schema created: {response}")
            
            # 2. Upload trace metrics
            print("Uploading trace metrics...")
            from ragaai_catalyst.tracers.agentic_tracing.upload.upload_trace_metric import upload_trace_metric
            response = upload_trace_metric(
                json_file_path=trace_file,
                dataset_name=self.dataset_name,
                project_name=self.project_name,
            )
            print(f"Metrics uploaded: {response}")
            
            # 3. Get code hash and zip path if available
            code_hash = None
            zip_path = None
            try:
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                    code_hash = data.get("metadata", {}).get("system_info", {}).get("source_code")
                    if code_hash:
                        zip_path = os.path.join(trace_dir, f"{code_hash}.zip")
                        print(f"Found code hash: {code_hash}")
                        print(f"Zip path: {zip_path}")
            except Exception as e:
                print(f"Error getting code hash: {e}")
            
            # 4. Upload agentic traces
            print("Uploading agentic traces...")
            from ragaai_catalyst.tracers.agentic_tracing.upload.upload_agentic_traces import UploadAgenticTraces
            from ragaai_catalyst import RagaAICatalyst
            upload_traces = UploadAgenticTraces(
                json_file_path=trace_file,
                project_name=self.project_name,
                project_id=self.project_id,
                dataset_name=self.dataset_name,
                user_detail=self.user_details,
                base_url=RagaAICatalyst.BASE_URL,
            )
            upload_traces.upload_agentic_traces()
            print("Agentic traces uploaded successfully")
            
            # 5. Upload code hash if available
            if code_hash and zip_path and os.path.exists(zip_path):
                print(f"Uploading code hash: {code_hash}")
                from ragaai_catalyst.tracers.agentic_tracing.upload.upload_code import upload_code
                response = upload_code(
                    hash_id=code_hash,
                    zip_path=zip_path,
                    project_name=self.project_name,
                    dataset_name=self.dataset_name,
                )
                print(f"Code uploaded: {response}")
            
            print("Upload completed successfully - check UI now")
            return True
        except Exception as e:
            print(f"Error during direct upload: {e}")
            import traceback
            traceback.print_exc()
            return False
