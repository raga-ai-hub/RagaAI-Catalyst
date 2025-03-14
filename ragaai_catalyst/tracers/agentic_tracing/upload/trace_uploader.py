"""
trace_uploader.py - A dedicated process for handling trace uploads
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import tempfile
from pathlib import Path
import multiprocessing
import queue
from datetime import datetime
import atexit
import glob
from logging.handlers import RotatingFileHandler
import concurrent.futures
from typing import Dict, Any, Optional

# Set up logging
log_dir = os.path.join(tempfile.gettempdir(), "ragaai_logs")
os.makedirs(log_dir, exist_ok=True)

# Define maximum file size (e.g., 5 MB) and backup count
max_file_size = 5 * 1024 * 1024  # 5 MB
backup_count = 1  # Number of backup files to keep

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(log_dir, "trace_uploader.log"),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
    ]
)
logger = logging.getLogger("trace_uploader")

try:
    from ragaai_catalyst.tracers.agentic_tracing.upload.upload_agentic_traces import UploadAgenticTraces
    from ragaai_catalyst.tracers.agentic_tracing.upload.upload_code import upload_code
    from ragaai_catalyst.tracers.agentic_tracing.upload.upload_trace_metric import upload_trace_metric
    from ragaai_catalyst.tracers.agentic_tracing.utils.create_dataset_schema import create_dataset_schema_with_trace
    from ragaai_catalyst import RagaAICatalyst
    IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("RagaAI Catalyst imports not available - running in test mode")
    IMPORTS_AVAILABLE = False

# Define task queue directory
QUEUE_DIR = os.path.join(tempfile.gettempdir(), "ragaai_tasks")
os.makedirs(QUEUE_DIR, exist_ok=True)

# Status codes
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Global executor for handling uploads
_executor = None
# Dictionary to track futures and their associated task IDs
_futures: Dict[str, Any] = {}

def get_executor():
    """Get or create the thread pool executor"""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="trace_uploader")
    return _executor

def process_upload(task_id: str, filepath: str, hash_id: str, zip_path: str, 
                  project_name: str, project_id: str, dataset_name: str, 
                  user_details: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """
    Process a single upload task
    
    Args:
        task_id: Unique identifier for the task
        filepath: Path to the trace file
        hash_id: Hash ID for the code
        zip_path: Path to the code zip file
        project_name: Project name
        project_id: Project ID
        dataset_name: Dataset name
        user_details: User details dictionary
        base_url: Base URL for API calls
        
    Returns:
        Dict containing status and any error information
    """
    # Correct base_url
    base_url = base_url[0] if isinstance(base_url, tuple) else base_url

    logger.info(f"Processing upload task {task_id}")
    result = {
        "task_id": task_id,
        "status": STATUS_PROCESSING,
        "error": None,
        "start_time": datetime.now().isoformat()
    }
    
    # Save initial status to file
    save_task_status(result)
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            error_msg = f"Task filepath does not exist: {filepath}"
            logger.error(error_msg)
            result["status"] = STATUS_FAILED
            result["error"] = error_msg
            save_task_status(result)
            return result

        if not IMPORTS_AVAILABLE:
            logger.warning(f"Test mode: Simulating processing of task {task_id}")
            # time.sleep(2)  # Simulate work
            result["status"] = STATUS_COMPLETED
            save_task_status(result)
            return result
            
        # Step 1: Create dataset schema
        logger.info(f"Creating dataset schema for {dataset_name} with base_url: {base_url}")
        try:
            response = create_dataset_schema_with_trace(
                dataset_name=dataset_name,
                project_name=project_name,
                base_url=base_url
            )
            logger.info(f"Dataset schema created: {response}")
        except Exception as e:
            logger.error(f"Error creating dataset schema: {e}")
            # Continue with other steps
            
        # Step 2: Upload trace metrics
        if filepath and os.path.exists(filepath):
            logger.info(f"Uploading trace metrics for {filepath}")
            try:
                response = upload_trace_metric(
                    json_file_path=filepath,
                    dataset_name=dataset_name,
                    project_name=project_name,
                    base_url=base_url
                )
                logger.info(f"Trace metrics uploaded: {response}")
            except Exception as e:
                logger.error(f"Error uploading trace metrics: {e}")
                # Continue with other uploads
        else:
            logger.warning(f"Trace file {filepath} not found, skipping metrics upload")
        
        # Step 3: Upload agentic traces
        if filepath and os.path.exists(filepath):
            logger.info(f"Uploading agentic traces for {filepath}")
            try:
                upload_traces = UploadAgenticTraces(
                    json_file_path=filepath,
                    project_name=project_name,
                    project_id=project_id,
                    dataset_name=dataset_name,
                    user_detail=user_details,
                    base_url=base_url,   
                )
                upload_traces.upload_agentic_traces()
                logger.info("Agentic traces uploaded successfully")
            except Exception as e:
                logger.error(f"Error uploading agentic traces: {e}")
                # Continue with code upload
        else:
            logger.warning(f"Trace file {filepath} not found, skipping traces upload")
        
        # Step 4: Upload code hash
        if hash_id and zip_path and os.path.exists(zip_path):
            logger.info(f"Uploading code hash {hash_id}")
            try:
                response = upload_code(
                    hash_id=hash_id,
                    zip_path=zip_path,
                    project_name=project_name,
                    dataset_name=dataset_name,
                    base_url=base_url
                )
                logger.info(f"Code hash uploaded: {response}")
            except Exception as e:
                logger.error(f"Error uploading code hash: {e}")
        else:
            logger.warning(f"Code zip {zip_path} not found, skipping code upload")
        
        # Mark task as completed
        result["status"] = STATUS_COMPLETED
        result["end_time"] = datetime.now().isoformat()
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        result["status"] = STATUS_FAILED
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        
    # Save final status
    save_task_status(result)
    return result

def save_task_status(task_status: Dict[str, Any]):
    """Save task status to a file"""
    task_id = task_status["task_id"]
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    with open(status_path, "w") as f:
        json.dump(task_status, f, indent=2)

def submit_upload_task(filepath, hash_id, zip_path, project_name, project_id, dataset_name, user_details, base_url):
    """
    Submit a new upload task using futures.
    
    Args:
        filepath: Path to the trace file
        hash_id: Hash ID for the code
        zip_path: Path to the code zip file
        project_name: Project name
        project_id: Project ID
        dataset_name: Dataset name
        user_details: User details dictionary
        base_url: Base URL for API calls
        
    Returns:
        str: Task ID
    """
    logger.info(f"Submitting new upload task for file: {filepath}")
    logger.debug(f"Task details - Project: {project_name}, Dataset: {dataset_name}, Hash: {hash_id}, Base_URL: {base_url}")
    
    # Verify the trace file exists
    if not os.path.exists(filepath):
        logger.error(f"Trace file not found: {filepath}")
        return None

    # Create absolute path to the trace file
    filepath = os.path.abspath(filepath)
    logger.debug(f"Using absolute filepath: {filepath}")

    # Generate a unique task ID
    task_id = f"task_{int(time.time())}_{os.getpid()}_{hash(str(time.time()))}"
    
    # Submit the task to the executor
    executor = get_executor()
    future = executor.submit(
        process_upload,
        task_id=task_id,
        filepath=filepath,
        hash_id=hash_id,
        zip_path=zip_path,
        project_name=project_name,
        project_id=project_id,
        dataset_name=dataset_name,
        user_details=user_details,
        base_url=base_url
    )
    
    # Store the future for later status checks
    _futures[task_id] = future
    
    # Create initial status
    initial_status = {
        "task_id": task_id,
        "status": STATUS_PENDING,
        "error": None,
        "start_time": datetime.now().isoformat()
    }
    save_task_status(initial_status)
    
    return task_id

def get_task_status(task_id):
    """
    Get the status of a task by ID.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        dict: Task status information
    """
    logger.debug(f"Getting status for task {task_id}")
    
    # Check if we have a future for this task
    future = _futures.get(task_id)
    
    # If we have a future, check its status
    if future:
        if future.done():
            try:
                # Get the result (this will re-raise any exception that occurred)
                result = future.result(timeout=0)
                return result
            except concurrent.futures.TimeoutError:
                return {"status": STATUS_PROCESSING, "error": None}
            except Exception as e:
                logger.error(f"Error retrieving future result for task {task_id}: {e}")
                return {"status": STATUS_FAILED, "error": str(e)}
        else:
            return {"status": STATUS_PROCESSING, "error": None}
    
    # If we don't have a future, try to read from the status file
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file for task {task_id}: {e}")
            return {"status": "unknown", "error": f"Error reading status: {e}"}
    
    return {"status": "unknown", "error": "Task not found"}

def shutdown():
    """Shutdown the executor"""
    global _executor
    if _executor:
        logger.info("Shutting down executor")
        _executor.shutdown(wait=True)
        _executor = None

# Register shutdown handler
atexit.register(shutdown)

# For backward compatibility
def ensure_uploader_running():
    """
    Ensure the uploader is running.
    This is a no-op in the futures implementation, but kept for API compatibility.
    """
    get_executor()  # Just ensure the executor is created
    return True

# For backward compatibility with the old daemon mode
def run_daemon():
    """
    Run the uploader as a daemon process.
    This is a no-op in the futures implementation, but kept for API compatibility.
    """
    logger.info("Daemon mode not needed in futures implementation")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace uploader process")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon process")
    args = parser.parse_args()
    
    if args.daemon:
        logger.info("Daemon mode not needed in futures implementation")
    else:
        logger.info("Interactive mode not needed in futures implementation")