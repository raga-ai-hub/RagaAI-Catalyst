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

# Set up logging
log_dir = os.path.join(tempfile.gettempdir(), "ragaai_logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "trace_uploader.log"))
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

# Clean up any stale processes
def cleanup_stale_processes():
    """Check for stale processes but allow active uploads to complete"""
    pid_file = os.path.join(tempfile.gettempdir(), "trace_uploader.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            try:
                import psutil
                if psutil.pid_exists(old_pid):
                    p = psutil.Process(old_pid)
                    if "trace_uploader.py" in " ".join(p.cmdline()):
                        # Instead of terminating, just remove the PID file
                        # This allows the process to finish its current uploads
                        logger.info(f"Removing PID file for process {old_pid}")
                        os.remove(pid_file)
                        return
            except Exception as e:
                logger.warning(f"Error checking stale process: {e}")
            os.remove(pid_file)
        except Exception as e:
            logger.warning(f"Error reading PID file: {e}")

cleanup_stale_processes()

# Status codes
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class UploadTask:
    """Class representing a single upload task"""
    
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id or f"task_{int(time.time())}_{os.getpid()}_{hash(str(time.time()))}"
        self.status = STATUS_PENDING
        self.attempts = 0
        self.max_attempts = 3
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.error = None
        
        # Task details
        self.filepath = kwargs.get("filepath")
        self.hash_id = kwargs.get("hash_id")
        self.zip_path = kwargs.get("zip_path")
        self.project_name = kwargs.get("project_name")
        self.project_id = kwargs.get("project_id")
        self.dataset_name = kwargs.get("dataset_name")
        self.user_details = kwargs.get("user_details", {})
        self.base_url = kwargs.get("base_url")
        
    def to_dict(self):
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "filepath": self.filepath,
            "hash_id": self.hash_id,
            "zip_path": self.zip_path,
            "project_name": self.project_name,
            "project_id": self.project_id,
            "dataset_name": self.dataset_name,
            "user_details": self.user_details,
            "base_url": self.base_url
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create task from dictionary"""
        task = cls(task_id=data.get("task_id"))
        task.status = data.get("status", STATUS_PENDING)
        task.attempts = data.get("attempts", 0)
        task.max_attempts = data.get("max_attempts", 3)
        task.created_at = data.get("created_at")
        task.updated_at = data.get("updated_at")
        task.error = data.get("error")
        task.filepath = data.get("filepath")
        task.hash_id = data.get("hash_id")
        task.zip_path = data.get("zip_path")
        task.project_name = data.get("project_name")
        task.project_id = data.get("project_id")
        task.dataset_name = data.get("dataset_name")
        task.user_details = data.get("user_details", {})
        task.base_url = data.get("base_url")    
        return task
        
    def update_status(self, status, error=None):
        """Update task status"""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if error:
            self.error = str(error)
        self.save()
        
    def increment_attempts(self):
        """Increment the attempt counter"""
        self.attempts += 1
        self.updated_at = datetime.now().isoformat()
        self.save()
        
    def save(self):
        """Save task to disk"""
        task_path = os.path.join(QUEUE_DIR, f"{self.task_id}.json")
        with open(task_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def delete(self):
        """Delete task file from disk"""
        task_path = os.path.join(QUEUE_DIR, f"{self.task_id}.json")
        if os.path.exists(task_path):
            os.remove(task_path)
            
    @staticmethod
    def list_pending_tasks():
        """List all pending tasks"""
        tasks = []
        logger.info("Listing pending tasks from queue directory: {}".format(QUEUE_DIR))
        for filename in os.listdir(QUEUE_DIR):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(QUEUE_DIR, filename), "r") as f:
                        task_data = json.load(f)
                        task = UploadTask.from_dict(task_data)
                        if task.status in [STATUS_PENDING, STATUS_FAILED] and task.attempts < task.max_attempts:
                            # Verify files still exist
                            if (not task.filepath or os.path.exists(task.filepath)) and \
                               (not task.zip_path or os.path.exists(task.zip_path)):
                                tasks.append(task)
                            else:
                                # Files missing, mark as failed
                                task.update_status(STATUS_FAILED, "Required files missing")
                except Exception as e:
                    logger.error(f"Error loading task {filename}: {e}")
        return tasks


class TraceUploader:
    """
    Trace uploader process
    Handles the actual upload work in a separate process
    """
    
    def __init__(self):
        self.running = True
        self.processing = False
        
    def start(self):
        """Start the uploader loop"""
        logger.info("Trace uploader starting")
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Main processing loop
        while self.running:
            try:
                # Get pending tasks
                tasks = UploadTask.list_pending_tasks()
                if tasks:
                    logger.info(f"Found {len(tasks)} pending tasks")
                    for task in tasks:
                        if not self.running:
                            break
                        self.process_task(task)
                else:
                    # No tasks, sleep before checking again
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Error in uploader loop: {e}")
                time.sleep(5)
                
        logger.info("Trace uploader stopped")
        
    def process_task(self, task):
        """Process a single upload task"""
        logger.info(f"Starting to process task {task.task_id}")
        logger.debug(f"Task details: {task.to_dict()}")

        # Check if file exists
        if not os.path.exists(task.filepath):
            error_msg = f"Task filepath does not exist: {task.filepath}"
            logger.error(error_msg)
            task.update_status(STATUS_FAILED, error_msg)
            return

        if not IMPORTS_AVAILABLE:
            logger.warning(f"Test mode: Simulating processing of task {task.task_id}")
            time.sleep(2)  # Simulate work
            task.update_status(STATUS_COMPLETED)
            return
            
        logger.info(f"Processing task {task.task_id} (attempt {task.attempts+1}/{task.max_attempts})")
        self.processing = True
        task.update_status(STATUS_PROCESSING)
        task.increment_attempts()
        
        # Log memory state for debugging
        try:
            import psutil
            process = psutil.Process()
            logger.debug(f"Memory usage before processing: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        except ImportError:
            pass
        
        try:
            # Step 1: Create dataset schema
            logger.info(f"Creating dataset schema for {task.dataset_name} with base_url: {task.base_url}")
            response = create_dataset_schema_with_trace(
                dataset_name=task.dataset_name,
                project_name=task.project_name,
                base_url=task.base_url
            )
            logger.info(f"Dataset schema created: {response}")
            
            # Step 2: Upload trace metrics
            if task.filepath and os.path.exists(task.filepath):
                logger.info(f"Uploading trace metrics for {task.filepath}")
                try:
                    response = upload_trace_metric(
                        json_file_path=task.filepath,
                        dataset_name=task.dataset_name,
                        project_name=task.project_name,
                        base_url=task.base_url
                    )
                    logger.info(f"Trace metrics uploaded: {response}")
                except Exception as e:
                    logger.error(f"Error uploading trace metrics: {e}")
                    # Continue with other uploads
            else:
                logger.warning(f"Trace file {task.filepath} not found, skipping metrics upload")
            
            # Step 3: Upload agentic traces
            if task.filepath and os.path.exists(task.filepath):
                logger.info(f"Uploading agentic traces for {task.filepath}")
                try:
                    upload_traces = UploadAgenticTraces(
                        json_file_path=task.filepath,
                        project_name=task.project_name,
                        project_id=task.project_id,
                        dataset_name=task.dataset_name,
                        user_detail=task.user_details,
                        base_url=task.base_url,   
                    )
                    upload_traces.upload_agentic_traces()
                    logger.info("Agentic traces uploaded successfully")
                except Exception as e:
                    logger.error(f"Error uploading agentic traces: {e}")
                    # Continue with code upload
            else:
                logger.warning(f"Trace file {task.filepath} not found, skipping traces upload")
            
            # Step 4: Upload code hash
            if task.hash_id and task.zip_path and os.path.exists(task.zip_path):
                logger.info(f"Uploading code hash {task.hash_id}")
                try:
                    response = upload_code(
                        hash_id=task.hash_id,
                        zip_path=task.zip_path,
                        project_name=task.project_name,
                        dataset_name=task.dataset_name,
                        base_url=task.base_url
                    )
                    logger.info(f"Code hash uploaded: {response}")
                except Exception as e:
                    logger.error(f"Error uploading code hash: {e}")
            else:
                logger.warning(f"Code zip {task.zip_path} not found, skipping code upload")
            
            # Mark task as completed
            task.update_status(STATUS_COMPLETED)
            logger.info(f"Task {task.task_id} completed successfully")
            
            # Clean up task file
            task.delete()
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            if task.attempts >= task.max_attempts:
                task.update_status(STATUS_FAILED, str(e))
                logger.error(f"Task {task.task_id} failed after {task.attempts} attempts")
            else:
                task.update_status(STATUS_PENDING, str(e))
                logger.warning(f"Task {task.task_id} will be retried (attempt {task.attempts}/{task.max_attempts})")
        finally:
            self.processing = False
            
    def handle_signal(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        
    def cleanup(self):
        """Cleanup before exit"""
        logger.info("Performing cleanup before exit")
        self.running = False
        

def submit_upload_task(filepath, hash_id, zip_path, project_name, project_id, dataset_name, user_details, base_url):
    """
    Submit a new upload task to the queue.
    This function can be called from the main application.
    
    Returns:
        str: Task ID
    """
    logger.info(f"Submitting new upload task for file: {filepath}")
    logger.debug(f"Task details - Project: {project_name}, Dataset: {dataset_name}, Hash: {hash_id}, Base_URL: {base_url}")
    
    # Verify the trace file exists
    if not os.path.exists(filepath):
        logger.error(f"Trace file not found: {filepath}")
        return None

    # Create task with absolute path to the trace file
    filepath = os.path.abspath(filepath)
    logger.debug(f"Using absolute filepath: {filepath}")

    task = UploadTask(
        filepath=filepath,
        hash_id=hash_id,
        zip_path=zip_path,
        project_name=project_name,
        project_id=project_id,
        dataset_name=dataset_name,
        user_details=user_details,
        base_url=base_url
    )
    
    # Save the task with proper error handling
    task_path = os.path.join(QUEUE_DIR, f"{task.task_id}.json")
    logger.debug(f"Saving task to: {task_path}")
    
    try:
        # Ensure queue directory exists
        os.makedirs(QUEUE_DIR, exist_ok=True)
        
        with open(task_path, "w") as f:
            json.dump(task.to_dict(), f, indent=2)
            
        logger.info(f"Task {task.task_id} created successfully for trace file: {filepath}")
    except Exception as e:
        logger.error(f"Error creating task file: {e}", exc_info=True)
        return None
    
    # Ensure uploader process is running
    logger.info("Starting uploader process...")
    pid = ensure_uploader_running()
    if pid:
        logger.info(f"Uploader process running with PID {pid}")
    else:
        logger.warning("Failed to start uploader process, but task was queued")
    
    return task.task_id


def get_task_status(task_id):
    """
    Get the status of a task by ID.
    This function can be called from the main application.
    
    Returns:
        dict: Task status information
    """
    task_path = os.path.join(QUEUE_DIR, f"{task_id}.json")
    if not os.path.exists(task_path):
        # Check if it might be in completed directory
        completed_path = os.path.join(QUEUE_DIR, "completed", f"{task_id}.json")
        if os.path.exists(completed_path):
            with open(completed_path, "r") as f:
                return json.load(f)
        return {"status": "unknown", "error": "Task not found"}
    
    with open(task_path, "r") as f:
        return json.load(f)


def ensure_uploader_running():
    """
    Ensure the uploader process is running.
    Starts it if not already running.
    """
    logger.info("Checking if uploader process is running...")
    
    # Check if we can find a running process
    pid_file = os.path.join(tempfile.gettempdir(), "trace_uploader.pid")
    logger.debug(f"PID file location: {pid_file}")
    
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid_str = f.read().strip()
                logger.debug(f"Read PID from file: {pid_str}")
                pid = int(pid_str)
            
            # Check if process is actually running
            # Use platform-specific process check
            is_running = False
            try:
                if os.name == 'posix':  # Unix/Linux/Mac
                    logger.debug(f"Checking process {pid} on Unix/Mac")
                    os.kill(pid, 0)  # This raises an exception if process doesn't exist
                    is_running = True
                else:  # Windows
                    logger.debug(f"Checking process {pid} on Windows")
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    SYNCHRONIZE = 0x00100000
                    process = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
                    if process:
                        kernel32.CloseHandle(process)
                        is_running = True
            except (ImportError, AttributeError) as e:
                logger.debug(f"Platform-specific check failed: {e}, falling back to cross-platform check")
                # Fall back to cross-platform check
                try:
                    import psutil
                    is_running = psutil.pid_exists(pid)
                    logger.debug(f"psutil check result: {is_running}")
                except ImportError:
                    logger.debug("psutil not available, using basic process check")
                    # If psutil is not available, make a best guess
                    try:
                        os.kill(pid, 0)
                        is_running = True
                    except Exception as e:
                        logger.debug(f"Basic process check failed: {e}")
                        is_running = False
            
            if is_running:
                logger.debug(f"Uploader process already running with PID {pid}")
                return pid
        except (ProcessLookupError, ValueError, PermissionError):
            # Process not running or other error, remove stale PID file
            try:
                os.remove(pid_file)
            except:
                pass
    
    # Start new process
    logger.info("Starting new uploader process")
    
    # Get the path to this script
    script_path = os.path.abspath(__file__)
    
    # Start detached process in a platform-specific way
    try:
        # First, try the preferred method for each platform
        if os.name == 'posix':  # Unix/Linux/Mac
            import subprocess
            
            # Use double fork method on Unix systems
            try:
                # First fork
                pid = os.fork()
                if pid > 0:
                    # Parent process, return
                    return pid
                    
                # Decouple from parent environment
                os.chdir('/')
                os.setsid()
                os.umask(0)
                
                # Second fork
                pid = os.fork()
                if pid > 0:
                    # Exit from second parent
                    os._exit(0)
                    
                # Redirect standard file descriptors
                sys.stdout.flush()
                sys.stderr.flush()
                si = open(os.devnull, 'r')
                so = open(os.path.join(tempfile.gettempdir(), 'trace_uploader_stdout.log'), 'a+')
                se = open(os.path.join(tempfile.gettempdir(), 'trace_uploader_stderr.log'), 'a+')
                os.dup2(si.fileno(), sys.stdin.fileno())
                os.dup2(so.fileno(), sys.stdout.fileno())
                os.dup2(se.fileno(), sys.stderr.fileno())
                
                # Execute the daemon process
                os.execl(sys.executable, sys.executable, script_path, '--daemon')
                
            except (AttributeError, OSError):
                # Fork not available, try subprocess
                process = subprocess.Popen(
                    [sys.executable, script_path, "--daemon"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    start_new_session=True  # Detach from parent
                )
                pid = process.pid
                
        else:  # Windows
            import subprocess
            # Use the DETACHED_PROCESS flag on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            
            # Windows-specific flags
            DETACHED_PROCESS = 0x00000008
            CREATE_NO_WINDOW = 0x08000000
            
            process = subprocess.Popen(
                [sys.executable, script_path, "--daemon"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                startupinfo=startupinfo,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW
            )
            pid = process.pid
        
        # Write PID to file
        with open(pid_file, "w") as f:
            f.write(str(pid))
            
        logger.info(f"Started uploader process with PID {pid}")
        return pid
        
    except Exception as e:
        logger.error(f"Error starting uploader process using primary method: {e}")
        
        # Fallback method using multiprocessing (works on most platforms)
        try:
            logger.info("Trying fallback method with multiprocessing")
            import multiprocessing
            
            def run_uploader():
                """Run the uploader in a separate process"""
                # Redirect output
                sys.stdout = open(os.path.join(tempfile.gettempdir(), 'trace_uploader_stdout.log'), 'a+')
                sys.stderr = open(os.path.join(tempfile.gettempdir(), 'trace_uploader_stderr.log'), 'a+')
                
                # Run daemon
                run_daemon()
                
            # Start process
            process = multiprocessing.Process(target=run_uploader)
            process.daemon = True  # Daemonize it
            process.start()
            pid = process.pid
            
            # Write PID to file
            with open(pid_file, "w") as f:
                f.write(str(pid))
                
            logger.info(f"Started uploader process with fallback method, PID {pid}")
            return pid
            
        except Exception as e2:
            logger.error(f"Error starting uploader process using fallback method: {e2}")
            
            # Super fallback - run in the current process if all else fails
            # This is not ideal but better than failing completely
            try:
                logger.warning("Using emergency fallback - running in current process thread")
                import threading
                
                thread = threading.Thread(target=run_daemon, daemon=True)
                thread.start()
                
                # No real PID since it's a thread, but we'll create a marker file
                with open(pid_file, "w") as f:
                    f.write(f"thread_{id(thread)}")
                    
                return None
            except Exception as e3:
                logger.error(f"All methods failed to start uploader: {e3}")
                return None


def run_daemon():
    """Run the uploader as a daemon process"""
    # Write PID to file
    pid_file = os.path.join(tempfile.gettempdir(), "trace_uploader.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
        
    try:
        uploader = TraceUploader()
        uploader.start()
    finally:
        # Clean up PID file
        if os.path.exists(pid_file):
            os.remove(pid_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace uploader process")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon process")
    parser.add_argument("--test", action="store_true", help="Submit a test task")
    args = parser.parse_args()
    
    if args.daemon:
        run_daemon()
    elif args.test:
        # Submit a test task
        test_file = os.path.join(tempfile.gettempdir(), "test_trace.json")
        with open(test_file, "w") as f:
            f.write("{}")
        
        task_id = submit_upload_task(
            filepath=test_file,
            hash_id="test_hash",
            zip_path=test_file,
            project_name="test_project",
            project_id="test_id",
            dataset_name="test_dataset",
            user_details={"id": "test_user"}
        )
        print(f"Submitted test task with ID: {task_id}")
    else:
        print("Use --daemon to run as daemon or --test to submit a test task")