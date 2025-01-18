import os
import sys
import time
import psutil
import socket
import logging
import requests
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_free_port(start_port=8000, max_port=8100):
    """Finds a free port between start_port and max_port."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    return None


def is_port_free(port):
    """Checks if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def launch_dashboard(port=3005):
    """Launches the dashboard on a specified port."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    agentneo_dir = os.path.dirname(parent_dir)

    if not is_port_free(port):
        logging.info(f"Port {port} is busy. Finding an available port...")
        free_port = find_free_port(port + 1)
        if free_port is None:
            logging.error(f"No free ports available starting from {port}")
            return
        port = free_port
        logging.info(f"Using port {port}")

    # Set the environment variable with the port
    os.environ["AGENTNEO_DASHBOARD_PORT"] = str(port)

    # Start the dashboard server in a new detached subprocess
    command = [
        sys.executable,
        "-m",
        "agentneo.server.dashboard_server",
        "--port",
        str(port),
    ]

    logging.debug(f"Command to be executed: {' '.join(command)}")

    try:
        if sys.platform == "win32":
            # Windows
            DETACHED_PROCESS = 0x00000008
            process = subprocess.Popen(
                command,
                creationflags=DETACHED_PROCESS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=agentneo_dir,
            )
        else:
            # Unix/Linux/Mac
            process = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=agentneo_dir,
            )

        try:
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode is not None:
                logging.error(
                    f"Dashboard failed to start. Return code: {process.returncode}"
                )
                logging.error(f"STDOUT: {stdout.decode('utf-8')}")
                logging.error(f"STDERR: {stderr.decode('utf-8')}")
                return
        except subprocess.TimeoutExpired:
            # Process is still running, which is good
            logging.info("Dashboard process started successfully")

        # Check if the server is responding
        max_retries = 5
        for _ in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    logging.info(
                        f"Dashboard launched successfully. Access it at: http://localhost:{port}"
                    )
                    return
            except requests.RequestException:
                time.sleep(1)

        logging.error(
            "Dashboard started but is not responding. It may have encountered an error."
        )
    except Exception as e:
        logging.error(f"Failed to launch dashboard: {e}", exc_info=True)
        return

    logging.info(
        f"Dashboard launch attempt completed. If successful, access it at: http://localhost:{port}"
    )


def get_process_by_port(port):
    """Find process using the specified port."""
    for proc in psutil.process_iter(["pid", "name", "connections"]):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def close_dashboard(port=3005):
    """Closes the dashboard by sending a shutdown request or killing the process if necessary."""
    try:
        # First try graceful shutdown
        response = requests.post(f"http://localhost:{port}/shutdown", timeout=5)
        if response.status_code == 200:
            logging.info("Dashboard closed successfully.")
            return True
    except Exception as e:
        logging.warning(f"Graceful shutdown failed: {e}")

    # If graceful shutdown fails, try to force kill the process
    try:
        process = get_process_by_port(port)
        if process:
            process.terminate()  # Try graceful termination first
            try:
                process.wait(timeout=3)  # Wait for process to terminate
            except psutil.TimeoutExpired:
                process.kill()  # Force kill if termination doesn't work
            logging.info(f"Dashboard process on port {port} forcefully terminated.")
            return True
        else:
            logging.warning(f"No process found using port {port}")
    except Exception as e:
        logging.error(f"Failed to forcefully terminate dashboard process: {e}")

    return False
