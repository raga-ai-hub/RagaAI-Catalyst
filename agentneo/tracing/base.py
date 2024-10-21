import asyncio
import platform
import cpuinfo
import json
import sys
from pathlib import Path
import contextvars
from typing import Optional
import traceback
import psutil
import GPUtil
import pkg_resources
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import logging


from ..data import (
    Base,
    ProjectInfoModel,
    SystemInfoModel,
    ErrorModel,
    TraceModel,
    LLMCallModel,
)


class BaseTracer:
    def __init__(self, session):
        self.user_session = session
        project_name = session.project_name

        # Setup DB
        self.db_path = session.db_path
        self.engine = create_engine(self.db_path)

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.project_info = self._get_project(project_name)
        if self.project_info is None:
            raise ValueError(
                f"Project '{project_name}' does not exist. Please create a project first."
            )
        self.project_id = self.project_info.id

        self.trace_data = {
            "project_info": {
                "project_name": project_name,
                "start_time": self.project_info.start_time,
            },
            "llm_calls": [],
            "tool_calls": [],
            "agent_calls": [],
            "errors": [],
        }

        self.trace_id = None
        self.trace = None

        # Initialize context variables as instance variables
        self.current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
        self.current_llm_call_ids = contextvars.ContextVar(
            "current_llm_call_ids", default=None
        )
        self.current_tool_call_ids = contextvars.ContextVar(
            "current_tool_call_ids", default=None
        )
        self.current_llm_call_name = contextvars.ContextVar(
            "current_llm_call_name", default=None
        )
        self.current_user_interaction_ids = contextvars.ContextVar(
            "current_user_interaction_ids", default=None
        )

    def __enter__(self):
        # Start the tracer when entering the context

        self.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # End the tracer when exiting the context
        self.stop()

    async def __aenter__(self):
        # Start the tracer when entering the async context
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, tb):
        # End the tracer when exiting the async context
        self.stop()

    def start(self):
        print("Tracing Started.")

        # Create a new trace
        with self.Session() as session:
            trace = TraceModel(project_id=self.project_id, start_time=datetime.now())
            session.add(trace)
            session.commit()
            self.trace_id = trace.id
            self.trace = trace

        # Save the system information
        self._save_system_info()

    def stop(self):
        with self.Session() as session:
            project = session.query(ProjectInfoModel).get(self.project_id)
            if project is None:
                raise ValueError(f"Project with id {self.project_id} not found")

            trace = session.query(TraceModel).get(self.trace_id)
            if trace is None:
                raise ValueError(f"Trace with id {self.trace_id} not found")

            trace.end_time = datetime.now()
            trace.duration = (trace.end_time - trace.start_time).total_seconds()

            project.end_time = trace.end_time
            if project.duration is None:
                project.duration = 0
            project.duration += trace.duration  # Accumulate duration

            llm_calls = (
                session.query(LLMCallModel).filter_by(trace_id=self.trace_id).all()
            )
            trace_cost = sum(
                json.loads(llm_call.cost).get("total", 0) for llm_call in llm_calls
            )
            trace_tokens = sum(
                json.loads(llm_call.token_usage).get("total", 0)
                for llm_call in llm_calls
            )

            # Accumulate costs and tokens instead of overwriting
            if project.total_cost is None:
                project.total_cost = 0
            project.total_cost += trace_cost

            if project.total_tokens is None:
                project.total_tokens = 0
            project.total_tokens += trace_tokens

            session.commit()

            end_time = trace.end_time
            duration = trace.duration
            total_cost = project.total_cost
            total_tokens = project.total_tokens

        self.trace_data["project_info"].update(
            {
                "end_time": end_time,
                "duration": duration,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            }
        )

        print(f"Tracing Completed.\nData saved to the database and JSON file.\n")

    def _get_project(self, project_name: str) -> Optional[ProjectInfoModel]:
        with self.Session() as session:
            try:
                project = (
                    session.query(ProjectInfoModel)
                    .filter_by(project_name=project_name)
                    .first()
                )
                if project:
                    print(f"Project '{project_name}' found.")
                    return project
                else:
                    print(f"Project '{project_name}' does not exist.")
                    return None
            except SQLAlchemyError as e:
                print(f"An error occurred while accessing the database: {e}")
                raise

    def _save_system_info(self):
        os_name = platform.system()
        if os_name == "Darwin":
            os_version = platform.mac_ver()[0]
        else:
            os_version = platform.version()

        # Get GPU information
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [
                {"name": gpu.name, "memory_total": gpu.memoryTotal} for gpu in gpus
            ]
        except Exception as e:
            logging.warning(f"Failed to get GPU information: {e}")
            gpu_info = None

        # Get disk information
        disk = psutil.disk_usage("/")
        disk_info = {
            "total": disk.total / (1024**3),  # GB
            "available": disk.free / (1024**3),  # GB
        }

        system_info = SystemInfoModel(
            project_id=self.project_id,
            trace_id=self.trace_id,
            os_name=os_name,
            os_version=os_version,
            python_version=platform.python_version(),
            cpu_info=cpuinfo.get_cpu_info()["brand_raw"],
            gpu_info=json.dumps(gpu_info) if gpu_info else None,
            disk_info=json.dumps(disk_info),
            memory_total=psutil.virtual_memory().total / (1024**3),  # GB
            installed_packages=json.dumps(
                {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            ),
        )

        with self.Session() as session:
            session.add(system_info)
            session.commit()

            self.trace_data["system_info"] = {
                "os_name": system_info.os_name,
                "os_version": system_info.os_version,
                "python_version": system_info.python_version,
                "cpu_info": system_info.cpu_info,
                "gpu_info": (
                    json.loads(system_info.gpu_info) if system_info.gpu_info else None
                ),
                "disk_info": json.loads(system_info.disk_info),
                "memory_total": system_info.memory_total,
                "installed_packages": json.loads(system_info.installed_packages),
            }

    def _save_to_json(self, log_file_path):
        def default_converter(o):
            if isinstance(o, datetime):
                return o.isoformat()
            elif isinstance(o, timedelta):
                return str(o)
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            json.dump(self.trace_data, f, indent=2, default=default_converter)

    def _log_error(
        self, error: Exception, call_type: str, call_name: str, call_id: int = None
    ):
        error_info = {
            "type": call_type,
            "name": call_name,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(),
        }
        self.trace_data.setdefault("errors", []).append(error_info)
        print(f"Error in {call_type} '{call_name}': {error}")

        # Get current agent, tool, or LLM call IDs
        agent_id = self.current_agent_id.get()
        tool_call_id = None
        llm_call_id = None

        if call_type == "agent":
            agent_id = call_id
        elif call_type == "tool":
            tool_call_id = call_id
        elif call_type == "llm":
            llm_call_id = call_id

        # Save error to the database
        error_model = ErrorModel(
            project_id=self.project_id,
            trace_id=self.trace_id,
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            llm_call_id=llm_call_id,
            error_type=call_type,
            error_message=f"{call_name}: {str(error)}",
            timestamp=error_info["timestamp"],
        )
        with self.Session() as session:
            session.add(error_model)
            session.commit()
