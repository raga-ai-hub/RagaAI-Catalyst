# dashboard_server.py

import os
import sys
import time
import logging
import threading
from flask import Flask, send_from_directory
from waitress import serve
from flask import request, abort
from flask_cors import CORS
from flask_caching import Cache

from flask import jsonify
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.exc import SQLAlchemyError

from ..utils import get_db_path
from ..data import (
    ProjectInfoModel,
    TraceModel,
    AgentCallModel,
    LLMCallModel,
    ToolCallModel,
    UserInteractionModel,
    SystemInfoModel,
    ErrorModel,
    MetricModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Adjust the path to find the 'ui' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_folder = os.path.join(current_dir, "../ui/dist")

app = Flask(__name__, static_folder=ui_folder)
CORS(app)  # Enable CORS


db_path = get_db_path()
# Setup database connection
engine = create_engine(db_path)
Session = sessionmaker(bind=engine)


cache = Cache(
    app, config={"CACHE_TYPE": "simple", "CACHE_DEFAULT_TIMEOUT": 600}
)  # Cache for 10 minutes


# Remove X-Frame-Options header
@app.after_request
def add_header(response):
    response.headers.pop("X-Frame-Options", None)
    return response


@app.route("/api/projects", methods=["GET"])
def get_projects():
    try:
        with Session() as session:
            projects = session.query(ProjectInfoModel).all()
            return jsonify(
                [
                    {
                        "id": p.id,
                        "project_name": p.project_name,
                        "start_time": p.start_time,
                        "end_time": p.end_time,
                        "duration": p.duration,
                        "total_cost": p.total_cost,
                        "total_tokens": p.total_tokens,
                    }
                    for p in projects
                ]
            )
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):
    try:
        with Session() as session:
            project = (
                session.query(ProjectInfoModel)
                .options(joinedload(ProjectInfoModel.system_info))  
                .get(project_id)
            )
            if project is None:
                return jsonify({"error": "Project not found"}), 404
            
            # Add system_info to the response
            return jsonify({
                "id": project.id,
                "project_name": project.project_name,
                "start_time": project.start_time,
                "end_time": project.end_time,
                "duration": project.duration,
                "total_cost": project.total_cost,
                "total_tokens": project.total_tokens,
                "system_info": {
                    "os_name": project.system_info.os_name if project.system_info else None,
                    "os_version": project.system_info.os_version if project.system_info else None,
                    "python_version": project.system_info.python_version if project.system_info else None,
                    "cpu_info": project.system_info.cpu_info if project.system_info else None,
                    "gpu_info": project.system_info.gpu_info if project.system_info else None,
                    "disk_info": project.system_info.disk_info if project.system_info else None,
                    "memory_total": project.system_info.memory_total if project.system_info else None,
                    "installed_packages": project.system_info.installed_packages if project.system_info else None,
                } if project.system_info else None
            })
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<int:project_id>/traces", methods=["GET"])
def get_project_traces(project_id):
    try:
        with Session() as session:
            traces = (
                session.query(TraceModel)
                .filter_by(project_id=project_id)
                .options(
                    joinedload(TraceModel.agent_calls),
                    joinedload(TraceModel.llm_calls),
                    joinedload(TraceModel.tool_calls),
                    joinedload(TraceModel.user_interactions),
                    joinedload(TraceModel.errors),
                )
                .all()
            )
            return jsonify(
                [
                    {
                        "id": t.id,
                        "start_time": t.start_time,
                        "end_time": t.end_time,
                        "duration": t.duration,
                        "total_agent_calls": len(t.agent_calls),
                        "total_llm_calls": len(t.llm_calls),
                        "total_tool_calls": len(t.tool_calls),
                        "total_user_interactions": len(t.user_interactions),
                        "total_errors": len(t.errors),
                    }
                    for t in traces
                ]
            )
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis_traces/<int:trace_id>", methods=["GET"])
def get_analysis_trace(trace_id):
    try:
        with Session() as session:
            trace = (
                session.query(TraceModel)
                .options(
                    joinedload(TraceModel.llm_calls),
                    joinedload(TraceModel.tool_calls),
                    joinedload(TraceModel.agent_calls),
                    joinedload(TraceModel.user_interactions),
                    joinedload(TraceModel.errors),
                    joinedload(TraceModel.system_info),
                    joinedload(TraceModel.metrics),
                )
                .get(trace_id)
            )
            if trace is None:
                return jsonify({"error": "Trace not found"}), 404
            return jsonify(
                {
                    "id": trace.id,
                    "project_id": trace.project_id,
                    "start_time": trace.start_time,
                    "end_time": trace.end_time,
                    "duration": trace.duration,
                    "llm_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "model": call.model,
                            "input_prompt": call.input_prompt,
                            "output": call.output,
                            "tool_call": call.tool_call,
                            "start_time": call.start_time,
                            "end_time": call.end_time,
                            "duration": call.duration,
                            "token_usage": call.token_usage,
                            "cost": call.cost,
                            "memory_used": call.memory_used,
                        }
                        for call in trace.llm_calls
                    ],
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "input_parameters": call.input_parameters,
                            "output": call.output,
                            "start_time": call.start_time,
                            "end_time": call.end_time,
                            "duration": call.duration,
                            "memory_used": call.memory_used,
                            "network_calls": call.network_calls,
                        }
                        for call in trace.tool_calls
                    ],
                    "agent_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "start_time": call.start_time,
                            "end_time": call.end_time,
                            "llm_call_ids": call.llm_call_ids,
                            "tool_call_ids": call.tool_call_ids,
                            "user_interaction_ids": call.user_interaction_ids,
                        }
                        for call in trace.agent_calls
                    ],
                    "user_interactions": [
                        {
                            "id": interaction.id,
                            "interaction_type": interaction.interaction_type,
                            "content": interaction.content,
                            "timestamp": interaction.timestamp,
                        }
                        for interaction in trace.user_interactions
                    ],
                    "errors": [
                        {
                            "id": error.id,
                            "error_type": error.error_type,
                            "error_message": error.error_message,
                            "timestamp": error.timestamp,
                        }
                        for error in trace.errors
                    ],
                    "system_info": (
                        {
                            "os_name": trace.system_info.os_name,
                            "os_version": trace.system_info.os_version,
                            "python_version": trace.system_info.python_version,
                            "cpu_info": trace.system_info.cpu_info,
                            "gpu_info": trace.system_info.gpu_info,
                            "disk_info": trace.system_info.disk_info,
                            "memory_total": trace.system_info.memory_total,
                            "installed_packages": trace.system_info.installed_packages,
                        }
                        if trace.system_info
                        else None
                    ),
                    "metrics": [
                        {
                            "id": metric.id,
                            "metric_name": metric.metric_name,
                            "score": metric.score,
                            "reason": metric.reason,
                            "result_detail": metric.result_detail,
                            "config": metric.config,
                            "start_time": metric.start_time,
                            "end_time": metric.end_time,
                            "duration": metric.duration,
                            "timestamp": metric.timestamp,
                        }
                        for metric in trace.metrics
                    ],
                }
            )
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500
    
    

@cache.memoize(timeout=600)
@app.route("/api/traces/<int:trace_id>", methods=["GET"])
def get_trace(trace_id):
    start_time = time.time()
    try:
        with Session() as session:
            trace = (
                session.query(TraceModel)
                .options(
                    selectinload(TraceModel.agent_calls).selectinload(
                        AgentCallModel.llm_calls
                    ),
                    selectinload(TraceModel.agent_calls).selectinload(
                        AgentCallModel.tool_calls
                    ),
                    selectinload(TraceModel.agent_calls).selectinload(
                        AgentCallModel.user_interactions
                    ),
                    selectinload(TraceModel.agent_calls).selectinload(
                        AgentCallModel.errors
                    ),
                    selectinload(TraceModel.llm_calls),
                    selectinload(TraceModel.tool_calls),
                    selectinload(TraceModel.user_interactions),
                    selectinload(TraceModel.errors),
                    selectinload(TraceModel.system_info),
                    selectinload(TraceModel.metrics),
                )
                .get(trace_id)
            )
            if trace is None:
                return jsonify({"error": "Trace not found"}), 404

            def format_agent_call(agent_call):
                return {
                    "id": agent_call.id,
                    "name": agent_call.name,
                    "start_time": agent_call.start_time,
                    "end_time": agent_call.end_time,
                    "llm_calls": [
                        format_llm_call(call) for call in agent_call.llm_calls
                    ],
                    "tool_calls": [
                        format_tool_call(call) for call in agent_call.tool_calls
                    ],
                    "user_interactions": [
                        format_user_interaction(ui)
                        for ui in agent_call.user_interactions
                    ],
                    "errors": [format_error(error) for error in agent_call.errors],
                }

            def format_llm_call(call):
                return {
                    "id": call.id,
                    "name": call.name,
                    "model": call.model,
                    "input_prompt": call.input_prompt,
                    "output": call.output,
                    "tool_call": call.tool_call,
                    "start_time": call.start_time,
                    "end_time": call.end_time,
                    "duration": call.duration,
                    "token_usage": call.token_usage,
                    "cost": call.cost,
                    "memory_used": call.memory_used,
                    "errors": [format_error(error) for error in call.errors],
                    # "user_interactions": [
                    #     format_user_interaction(ui) for ui in call.user_interactions
                    # ],
                }

            def format_tool_call(call):
                return {
                    "id": call.id,
                    "name": call.name,
                    "input_parameters": call.input_parameters,
                    "output": call.output,
                    "start_time": call.start_time,
                    "end_time": call.end_time,
                    "duration": call.duration,
                    "memory_used": call.memory_used,
                    "network_calls": call.network_calls,
                    "errors": [format_error(error) for error in call.errors],
                    # "user_interactions": [
                    #     format_user_interaction(ui) for ui in call.user_interactions
                    # ],
                }

            def format_user_interaction(ui):
                return {
                    "id": ui.id,
                    "interaction_type": ui.interaction_type,
                    "content": ui.content,
                    "timestamp": ui.timestamp,
                }

            def format_error(error):
                return {
                    "id": error.id,
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "timestamp": error.timestamp,
                }

            return jsonify(
                {
                    "id": trace.id,
                    "project_id": trace.project_id,
                    "start_time": trace.start_time,
                    "end_time": trace.end_time,
                    "duration": trace.duration,
                    "agent_calls": [
                        format_agent_call(call) for call in trace.agent_calls
                    ],
                    "llm_calls": [
                        format_llm_call(call)
                        for call in trace.llm_calls
                        if call.agent_id is None
                    ],
                    "tool_calls": [
                        format_tool_call(call)
                        for call in trace.tool_calls
                        if call.agent_id is None
                    ],
                    "user_interactions": [
                        format_user_interaction(ui)
                        for ui in trace.user_interactions
                        if ui.agent_id is None
                    ],
                    "errors": [
                        format_error(error)
                        for error in trace.errors
                        if error.agent_id is None
                        and error.tool_call_id is None
                        and error.llm_call_id is None
                    ],
                    "system_info": (
                        {
                            "os_name": trace.system_info.os_name,
                            "os_version": trace.system_info.os_version,
                            "python_version": trace.system_info.python_version,
                            "cpu_info": trace.system_info.cpu_info,
                            "gpu_info": trace.system_info.gpu_info,
                            "disk_info": trace.system_info.disk_info,
                            "memory_total": trace.system_info.memory_total,
                            "installed_packages": trace.system_info.installed_packages,
                        }
                        if trace.system_info
                        else None
                    ),
                }
            )
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"get_trace({trace_id}) took {duration:.2f} seconds")

            return response
    except SQLAlchemyError as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(
            f"get_trace({trace_id}) failed after {duration:.2f} seconds: {str(e)}"
        )
        return jsonify({"error": str(e)}), 500
    


@app.route("/api/projects/<int:project_id>/evaluation", methods=["GET"])
def get_evaluation_data(project_id):
    trace_id = request.args.get('trace_id')
    try:
        with Session() as session:
            # First get all traces for the project
            trace_ids = session.query(TraceModel.id).filter(TraceModel.project_id == project_id)
            
            # Then query metrics for these traces
            query = session.query(MetricModel).filter(MetricModel.trace_id.in_(trace_ids))
            
            if trace_id and trace_id != 'all':
                query = query.filter(MetricModel.trace_id == trace_id)
            
            metrics = query.all()
            
            return jsonify([{
                'trace_id': metric.trace_id,
                'metric_name': metric.metric_name,
                'score': metric.score,
                'reason': metric.reason,
                'result_detail': metric.result_detail,
                'config': metric.config,
                'start_time': metric.start_time,
                'end_time': metric.end_time,
                'duration': metric.duration,
                'timestamp': metric.timestamp
            } for metric in metrics])
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500



@app.route("/health")
def health_check():
    return "OK", 200


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    try:
        cache.clear()
        return jsonify({"message": "Cache cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/port")
def get_port():
    return jsonify({"port": os.environ.get("AGENTNEO_DASHBOARD_PORT", "3000")})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    """Shutdown endpoint that works with Waitress."""
    if request.remote_addr not in ["127.0.0.1", "::1", "localhost"]:
        abort(403, "Shutdown only allowed from localhost")

    def shutdown_server():
        time.sleep(0.5)  # Brief delay to allow response to be sent
        os._exit(0)

    threading.Thread(target=shutdown_server).start()
    return jsonify({"message": "Server shutting down..."}), 200


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard Server")
    parser.add_argument("--port", type=int, default=3000, help="Port number")
    args = parser.parse_args()

    port = args.port
    os.environ["AGENTNEO_DASHBOARD_PORT"] = str(port)

    # Start the server
    logging.info(f"Starting dashboard server on port {port}")
    serve(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
