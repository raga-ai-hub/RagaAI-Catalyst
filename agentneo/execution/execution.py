from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
import json
import ast

from ..data import (
    ProjectInfoModel,
    TraceModel,
    MetricModel,
)

from .metrics import (
    execute_goal_decomposition_efficiency_metric,
    execute_goal_fulfillment_metric,
    execute_tool_correctness_metric,
    execute_tool_call_success_rate_metric,
    execute_tool_selection_accuracy_metric,
    execute_tool_usage_efficiency_metric,
    execute_plan_adaptibility_metric,
)

from datetime import datetime

class Execution:
    def __init__(self, session, trace_id):
        self.user_session = session
        self.project_name = session.project_name
        self.trace_id = trace_id

        # Setup DB
        self.db_path = self.user_session.db_path
        self.engine = create_engine(self.db_path)
        self.session = Session(bind=self.engine)

        self.trace_data = self.get_trace_data()

    def execute(self, metric_list=[], config={}, metadata={}):
        for metric in metric_list:
            start_time = datetime.now()   
            result = self._execute_metric(metric, config, metadata)   
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self._save_metric_result(metric, result, start_time, end_time, duration)

        self.session.commit()
        self.session.close()

    def _execute_metric(self, metric, config, metadata):
        if metric == 'goal_decomposition_efficiency':
            return execute_goal_decomposition_efficiency_metric(
                trace_json=self.trace_data,
                config=config,
                metadata=metadata,
            )
        elif metric == 'goal_fulfillment_rate':
            return execute_goal_fulfillment_metric(
                trace_json=self.trace_data,
                config=config,
            )
        elif metric == 'tool_correctness_metric':
            return execute_tool_correctness_metric(
                trace_json=self.trace_data,
                config=config,
            )
        elif metric == 'tool_call_success_rate_metric':
            return execute_tool_call_success_rate_metric(
                trace_json=self.trace_data,
                config=config,
            )
        else:
            raise ValueError("provided metric name is not supported.")
        

    def _save_metric_result(self, metric, result, start_time, end_time, duration):
        metric_entry = MetricModel(
            trace_id=self.trace_id,
            metric_name=metric,
            score = result['result']['score'],
            reason = result['result']['reason'],
            result_detail = result,
            config = result['config'],
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
        self.session.add(metric_entry)

    def get_results(self):
        results = self.session.query(MetricModel).filter_by(trace_id=self.trace_id).all()
        results_list = []
        for result in results:
            result_dict = {
                'metric_name': result.metric_name,
                'score' : result.score,
                'reason' : result.reason,
                'result_detail': result.result_detail,
                'config' : result.config,
                'start_time': result.start_time.isoformat() if result.start_time else None,
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration': result.duration
            }
            results_list.append(result_dict)
        
        return results_list

    def get_trace_data(self):
        try:
            # Query the trace with the specific ID
            trace = self.session.query(TraceModel).filter_by(id=self.trace_id).one_or_none()

            json_data = []
            if trace:
                # Serialize the trace and write it as JSON
                trace_dict = self.serialize_trace(trace)
                json_data.append(trace_dict)  # Store the trace dictionary
            else:
                raise ValueError(f"No trace found with ID {self.trace_id}")
            
            return json_data[0]

        except:
            raise ValueError("Unable to load the trace data.")


    def serialize_trace(self, trace):
        """Convert a TraceModel object into a dictionary, including all related objects."""
        return {
            'id': trace.id,
            'project_id': trace.project_id,
            'start_time': trace.start_time.isoformat() if trace.start_time else None,
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'duration': trace.duration,
            'project': trace.project.project_name if trace.project else None,
            'system_info': {
                'os_name': trace.system_info.os_name,
                'os_version': trace.system_info.os_version,
                'python_version': trace.system_info.python_version,
                'cpu_info': trace.system_info.cpu_info,
                'gpu_info': trace.system_info.gpu_info,
                'disk_info': self.parse_json_field(trace.system_info.disk_info),
                'memory_total': trace.system_info.memory_total,
                'installed_packages': self.parse_json_field(trace.system_info.installed_packages)
            } if trace.system_info else None,
            'agent_calls': [
                {
                    'id': agent_call.id,
                    'name': agent_call.name,
                    # 'input_parameters': self.parse_json_field(agent_call.input_parameters),
                    # 'output': self.parse_json_field(agent_call.output),
                    'start_time': agent_call.start_time.isoformat() if agent_call.start_time else None,
                    'end_time': agent_call.end_time.isoformat() if agent_call.end_time else None,
                    # 'duration': agent_call.duration,
                    # 'memory_used': agent_call.memory_used
                    'llm_call_ids': self.parse_json_field(agent_call.llm_call_ids),
                    'tool_call_ids': self.parse_json_field(agent_call.tool_call_ids),
                    'user_interaction_ids': self.parse_json_field(agent_call.user_interaction_ids),
                } for agent_call in trace.agent_calls
            ],
            'llm_calls': [
                {
                    'id': llm_call.id,
                    'name': llm_call.name,
                    'model': llm_call.model,
                    'input_prompt': self.parse_json_field(llm_call.input_prompt),
                    'output': self.parse_json_field(llm_call.output),
                    'tool_call': self.parse_json_field(llm_call.tool_call),
                    'start_time': llm_call.start_time.isoformat() if llm_call.start_time else None,
                    'end_time': llm_call.end_time.isoformat() if llm_call.end_time else None,
                    'duration': llm_call.duration,
                    'token_usage': self.parse_json_field(llm_call.token_usage),
                    'cost': self.parse_json_field(llm_call.cost),
                    'memory_used': llm_call.memory_used
                } for llm_call in trace.llm_calls
            ],
            'tool_calls': [
                {
                    'id': tool_call.id,
                    'name': tool_call.name,
                    'input_parameters': self.parse_json_field(tool_call.input_parameters),
                    'output': self.parse_json_field(tool_call.output),
                    'start_time': tool_call.start_time.isoformat() if tool_call.start_time else None,
                    'end_time': tool_call.end_time.isoformat() if tool_call.end_time else None,
                    'duration': tool_call.duration,
                    'memory_used': tool_call.memory_used,
                    'network_calls': self.parse_json_field(tool_call.network_calls)
                } for tool_call in trace.tool_calls
            ],
            'errors': [
                {
                    'id': error.id,
                    'error_type': error.error_type,
                    'error_message': error.error_message,
                    'timestamp': error.timestamp.isoformat() if error.timestamp else None
                } for error in trace.errors
            ],
            'user_interactions': [
                {
                    'id': interaction.id,
                    'interaction_type': interaction.interaction_type,
                    'content': interaction.content,
                    'timestamp': interaction.timestamp.isoformat() if interaction.timestamp else None
                } for interaction in trace.user_interactions
            ]
        }
    

    def parse_json_field(self, field):
        """Parse a JSON string field into a Python object, if necessary."""
        if isinstance(field, str):
            try:
                return json.loads(field)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(field)
                except:
                    return field
        elif isinstance(field, (list, dict)):
            return field
        return field
