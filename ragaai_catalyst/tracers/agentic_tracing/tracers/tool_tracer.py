import os
import uuid
from datetime import datetime
import psutil
import functools
from typing import Optional, Any, Dict, List
from ..utils.unique_decorator import generate_unique_hash_simple
import contextvars
import asyncio
from ..utils.file_name_tracker import TrackName
from ..utils.span_attributes import SpanAttributes
import logging

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


class ToolTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.current_tool_name = contextvars.ContextVar("tool_name", default=None)
        self.current_tool_id = contextvars.ContextVar("tool_id", default=None)
        self.component_network_calls = {}
        self.component_user_interaction = {}
        self.gt = None

        # add auto_instrument option
        self.auto_instrument_tool = False
        self.auto_instrument_user_interaction = False
        self.auto_instrument_file_io = False
        self.auto_instrument_network = False

    # take care of auto_instrument
    def instrument_tool_calls(self):
        self.auto_instrument_tool = True

    def instrument_user_interaction_calls(self):
        self.auto_instrument_user_interaction = True
        
    def instrument_file_io_calls(self):
        self.auto_instrument_file_io = True

    def instrument_network_calls(self):
        self.auto_instrument_network = True

    def trace_tool(
        self,
        name: str,
        tool_type: str = "generic",
        version: str = "1.0.0",
        tags: List[str] = [],
        metadata: Dict[str, Any] = {},
        metrics: List[Dict[str, Any]] = [],
        feedback: Optional[Any] = None,
    ):
        if name not in self.span_attributes_dict:
            self.span_attributes_dict[name] = SpanAttributes(name)
        if tags:
            self.span(name).add_tags(tags)
        if metadata:
            self.span(name).add_metadata(metadata)
        if metrics:
            if isinstance(metrics, dict):
                metrics = [metrics]
            try:
                for metric in metrics:
                    self.span(name).add_metrics(
                        name=metric["name"],
                        score=metric["score"],
                        reasoning=metric.get("reasoning", ""),
                        cost=metric.get("cost", None),
                        latency=metric.get("latency", None),
                        metadata=metric.get("metadata", {}),
                        config=metric.get("config", {}),
                    )
            except ValueError as e:
                    logger.error(f"Validation Error: {e}")
            except Exception as e:
                logger.error(f"Error adding metric: {e}")
            
        if feedback:
            self.span(name).add_feedback(feedback)

        def decorator(func):
            # Add metadata attribute to the function
            metadata = {
                "name": name,
                "tool_type": tool_type,
                "version": version,
                "is_active": self.is_active,
            }

            # Check if the function is async
            is_async = asyncio.iscoroutinefunction(func)

            @self.file_tracker.trace_decorator
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async_wrapper.metadata = metadata
                self.gt = kwargs.get("gt", None) if kwargs else None
                return await self._trace_tool_execution(
                    func, name, tool_type, version, *args, **kwargs
                )

            @self.file_tracker.trace_decorator
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                sync_wrapper.metadata = metadata
                self.gt = kwargs.get("gt", None) if kwargs else None
                return self._trace_sync_tool_execution(
                    func, name, tool_type, version, *args, **kwargs
                )

            wrapper = async_wrapper if is_async else sync_wrapper
            wrapper.metadata = metadata
            return wrapper

        return decorator

    def _trace_sync_tool_execution(
        self, func, name, tool_type, version, *args, **kwargs
    ):
        """Synchronous version of tool tracing"""
        if not self.is_active:
            return func(*args, **kwargs)

        if not self.auto_instrument_tool:
            return func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash_simple(func)

        # Set current tool name and store the token
        name_token = self.current_tool_name.set(name)
        id_token = self.current_tool_id.set(component_id)

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the tool
            result = func(*args, **kwargs)

            # Calculate resource usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create tool component
            tool_component = self.create_tool_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                tool_type=tool_type,
                version=version,
                memory_used=memory_used,
                start_time=start_time,
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
            )

            self.add_component(tool_component)

            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            # End tracking network calls for this component
            self.end_component(component_id)

            tool_component = self.create_tool_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                tool_type=tool_type,
                version=version,
                memory_used=0,
                start_time=start_time,
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error_component,
            )

            self.add_component(tool_component)

            raise
        finally:
            # Reset the tool name and id context
            if name_token:
                self.current_tool_name.reset(name_token)
            if id_token:
                self.current_tool_id.reset(id_token)

    async def _trace_tool_execution(
        self, func, name, tool_type, version, *args, **kwargs
    ):
        """Asynchronous version of tool tracing"""
        if not self.is_active:
            return await func(*args, **kwargs)

        if not self.auto_instrument_tool:
            return await func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash_simple(func)

        # Set current tool name and store the token
        name_token = self.current_tool_name.set(name)
        id_token = self.current_tool_id.set(component_id)

        self.start_component(component_id)
        try:
            # Execute the tool
            result = await func(*args, **kwargs)

            # Calculate resource usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)
            self.end_component(component_id)

            # Create tool component
            tool_component = self.create_tool_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                tool_type=tool_type,
                version=version,
                start_time=start_time,
                memory_used=memory_used,
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
            )
            self.add_component(tool_component)

            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            tool_component = self.create_tool_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                tool_type=tool_type,
                version=version,
                start_time=start_time,
                memory_used=0,
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error_component,
            )
            self.add_component(tool_component)

            raise
        finally:
            # Reset the tool name and id context
            if name_token:
                self.current_tool_name.reset(name_token)
            if id_token:
                self.current_tool_id.reset(id_token)

    def create_tool_component(self, **kwargs):
        """Create a tool component according to the data structure"""
        network_calls = []
        if self.auto_instrument_network:
            network_calls = self.component_network_calls.get(kwargs["component_id"], [])
        interactions = []
        if self.auto_instrument_user_interaction:
            input_output_interactions = []
            for interaction in self.component_user_interaction.get(kwargs["component_id"], []):
                if interaction["interaction_type"] in ["input", "output"]:
                    input_output_interactions.append(interaction)
            if input_output_interactions!=[]:
                interactions.extend(input_output_interactions) 
        if self.auto_instrument_file_io:
            file_io_interactions = []
            for interaction in self.component_user_interaction.get(kwargs["component_id"], []):
                if interaction["interaction_type"] in ["file_read", "file_write"]:
                    file_io_interactions.append(interaction)
            if file_io_interactions!=[]:
                interactions.extend(file_io_interactions)

        # Get tags, metrics
        name = kwargs["name"]
        # tags
        tags = []
        if name in self.span_attributes_dict:
            tags = self.span_attributes_dict[name].tags or []

        # metrics
        metrics = []
        if name in self.span_attributes_dict:
            raw_metrics = self.span_attributes_dict[name].metrics or []
            for metric in raw_metrics:
                base_metric_name = metric["name"]
                counter = sum(1 for x in self.visited_metrics if x.startswith(base_metric_name))
                metric_name = f'{base_metric_name}_{counter}' if counter > 0 else base_metric_name
                self.visited_metrics.append(metric_name)
                metric["name"] = metric_name  
                metrics.append(metric)

        start_time = kwargs["start_time"]
        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "source_hash_id": None,
            "type": "tool",
            "name": kwargs["name"],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().astimezone().isoformat(),
            "error": kwargs.get("error"),
            "parent_id": self.current_agent_id.get(),
            "info": {
                "tool_type": kwargs["tool_type"],
                "version": kwargs["version"],
                "memory_used": kwargs["memory_used"],
                "tags": tags,
            },
            "data": {
                "input": kwargs["input_data"],
                "output": kwargs["output_data"],
                "memory_used": kwargs["memory_used"],
            },
            "metrics": metrics,
            "network_calls": network_calls,
            "interactions": interactions,
        }

        if self.gt:
            component["data"]["gt"] = self.gt

        # Reset the SpanAttributes context variable
        self.span_attributes_dict[kwargs["name"]] = SpanAttributes(kwargs["name"])

        return component

    def start_component(self, component_id):
        self.component_network_calls[component_id] = []

    def end_component(self, component_id):
        pass

    def _sanitize_input(self, args: tuple, kwargs: dict) -> Dict:
        """Sanitize and format input data"""
        return {
            "args": [
                (
                    str(arg)
                    if not isinstance(arg, (int, float, bool, str, list, dict))
                    else arg
                )
                for arg in args
            ],
            "kwargs": {
                k: (
                    str(v)
                    if not isinstance(v, (int, float, bool, str, list, dict))
                    else v
                )
                for k, v in kwargs.items()
            },
        }

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize and format output data"""
        if isinstance(output, (int, float, bool, str, list, dict)):
            return output
        return str(output)
