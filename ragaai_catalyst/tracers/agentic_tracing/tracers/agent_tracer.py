import os
import functools
import uuid
from datetime import datetime
import psutil
from typing import Optional, Any, Dict, List
from ..utils.unique_decorator import mydecorator, generate_unique_hash_simple
import contextvars
import asyncio
from ..utils.file_name_tracker import TrackName
from ..utils.span_attributes import SpanAttributes
from .base import BaseTracer
import logging

logger = logging.getLogger(__name__)
logging_level = (
    logger.setLevel(logging.DEBUG)
    if os.getenv("DEBUG")
    else logger.setLevel(logging.INFO)
)


class AgentTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.current_agent_id = contextvars.ContextVar("agent_id", default=None)
        self.current_agent_name = contextvars.ContextVar("agent_name", default=None)
        self.agent_children = contextvars.ContextVar("agent_children", default=[])
        self.component_network_calls = contextvars.ContextVar(
            "component_network_calls", default={}
        )
        self.component_user_interaction = contextvars.ContextVar(
            "component_user_interaction", default={}
        )
        self.version = contextvars.ContextVar("version", default=None)
        self.agent_type = contextvars.ContextVar("agent_type", default="generic")
        self.capabilities = contextvars.ContextVar("capabilities", default=[])
        self.start_time = contextvars.ContextVar("start_time", default=None)
        self.input_data = contextvars.ContextVar("input_data", default=None)
        self.gt = None

        self.span_attributes_dict = {}

        # Add auto instrument flags
        self.auto_instrument_agent = False
        self.auto_instrument_user_interaction = False
        self.auto_instrument_file_io = False
        self.auto_instrument_network = False

    def trace_agent(
            self,
            name: str,
            agent_type: str = None,
            version: str = None,
            capabilities: List[str] = None,
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

        def decorator(target):
            # Check if target is a class
            is_class = isinstance(target, type)
            tracer = self  # Store reference to tracer instance
            top_level_hash_id = generate_unique_hash_simple(
                target
            )  # Generate hash based on the decorated target code
            self.version.set(version)
            self.agent_type.set(agent_type)
            self.capabilities.set(capabilities)

            if is_class:
                # Store original __init__
                original_init = target.__init__

                def wrapped_init(self, *args, **kwargs):
                    gt = kwargs.get("gt") if kwargs else None
                    if gt is not None:
                        span = self.span(name)
                        span.add_gt(gt)

                    if not hasattr(self, '_agent_component_id'):
                        component_id = str(uuid.uuid4())
                        self._agent_component_id = component_id
                        
                        # Get parent agent ID if exists
                        parent_agent_id = tracer.current_agent_id.get()
                        
                        agent_component = tracer.create_agent_component(
                            component_id=component_id,
                            hash_id=top_level_hash_id,
                            name=name,
                            agent_type=agent_type,
                            version=version,
                            capabilities=capabilities or [],
                            start_time=datetime.now().astimezone().isoformat(),
                            memory_used=0,
                            input_data=tracer._sanitize_input(args, kwargs),
                            output_data=None,
                            children=[],
                            parent_id=parent_agent_id,
                        )
                        
                        if not hasattr(tracer, "_agent_components"):
                            tracer._agent_components = {}
                        tracer._agent_components[component_id] = agent_component
                        
                        # For class agents, only add to parent's children if parent exists
                        if parent_agent_id and parent_agent_id in tracer._agent_components:
                            parent_component = tracer._agent_components[parent_agent_id]
                            if not hasattr(parent_component, "children"):
                                parent_component["children"] = []
                            if component_id not in parent_component["children"]:
                                parent_component["children"].append(component_id)
                    
                    token = tracer.current_agent_id.set(self._agent_component_id)
                    try:
                        original_init(self, *args, **kwargs)
                    finally:
                        tracer.current_agent_id.reset(token)

                # Wrap all public methods to track execution
                for attr_name in dir(target):
                    if not attr_name.startswith("_"):
                        attr_value = getattr(target, attr_name)
                        if callable(attr_value):

                            def wrap_method(method):
                                @functools.wraps(method)
                                def wrapped_method(self, *args, **kwargs):
                                    gt = kwargs.get("gt") if kwargs else None
                                    if gt is not None:
                                        span = tracer.span(name)
                                        span.add_gt(gt)
                                    # Set this agent as current during method execution
                                    token = tracer.current_agent_id.set(
                                        self._agent_component_id
                                    )

                                    # Store parent's children before setting new empty list
                                    parent_children = tracer.agent_children.get()
                                    children_token = tracer.agent_children.set([])

                                    try:
                                        start_time = datetime.now().astimezone().isoformat()
                                        result = method(self, *args, **kwargs)

                                        # Update agent component with method result
                                        if hasattr(tracer, "_agent_components"):
                                            component = tracer._agent_components.get(
                                                self._agent_component_id
                                            )
                                            if component:
                                                component["data"]["output"] = (
                                                    tracer._sanitize_output(result)
                                                )
                                                component["data"]["input"] = (
                                                    tracer._sanitize_input(args, kwargs)
                                                )
                                                component["start_time"] = (
                                                    start_time
                                                )

                                                # Get children accumulated during method execution
                                                children = tracer.agent_children.get()
                                                if children:
                                                    if (
                                                            "children"
                                                            not in component["data"]
                                                    ):
                                                        component["data"][
                                                            "children"
                                                        ] = []
                                                    component["data"][
                                                        "children"
                                                    ].extend(children)

                                                    # Add this component as a child to parent's children list
                                                    parent_children.append(component)
                                                    tracer.agent_children.set(
                                                        parent_children
                                                    )
                                        return result
                                    finally:
                                        tracer.current_agent_id.reset(token)
                                        tracer.agent_children.reset(children_token)

                                return wrapped_method

                            setattr(target, attr_name, wrap_method(attr_value))

                # Replace __init__ with wrapped version

                target.__init__ = wrapped_init

                # Wrap all methods to maintain parent-child relationship
                for attr_name, attr_value in target.__dict__.items():
                    if callable(attr_value) and not attr_name.startswith('__'):
                        original_method = attr_value
                        
                        def create_wrapper(method):
                            @self.file_tracker.trace_decorator
                            @functools.wraps(method)
                            def method_wrapper(self, *args, **kwargs):
                                gt = kwargs.get("gt") if kwargs else None
                                if gt is not None:
                                    span = tracer.span(name)
                                    span.add_gt(gt)
                                # Use the class instance's agent ID as parent
                                parent_id = getattr(self, '_agent_component_id', None)
                                if parent_id:
                                    if asyncio.iscoroutinefunction(method):
                                        return tracer._trace_agent_execution(
                                            method.__get__(self, type(self)),
                                            name,
                                            agent_type,
                                            version,
                                            capabilities,
                                            top_level_hash_id,
                                            *args,
                                            **kwargs,
                                        )
                                    else:
                                        return tracer._trace_sync_agent_execution(
                                            method.__get__(self, type(self)),
                                            name,
                                            agent_type,
                                            version,
                                            capabilities,
                                            top_level_hash_id,
                                            *args,
                                            **kwargs,
                                        )
                                else:
                                    return method(self, *args, **kwargs)
                            return method_wrapper
                        
                        setattr(target, attr_name, create_wrapper(original_method))

                return target
            else:
                # For non-class targets (e.g., functions), use existing function wrapping logic
                @functools.wraps(target)
                def wrapper(*args, **kwargs):
                    if asyncio.iscoroutinefunction(target):
                        return tracer._trace_agent_execution(
                            target,
                            name,
                            agent_type,
                            version,
                            capabilities,
                            top_level_hash_id,
                            *args,
                            **kwargs,
                        )
                    else:
                        return tracer._trace_sync_agent_execution(
                            target,
                            name,
                            agent_type,
                            version,
                            capabilities,
                            top_level_hash_id,
                            *args,
                            **kwargs,
                        )
                return wrapper

        return decorator

    def _trace_sync_agent_execution(
        self, func, name, agent_type, version, capabilities, top_level_hash_id, *args, **kwargs
    ):
        """Synchronous version of agent tracing"""
        if not self.is_active:
            return func(*args, **kwargs)

        if not self.auto_instrument_agent:
            return func(*args, **kwargs)

        start_time = datetime.now().astimezone().isoformat()
        self.start_time = start_time
        self.input_data = self._sanitize_input(args, kwargs)
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())

        # Extract ground truth if present
        ground_truth = kwargs.pop("gt", None) if kwargs else None
        if ground_truth is not None:
            span = self.span(name)
            span.add_gt(ground_truth)

        # Get parent agent ID if exists
        parent_agent_id = self.current_agent_id.get()

        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Initialize empty children list for this agent
        parent_children = self.agent_children.get()
        children_token = self.agent_children.set([])

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the agent
            result = func(*args, **kwargs)

            # Calculate resource usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Get children components collected during execution
            children = self.agent_children.get()

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create agent component with children and parent if exists
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=top_level_hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                memory_used=memory_used,
                input_data=self.input_data,
                output_data=self._sanitize_output(result),
                children=children,
                parent_id=parent_agent_id,
            )

            # Store component for updates
            if not hasattr(self, "_agent_components"):
                self._agent_components = {}
            self._agent_components[component_id] = agent_component

            # Only add to hierarchy if this is a root component (no parent)
            # or if the parent explicitly added it as a child
            if parent_agent_id:
                parent_children.append(agent_component)
                self.agent_children.set(parent_children)
            else:
                self.add_component(agent_component)

            return result
        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            # Get children even in case of error
            children = self.agent_children.get()

            # Create error component
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=top_level_hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                memory_used=0,
                input_data=self.input_data,
                output_data=None,
                error=error_component,
                children=children,
                parent_id=parent_agent_id,  # Add parent ID if exists
            )

            # Store component for updates
            if not hasattr(self, "_agent_components"):
                self._agent_components = {}
            self._agent_components[component_id] = agent_component

            # Only add to hierarchy if this is a root component (no parent)
            # or if the parent explicitly added it as a child
            if parent_agent_id:
                parent_children.append(agent_component)
                self.agent_children.set(parent_children)
            else:
                # Only add to root components if no parent
                self.add_component(agent_component, is_error=True)

            raise
        finally:
            self.current_agent_id.reset(agent_token)
            self.current_agent_name.reset(agent_name_token)
            self.agent_children.reset(children_token)

    async def _trace_agent_execution(
            self, func, name, agent_type, version, capabilities, hash_id, *args, **kwargs
    ):
        """Asynchronous version of agent tracing"""
        if not self.is_active:
            return await func(*args, **kwargs)

        if not self.auto_instrument_agent:
            return await func(*args, **kwargs)

        start_time = datetime.now().astimezone().isoformat()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())

        # Extract ground truth if present
        ground_truth = kwargs.pop("gt", None) if kwargs else None
        if ground_truth is not None:
            span = self.span(name)
            span.add_gt(ground_truth)

        # Get parent agent ID if exists
        parent_agent_id = self.current_agent_id.get()

        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Initialize empty children list for this agent
        parent_children = self.agent_children.get()
        children_token = self.agent_children.set([])
        self.start_component(component_id)

        try:
            # Execute the agent
            result = await func(*args, **kwargs)

            # Calculate resource usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Get children components collected during execution
            children = self.agent_children.get()

            self.end_component(component_id)

            # Create agent component with children and parent if exists
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                memory_used=memory_used,
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
                children=children,
                parent_id=parent_agent_id,
            )

            # Store component for updates
            if not hasattr(self, "_agent_components"):
                self._agent_components = {}
            self._agent_components[component_id] = agent_component

            # Only add to hierarchy if this is a root component (no parent)
            # or if the parent explicitly added it as a child
            if parent_agent_id:
                parent_children.append(agent_component)
                self.agent_children.set(parent_children)
            else:
                self.add_component(agent_component)

            return result
        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {},
            }

            # Get children even in case of error
            children = self.agent_children.get()

            # Create error component
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                memory_used=0,
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error_component,
                children=children,
                parent_id=parent_agent_id,  # Add parent ID if exists
            )

            # Store component for updates
            if not hasattr(self, "_agent_components"):
                self._agent_components = {}
            self._agent_components[component_id] = agent_component

            # Only add to hierarchy if this is a root component (no parent)
            # or if the parent explicitly added it as a child
            if parent_agent_id:
                parent_children.append(agent_component)
                self.agent_children.set(parent_children)
            else:
                # Only add to root components if no parent
                self.add_component(agent_component, is_error=True)

            raise
        finally:
            # Reset context variables
            self.current_agent_id.reset(agent_token)
            self.current_agent_name.reset(agent_name_token)
            self.agent_children.reset(children_token)

    def create_agent_component(self, **kwargs):
        """Create an agent component according to the data structure"""
        network_calls = []
        if self.auto_instrument_network:
            network_calls = self.component_network_calls.get(kwargs["component_id"], [])
        interactions = []
        if self.auto_instrument_user_interaction:
            input_output_interactions = []
            for interaction in self.component_user_interaction.get(kwargs["component_id"], []):
                if interaction["interaction_type"] in ["input", "output"]:
                    input_output_interactions.append(interaction)
            interactions.extend(input_output_interactions)
        if self.auto_instrument_file_io:
            file_io_interactions = []
            for interaction in self.component_user_interaction.get(kwargs["component_id"], []):
                if interaction["interaction_type"] in ["file_read", "file_write"]:
                    file_io_interactions.append(interaction)
            interactions.extend(file_io_interactions)

        # Get start time
        start_time = None
        if "start_time" in kwargs:
            start_time = kwargs["start_time"]

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

        # TODO agent_trace execute metric
        formatted_metrics = BaseTracer.get_formatted_metric(self.span_attributes_dict, self.project_id, name)
        if formatted_metrics:
            metrics.extend(formatted_metrics)

        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "source_hash_id": None,
            "type": "agent",
            "name": kwargs["name"],
            "start_time": start_time,
            "end_time": datetime.now().astimezone().isoformat(),
            "error": kwargs.get("error"),
            "parent_id": kwargs.get("parent_id"),
            "info": {
                "agent_type": kwargs["agent_type"],
                "version": kwargs["version"],
                "capabilities": kwargs["capabilities"],
                "memory_used": kwargs["memory_used"],
                "tags": tags,
            },
            "data": {
                "input": kwargs["input_data"],
                "output": kwargs["output_data"],
                "children": kwargs.get("children", []),
            },
            "metrics": metrics,
            "network_calls": network_calls,
            "interactions": interactions,
        }

        if name in self.span_attributes_dict:
            span_gt = self.span_attributes_dict[name].gt
            if span_gt is not None:
                component["data"]["gt"] = span_gt
            span_context = self.span_attributes_dict[name].context
            if span_context:
                component["data"]["context"] = span_context

        # Reset the SpanAttributes context variable
        self.span_attributes_dict[kwargs["name"]] = SpanAttributes(kwargs["name"])

        return component

    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        component_network_calls = self.component_network_calls.get()
        if component_id not in component_network_calls:
            component_network_calls[component_id] = []
        self.component_network_calls.set(component_network_calls)

    def end_component(self, component_id):
        """End tracking network calls for a component"""
        component_network_calls = self.component_network_calls.get()
        if component_id in component_network_calls:
            component_network_calls[component_id] = []
        self.component_network_calls.set(component_network_calls)

    def _sanitize_input(self, args: tuple, kwargs: dict) -> dict:
        """Sanitize and format input data, including handling of nested lists and dictionaries."""

        def sanitize_value(value):
            if isinstance(value, (int, float, bool, str)):
                return value
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, dict):
                return {key: sanitize_value(val) for key, val in value.items()}
            else:
                return str(value)  # Convert non-standard types to string

        return {
            "args": [sanitize_value(arg) for arg in args],
            "kwargs": {key: sanitize_value(val) for key, val in kwargs.items()},
        }

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize and format output data"""
        if isinstance(output, (int, float, bool, str, list, dict)):
            return output
        return str(output)

    def instrument_agent_calls(self):
        self.auto_instrument_agent = True

    def instrument_user_interaction_calls(self):
        self.auto_instrument_user_interaction = True

    def instrument_network_calls(self):
        self.auto_instrument_network = True

    def instrument_file_io_calls(self):
        self.auto_instrument_file_io = True
