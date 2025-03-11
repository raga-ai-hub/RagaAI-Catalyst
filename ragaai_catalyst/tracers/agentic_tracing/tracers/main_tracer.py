import contextvars
from typing import Optional, Dict
import json
from datetime import datetime
import uuid
import os
import builtins
from pathlib import Path
import logging

from .base import BaseTracer
from .llm_tracer import LLMTracerMixin
from .tool_tracer import ToolTracerMixin
from .agent_tracer import AgentTracerMixin
from .network_tracer import NetworkTracer
from .user_interaction_tracer import UserInteractionTracer
from .custom_tracer import CustomTracerMixin
from ..utils.span_attributes import SpanAttributes

from ..data.data_structure import (
    Trace,
    Metadata,
    SystemInfo,
    OSInfo,
    EnvironmentInfo,
    Resources,
    CPUResource,
    MemoryResource,
    DiskResource,
    NetworkResource,
    ResourceInfo,
    MemoryInfo,
    DiskInfo,
    NetworkInfo,
    Component,
    LLMComponent,
    AgentComponent,
    ToolComponent,
    NetworkCall,
    Interaction,
    Error,
)

from ....ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.upload_traces import UploadTraces


class AgenticTracing(
    BaseTracer, LLMTracerMixin, ToolTracerMixin, AgentTracerMixin, CustomTracerMixin
):
    def __init__(self, user_detail, auto_instrumentation=None):
        # Initialize all parent classes
        self.user_interaction_tracer = UserInteractionTracer()
        LLMTracerMixin.__init__(self)
        ToolTracerMixin.__init__(self)
        AgentTracerMixin.__init__(self)
        CustomTracerMixin.__init__(self)

        self.project_name = user_detail["project_name"]
        self.project_id = user_detail["project_id"]
        self.trace_user_detail = user_detail["trace_user_detail"]
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 10
        
        # Add warning flag
        self._warning_shown = False

        BaseTracer.__init__(self, user_detail)

        self.tools: Dict[str, Tool] = {}
        self.call_depth = contextvars.ContextVar("call_depth", default=0)
        self.current_component_id = contextvars.ContextVar(
            "current_component_id", default=None
        )
        self.network_tracer = NetworkTracer()

        # Handle auto_instrumentation
        if auto_instrumentation is None:
            # Default behavior: everything enabled
            self.is_active = True
            self.auto_instrument_llm = True
            self.auto_instrument_tool = True
            self.auto_instrument_agent = True
            self.auto_instrument_user_interaction = True
            self.auto_instrument_file_io = True
            self.auto_instrument_network = True
            self.auto_instrument_custom = True
        else:
            # Set global active state
            self.is_active = True

            # Set individual components
            if isinstance(auto_instrumentation, dict):
                self.auto_instrument_llm = auto_instrumentation.get("llm", True)
                self.auto_instrument_tool = auto_instrumentation.get("tool", True)
                self.auto_instrument_agent = auto_instrumentation.get("agent", True)
                self.auto_instrument_user_interaction = auto_instrumentation.get(
                    "user_interaction", True
                )
                self.auto_instrument_file_io = auto_instrumentation.get(
                    "file_io", True
                )
                self.auto_instrument_network = auto_instrumentation.get(
                    "network", True
                )
                self.auto_instrument_custom = auto_instrumentation.get("custom", True)
            else:
                # If boolean provided, apply to all components
                self.auto_instrument_llm = bool(auto_instrumentation)
                self.auto_instrument_tool = bool(auto_instrumentation)
                self.auto_instrument_agent = bool(auto_instrumentation)
                self.auto_instrument_user_interaction = bool(auto_instrumentation)
                self.auto_instrument_file_io = bool(auto_instrumentation)
                self.auto_instrument_network = bool(auto_instrumentation)
                self.auto_instrument_custom = bool(auto_instrumentation)

        self.current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
        self.agent_children = contextvars.ContextVar("agent_children", default=[])
        self.component_network_calls = {}  # Store network calls per component
        self.component_user_interaction = {}


    def start_component(self, component_id: str):
        """Start tracking network calls for a component"""
        self.component_network_calls[component_id] = []
        self.network_tracer.network_calls = []  # Reset network calls
        self.current_component_id.set(component_id)
        self.user_interaction_tracer.component_id.set(component_id)

    def end_component(self, component_id: str):
        """End tracking network calls for a component"""
        self.component_network_calls[component_id] = (
            self.network_tracer.network_calls.copy()
        )
        self.network_tracer.network_calls = []  # Reset for next component

        # Store user interactions for the component
        for interaction in self.user_interaction_tracer.interactions:
            interaction_component_id = interaction.get("component_id")
            if interaction_component_id not in self.component_user_interaction:
                self.component_user_interaction[interaction_component_id] = []
            if interaction not in self.component_user_interaction[interaction_component_id]:
                self.component_user_interaction[interaction_component_id].append(interaction)
                
        # Only reset component_id if it matches the current one
        # This ensures we don't reset a parent's component_id when a child component ends
        if self.current_component_id.get() == component_id:
            # Get the parent agent's component_id if it exists
            parent_agent_id = self.current_agent_id.get()
            # If there's a parent agent, set the component_id back to the parent's
            if parent_agent_id:
                self.current_component_id.set(parent_agent_id)
                self.user_interaction_tracer.component_id.set(parent_agent_id)
            else:
                # Only reset to None if there's no parent
                self.current_component_id.set(None)
                self.user_interaction_tracer.component_id.set(None)

    def start(self):
        """Start tracing"""
        self.is_active = True

        # Setup user interaction tracing
        self.user_interaction_tracer.project_id.set(self.project_id)
        self.user_interaction_tracer.trace_id.set(self.trace_id)
        self.user_interaction_tracer.tracer = self
        self.user_interaction_tracer.component_id.set(self.current_component_id.get())

        # Start base tracer (includes system info and resource monitoring)
        super().start()

        # Activate network tracing
        self.network_tracer.activate_patches()

        # take care of the auto instrumentation
        if self.auto_instrument_user_interaction:
            ToolTracerMixin.instrument_user_interaction_calls(self)
            LLMTracerMixin.instrument_user_interaction_calls(self)
            AgentTracerMixin.instrument_user_interaction_calls(self)
            CustomTracerMixin.instrument_user_interaction_calls(self)
            builtins.print = self.user_interaction_tracer.traced_print
            builtins.input = self.user_interaction_tracer.traced_input

        if self.auto_instrument_network:
            ToolTracerMixin.instrument_network_calls(self)
            LLMTracerMixin.instrument_network_calls(self)
            AgentTracerMixin.instrument_network_calls(self)
            CustomTracerMixin.instrument_network_calls(self)

        if self.auto_instrument_file_io:
            ToolTracerMixin.instrument_file_io_calls(self)
            LLMTracerMixin.instrument_file_io_calls(self)
            AgentTracerMixin.instrument_file_io_calls(self)
            CustomTracerMixin.instrument_file_io_calls(self)
            builtins.open = self.user_interaction_tracer.traced_open
            
        if self.auto_instrument_llm:
            self.instrument_llm_calls()

        if self.auto_instrument_tool:
            self.instrument_tool_calls()

        if self.auto_instrument_agent:
            self.instrument_agent_calls()

        if self.auto_instrument_custom:
            self.instrument_custom_calls()

    def stop(self):
        """Stop tracing and save results"""
        if self.is_active:
            # Restore original print and input functions
            builtins.print = self.user_interaction_tracer.original_print
            builtins.input = self.user_interaction_tracer.original_input
            builtins.open = self.user_interaction_tracer.original_open

            # Calculate final metrics before stopping
            self._calculate_final_metrics()

            # Deactivate network tracing
            self.network_tracer.deactivate_patches()

            # Clear visited metrics when stopping trace
            self.visited_metrics.clear()

            # Stop base tracer (includes saving to file)
            super().stop()

            # Cleanup
            self.unpatch_llm_calls()
            self.user_interaction_tracer.interactions = []  # Clear interactions list
            self.is_active = False

    def _calculate_final_metrics(self):
        """Calculate total cost and tokens from all components"""
        total_cost = 0.0
        total_tokens = 0

        processed_components = set()

        def process_component(component):
            nonlocal total_cost, total_tokens
            # Convert component to dict if it's an object
            comp_dict = (
                component.__dict__ if hasattr(component, "__dict__") else component
            )

            comp_id = comp_dict.get("id") or comp_dict.get("component_id")
            if comp_id in processed_components:
                return  # Skip if already processed
            processed_components.add(comp_id)

            if comp_dict.get("type") == "llm":
                info = comp_dict.get("info", {})
                if isinstance(info, dict):
                    # Extract cost
                    cost_info = info.get("cost", {})
                    if isinstance(cost_info, dict):
                        total_cost += cost_info.get("total_cost", 0)

                    # Extract tokens
                    token_info = info.get("tokens", {})
                    if isinstance(token_info, dict):
                        total_tokens += token_info.get("total_tokens", 0)
                    else:
                        token_info = info.get("token_usage", {})
                        if isinstance(token_info, dict):
                            total_tokens += token_info.get("total_tokens", 0)

            # Process children if they exist
            data = comp_dict.get("data", {})
            if isinstance(data, dict):
                children = data.get("children", [])
                if children:
                    for child in children:
                        process_component(child)

        # Process all root components
        for component in self.components:
            process_component(component)

        # Update metadata in trace
        if hasattr(self, "trace"):
            if isinstance(self.trace.metadata, dict):
                self.trace.metadata["total_cost"] = total_cost
                self.trace.metadata["total_tokens"] = total_tokens
            else:
                self.trace.metadata.total_cost = total_cost
                self.trace.metadata.total_tokens = total_tokens

    def add_component(self, component_data: dict, is_error: bool = False):
        """Add a component to the trace data"""
        # Convert dict to appropriate Component type
        filtered_data = {
            k: v
            for k, v in component_data.items()
            if k
            in [
                "id",
                "hash_id",
                "source_hash_id",
                "type",
                "name",
                "start_time",
                "end_time",
                "parent_id",
                "info",
                "extra_info",
                "data",
                "metadata",
                "metrics",
                "feedback",
                "network_calls",
                "interactions",
                "error",
            ]
        }

        if component_data == None or component_data == {} or component_data.get("type", None) == None:
            # Only show warning if it hasn't been shown before
            if not self._warning_shown:
                import toml
                import os
                from pathlib import Path

                # Load supported LLM calls from TOML file
                current_dir = Path(__file__).parent
                toml_path = current_dir / "../utils/supported_llm_provider.toml"
                try:
                    with open(toml_path, "r") as f:
                        config = toml.load(f)
                        supported_calls = ", ".join(config["supported_llm_calls"])
                except Exception as e:
                    supported_calls = "Error loading supported LLM calls"

                # ANSI escape codes for colors and formatting
                RED = "\033[91m"
                BOLD = "\033[1m"
                RESET = "\033[0m"
                BIG = "\033[1;2m"  # Makes text slightly larger in supported terminals

                warning_msg = f"""{RED}{BOLD}{BIG}
╔════════════════════════ COMPONENT DATA INCOMPLETE ════════════════════════╗
║                                                                          ║
║  Please ensure these requirements:                                       ║
║  ✗ trace_llm decorator must have a stand alone llm call                 ║
║  ✗ trace_tool decorator must be a stand alone tool/function call        ║
║  ✗ trace_agent decorator can have multiple/nested llm/tool/agent calls  ║
║                                                                          ║
║  Supported LLM calls:                                                    ║
║  {supported_calls}                                                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
{RESET}"""
                # Use logger.warning for the message
                logging.warning(warning_msg)
                self._warning_shown = True
            return

        if component_data["type"] == "llm":
            component = LLMComponent(**filtered_data)
        elif component_data["type"] == "agent":
            component = AgentComponent(**filtered_data)
        elif component_data["type"] == "tool":
            component = ToolComponent(**filtered_data)
        else:
            component = Component(**component_data)

        # Check if there's an active agent context
        current_agent_id = self.current_agent_id.get()
        if current_agent_id and component_data["type"] in ["llm", "tool", "custom"]:
            # Add this component as a child of the current agent
            current_children = self.agent_children.get()
            current_children.append(component_data)
            self.agent_children.set(current_children)
        else:
            # Add component to the main trace
            super().add_component(component)

        # Handle error case
        if is_error and not self.current_agent_id.get():
            self.stop()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        self.stop()
