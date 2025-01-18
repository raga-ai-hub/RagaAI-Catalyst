import contextvars
from typing import Optional, Dict

from .base import BaseTracer
from .llm_tracer import LLMTracerMixin
from .tool_tracer import ToolTracerMixin
from .agent_tracer import AgentTracerMixin
from .tool import Tool
from .network_tracer import NetworkTracer


class Tracer(LLMTracerMixin, ToolTracerMixin, AgentTracerMixin, BaseTracer):
    def __init__(
        self,
        session,
        auto_instrument_llm: bool = True,
    ):
        super().__init__(session)
        self.auto_instrument_llm = auto_instrument_llm
        self.tools: Dict[str, Tool] = {}
        self.call_depth = contextvars.ContextVar("call_depth", default=0)
        self.network_tracer = NetworkTracer()
        self.is_active = False  # Add tracking flag

    def start(self):
        # Start base tracer
        super().start()
        self.is_active = True
        # Instrument calls from mixins
        if self.auto_instrument_llm:
            self.instrument_llm_calls()

    def stop(self):
        # Unpatch methods from mixins
        if self.is_active:  # Only unpatch if currently active
            self.unpatch_llm_calls()
            self.is_active = False

        # Stop base tracer
        super().stop()

    # If you need an unpatch_methods method
    def unpatch_methods(self):
        # Unpatch methods from all mixins
        self.unpatch_llm_calls()
