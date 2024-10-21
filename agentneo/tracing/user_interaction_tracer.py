import builtins
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime

from ..data import UserInteractionModel


class UserInteractionTracer:
    def __init__(self, tracer):
        self.tracer = tracer
        self.original_input = builtins.input
        self.original_print = builtins.print

    def input(self, prompt=""):
        content = prompt
        user_input = self.original_input(prompt)
        self._log_interaction("input", user_input)
        return user_input

    def print(self, *args, **kwargs):
        content = " ".join(str(arg) for arg in args)
        self._log_interaction("output", content)
        self.original_print(*args, **kwargs)

    def _log_interaction(self, interaction_type, content):
        agent_id = self.tracer.current_agent_id.get()
        user_interaction = UserInteractionModel(
            project_id=self.tracer.project_id,
            trace_id=self.tracer.trace_id,
            agent_id=agent_id,
            interaction_type=interaction_type,
            content=content,
            timestamp=datetime.now(),
        )
        with self.tracer.Session() as session:
            session.add(user_interaction)
            session.commit()

        # Also add to trace data
        self.tracer.trace_data.setdefault("user_interactions", []).append(
            {
                "interaction_type": interaction_type,
                "content": content,
                "timestamp": datetime.now(),
                "agent_id": agent_id,
            }
        )

    @contextmanager
    def capture(self):
        builtins.input = self.input
        builtins.print = self.print
        try:
            yield
        finally:
            builtins.input = self.original_input
            builtins.print = self.original_print

    @asynccontextmanager
    async def async_capture(self):
        builtins.input = self.input
        builtins.print = self.print
        try:
            yield
        finally:
            builtins.input = self.original_input
            builtins.print = self.original_print
