import builtins
from datetime import datetime
import contextvars
import inspect
import uuid

class UserInteractionTracer:
    def __init__(self, *args, **kwargs):
        self.project_id = contextvars.ContextVar("project_id", default=None)
        self.trace_id = contextvars.ContextVar("trace_id", default=None)
        self.tracer = contextvars.ContextVar("tracer", default=None)
        self.component_id = contextvars.ContextVar("component_id", default=None)
        self.original_input = builtins.input
        self.original_print = builtins.print
        self.interactions = []

    def traced_input(self, prompt=""):
        # Get caller information
        if prompt:
            self.traced_print(prompt, end="")
        try:
            content = self.original_input()
        except EOFError:
            content = ""  # Return empty string on EOF
            
        self.interactions.append({
            "id": str(uuid.uuid4()),
            "interaction_type": "input",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        return content

    def traced_print(self, *args, **kwargs):
        content = " ".join(str(arg) for arg in args)
        
        self.interactions.append({
            "id": str(uuid.uuid4()),
            "interaction_type": "output",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        return self.original_print(*args, **kwargs)
