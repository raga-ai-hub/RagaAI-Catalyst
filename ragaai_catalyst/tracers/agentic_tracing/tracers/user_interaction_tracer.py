import builtins
from datetime import datetime
import contextvars
import inspect
import uuid
from typing import Optional, Any

class TracedFile:
    def __init__(self, file_obj, file_path: str, tracer):
        self._file = file_obj
        self._file_path = file_path
        self._tracer = tracer

    def write(self, content: str) -> int:
        self._tracer.trace_file_operation("write", self._file_path, content=content)
        return self._file.write(content)

    def read(self, size: Optional[int] = None) -> str:
        content = self._file.read() if size is None else self._file.read(size)
        self._tracer.trace_file_operation("read", self._file_path, content=content)
        return content

    def close(self) -> None:
        return self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file, name)

class UserInteractionTracer:
    def __init__(self, *args, **kwargs):
        self.project_id = contextvars.ContextVar("project_id", default=None)
        self.trace_id = contextvars.ContextVar("trace_id", default=None)
        self.tracer = contextvars.ContextVar("tracer", default=None)
        self.component_id = contextvars.ContextVar("component_id", default=None)
        self.original_input = builtins.input
        self.original_print = builtins.print
        self.original_open = builtins.open
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
            "component_id": self.component_id.get(),
            "interaction_type": "input",
            "content": content,
            "timestamp": datetime.now().astimezone().isoformat()
        })
        return content

    def traced_print(self, *args, **kwargs):
        content = " ".join(str(arg) for arg in args)
        
        self.interactions.append({
            "id": str(uuid.uuid4()),
            "component_id": self.component_id.get(),
            "interaction_type": "output",
            "content": content,
            "timestamp": datetime.now().astimezone().isoformat()
        })
        return self.original_print(*args, **kwargs)

    def traced_open(self, file: str, mode: str = 'r', *args, **kwargs):
        # Skip tracing for system and virtual environment paths
        system_paths = [
            'site-packages',
            'dist-packages',
            '/proc/',
            '/sys/',
            '/var/lib/',
            '/usr/lib/',
            '/System/Library'
        ]
        
        file_str = str(file)
        if any(path in file_str for path in system_paths):
            return self.original_open(file, mode, *args, **kwargs)
            
        file_obj = self.original_open(file, mode, *args, **kwargs)
        return TracedFile(file_obj, file, self)

    def trace_file_operation(self, operation: str, file_path: str, **kwargs):
        interaction_type = f"file_{operation}"
        
        # Check for existing interaction with same file_path and operation
        for existing in reversed(self.interactions):
            if (existing.get("file_path") == file_path and 
                existing.get("interaction_type") == interaction_type):
                # Merge content if it exists
                if "content" in kwargs and "content" in existing:
                    existing["content"] += kwargs["content"]
                    return
                break
        
        # If no matching interaction found or couldn't merge, create new one
        interaction = {
            "id": str(uuid.uuid4()),
            "component_id": self.component_id.get(),
            "interaction_type": interaction_type,
            "file_path": file_path,
            "timestamp": datetime.now().astimezone().isoformat()
        }
        interaction.update(kwargs)
        self.interactions.append(interaction)

    def __enter__(self):
        builtins.input = self.traced_input
        builtins.print = self.traced_print
        builtins.open = self.traced_open
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.input = self.original_input
        builtins.print = self.original_print
        builtins.open = self.original_open
