from importlib.util import find_spec
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


class OpenAI:
    def __init__(self) -> None:
        if find_spec("openai") is None:
            raise ModuleNotFoundError(
                "Missing `openai` package. Install with `pip install openai`."
            )

    def get(self):
        return OpenAIInstrumentor
