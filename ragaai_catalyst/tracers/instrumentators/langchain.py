from importlib.util import find_spec
from opentelemetry.instrumentation.langchain import LangchainInstrumentor


class Langchain:
    def __init__(self) -> None:
        # Check if the necessary part of the 'opentelemetry' package is installed
        if find_spec("opentelemetry.instrumentation.langchain") is None:
            raise ModuleNotFoundError(
                "Missing `opentelemetry-instrumentation-langchain` component. Install with `pip install opentelemetry-instrumentation-langchain`."
            )

    def get(self):
        return LangchainInstrumentor
