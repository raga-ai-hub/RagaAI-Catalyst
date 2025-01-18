from importlib.util import find_spec


class LlamaIndex:
    def __init__(self) -> None:
        if find_spec("llamaindex") is None:
            raise ModuleNotFoundError(
                "Missing `llamaindex` package. Install with `pip install llamaindex`."
            )

    def get(self):
        from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

        return LlamaIndexInstrumentor
