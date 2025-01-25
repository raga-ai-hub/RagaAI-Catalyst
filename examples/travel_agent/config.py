import sys 
import os
from dotenv import load_dotenv

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
import uuid


def initialize_tracing():
    catalyst = RagaAICatalyst(
        access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
        # base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
    )

    tracer = Tracer(
        project_name="Trace_testing",
        dataset_name="travel_agent_dataset",
        tracer_type="Agentic",
    )

    init_tracing(catalyst=catalyst, tracer=tracer)
    return tracer
