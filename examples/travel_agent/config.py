import sys 

sys.path.append(".")

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers import Tracer
import uuid


def initialize_catalyst():
    return RagaAICatalyst(
        access_key="jqa9FHN8B313H5mQS14X",
        secret_key="BTzSSKKzFbX8YKoHmIMNpcegsKXF9yRhnYFvyMAF",
        # base_url="https://llm-dev5.ragaai.ai/api",
    )

def initialize_tracer():
    trace_name = f"travel_agent_{uuid.uuid4().hex[:8]}"
    user_detail = {
        "project_name": "Alteryx-Sanity-Check",
        "dataset_name": "testing-dataset1",
        # "project_id": "test-project-1",
        # "trace_name": trace_name,
        # "trace_user_detail": {
        #     "user_id": "test-user",
        #     "session_id": str(uuid.uuid4()),
        # },
        # "interval_time": 2,
    }
    
    return Tracer(
        project_name=user_detail["project_name"],
        dataset_name=user_detail["dataset_name"],
        # trace_name=trace_name,
        tracer_type="Agentic",
        # auto_instrumentation={
        #     'llm': True,
        #     'tool': True,
        #     'agent': True,
        #     'user_interaction': True,
        #     'file_io': True,
        #     'network': True,
        #     'custom': True
        # }
    )
