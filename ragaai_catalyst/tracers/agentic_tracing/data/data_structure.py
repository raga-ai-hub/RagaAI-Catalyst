from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid

@dataclass
class OSInfo:
    name: str
    version: str
    platform: str
    kernel_version: str

@dataclass
class EnvironmentInfo:
    name: str
    version: str
    packages: List[str]
    env_path: str
    command_to_run: str

@dataclass
class SystemInfo:
    id: str
    os: OSInfo
    environment: EnvironmentInfo
    source_code: str

@dataclass
class ResourceInfo:
    name: str
    cores: int
    threads: int

@dataclass
class CPUResource:
    info: ResourceInfo
    interval: str
    values: List[float]

@dataclass
class MemoryInfo:
    total: float
    free: float

@dataclass
class MemoryResource:
    info: MemoryInfo
    interval: str
    values: List[float]

@dataclass
class DiskInfo:
    total: float
    free: float

@dataclass
class DiskResource:
    info: DiskInfo
    interval: str
    read: List[float]
    write: List[float]

@dataclass
class NetworkInfo:
    upload_speed: float
    download_speed: float

@dataclass
class NetworkResource:
    info: NetworkInfo
    interval: str
    uploads: List[float]
    downloads: List[float]

@dataclass
class Resources:
    cpu: CPUResource
    memory: MemoryResource
    disk: DiskResource
    network: NetworkResource

@dataclass
class Metadata:
    cost: Dict[str, Any]
    tokens: Dict[str, Any]
    system_info: SystemInfo
    resources: Resources

@dataclass
class NetworkCall:
    url: str
    method: str
    status_code: int
    response_time: float
    bytes_sent: int
    bytes_received: int
    protocol: str
    connection_id: str
    parent_id: str
    request: Dict[str, Any]
    response: Dict[str, Any]

class Interaction:
    def __init__(self, id, type: str, content: str, timestamp: str):
        self.id = id
        self.type = type
        self.content = content
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "id": self.id,
            "interaction_type": self.type,
            "content": self.content,
            "timestamp": self.timestamp
        }

@dataclass
class Error:
    code: int
    type: str
    message: str
    details: Dict[str, Any]

@dataclass
class LLMParameters:
    temperature: float
    top_p: float
    max_tokens: int

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Cost:
    prompt_cost: float
    completion_cost: float
    total_cost: float

@dataclass
class LLMInfo:
    model: str
    parameters: LLMParameters
    token_usage: TokenUsage
    cost: Cost

@dataclass
class AgentInfo:
    agent_type: str
    version: str
    capabilities: List[str]

@dataclass
class ToolInfo:
    tool_type: str
    version: str
    memory_used: int

@dataclass
class LLMCall:
    name: str
    model_name: str
    input_prompt: str
    output_response: str
    tool_call: Dict
    token_usage: Dict[str, int]
    cost: Dict[str, float]
    start_time: float = field(default=0)
    end_time: float = field(default=0)
    duration: float = field(default=0)

class Component:
    def __init__(
        self,
        id: str,
        hash_id: str,
        source_hash_id: str,
        type: str,
        name: str,
        start_time: str,
        end_time: str,
        parent_id: int,
        info: Dict[str, Any],
        extra_info: Optional[Dict[str, Any]] = None,
        data: Dict[str, Any]={},
        metadata: Optional[Dict[str, Any]] = None, 
        metrics: Optional[List[Dict[str, Any]]] = None, 
        feedback: Optional[Any] = None,
        network_calls: Optional[List[NetworkCall]] = None,
        interactions: Optional[List[Union[Interaction, Dict]]] = None,
        error: Optional[Dict[str, Any]] = None):

        self.id = id
        self.hash_id = hash_id
        self.source_hash_id = source_hash_id
        self.type = type
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.parent_id = parent_id
        self.info = info
        self.extra_info = extra_info
        self.data = data
        self.metadata = metadata or {}
        self.metrics = metrics or []
        self.feedback = feedback
        self.network_calls = network_calls or []
        self.interactions = []
        self.error = error
        if interactions:
            for interaction in interactions:
                if isinstance(interaction, dict):
                    self.interactions.append(
                        Interaction(
                            id=interaction.get("id", str(uuid.uuid4())),
                            type=interaction.get("interaction_type", ""),
                            content=str(interaction.get("content", "")),
                            timestamp=interaction.get("timestamp", datetime.now().astimezone().isoformat())
                        )
                    )
                else:
                    self.interactions.append(interaction)

    def to_dict(self):
        return {
            "id": self.id,
            "hash_id": self.hash_id,
            "source_hash_id": self.source_hash_id,
            "type": self.type,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "parent_id": self.parent_id,
            "info": self.info,
            "extra_info": self.extra_info,
            "error": self.error,
            "data": self.data,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "feedback": self.feedback,
            "network_calls": [call.to_dict() if hasattr(call, 'to_dict') else call for call in self.network_calls],
            "interactions": self.interactions
        }

class LLMComponent(Component):
    def __init__(self, id: str, hash_id: str, source_hash_id: str, type: str, name: str, start_time: str, end_time: str, parent_id: int, info: Dict[str, Any], extra_info: Optional[Dict[str, Any]] = None, data: Dict[str, Any]={}, metadata: Optional[Dict[str, Any]] = None, metrics: Optional[List[Dict[str, Any]]] = None, feedback: Optional[Any] = None, network_calls: Optional[List[NetworkCall]] = None, interactions: Optional[List[Union[Interaction, Dict]]] = None, error: Optional[Dict[str, Any]] = None):
        super().__init__(id, hash_id, source_hash_id, type, name, start_time, end_time, parent_id, info, extra_info, data, metadata, metrics, feedback, network_calls, interactions, error)

class AgentComponent(Component):
    def __init__(self, id: str, hash_id: str, source_hash_id: str, type: str, name: str, start_time: str, end_time: str, parent_id: int, info: Dict[str, Any], extra_info: Optional[Dict[str, Any]] = None, data: Dict[str, Any]={}, metadata: Optional[Dict[str, Any]] = None, metrics: Optional[List[Dict[str, Any]]] = None, feedback: Optional[Any] = None, network_calls: Optional[List[NetworkCall]] = None, interactions: Optional[List[Union[Interaction, Dict]]] = None, error: Optional[Dict[str, Any]] = None):
        super().__init__(id, hash_id, source_hash_id, type, name, start_time, end_time, parent_id, info, extra_info, data, metadata, metrics, feedback, network_calls, interactions, error)

class ToolComponent(Component):
    def __init__(self, id: str, hash_id: str, source_hash_id: str, type: str, name: str, start_time: str, end_time: str, parent_id: int, info: Dict[str, Any], extra_info: Optional[Dict[str, Any]] = None, data: Dict[str, Any]={}, metadata: Optional[Dict[str, Any]] = None, metrics: Optional[List[Dict[str, Any]]] = None, feedback: Optional[Any] = None, network_calls: Optional[List[NetworkCall]] = None, interactions: Optional[List[Union[Interaction, Dict]]] = None, error: Optional[Dict[str, Any]] = None):
        super().__init__(id, hash_id, source_hash_id, type, name, start_time, end_time, parent_id, info, extra_info, data, metadata, metrics, feedback, network_calls, interactions, error)


@dataclass
class ComponentInfo:
    tool_type: Optional[str] = None
    agent_type: Optional[str] = None
    version: str = ""
    capabilities: Optional[List[str]] = None
    memory_used: Optional[int] = None
    model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, int]] = None
    cost: Optional[Dict[str, float]] = None

class Trace:
    def __init__(self, id: str, trace_name: str, project_name: str, start_time: str, end_time: str, metadata: Optional[Metadata] = None, data: Optional[List[Dict[str, Any]]] = None, replays: Optional[Dict[str, Any]] = None, metrics: Optional[List[Dict[str, Any]]] = None):
        self.id = id
        self.trace_name = trace_name
        self.project_name = project_name
        self.start_time = start_time
        self.end_time = end_time
        self.metadata = metadata or Metadata()
        self.data = data or []
        self.replays = replays
        self.metrics = metrics or []

    def to_dict(self):
        return {
            "id": self.id,
            "trace_name": self.trace_name,
            "project_name": self.project_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "data": self.data,
            "replays": self.replays,
            "metrics": self.metrics
        }