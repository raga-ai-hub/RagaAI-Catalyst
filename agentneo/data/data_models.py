from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class UserInteractionModel(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agent_call.id"), nullable=True)
    interaction_type = Column(String, nullable=False)  # 'input' or 'output'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    trace = relationship("TraceModel", back_populates="user_interactions")
    project = relationship("ProjectInfoModel", back_populates="user_interactions")
    agent = relationship("AgentCallModel", back_populates="user_interactions")


class TraceModel(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)

    project = relationship("ProjectInfoModel", back_populates="traces")
    system_info = relationship("SystemInfoModel", uselist=False, back_populates="trace")
    errors = relationship("ErrorModel", back_populates="trace")
    llm_calls = relationship("LLMCallModel", back_populates="trace")
    tool_calls = relationship("ToolCallModel", back_populates="trace")
    agent_calls = relationship("AgentCallModel", back_populates="trace")
    user_interactions = relationship(
        "UserInteractionModel", order_by=UserInteractionModel.id, back_populates="trace"
    )


class ProjectInfoModel(Base):
    __tablename__ = "project_info"

    id = Column(Integer, primary_key=True)
    project_name = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    traces = relationship("TraceModel", back_populates="project")
    user_interactions = relationship(
        "UserInteractionModel",
        order_by=UserInteractionModel.id,
        back_populates="project",
    )


class SystemInfoModel(Base):
    __tablename__ = "system_info"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    os_name = Column(String, nullable=False)
    os_version = Column(String, nullable=False)
    python_version = Column(String, nullable=False)
    cpu_info = Column(String, nullable=False)
    gpu_info = Column(String, nullable=True)
    disk_info = Column(String, nullable=True)
    memory_total = Column(Float, nullable=False)
    installed_packages = Column(String, nullable=False)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)

    trace = relationship("TraceModel", back_populates="system_info")
    project = relationship("ProjectInfoModel", back_populates="system_info")


class ErrorModel(Base):
    __tablename__ = "errors"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agent_call.id"), nullable=True)
    tool_call_id = Column(Integer, ForeignKey("tool_call.id"), nullable=True)
    llm_call_id = Column(Integer, ForeignKey("llm_call.id"), nullable=True)
    error_type = Column(String, nullable=False)  # 'LLM', 'Tool', or 'Agent'
    error_message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    trace = relationship("TraceModel", back_populates="errors")
    project = relationship("ProjectInfoModel", back_populates="errors")
    agent = relationship("AgentCallModel", back_populates="errors")
    tool_call = relationship("ToolCallModel", back_populates="errors")
    llm_call = relationship("LLMCallModel", back_populates="errors")


class LLMCallModel(Base):
    __tablename__ = "llm_call"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agent_call.id"), nullable=True)
    name = Column(String, nullable=False)
    model = Column(String, nullable=True)
    input_prompt = Column(String, nullable=False)
    output = Column(String, nullable=False)
    tool_call = Column(String, nullable=True)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    token_usage = Column(JSON, nullable=False)
    cost = Column(JSON, nullable=False)
    memory_used = Column(Integer, nullable=False)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)

    trace = relationship("TraceModel", back_populates="llm_calls")
    project = relationship("ProjectInfoModel", back_populates="llm_calls")
    agent = relationship("AgentCallModel", back_populates="llm_calls")


class ToolCallModel(Base):
    __tablename__ = "tool_call"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agent_call.id"), nullable=True)
    name = Column(String, nullable=False)
    input_parameters = Column(String, nullable=False)
    output = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    memory_used = Column(Integer, nullable=False)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)
    network_calls = Column(JSON, nullable=True)
    trace = relationship("TraceModel", back_populates="tool_calls")
    project = relationship("ProjectInfoModel", back_populates="tool_calls")
    agent = relationship("AgentCallModel", back_populates="tool_calls")


class AgentCallModel(Base):
    __tablename__ = "agent_call"

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    project_id = Column(Integer, ForeignKey("project_info.id"), nullable=False)
    name = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)
    llm_call_ids = Column(JSON, nullable=True)
    tool_call_ids = Column(JSON, nullable=True)
    user_interaction_ids = Column(JSON, nullable=True)

    trace = relationship("TraceModel", back_populates="agent_calls")
    project = relationship("ProjectInfoModel", back_populates="agent_calls")
    llm_calls = relationship("LLMCallModel", back_populates="agent")
    tool_calls = relationship("ToolCallModel", back_populates="agent")
    user_interactions = relationship(
        "UserInteractionModel", order_by=UserInteractionModel.id, back_populates="agent"
    )


class MetricModel(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, ForeignKey("traces.id"), nullable=False)
    metric_name = Column(String, nullable=False)
    score = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    result_detail = Column(JSON, nullable=False)
    config = Column(JSON, nullable=False)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)

    trace = relationship("TraceModel", back_populates="metrics")

TraceModel.metrics = relationship(
    "MetricModel", order_by=MetricModel.id, back_populates="trace"
)



# Establish relationships
ProjectInfoModel.llm_calls = relationship(
    "LLMCallModel", order_by=LLMCallModel.id, back_populates="project"
)
ProjectInfoModel.tool_calls = relationship(
    "ToolCallModel", order_by=ToolCallModel.id, back_populates="project"
)
ProjectInfoModel.agent_calls = relationship(
    "AgentCallModel", order_by=AgentCallModel.id, back_populates="project"
)
ProjectInfoModel.system_info = relationship(
    "SystemInfoModel", uselist=False, back_populates="project"
)
ProjectInfoModel.errors = relationship(
    "ErrorModel", order_by=ErrorModel.id, back_populates="project"
)

AgentCallModel.errors = relationship(
    "ErrorModel", order_by=ErrorModel.id, back_populates="agent"
)

ToolCallModel.errors = relationship(
    "ErrorModel", order_by=ErrorModel.id, back_populates="tool_call"
)

LLMCallModel.errors = relationship(
    "ErrorModel", order_by=ErrorModel.id, back_populates="llm_call"
)
