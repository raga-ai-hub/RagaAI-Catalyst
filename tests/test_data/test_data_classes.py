import pytest
from datetime import datetime
from agentneo.data.data_classes import ProjectInfo, SystemInfo, LLMCall, ToolCall, AgentCall  

# Sample data
@pytest.fixture
def sample_project_info():
    return ProjectInfo(
        project_name="Test Project",
        start_time=1609.0,
        end_time=1609.0,
        duration=3600.0,
        total_cost=100.0,
        total_tokens=1000
    )

@pytest.fixture
def sample_system_info():
    return SystemInfo(
        project_id=1,
        os_name="Linux",
        os_version="5.4.0",
        python_version="3.8.5",
        cpu_info="Intel i7",
        memory_total=16.0,
        installed_packages="pytest==6.0.0"
    )

@pytest.fixture
def sample_llm_call():
    return LLMCall(
        name="Test LLM",
        model_name="gpt-3",
        input_prompt="Hello",
        output_response="Hi there!",
        tool_call={},
        token_usage={"input_tokens": 10, "output_tokens": 15},
        cost={"usage_cost": 0.02},
        start_time=1609.0,
        end_time=1609.0,
        duration=3600.0
    )

@pytest.fixture
def sample_tool_call():
    return ToolCall(
        name="Sample Tool",
        input_parameters={"param1": "value1"},
        output="Result",
        start_time=1609.0,
        end_time=1609.0,
        duration=3600.0
    )

@pytest.fixture
def sample_agent_call():
    return AgentCall(
        name="Sample Agent",
        input_parameters={"param1": "value1"},
        output="Agent Result",
        start_time=1609.0,
        end_time=1609.0,
        duration=3600.0,
        tool_calls=[{"name": "Tool1", "output": "Output1"}],
        llm_calls=[{"name": "LLM1", "output": "Response1"}],
        errors=None
    )


# Test cases

def test_project_info_initialization(sample_project_info):
    assert sample_project_info.project_name == "Test Project"
    assert sample_project_info.start_time == 1609.0
    assert sample_project_info.end_time == 1609.0
    assert sample_project_info.duration == 3600.0
    assert sample_project_info.total_cost == 100.0
    assert sample_project_info.total_tokens == 1000

def test_system_info_initialization(sample_system_info):
    assert sample_system_info.project_id == 1
    assert sample_system_info.os_name == "Linux"
    assert sample_system_info.os_version == "5.4.0"
    assert sample_system_info.python_version == "3.8.5"
    assert sample_system_info.cpu_info == "Intel i7"
    assert sample_system_info.memory_total == 16.0
    assert sample_system_info.installed_packages == "pytest==6.0.0"

def test_llm_call_initialization(sample_llm_call):
    assert sample_llm_call.name == "Test LLM"
    assert sample_llm_call.model_name == "gpt-3"
    assert sample_llm_call.input_prompt == "Hello"
    assert sample_llm_call.output_response == "Hi there!"
    assert sample_llm_call.tool_call == {}
    assert sample_llm_call.token_usage == {"input_tokens": 10, "output_tokens": 15}
    assert sample_llm_call.cost == {"usage_cost": 0.02}
    assert sample_llm_call.start_time == 1609.0
    assert sample_llm_call.end_time == 1609.0
    assert sample_llm_call.duration == 3600.0

def test_tool_call_initialization(sample_tool_call):
    assert sample_tool_call.name == "Sample Tool"
    assert sample_tool_call.input_parameters == {"param1": "value1"}
    assert sample_tool_call.output == "Result"
    assert sample_tool_call.start_time == 1609.0
    assert sample_tool_call.end_time == 1609.0
    assert sample_tool_call.duration == 3600.0

def test_agent_call_initialization(sample_agent_call):
    assert sample_agent_call.name == "Sample Agent"
    assert sample_agent_call.input_parameters == {"param1": "value1"}
    assert sample_agent_call.output == "Agent Result"
    assert sample_agent_call.start_time == 1609.0
    assert sample_agent_call.end_time == 1609.0
    assert sample_agent_call.duration == 3600.0
    assert sample_agent_call.tool_calls == [{"name": "Tool1", "output": "Output1"}]
    assert sample_agent_call.llm_calls == [{"name": "LLM1", "output": "Response1"}]
    assert sample_agent_call.errors is None
