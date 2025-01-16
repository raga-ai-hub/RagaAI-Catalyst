import pytest
from agentneo.agentneo import AgentNeo
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest import mock

@pytest.fixture(scope="module")
def setup_engine():
    """Fixture to create a temporary SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    AgentNeo.Session = sessionmaker(bind=engine)
    AgentNeo.engine = engine
    # Create all tables
    from agentneo.data.data_models import Base
    Base.metadata.create_all(engine)
    yield engine

@pytest.fixture
def agentneo_instance(setup_engine):
    """Fixture to provide a clean instance of AgentNeo for each test."""
    return AgentNeo()

def test_connect_project(agentneo_instance):
    """Test connecting to an existing project."""
    project_name = "Connect Project"

    # Try creating the project; if it already exists, pass the exception
    try:
        agentneo_instance.create_project(project_name)
    except ValueError as e:
        assert str(e) == f"Project '{project_name}' already exists."

    # Connect to the project (should succeed if already exists or just created)
    connected_id = agentneo_instance.connect_project(project_name)
    assert connected_id == agentneo_instance.project_id
    assert agentneo_instance.project_name == project_name

def test_create_project(agentneo_instance):
    """Test creating a new project."""
    project_name = "Test Project"

    # Create project
    try:
        project_id = agentneo_instance.create_project(project_name)
        assert project_id is not None
        assert agentneo_instance.project_name == project_name
    except:
        # Test for duplicate project creation
        with pytest.raises(ValueError, match=f"Project '{project_name}' already exists."):
            agentneo_instance.create_project(project_name)

def test_list_projects(agentneo_instance):
    """Test listing projects."""
    project_names = ["Project 1", "Project 2", "Project 3"]

    for name in project_names:
        try:
            agentneo_instance.create_project(name)
        except ValueError as e:
            assert str(e) == f"Project '{name}' already exists."

    # List projects
    projects = AgentNeo.list_projects()
    assert len(projects) >= len(project_names)

    # Verify projects are listed
    project_names_in_list = [p["name"] for p in projects]
    for name in project_names:
        assert name in project_names_in_list

    # Test limiting the number of listed projects
    limited_projects = AgentNeo.list_projects(num_projects=2)
    assert len(limited_projects) == 2

@mock.patch("agentneo.server.dashboard.launch_dashboard")
def test_launch_dashboard(mock_launch_dashboard):
    """Test launching the dashboard."""
    AgentNeo.launch_dashboard(port=4000)
    mock_launch_dashboard.assert_called_once_with(4000)
