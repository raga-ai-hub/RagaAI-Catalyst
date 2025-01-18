# Agentic Tracing

This module provides tracing functionality for agentic AI systems, helping track and analyze various aspects of AI agent behavior including LLM interactions, tool usage, and network activities.

## Directory Structure

```
agentic_tracing/
├── tracers/                  # Core tracing implementations
│   ├── main_tracer.py       # Main tracing functionality
│   ├── agent_tracer.py      # Agent behavior tracing
│   ├── base.py              # Base tracing classes
│   ├── llm_tracer.py        # Language model interaction tracing
│   ├── network_tracer.py    # Network activity tracing
│   ├── tool_tracer.py       # Tool usage tracing
│   ├── user_interaction_tracer.py # User interaction tracing
│   └── __init__.py          # Tracer module initialization
├── data/                     # Data structures and classes
│   ├── data_classes.py      # Data class definitions
│   └── __init__.py          # Data module initialization
├── utils/                    # Utility functions and helpers
│   ├── api_utils.py         # API-related utilities
│   ├── file_name_tracker.py # Tracks file names and paths
│   ├── generic.py           # Generic utility functions
│   ├── llm_utils.py         # LLM-specific utilities
│   ├── model_costs.json     # Model cost configurations
│   ├── trace_utils.py       # General tracing utilities
│   ├── unique_decorator.py  # Unique ID generation
│   ├── zip_list_of_unique_files.py # File handling utilities
│   └── __init__.py          # Utils module initialization
├── tests/                    # Test suites and examples
│   ├── ai_travel_agent.py   # Travel agent test implementation
│   ├── unique_decorator_test.py # Tests for unique decorator
│   ├── TravelPlanner.ipynb  # Travel planner example notebook
│   ├── FinancialAnalysisSystem.ipynb # Financial analysis example
│   ├── GameActivityEventPlanner.ipynb # Game event planner example
│   └── __init__.py          # Tests module initialization
├── upload/                   # Upload functionality
│   ├── upload_code.py       # Code upload utilities
│   └── __init__.py          # Upload module initialization
└── __init__.py              # Package initialization
```

## Components

### Tracers
Different types of tracers for various aspects of agent behavior:
- Main Tracer: Core tracing functionality for managing and coordinating different trace types
- Agent Tracer: Tracks agent behavior, decisions, and state changes
- Base Tracer: Provides base classes and common functionality for all tracers
- LLM Tracer: Monitors language model interactions, including:
  - Token usage tracking
  - Cost calculation
  - Input/output monitoring
  - Model parameter tracking
- Network Tracer: Tracks network activities and API calls
- Tool Tracer: Monitors tool usage and execution
- User Interaction Tracer: Tracks user interactions and feedback

### Data
Core data structures and classes:
- Data Classes: Defines structured data types for:
  - LLM calls
  - Network requests
  - Tool executions
  - Trace components
  - Agent states
  - User interactions

### Utils
Helper functions and utilities:
- API Utils: Handles API-related operations and configurations
- LLM Utils: Utilities for handling LLM-specific operations:
  - Model name extraction
  - Token usage calculation
  - Cost computation
  - Parameter sanitization
- Generic Utils: Common utility functions used across modules
- Trace Utils: General tracing utilities
- File Name Tracker: Manages file paths and names
- Unique Decorator: Generates unique identifiers for trace components
- Model Costs: Configuration for different model pricing
- Zip List of Unique Files: Handles file compression and unique file management

### Tests
Test suites and example implementations:
- AI Travel Agent: Test implementation of a travel planning agent
- Unique Decorator Tests: Unit tests for unique ID generation
- Example Notebooks:
  - Travel Planner: Example of travel planning implementation
  - Financial Analysis: Example of financial system analysis
  - Game Event Planner: Example of game activity planning

### Upload
Components for uploading and managing trace data:
- Code Upload: Handles uploading of traced code and execution data
- Supports various data formats and trace types
