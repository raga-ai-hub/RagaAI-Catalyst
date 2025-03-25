
## Components

### 1. Workflow States
- **PlanExecute**: Manages the workflow state
- **Plan**: Defines the step-by-step execution plan
- **Response**: Handles user responses
- **Act**: Determines next actions

### 2. Key Functions
- `setup_tools()`: Initializes available tools
- `setup_agent()`: Creates the execution agent
- `execute_step()`: Handles step execution
- `plan_step()`: Creates initial plans
- `replan_step()`: Adjusts plans as needed

# Plan-Execute Workflow with RagaAI Catalyst

[Previous sections remain the same...]

## Quick Start Guide

### 1. Clone the Repository
```bash
# Clone the RagaAI-Catalyst repository
git clone https://github.com/raga-ai-hub/RagaAI-Catalyst.git

# Navigate to the plan-and-execute example directory
cd RagaAI-Catalyst2/examples/langgraph/plan_and_execute
```

### 2. Set Up Python Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `plan_and_execute` directory:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Tavily Search Configuration
TAVILY_API_KEY=your_tavily_api_key

# RagaAI Catalyst Configuration
CATALYST_ACCESS_KEY=your_catalyst_access_key
CATALYST_SECRET_KEY=your_catalyst_secret_key
CATALYST_BASE_URL=https://llm-dev5.ragaai.ai/api
```

### 4. Run the Script
```bash
# Execute the plan_execute.py script
python plan_execute.py
```


The script will:
1. Initialize the RagaAI Catalyst tracer
2. Set up the workflow components
3. Display the workflow graph (in notebook environments)
4. Execute the sample query
5. Show step-by-step execution progress
6. Upload tracing data to RagaAI Catalyst


