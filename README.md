# RagaAI Catalyst&nbsp; ![GitHub release (latest by date)](https://img.shields.io/github/v/release/raga-ai-hub/ragaai-catalyst) ![GitHub stars](https://img.shields.io/github/stars/raga-ai-hub/ragaai-catalyst?style=social)  ![Issues](https://img.shields.io/github/issues/raga-ai-hub/ragaai-catalyst) 

RagaAI Catalyst is a comprehensive platform designed to enhance the management and optimization of LLM projects. It offers a wide range of features, including project management, dataset management, evaluation management, trace management, prompt management, synthetic data generation, and guardrail management. These functionalities enable you to efficiently evaluate, and safeguard your LLM applications.

## Table of Contents

- [RagaAI Catalyst](#ragaai-catalyst)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Project Management](#project-management)
    - [Dataset Management](#dataset-management)
    - [Evaluation Management](#evaluation)
    - [Trace Management](#trace-management)
    - [Prompt Management](#prompt-management)
    - [Synthetic Data Generation](#synthetic-data-generation)
    - [Guardrail Management](#guardrail-management)
    - [Agentic Tracing](#agentic-tracing)
    - [Red-teaming](#red-teaming)

## Installation

To install RagaAI Catalyst, you can use pip:

```bash
pip install ragaai-catalyst
```

## Configuration

Before using RagaAI Catalyst, you need to set up your credentials. You can do this by setting environment variables or passing them directly to the `RagaAICatalyst` class:

```python
from ragaai_catalyst import RagaAICatalyst

catalyst = RagaAICatalyst(
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    base_url="BASE_URL"
)
```
**Note**: Authetication to RagaAICatalyst is necessary to perform any operations below 


## Usage

### Project Management

Create and manage projects using RagaAI Catalyst:

```python
# Create a project
project = catalyst.create_project(
    project_name="Test-RAG-App-1",
    usecase="Chatbot"
)

# Get project usecases
catalyst.project_use_cases()

# List projects
projects = catalyst.list_projects()
print(projects)
```

### Dataset Management
Manage datasets efficiently for your projects:

```py
from ragaai_catalyst import Dataset

# Initialize Dataset management for a specific project
dataset_manager = Dataset(project_name="project_name")

# List existing datasets
datasets = dataset_manager.list_datasets()
print("Existing Datasets:", datasets)

# Create a dataset from CSV
dataset_manager.create_from_csv(
    csv_path='path/to/your.csv',
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)

# Get project schema mapping
dataset_manager.get_schema_mapping()

```

For more detailed information on Dataset Management, including CSV schema handling and advanced usage, please refer to the [Dataset Management documentation](docs/dataset_management.md).


### Evaluation

Create and manage metric evaluation of your RAG application:

```python
from ragaai_catalyst import Evaluation

# Create an experiment
evaluation = Evaluation(
    project_name="Test-RAG-App-1",
    dataset_name="MyDataset",
)

# Get list of available metrics
evaluation.list_metrics()

# Add metrics to the experiment

schema_mapping={
    'Query': 'prompt',
    'response': 'response',
    'Context': 'context',
    'expectedResponse': 'expected_response'
}

# Add single metric
evaluation.add_metrics(
    metrics=[
      {"name": "Faithfulness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "Faithfulness_v1", "schema_mapping": schema_mapping},
    
    ]
)

# Add multiple metrics
evaluation.add_metrics(
    metrics=[
        {"name": "Faithfulness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.323}}, "column_name": "Faithfulness_gte", "schema_mapping": schema_mapping},
        {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"lte": 0.323}}, "column_name": "Hallucination_lte", "schema_mapping": schema_mapping},
        {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"eq": 0.323}}, "column_name": "Hallucination_eq", "schema_mapping": schema_mapping},
    ]
)

# Get the status of the experiment
status = evaluation.get_status()
print("Experiment Status:", status)

# Get the results of the experiment
results = evaluation.get_results()
print("Experiment Results:", results)
```



### Trace Management

Record and analyze traces of your RAG application:

```python
from ragaai_catalyst import Tracer

# Start a trace recording
tracer = Tracer(
    project_name="Test-RAG-App-1",
    dataset_name="tracer_dataset_name",
    metadata={"key1": "value1", "key2": "value2"},
    tracer_type="langchain",
    pipeline={
        "llm_model": "gpt-4o-mini",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
).start()

# Your code here


# Stop the trace recording
tracer.stop()

# Get upload status
tracer.get_upload_status()
```
For more detailed information on Trace Management, please refer to the [Trace Management documentation](docs/trace_management.md).


### Prompt Management

Manage and use prompts efficiently in your projects:

```py
from ragaai_catalyst import PromptManager

# Initialize PromptManager
prompt_manager = PromptManager(project_name="Test-RAG-App-1")

# List available prompts
prompts = prompt_manager.list_prompts()
print("Available prompts:", prompts)

# Get default prompt by prompt_name
prompt_name = "your_prompt_name"
prompt = prompt_manager.get_prompt(prompt_name)

# Get specific version of prompt by prompt_name and version
prompt_name = "your_prompt_name"
version = "v1"
prompt = prompt_manager.get_prompt(prompt_name,version)

# Get variables in a prompt
variable = prompt.get_variables()
print("variable:",variable)

# Get prompt content
prompt_content = prompt.get_prompt_content()
print("prompt_content:", prompt_content)

# Compile the prompt with variables
compiled_prompt = prompt.compile(query="What's the weather?", context="sunny", llm_response="It's sunny today")
print("Compiled prompt:", compiled_prompt)

# implement compiled_prompt with openai
import openai
def get_openai_response(prompt):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt
    )
    return response.choices[0].message.content
openai_response = get_openai_response(compiled_prompt)
print("openai_response:", openai_response)

# implement compiled_prompt with litellm
import litellm
def get_litellm_response(prompt):
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=prompt
    )
    return response.choices[0].message.content
litellm_response = get_litellm_response(compiled_prompt)
print("litellm_response:", litellm_response)

```
For more detailed information on Prompt Management, please refer to the [Prompt Management documentation](docs/prompt_management.md).


### Synthetic Data Generation

```py
from ragaai_catalyst import SyntheticDataGeneration

# Initialize Synthetic Data Generation
sdg = SyntheticDataGeneration()

# Process your file
text = sdg.process_document(input_data="file_path")

# Generate results
result = sdg.generate_qna(text, question_type ='complex',model_config={"provider":"openai","model":"gpt-4o-mini"},n=5)

print(result.head())

# Get supported Q&A types
sdg.get_supported_qna()

# Get supported providers
sdg.get_supported_providers()
```



### Guardrail Management

```py
from ragaai_catalyst import GuardrailsManager

# Initialize Guardrails Manager
gdm = GuardrailsManager(project_name=project_name)

# Get list of Guardrails available
guardrails_list = gdm.list_guardrails()
print('guardrails_list:', guardrails_list)

# Get list of fail condition for guardrails
fail_conditions = gdm.list_fail_condition()
print('fail_conditions;', fail_conditions)

#Get list of deployment ids
deployment_list = gdm.list_deployment_ids()
print('deployment_list:', deployment_list)

# Get specific deployment id with guardrails information
deployment_id_detail = gdm.get_deployment(17)
print('deployment_id_detail:', deployment_id_detail)

# Add guardrails to a deployment id
guardrails_config = {"guardrailFailConditions": ["FAIL"],
                     "deploymentFailCondition": "ALL_FAIL",
                     "alternateResponse": "Your alternate response"}

guardrails = [
    {
      "displayName": "Response_Evaluator",
      "name": "Response Evaluator",
      "config":{
          "mappings": [{
                        "schemaName": "Text",
                        "variableName": "Response"
                    }],
          "params": {
                    "isActive": {"value": False},
                    "isHighRisk": {"value": True},
                    "threshold": {"eq": 0},
                    "competitors": {"value": ["Google","Amazon"]}
                }
      }
    },
    {
      "displayName": "Regex_Check",
      "name": "Regex Check",
      "config":{
          "mappings": [{
                        "schemaName": "Text",
                        "variableName": "Response"
                    }],
          "params":{
              "isActive": {"value": False},
              "isHighRisk": {"value": True},
              "threshold": {"lt1": 1}
          }
      }
    }
]

gdm.add_guardrails(deployment_id, guardrails, guardrails_config)


# Import GuardExecutor
from ragaai_catalyst import GuardExecutor

# Initialise GuardExecutor with required params and Evaluate
executor = GuardExecutor(deployment_id,gdm,field_map={'context':'document'})


message={'role':'user',
         'content':'What is the capital of France'
        }
prompt_params={'document':' France'}

model_params = {'temperature':.7,'model':'gpt-4o-mini'}
llm_caller = 'litellm'

executor([message],prompt_params,model_params,llm_caller)

```

### Agentic Tracing

The Agentic Tracing module provides comprehensive monitoring and analysis capabilities for AI agent systems. It helps track various aspects of agent behavior including:

- LLM interactions and token usage
- Tool utilization and execution patterns
- Network activities and API calls
- User interactions and feedback
- Agent decision-making processes

The module includes utilities for cost tracking, performance monitoring, and debugging agent behavior. This helps in understanding and optimizing AI agent performance while maintaining transparency in agent operations.

```python
from ragaai_catalyst import Tracer

agentic_tracing_dataset_name = "agentic_tracing_dataset_name"

tracer = Tracer(
    project_name=agentic_tracing_project_name,
    dataset_name=agentic_tracing_dataset_name,
    tracer_type="Agentic",
)

from ragaai_catalyst import init_tracing
init_tracing(catalyst=catalyst, tracer=tracer)

from ragaai_catalyst import trace_llm, trace_tool, trace_agent, current_span

from openai import OpenAI


@trace_llm(name="llm_call", tags=["default_llm_call"])
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.85,
    )
    # Span level context
    current_span().add_context("name = span level in summary_agent, context = some span level context")

    # Span level execute metrics
    current_span().execute_metrics(
        name="Hallucination",
        model="gpt-4o",
        provider="openai"
    )
    response_data = response.choices[0].message.content.strip()
    print('response_data: ', response_data)
    return response_data

class SummaryAgent:
    def __init__(self, persona="Summary Agent"):
        self.persona = persona

    @trace_agent(name="summary_agent")
    def summarize(self, text):
        prompt = f"Please summarize this text concisely: {text}"

        # Span level metric
        current_span().add_metrics(name='Accuracy', score=0.5, reasoning='some reasoning')

        # Span level context
        current_span().add_context("name = span level in summary_agent, context = some span level context")

        summary = llm_call(prompt)
        return summary


class AnalysisAgent:
    def __init__(self, persona="Analysis Agent"):
        self.persona = persona
        self.summary_agent = SummaryAgent()

    @trace_agent(name="analysis_agent")
    def analyze(self, text):
        summary = self.summary_agent.summarize(text)

        prompt = f"Given this summary: {summary}\nProvide a brief analysis of the main points."

        # Span level metric
        current_span().add_metrics(name='correctness', score=0.5, reasoning='some reasoning')
        analysis = llm_call(prompt)

        return {
            "summary": summary,
            "analysis": analysis
        }

class RecommendationAgent:
    def __init__(self, persona="Recommendation Agent"):
        self.persona = persona
        self.analysis_agent = AnalysisAgent()

    @trace_agent(name="recommendation_agent", tags=['coordinator_agent'])
    def recommend(self, text):
        analysis_result = self.analysis_agent.analyze(text)

        prompt = f"""Given this summary: {analysis_result['summary']}
        And this analysis: {analysis_result['analysis']}
        Provide 2-3 actionable recommendations."""

        recommendations = llm_call(prompt)

        return {
            "summary": analysis_result["summary"],
            "analysis": analysis_result["analysis"],
            "recommendations": recommendations
        }
#Defining agent tracer
@trace_agent(name="get_recommendation", tags=['coordinator_agent'])
def get_recommendation(agent, text):
    recommendation = agent.recommend(text)
    return recommendation

def main():
    text = """
    Artificial Intelligence has transformed various industries in recent years.
    From healthcare to finance, AI applications are becoming increasingly prevalent.
    Machine learning models are being used to predict market trends, diagnose diseases,
    and automate routine tasks. The impact of AI on society continues to grow,
    raising both opportunities and challenges for the future.
    """

    recommendation_agent = RecommendationAgent()
    result = get_recommendation(recommendation_agent, text)


    # Trace level metric
    tracer.add_metrics(name='hallucination_1', score=0.5, reasoning='some reasoning')

# Run tracer
with tracer:
    main()
```

### Red-teaming

The Red-teaming module provides comprehensive scans for model vulnerabilities:

- Initialize RedTeaming object requiring optional `provider` (defaulting to OpenAI), `model`, `api_key`, `api_base` and `api_version`. 
User can set API keys in the environment variables, or optionally pass them to the constructor.

1. View all supported evaluators
```python
from ragaai_catalyst import RedTeaming
rt = RedTeaming()

supported_evaluators = rt.get_supported_evaluators()
```

2. Run scan: returns a scan dataframe for the model
```python
import pandas as pd
from ragaai_catalyst import RedTeaming

rt = RedTeaming("openai", "gpt-4o-mini", "my-api-key")

def mock_llm_call(query):
  pass # llm call for the query

def model(df: pd.DataFrame):
  # Function which takes in an input dataframe, and returns a list containing LLM outputs for the inputs  
  return [mock_llm_call({"query": question}) for question in df["question"]]


scan_df = rt.run_scan(model=model, evaluators=["llm"], save_report=True)
```