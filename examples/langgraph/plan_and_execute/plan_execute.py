import getpass
import os
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key=os.environ['CATALYST_ACCESS_KEY'], 
    secret_key=os.environ['CATALYST_SECRET_KEY'], 
    base_url=os.environ['CATALYST_BASE_URL']
)

tracer = Tracer(
    project_name='financial_expert4',
    dataset_name='dataset',
    tracer_type="Agentic",
)
init_tracing(catalyst=catalyst, tracer=tracer)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

# Type definitions
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# Prompt templates
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

@tracer.trace_custom("create_tools")
def setup_tools():
    return [TavilySearchResults(max_results=3)]

@tracer.trace_custom("create_agent")
def setup_agent(tools):
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    prompt = "You are a helpful assistant."
    return create_react_agent(llm, tools, prompt=prompt)

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

@tracer.trace_custom("execute_step")
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

@tracer.trace_custom("plan_step")
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

@tracer.trace_custom("replan_step")
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

@tracer.trace_custom("workflow_setup")
def setup_workflow():
    workflow = StateGraph(PlanExecute)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    
    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )
    
    return workflow.compile()

async def main():
    # Setup tools and agent
    tools = setup_tools()
    global agent_executor
    agent_executor = setup_agent(tools)
    
    # Setup prompts
    global planner, replanner
    planner = planner_prompt | ChatOpenAI(
        model="gpt-4", temperature=0
    ).with_structured_output(Plan)
    
    replanner = replanner_prompt | ChatOpenAI(
        model="gpt-4", temperature=0
    ).with_structured_output(Act)
    
    # Setup and compile workflow
    app = setup_workflow()
    
    # Optional: Display workflow graph
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except:
        print("Could not display workflow graph")
    
    # Run the workflow
    config = {"recursion_limit": 50}
    inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
    
    print("\nExecuting workflow...")
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    import asyncio
    
    print("Starting Plan-Execute workflow with RagaAI Catalyst integration...")
    
    with tracer:
        asyncio.run(main())
    
    print("\nChecking trace upload status...")
    tracer.get_upload_status()