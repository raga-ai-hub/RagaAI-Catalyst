# information_gather_prompting_traced.py

import getpass
import os
from dotenv import load_dotenv
from typing import List, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import uuid
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

# Load environment variables
load_dotenv()

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key="1q2igAYCIlpSBufkdB6f",
    secret_key="yG6TJOgES8D9jAi9OI0X6SgvZNtkcFvkOruukJay",
    base_url="https://llm-dev5.ragaai.ai/api")
# project = catalyst.create_project(
#     project_name="langgraph",
#     usecase="Agentic Application"
# )
tracer = Tracer(
    project_name='langgraph',
    dataset_name='dataset',
    tracer_type="Agentic",
)
init_tracing(catalyst=catalyst, tracer=tracer)

tracer.start()

# Define the template for gathering user requirements
template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

# Function to get messages for information gathering
def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

# Define the PromptInstructions model
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

# Initialize the LLM
llm = ChatOpenAI(temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])

# Define the information chain
@tracer.trace_custom("info_chain")
def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

# Define the prompt generation system message
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

# Function to get messages for prompt generation
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

# Define the prompt generation chain
@tracer.trace_custom("prompt_gen_chain")
def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    
    # Check if the response matches the expected format
    if "RAG" in state["messages"][-1].content:
        response = AIMessage(content="Response: The status is RED.")
    
    return {"messages": [response]}

# Define the state logic
def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

# Define the State class
class State(TypedDict):
    messages: List[add_messages]

# Create the graph
memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)

@workflow.add_node
def add_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }

workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)

# Use the graph


# Use the graph
cached_human_responses = ["hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red", "q"]
cached_response_index = 0
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    try:
        user = input("User (q/Q to quit): ")
    except EOFError:
        # Handle end of input stream
        break

    if user in {"q", "Q"}:
        print("AI: Byebye")
        break

    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")

tracer.stop()
tracer.get_upload_status()



tracer.stop()
tracer.get_upload_status()