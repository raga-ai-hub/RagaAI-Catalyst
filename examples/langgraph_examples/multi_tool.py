from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
import os
from langchain_core.messages import ToolMessage
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Initialize tracer
tracer = Tracer(
    project_name="Langgraph_testing",
    dataset_name="multi_tools",
    tracer_type="Agentic",
)

tracer.start()

# Initialize multiple tools
arxiv_tool = ArxivQueryRun(max_results=2)
ddg_tool = DuckDuckGoSearchRun()

tools = [
    arxiv_tool,      # For academic papers from ArXiv
    ddg_tool,        # For web search results
]

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph
graph_builder = StateGraph(State)

# Setup LLM with tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(
    tools,
    tool_choice="auto",  # Let the model choose which tool to use
)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Initialize tool node
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    """Route to tools or end depending on whether tools were requested."""
    messages = state["messages"]
    if not messages:
        return "END"
    
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "END"

# Add edges to the graph
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "tools",
        "END": END
    }
)

graph_builder.add_edge("tools", "chatbot")

# Set the entry point
graph_builder.set_entry_point("chatbot")

# Compile the graph
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    """Stream updates from the graph."""
    config = {"messages": [{"role": "user", "content": user_input}]}
    for output in graph.stream(config):
        print(f"Assistant: {output}")

print("Multi-Tool Research Assistant Ready! (Type 'quit' to exit)")
print("Available tools:")
print("1. ArXiv - Find academic papers")
print("2. DuckDuckGo Search - Search the web")
print("\nExample queries:")
print("- 'Find recent papers and web results about LangChain'")
print("- 'Search for tutorials on Python async programming and related research papers'")
print("- 'What are the latest developments in quantum computing? Include papers and web results'")

while True:
    try:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        stream_graph_updates(user_input)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")

tracer.stop()
