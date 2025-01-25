# Required imports
from typing import Annotated, Literal, Optional, Callable
from typing_extensions import TypedDict
from datetime import datetime, date
from pydantic import BaseModel, Field
import sqlite3
import pytz
import os
import sys
sys.path.append(".")

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition

from dotenv import load_dotenv
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst

load_dotenv()


# State definition
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental", 
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]

# Base Assistant class
class Assistant:
    def __init__(self, runnable: Runnable, name: str = "Unknown"):
        self.runnable = runnable
        self.name = name
        
    def __call__(self, state: State, config: RunnableConfig):
        # Start tracing for this assistant
        with tracer.start_trace(f"{self.name}_execution") as trace:
            trace.add_input({"state": state, "config": config})
            
            while True:
                result = self.runnable.invoke(state)
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
                    
            trace.add_output({"result": result})
            return {"messages": result}

# Tool definitions for Complete/Escalate functionality
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant."""
    cancel: bool = True
    reason: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task."
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task."
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information."
            }
        }

# Specialized Assistant Tool Definitions
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates."""
    request: str = Field(description="Any necessary followup questions the update flight assistant should clarify before proceeding.")

class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""
    location: str = Field(description="The location where the user wants to rent a car.")
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(description="Any additional information or requests from the user regarding the car rental.")

class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""
    location: str = Field(description="The location where the user wants to book a hotel.")
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(description="Any additional information or requests from the user regarding the hotel booking.")

class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""
    location: str = Field(description="The location where the user wants to book a recommended trip.")
    request: str = Field(description="Any additional information or requests from the user regarding the trip recommendation.")

# LLM and Prompt Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Primary Assistant Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# Primary Assistant Runnable
assistant_runnable = (
    primary_assistant_prompt
    | llm.bind_tools(
        [
            ToFlightBookingAssistant,
            ToBookCarRental,
            ToHotelBookingAssistant,
            ToBookExcursion,
            TavilySearchResults(max_results=3),
        ]
    )
)

# Mock function to fetch user flight information
def fetch_user_flight_information():
    """Mock function to fetch user flight information from a database."""
    return """
    Booking Reference: ABCD123
    Flight: LX123
    From: Zurich (ZRH)
    To: New York (JFK)
    Date: 2024-02-15
    Status: Confirmed
    """

# Flight Update Assistant Prompt and Runnable
update_flight_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized flight update assistant for Swiss Airlines. "
        "Your role is to help customers update or modify their flight bookings. "
        "You have access to the customer's current flight information and can make changes as needed. "
        "Always verify the customer's request and confirm the changes before proceeding."
        "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
    ),
    ("placeholder", "{messages}"),
])

update_flight_runnable = (
    update_flight_prompt 
    | llm.bind_tools([CompleteOrEscalate])
)

# Graph Node Helper Functions
def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """Creates an entry node for a specialized workflow."""
    def entry_node(state: State) -> dict:
        with tracer.start_trace(f"entry_node_{new_dialog_state}") as trace:
            trace.add_input({"state": state, "assistant_name": assistant_name, "new_dialog_state": new_dialog_state})
            
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            result = {
                "messages": [
                    ToolMessage(
                        content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                        f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                        " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                        " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                        " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": new_dialog_state,
            }
            
            trace.add_output({"result": result})
            return result
    return entry_node

def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant."""
    with tracer.start_trace("pop_dialog_state") as trace:
        trace.add_input({"state": state})
        
        messages = []
        if state["messages"][-1].tool_calls:
            messages.append(
                ToolMessage(
                    content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            )
            
        result = {
            "dialog_state": "pop",
            "messages": messages,
        }
        
        trace.add_output({"result": result})
        return result

# Graph Building
def build_support_bot():
    builder = StateGraph(State)
    
    # Add initial user info node
    def user_info(state: State):
        with tracer.start_trace("fetch_user_info") as trace:
            trace.add_input({"state": state})
            result = {"user_info": fetch_user_flight_information()}
            trace.add_output({"result": result})
            return result
        
    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")
    
    # Add specialized workflow nodes
    # Flight booking assistant
    builder.add_node("enter_update_flight", create_entry_node("Flight Updates & Booking Assistant", "update_flight"))
    builder.add_node("update_flight", Assistant(update_flight_runnable, name="update_flight"))
    builder.add_node("update_flight_sensitive_tools", Assistant(update_flight_runnable, name="update_flight_sensitive"))
    builder.add_edge("enter_update_flight", "update_flight")
    builder.add_edge("update_flight", "update_flight_sensitive_tools")
    
    # Car rental workflow
    builder.add_node("enter_book_car_rental", create_entry_node("Car Rental Assistant", "book_car_rental"))
    builder.add_node("book_car_rental", Assistant(update_flight_runnable, name="book_car_rental"))  # TODO: Replace with car rental runnable
    builder.add_node("book_car_rental_sensitive_tools", Assistant(update_flight_runnable, name="book_car_rental_sensitive"))
    builder.add_edge("enter_book_car_rental", "book_car_rental")
    builder.add_edge("book_car_rental", "book_car_rental_sensitive_tools")
    
    # Hotel booking workflow
    builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel"))
    builder.add_node("book_hotel", Assistant(update_flight_runnable, name="book_hotel"))  # TODO: Replace with hotel booking runnable
    builder.add_node("book_hotel_sensitive_tools", Assistant(update_flight_runnable, name="book_hotel_sensitive"))
    builder.add_edge("enter_book_hotel", "book_hotel")
    builder.add_edge("book_hotel", "book_hotel_sensitive_tools")
    
    # Excursion booking workflow
    builder.add_node("enter_book_excursion", create_entry_node("Excursion Booking Assistant", "book_excursion"))
    builder.add_node("book_excursion", Assistant(update_flight_runnable, name="book_excursion"))  # TODO: Replace with excursion booking runnable
    builder.add_node("book_excursion_sensitive_tools", Assistant(update_flight_runnable, name="book_excursion_sensitive"))
    builder.add_edge("enter_book_excursion", "book_excursion")
    builder.add_edge("book_excursion", "book_excursion_sensitive_tools")
    
    # Add primary assistant node
    builder.add_node("primary_assistant", Assistant(assistant_runnable, name="primary_assistant"))
    
    # Add routing logic
    def route_to_workflow(state: State):
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]
        
    builder.add_conditional_edges("fetch_user_info", route_to_workflow)
    builder.add_conditional_edges("primary_assistant", route_to_workflow)
    
    # Add edges from sensitive tools back to primary assistant
    for node in ["update_flight_sensitive_tools", "book_car_rental_sensitive_tools", 
                "book_hotel_sensitive_tools", "book_excursion_sensitive_tools"]:
        builder.add_edge(node, "primary_assistant")
    
    # Compile graph
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=[
            "update_flight_sensitive_tools",
            "book_car_rental_sensitive_tools", 
            "book_hotel_sensitive_tools",
            "book_excursion_sensitive_tools",
        ],
    )
    
    return graph

# Initialize RagaAI Catalyst and Tracer
catalyst = RagaAICatalyst(
    access_key="jqa9FHN8B313H5mQS14X",
    secret_key="BTzSSKKzFbX8YKoHmIMNpcegsKXF9yRhnYFvyMAF",
    # base_url=os.environ["RAGA_BASE_URL"],
)

tracer = Tracer(
    project_name="Support_Bot_Langgraph",
    dataset_name="Test_Dataset1",
    tracer_type="agentic",
)

# Usage example:
if __name__ == "__main__":
    with tracer:
        support_bot = build_support_bot()
        
        # Configure the bot
        config = {
            "configurable": {
                "passenger_id": "3442 587242",
                "thread_id": "test-thread"
            }
        }
        
        # Process a user message
        events = support_bot.stream(
            {"messages": ("user", "Hi, when is my flight?")}, 
            config,
            stream_mode="values"
        )
        
        # Handle the response
        for event in events:
            print(event)