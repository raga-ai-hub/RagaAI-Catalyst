import asyncio
import json
import os
 
from fastapi.responses import StreamingResponse
# os.environ["DEBUG"] = "1"
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.core.workflow import Event, Workflow
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    step
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import BaseReasoningStep, ActionReasoningStep
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.tools import ToolSelection
from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing, trace_agent, current_span, trace_tool, trace_custom
import uvicorn
from llama_index.llms.azure_openai import AzureOpenAI


from dotenv import load_dotenv

load_dotenv() 

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ["AZURE_DEPLOYMENT"]
subscription_key = os.environ["AZURE_SUBSCRIPTION_KEY"]
model = "gpt-4o-mini"

FI_LLM = AzureOpenAI(
    azure_endpoint=endpoint,
    model = model,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
    engine=deployment
)

 
# Initialize RagaAI Catalyst tracing
def initialize_tracing():
    catalyst = RagaAICatalyst(
        access_key="", # Replace with Boeing credentials
        secret_key="",
        base_url="",
        
    )
    tracer = Tracer(
        project_name="test_stream", # Match UI project name
        dataset_name="fleety_chat_after_fix", # Unique dataset name
        tracer_type="Agentic"
    )
    init_tracing(catalyst=catalyst, tracer=tracer)
    return tracer
# FastAPI app
app = FastAPI(title="ReAct Agent API")
# Initialize tracing
tracer = initialize_tracing()
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
class PrepEvent(Event):
    pass
class InputEvent(Event):
    input: list[ChatMessage]
class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]
class FunctionOutputEvent(Event):
    output: ToolOutput
class ProgressEvent(Event):
    msg: str
from typing import Any, List
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.openai import OpenAI
class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or OpenAI()
        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter.from_defaults(
            context=extra_context or ""
        )
        self.output_parser = ReActOutputParser()
        self.sources = []
    @step
    @trace_custom("Prepare new message")
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources
        self.sources = []
        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)
        # clear current reasoning
        await ctx.set("current_reasoning", [])
        return PrepEvent()
    @step
    # @trace_custom("Prepare chat history")
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        # get chat history
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)
    @step
    @trace_agent("LLM Call")
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        # span = current_span()
        # span.add_gt("Test GT")
        # span.add_context("test context")
        # span.add_metrics(name="Test custom metric", score=1.0, reasoning="test reasoning") #for local metrics
        response = await self.llm.achat(chat_history)
        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get("current_reasoning", default=[])).append(
                reasoning_step
            )
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [*self.sources],
                        "reasoning": await ctx.get(
                            "current_reasoning", default=[]
                        ),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                ctx.write_event_to_stream(
                    ProgressEvent(
                        msg=reasoning_step.thought
                    )
                )
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
        # if no tool calls or final response, iterate again
        return PrepEvent()
    @step
    @trace_tool("Tool call")
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
        # prep the next iteraiton
        return PrepEvent()
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    return x + y
def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    return x * y
tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(multiply),
]
agent = ReActAgent(
    llm=FI_LLM, tools=tools, timeout=120, verbose=True
)
@app.post("/run/")
async def run_agent(payload: dict, background_tasks: BackgroundTasks):
    """Endpoint to run the ReAct agent with user input."""
    tracer.start()
    input = payload.get("input")  # Extract input from the payload
    handler = agent.run(input=input)
    return StreamingResponse(event_generator(handler), media_type="text/event-stream")
 
 
async def event_generator(handler):
    """Stream workflow events"""
    try:
        async for event in handler.stream_events():
            if isinstance(event, ProgressEvent):
                yield f"data: {json.dumps({'type': 'thought', 'msg': event.msg})}\n\n"
                    
        result = await handler
        yield f"data: {json.dumps({'type': 'answer', 'result': {'answer':result['response']}})}\n\n"
    except asyncio.CancelledError:
        print("Streaming cancelled by the client.")
    except Exception as e:
        print(f"Error in event_generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
    finally:
        tracer.stop()
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)