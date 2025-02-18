from configparser import InterpolationMissingOptionError
import json
from datetime import datetime
from typing import Any, Optional, Dict, List, ClassVar
from pydantic import Field
# from treelib import Tree

from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler

from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
    AgentRunStepStartEvent,
    AgentRunStepEndEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatErrorEvent,
    StreamChatDeltaReceivedEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
    EmbeddingEndEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMChatInProgressEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryStartEvent,
    QueryEndEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankStartEvent,
    ReRankEndEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalStartEvent,
    RetrievalEndEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
    GetResponseEndEvent,
    GetResponseStartEvent,
)

import uuid

from .utils.extraction_logic_llama_index import extract_llama_index_data
from .utils.convert_llama_instru_callback import convert_llamaindex_instrumentation_to_callback

class EventHandler(BaseEventHandler):
    """Example event handler.

    This event handler is an example of how to create a custom event handler.

    In general, logged events are treated as single events in a point in time,
    that link to a span. The span is a collection of events that are related to
    a single task. The span is identified by a unique span_id.

    While events are independent, there is some hierarchy.
    For example, in query_engine.query() call with a reranker attached:
    - QueryStartEvent
    - RetrievalStartEvent
    - EmbeddingStartEvent
    - EmbeddingEndEvent
    - RetrievalEndEvent
    - RerankStartEvent
    - RerankEndEvent
    - SynthesizeStartEvent
    - GetResponseStartEvent
    - LLMPredictStartEvent
    - LLMChatStartEvent
    - LLMChatEndEvent
    - LLMPredictEndEvent
    - GetResponseEndEvent
    - SynthesizeEndEvent
    - QueryEndEvent
    """

    events: List[BaseEvent] = []
    current_trace: List[Dict[str, Any]] = []  # Store events for the current trace


    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "EventHandler"

    def handle(self, event: BaseEvent) -> None:
        """Logic for handling event."""
        # print("-----------------------")
        # # all events have these attributes
        # print(event.id_)
        # print(event.timestamp)
        # print(event.span_id)

        # Prepare event details dictionary
        event_details = {
            "id": event.id_,
            "timestamp": event.timestamp,
            "span_id": event.span_id,
            "event_type": event.class_name(),
        }

        # event specific attributes
        # print(f"Event type: {event.class_name()}")
        if isinstance(event, AgentRunStepStartEvent):
            event_details.update({
                "task_id": event.task_id,
                "step": event.step,
                "input": event.input,
            })
        if isinstance(event, AgentRunStepEndEvent):
            event_details.update({
                "step_output": event.step_output,
            })
        if isinstance(event, AgentChatWithStepStartEvent):
            event_details.update({
                "user_msg": event.user_msg,
            })
        if isinstance(event, AgentChatWithStepEndEvent):
            event_details.update({
                "response": event.response,
            })
        if isinstance(event, AgentToolCallEvent):
            event_details.update({
                "arguments": event.arguments,
                "tool_name": event.tool.name,
                "tool_description": event.tool.description,
                "tool_openai": event.tool.to_openai_tool(),
            })
        if isinstance(event, StreamChatDeltaReceivedEvent):
            event_details.update({
                "delta": event.delta,
            })
        if isinstance(event, StreamChatErrorEvent):
            event_details.update({
                "exception": event.exception,
            })
        if isinstance(event, EmbeddingStartEvent):
            event_details.update({
                "model_dict": event.model_dict,
            })
        if isinstance(event, EmbeddingEndEvent):
            event_details.update({
                "chunks": event.chunks,
                "embeddings": event.embeddings[0][:5],
            })
        if isinstance(event, LLMPredictStartEvent):
            event_details.update({
                "template": event.template,
                "template_args": event.template_args,
            })
        if isinstance(event, LLMPredictEndEvent):
            event_details.update({
                "output": event.output,
            })
        if isinstance(event, LLMStructuredPredictStartEvent):
            event_details.update({
                "template": event.template,
                "template_args": event.template_args,
                "output_cls": event.output_cls,
            })
        if isinstance(event, LLMStructuredPredictEndEvent):
            event_details.update({
                "output": event.output,
            })
        if isinstance(event, LLMCompletionStartEvent):
            event_details.update({
                "model_dict": event.model_dict,
                "prompt": event.prompt,
                "additional_kwargs": event.additional_kwargs,
            })
        if isinstance(event, LLMCompletionEndEvent):
            event_details.update({
                "response": event.response,
                "prompt": event.prompt,
            })
        if isinstance(event, LLMChatInProgressEvent):
            event_details.update({
                "messages": event.messages,
                "response": event.response,
            })
        if isinstance(event, LLMChatStartEvent):
            event_details.update({
                "messages": event.messages,
                "additional_kwargs": event.additional_kwargs,
                "model_dict": event.model_dict,
            })
        if isinstance(event, LLMChatEndEvent):
            event_details.update({
                "messages": event.messages,
                "response": event.response,
            })
        if isinstance(event, RetrievalStartEvent):
            event_details.update({
                "str_or_query_bundle": event.str_or_query_bundle,
            })
        if isinstance(event, RetrievalEndEvent):
            event_details.update({
                "str_or_query_bundle": event.str_or_query_bundle,
                "nodes": event.nodes,
                "text": event.nodes[0].text
            })
        if isinstance(event, ReRankStartEvent):
            event_details.update({
                "query": event.query,
                "nodes": event.nodes,
                "top_n": event.top_n,
                "model_name": event.model_name,
            })
        if isinstance(event, ReRankEndEvent):
            event_details.update({
                "nodes": event.nodes,
            })
        if isinstance(event, QueryStartEvent):
            event_details.update({
                "query": event.query,
            })
        if isinstance(event, QueryEndEvent):
            event_details.update({
                "response": event.response,
                "query": event.query,
            })
        if isinstance(event, SpanDropEvent):
            event_details.update({
                "err_str": event.err_str,
            })
        if isinstance(event, SynthesizeStartEvent):
            event_details.update({
                "query": event.query,
            })
        if isinstance(event, SynthesizeEndEvent):
            event_details.update({
                "response": event.response,
                "query": event.query,
            })
        if isinstance(event, GetResponseStartEvent):
            event_details.update({
                "query_str": event.query_str,
            })

        # Append event details to current_trace
        self.current_trace.append(event_details)

        self.events.append(event)

    def _get_events_by_span(self) -> Dict[str, List[BaseEvent]]:
        events_by_span: Dict[str, List[BaseEvent]] = {}
        for event in self.events:
            if event.span_id in events_by_span:
                events_by_span[event.span_id].append(event)
            else:
                events_by_span[event.span_id] = [event]
        return events_by_span

    # def _get_event_span_trees(self) -> List[Tree]:
    #     events_by_span = self._get_events_by_span()

    #     trees = []
    #     tree = Tree()

    #     for span, sorted_events in events_by_span.items():
    #         # create root node i.e. span node
    #         tree.create_node(
    #             tag=f"{span} (SPAN)",
    #             identifier=span,
    #             parent=None,
    #             data=sorted_events[0].timestamp,
    #         )

    #         for event in sorted_events:
    #             tree.create_node(
    #                 tag=f"{event.class_name()}: {event.id_}",
    #                 identifier=event.id_,
    #                 parent=event.span_id,
    #                 data=event.timestamp,
    #             )

    #         trees.append(tree)
    #         tree = Tree()
    #     return trees

    # def print_event_span_trees(self) -> None:
    #     """Method for viewing trace trees."""
    #     trees = self._get_event_span_trees()
    #     for tree in trees:
    #         print(
    #             tree.show(
    #                 stdout=False, sorting=True, key=lambda node: node.data
    #             )
    #         )
    #         print("")



class SpanHandler(BaseSpanHandler[SimpleSpan]):
    # span_dict = {}
    span_dict: ClassVar[Dict[str, List[SimpleSpan]]] = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Create a span."""
        # logic for creating a new MyCustomSpan
        if id_ not in self.span_dict:
            self.span_dict[id_] = []
        self.span_dict[id_].append(
            SimpleSpan(id_=id_, parent_id=parent_span_id)
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()



class LlamaIndexInstrumentationTracer:
    def __init__(self, user_detail):
        """Initialize the LlamaIndexTracer with handlers but don't start tracing yet."""
        # Initialize the root dispatcher
        self.root_dispatcher = get_dispatcher()

        # Initialize handlers
        self.json_event_handler = EventHandler()
        self.span_handler = SpanHandler()
        self.simple_span_handler = SimpleSpanHandler()

        self.is_tracing = False  # Flag to check if tracing is active

        self.user_detail = user_detail

    def start(self):
        """Start tracing by registering handlers."""
        if self.is_tracing:
            print("Tracing is already active.")
            return

        # Register handlers
        self.root_dispatcher.add_span_handler(self.span_handler)
        self.root_dispatcher.add_span_handler(self.simple_span_handler)
        self.root_dispatcher.add_event_handler(self.json_event_handler)

        self.is_tracing = True
        print("Tracing started.")

    def stop(self):
        """Stop tracing by unregistering handlers."""
        if not self.is_tracing:
            print("Tracing is not active.")
            return

        # Write current_trace to a JSON file
        final_traces = {
            "project_id": self.user_detail["project_id"],
            "trace_id": str(uuid.uuid4()),
            "session_id": None,
            "trace_type": "llamaindex",
            "metadata": self.user_detail["trace_user_detail"]["metadata"],
            "pipeline": self.user_detail["trace_user_detail"]["pipeline"],
            "traces": self.json_event_handler.current_trace,

        }

        with open('new_llamaindex_traces.json', 'w') as f:
            json.dump([final_traces], f, default=str, indent=4)
        
        llamaindex_instrumentation_data = extract_llama_index_data([final_traces])
        converted_back_to_callback = convert_llamaindex_instrumentation_to_callback(llamaindex_instrumentation_data)

         # Just indicate tracing is stopped
        self.is_tracing = False
        print("Tracing stopped.")
        return converted_back_to_callback