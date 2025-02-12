import sys

from ragaai_catalyst.tracers.distributed import trace_agent 
sys.path.append('.')

import os
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing, trace_llm, current_span, trace_agent

from dotenv import load_dotenv
load_dotenv(override=True)

catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Initialize tracer
tracer = Tracer(
    project_name="Execute_Metric_Test1",
    dataset_name="joke_generation_workflow_async1",
    tracer_type="Agentic",
)

init_tracing(catalyst=catalyst, tracer=tracer)

class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = OpenAI()

    @step
    @trace_agent("generate joke")
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        prompt = f"Write your best joke about {topic}."
        
        # Get the current span and store its attributes
        span = current_span()
        span.add_context(context="joke generation")
        span.add_metrics(
            name="toxicity",
            score=0.4,
            reasoning="Reason for toxicity",
        )

        # First execute_metrics call
        span.execute_metrics(
            name="Hallucination",
            display_name="hallucination_generate_joke",
            provider="openai",
            model="gpt-4o-mini",
            mapping={
                "prompt": prompt,
                "response": "vnvnvs",
                "context": "Some Context"
            }
        )
        
        # Perform async operation
        response = await self.llm.acomplete(prompt)
        
        # Get fresh span after async operation
        span.execute_metrics(
            name="Hallucination",
            display_name="hallucination_generate_joke_2",
            provider="openai",
            model="gpt-4o-mini",
            mapping={
                "prompt": prompt,
                "response": response.text,
                "context": "Some Context"
            }
        )

        return JokeEvent(joke=str(response))

    @step
    @trace_agent("criticise joke")
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))
    

async def main():
    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(topic="climate change")
    print(str(result))

if __name__ == "__main__":
    import asyncio
    with tracer:
        asyncio.run(main())