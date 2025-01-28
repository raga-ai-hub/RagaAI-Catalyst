import sys
import asyncio
import json
from typing import Dict, Any
from argparse import ArgumentParser
# sys.path.append("/Users/ragaai_user/work/ragaai-catalyst")
from dotenv import load_dotenv
load_dotenv()

from agents.coordinator import CoordinatorAgent

from ragaai_catalyst import Tracer, RagaAICatalyst, init_tracing

catalyst = RagaAICatalyst(
    access_key="saLy6KmMVlfAzunuQGS9",
    secret_key="lm39fd4KXffM6gzLjnY9G7QReffhH4RGZPursp3A",
    base_url="http://52.172.168.127/api"
    # access_key="yKmaYjirGa6xoaLZQnCL",
    # secret_key="nXPYVEndpMkxLWJI8KllgRrx7DHkSRbtq1a0U9yO",
    # base_url="https://llm-dev5.ragaai.ai/api"
)
# Initialize tracer
tracer = Tracer(
    project_name="testing-multi-file-tracing",
    dataset_name="trial_01",
    tracer_type="Agentic",
    metadata={
        "model": "gpt-3.5-turbo",
        "environment": "production"
    },
    pipeline={
        "llm_model": "gpt-3.5-turbo",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)

init_tracing(tracer=tracer, catalyst=catalyst)

@tracer.trace_agent('Conduct Research')
async def conduct_research(research_question: str, parameters: Dict[str, Any] = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Conduct research using the autonomous research agent system.
    
    Args:
        research_question: The research question to investigate
        parameters: Optional parameters to configure the research process
        
    Returns:
        Dictionary containing research results
    """
    # Initialize coordinator
    coordinator = CoordinatorAgent(config=config)
    
    # Prepare input data
    input_data = {
        "research_question": research_question,
        "parameters": parameters or {}
    }
    
    # Run research process
    results = await process_research(coordinator, input_data)
    return results

@tracer.trace_agent('Process Research')
async def process_research(coordinator: CoordinatorAgent, input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Run research process
    results = await coordinator.process(input_data)
    return results

@tracer.trace_tool('Display Results')
def display_results(results: Dict[str, Any]) -> None:
    """Display research results in a formatted manner.
    
    Args:
        results: Dictionary containing research results
    """
    print("\nKey Findings:")
    for finding in results.get("findings", []):
        print(f"- {finding.get('summary', '')}")
    
    print("\nConclusions:")
    for conclusion in results.get("conclusions", []):
        print(f"- {conclusion.get('conclusion', '')}")
    
    print("\nRecommendations:")
    for recommendation in results.get("recommendations", []):
        print(f"- {recommendation.get('recommendation', '')}")

@tracer.trace_agent('Main')
async def main():
    # Load environment variables
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--provider", type=str, default='openai')
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--async_llm", type=bool, default=False)
    parser.add_argument("--syntax", type=str, default="chat")
    args = parser.parse_args()
    config = {
        "model": args.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "async_llm": args.async_llm,
        "syntax": args.syntax
    }
    # if args.config:
    #     with open(args.config, "r") as f:
    #         config.update(json.load(f))
    
    # Define research question and parameters
    research_question = "What are the latest developments in few-shot learning for NLP tasks?"
    parameters = {
        "max_sources": 10,
        "time_range": "last_year",
        "focus_areas": ["academic_papers", "tech_blogs", "conferences"]
    }
    
    try:
        # Conduct research
        print(f"Starting research on: {research_question}")
        results = await conduct_research(research_question, parameters, config)
        
        # Display results
        display_results(results)
        
        # Save results to file
        output_file = "research_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    except Exception as e:
        print(f"Error during research: {str(e)}")

if __name__ == "__main__":
    with tracer:
        # Run main function
        asyncio.run(main())

    # asyncio.run(main())