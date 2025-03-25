import sys
sys.path.append('.')

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from typing import Any
import json
from datetime import datetime

load_dotenv()

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'),
    secret_key=os.getenv("CATALYST_SECRET_KEY"),
    base_url=os.getenv('CATALYST_BASE_URL'))

tracer = Tracer(
    project_name='financial_expert4',
    dataset_name='dataset',
    tracer_type="Agentic",
)
init_tracing(catalyst=catalyst, tracer=tracer)

def write_to_file(filename: str, content: str) -> str:
    """Write content to a file with the specified filename."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Content successfully written to {filename}"

@tracer.trace_agent("market_research_analyst")
def create_market_researcher_agent():
    return Agent(
        role="Market Research Analyst",
        goal="Identify market opportunities and analyze potential business ideas",
        backstory="You are an experienced market analyst with expertise in identifying profitable business opportunities and market gaps.",
        verbose=True,
        allow_delegation=False
    )

@tracer.trace_agent("buisness_strategist")
def create_buisness_strategy():
    return Agent(
        role="Business Strategist",
        goal="Develop a comprehensive business strategy and revenue model",
        backstory="You are a strategic thinker with experience in business model development and strategic planning.",
        verbose=True,
        allow_delegation=False
    )

@tracer.trace_agent("financial_planner")
def create_financial_planner():
    return Agent(
        role="Financial Planner",
        goal="Create detailed financial projections and write the complete business plan",
        backstory="You are a financial expert skilled in creating business plans and financial forecasts.",
        verbose=True,
        allow_delegation=False
    )

@tracer.trace_custom("research_task")
def create_research_task(market_researcher):
    return Task(
        description="""Conduct market research and propose a innovative business idea. 
                    Include target market, problem being solved, and unique value proposition.""",
        expected_output="A detailed market analysis and business idea proposal (2-3 paragraphs).",
        agent=market_researcher
    )

@tracer.trace_custom("strategy_task")
def create_strategy_task(business_strategist, research_task):
    return Task(
        description="""Develop a business strategy including:
                    - Business model
                    - Revenue streams
                    - Marketing approach
                    - Competitive analysis""",
        expected_output="A comprehensive business strategy document with all key components.",
        agent=business_strategist,
        context=[research_task]
    )

@tracer.trace_custom("planning_task")
def create_planning_task(financial_planner, strategy_task):
    return Task(
        description="""Create a complete business plan including:
                    - Executive summary
                    - Financial projections
                    - Implementation timeline
                    Save the final plan as 'business_plan.md'""",
        expected_output="A markdown file containing the complete business plan.",
        agent=financial_planner,
        context=[strategy_task]
    )

def main():
    # Create agents
    market_researcher = create_market_researcher_agent()
    business_strategist = create_buisness_strategy()
    financial_planner = create_financial_planner()

    # Create tasks
    research_task = create_research_task(market_researcher)
    strategy_task = create_strategy_task(business_strategist, research_task)
    planning_task = create_planning_task(financial_planner, strategy_task)

    # Create and configure crew
    crew = Crew(
        agents=[market_researcher, business_strategist, financial_planner],
        tasks=[research_task, strategy_task, planning_task],
        process=Process.sequential,
        verbose=True
    )

    print("Starting the CrewAI Business Plan Generation process...")
    result = crew.kickoff()

    print("\nProcess completed! Final output:")
    print(result)

    try:
        with open("business_plan.md", "r") as file:
            print("\nGenerated Business Plan Content:")
            print(file.read())
    except FileNotFoundError:
        print("Business plan file not found. Check the financial planner agent's execution.")
    
    return result

if __name__ == "__main__":
    with tracer:
        main()
    tracer.get_upload_status()