import sys
sys.path.append('.')

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from typing import Any

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
@tool

def write_to_file(filename: str, content: str) -> str:
    """Write content to a file with the specified filename."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Content successfully written to {filename}"

@tracer.trace_agent('brainstormer')
def create_brainstormer():
    return Agent(
        role="Idea Generator",
        goal="Come up with a creative premise for a sci-fi story set in 2050",
        backstory="You are a visionary thinker who loves crafting imaginative sci-fi concepts.",
        verbose=True,
        allow_delegation=False
    )

@tracer.trace_agent('outliner')
def create_outliner():
    return Agent(
        role="Story Outliner",
        goal="Create a structured outline based on the brainstormed premise",
        backstory="You are an expert at organizing ideas into compelling story frameworks.",
        verbose=True,
        allow_delegation=False
    )

@tracer.trace_agent('writer')
def create_writer():
    return Agent(
        role="Story Writer",
        goal="Write a short sci-fi story based on the outline and save it to a file",
        backstory="You are a skilled writer with a flair for vivid sci-fi narratives.",
        verbose=True,
        tools=[write_to_file],
        allow_delegation=False
    )

@tracer.trace_custom('brainstorm_task')
def create_brainstorm_task(brainstormer):
    return Task(
        description="Generate a unique sci-fi story premise set in 2050. Include a setting, main character, and conflict.",
        expected_output="A one-paragraph premise (e.g., 'In 2050, on a floating city above Venus, a rogue AI engineer battles a sentient cloud threatening humanity').",
        agent=brainstormer
    )

@tracer.trace_custom('outline_task')
def create_outline_task(outliner, brainstorm_task):
    return Task(
        description="Take the premise and create a simple story outline with 3 sections: Beginning, Middle, End.",
        expected_output="A bullet-point outline (e.g., '- Beginning: Engineer discovers the sentient cloud...').",
        agent=outliner,
        context=[brainstorm_task]
    )

@tracer.trace_custom('writing_task')
def create_writing_task(writer, outline_task):
    return Task(
        description="""Write a short (300-500 word) sci-fi story based on the outline. 
                    Then use the FileWriteTool to save it as 'sci_fi_story.md'.""",
        expected_output="A markdown file containing the full story.",
        agent=writer,
        context=[outline_task]
    )

def main():
    # Create agents
    brainstormer = create_brainstormer()
    outliner = create_outliner()
    writer = create_writer()

    # Create tasks
    brainstorm_task = create_brainstorm_task(brainstormer)
    outline_task = create_outline_task(outliner, brainstorm_task)
    writing_task = create_writing_task(writer, outline_task)

    # Create and configure crew
    crew = Crew(
        agents=[brainstormer, outliner, writer],
        tasks=[brainstorm_task, outline_task, writing_task],
        process=Process.sequential,
        verbose=True
    )

    print("Starting the CrewAI Story Generation process...")
    result = crew.kickoff()

    print("\nProcess completed! Final output:")
    print(result)

    try:
        with open("sci_fi_story.md", "r") as file:
            print("\nGenerated Story Content:")
            print(file.read())
    except FileNotFoundError:
        print("Story file not found. Check the writer agent's execution.")
    
    return result

if __name__ == "__main__":
    with tracer:
        main()
    tracer.get_upload_status()