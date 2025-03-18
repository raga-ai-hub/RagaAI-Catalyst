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

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/crewai",
)
init_tracing(catalyst=catalyst, tracer=tracer)

@tool
def write_to_file(filename: str, content: str) -> str:
    """Write content to a file with the specified filename."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Content successfully written to {filename}"

brainstormer = Agent(
    role="Idea Generator",
    goal="Come up with a creative premise for a sci-fi story set in 2050",
    backstory="You are a visionary thinker who loves crafting imaginative sci-fi concepts.",
    verbose=True,
    allow_delegation=False
)

outliner = Agent(
    role="Story Outliner",
    goal="Create a structured outline based on the brainstormed premise",
    backstory="You are an expert at organizing ideas into compelling story frameworks.",
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="Story Writer",
    goal="Write a short sci-fi story based on the outline and save it to a file",
    backstory="You are a skilled writer with a flair for vivid sci-fi narratives.",
    verbose=True,
    tools=[write_to_file],
    allow_delegation=False
)

brainstorm_task = Task(
    description="Generate a unique sci-fi story premise set in 2050. Include a setting, main character, and conflict.",
    expected_output="A one-paragraph premise (e.g., 'In 2050, on a floating city above Venus, a rogue AI engineer battles a sentient cloud threatening humanity').",
    agent=brainstormer
)

outline_task = Task(
    description="Take the premise and create a simple story outline with 3 sections: Beginning, Middle, End.",
    expected_output="A bullet-point outline (e.g., '- Beginning: Engineer discovers the sentient cloud...').",
    agent=outliner,
    context=[brainstorm_task]  
)

writing_task = Task(
    description="""Write a short (300-500 word) sci-fi story based on the outline. 
                  Then use the FileWriteTool to save it as 'sci_fi_story.md'.""",
    expected_output="A markdown file containing the full story.",
    agent=writer,
    context=[outline_task]  
)

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