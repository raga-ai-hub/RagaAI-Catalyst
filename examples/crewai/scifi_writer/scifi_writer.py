from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process


load_dotenv()

class FileWriteTool:
    def __init__(self, filename):
        self.filename = filename

    def write(self, content):
        with open(self.filename, "w") as f:
            f.write(content)
        return f"Content written to {self.filename}"

file_write_tool = FileWriteTool("sci_fi_story.md")

brainstormer = Agent(
    role="Idea Generator",
    goal="Come up with a creative premise for a sci-fi story set in 2050",
    backstory="You are a visionary thinker who loves crafting imaginative sci-fi concepts.",
    verbose=True,
    tools=[],
    allow_delegation=False
)

outliner = Agent(
    role="Story Outliner",
    goal="Create a structured outline based on the brainstormed premise",
    backstory="You are an expert at organizing ideas into compelling story frameworks.",
    verbose=True,
    tools=[],
    allow_delegation=False
)

writer = Agent(
    role="Story Writer",
    goal="Write a short sci-fi story based on the outline and save it to a file",
    backstory="You are a skilled writer with a flair for vivid sci-fi narratives.",
    verbose=True,
    tools=[file_write_tool],
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
    agent=outliner
)

writing_task = Task(
    description="Write a short (300-500 word) sci-fi story based on the outline. Save it as 'sci_fi_story.md' using the provided tool.",
    expected_output="A markdown file containing the full story.",
    agent=writer
)

crew = Crew(
    agents=[brainstormer, outliner, writer],
    tasks=[brainstorm_task, outline_task, writing_task],
    process=Process.sequential,
    verbose=2
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