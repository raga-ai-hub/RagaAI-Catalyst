import os
import time
from typing import List, Optional, Callable, Any
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import Agent, Runner, ModelSettings, set_tracing_export_api_key

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

load_dotenv()
set_tracing_export_api_key(os.getenv('OPENAI_API_KEY'))

def initialize_catalyst():
    """Initialize RagaAI Catalyst using environment credentials."""
    catalyst = RagaAICatalyst(
        access_key=os.getenv('CATALYST_ACCESS_KEY'), 
        secret_key=os.getenv('CATALYST_SECRET_KEY'), 
        base_url=os.getenv('CATALYST_BASE_URL')
    )
    
    tracer = Tracer(
        project_name=os.environ.get('PROJECT_NAME', 'email-extraction'),
        dataset_name=os.environ.get('DATASET_NAME', 'email-data'),
        tracer_type="agentic/openai_agents",
    )
    
    init_tracing(catalyst=catalyst, tracer=tracer)

class Person(BaseModel):
    """Person data model for email sender and recipients."""
    name: str
    role: Optional[str] = None
    contact: Optional[str] = None

class Meeting(BaseModel):
    """Meeting data model for scheduled meetings in emails."""
    date: str
    time: str
    location: Optional[str] = None
    duration: Optional[str] = None

class Task(BaseModel):
    """Task data model for action items in emails."""
    description: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = None

class EmailData(BaseModel):
    """Complete email data model with structured information."""
    subject: str
    sender: Person
    recipients: List[Person]
    main_points: List[str]
    meetings: List[Meeting]
    tasks: List[Task]
    next_steps: Optional[str] = None

def initialize_agent(agent_name: str, agent_instructions: str|Callable, handoff_description: Optional[str]=None, handoffs: List[Agent]=list(), model_name: str='gpt-4o', temperature: float=0.3, max_tokens: int=1000, output_type: Optional[type[Any]]=None):
    """Initialize the OpenAI agent for email extraction."""
    # Initialize the agent with appropriate configuration
    # This could include model selection, temperature settings, etc.
    model_settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens
    )
    agent = Agent(
        name=agent_name,
        instructions=agent_instructions,
        handoff_description=handoff_description,
        handoffs=handoffs,
        model=model_name,
        model_settings=model_settings, 
        output_type=output_type
    )
    return agent

email_extractor = initialize_agent(
    agent_name="Email Extractor",
    agent_instructions="You are an expert at extracting structured information from emails.",
    model_name="gpt-4o",
    temperature=0.2,
    output_type=EmailData
)

async def extract_email_data(email_text: str) -> EmailData:
    """
    Extract structured data from an email using an OpenAI agent.
    
    Args:
        email_text: The raw email text to process
        
    Returns:
        EmailData object containing structured information from the email
    """
    runner = Runner()
    extraction_prompt = f"Please extract information from this email:\n\n{email_text}"
    result = await runner.run(
        email_extractor,
        extraction_prompt
    )
    return result.final_output

sample_email = """
From: Alex Johnson <alex.j@techcorp.com>
To: Team Development <team-dev@techcorp.com>
CC: Sarah Wong <sarah.w@techcorp.com>, Miguel Fernandez <miguel.f@techcorp.com>
Subject: Project Phoenix Update and Next Steps

Hi team,

I wanted to follow up on yesterday's discussion about Project Phoenix and outline our next steps.

Key points from our discussion:
- The beta testing phase has shown promising results with 85% positive feedback
- We're still facing some performance issues on mobile devices
- The client has requested additional features for the dashboard

Let's schedule a follow-up meeting this Friday, June 15th at 2:00 PM in Conference Room B. The meeting should last about 1.5 hours, and we'll need to prepare the updated project timeline.

Action items:
1. Sarah to address the mobile performance issues by June 20th (High priority)
2. Miguel to create mock-ups for the new dashboard features by next Monday
3. Everyone to review the beta testing feedback document and add comments by EOD tomorrow

If you have any questions before Friday's meeting, feel free to reach out.

Best regards,
Alex Johnson
Senior Project Manager
(555) 123-4567
"""

def display_email_data(email_data: EmailData):
    """
    Display the extracted email data in a formatted way.
    
    Args:
        email_data: The structured EmailData object to display
    """
    print(f"Subject: {email_data.subject}")
    print(f"From: {email_data.sender.name} ({email_data.sender.role})")
    
    print("\nMain points:")
    for point in email_data.main_points:
        print(f"- {point}")
    
    print("\nMeetings:")
    for meeting in email_data.meetings:
        print(f"- {meeting.date} at {meeting.time}, Location: {meeting.location}")
    
    print("\nTasks:")
    for task in email_data.tasks:
        print(f"- {task.description}")
        print(
            f"  Assignee: {task.assignee}, Deadline: {task.deadline}, Priority: {task.priority}"
        )
    
    if email_data.next_steps:
        print(f"\nNext Steps: {email_data.next_steps}")

async def process_email(email_text: str):
    """
    Process an email to extract structured data and display the results.
    
    Args:
        email_text: The raw email text to process
        
    Returns:
        The structured EmailData object
    """
    if os.getenv('CATALYST_ACCESS_KEY'):
        initialize_catalyst()
    
    start_time = time.time()
    email_data = await extract_email_data(email_text)
    duration = time.time() - start_time
    
    print(f"Email processing completed in {duration:.2f} seconds")
    display_email_data(email_data)
    
    return email_data

if __name__ == "__main__":
    import asyncio
    
    asyncio.run(process_email(sample_email))