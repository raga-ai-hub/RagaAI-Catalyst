# Travel Agent Planner with AgentNeo Integration

# Setup and Imports
import os
import requests
from dotenv import load_dotenv
from litellm import completion
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize AgentNeo Package
from agentneo import AgentNeo, Tracer, Evaluation,launch_dashboard

neo_session = AgentNeo(session_name="test")
project_name = "ai_travel_agent_demo3"

try:
    neo_session.create_project(project_name=project_name)
    print("Project created successfully")
except:
    neo_session.connect_project(project_name=project_name)
    print("Project connected successfully")

tracer = Tracer(session=neo_session)
tracer.start()

print("GOOGLE_API_KEY:", os.environ.get("GOOGLE_API_KEY"))

# Travel Agent Tools
@tracer.trace_llm(name="llm_call")
def llm_call(prompt, model_name="gemini-1.5-flash-002"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Error during model content generation: {e}")

@tracer.trace_tool(name="flight_price_estimator_tool")
def flight_price_estimator_tool(origin, destination):
    return f"Estimated price from {origin} to {destination}: $500-$1000"

@tracer.trace_agent(name="itinerary_agent")
class ItineraryAgent:
    def __init__(self, persona="Itinerary Agent"):
        self.persona = persona

    def plan_itinerary(self, user_preferences, duration=3):
        itinerary_prompt = f"""
You are a travel expert named {self.persona}.
Based on the following user preferences, create a {duration}-day travel itinerary.

User Preferences:
{user_preferences}

Itinerary:
"""
        return llm_call(itinerary_prompt)

@tracer.trace_agent(name="travel_agent")
def travel_agent():
    print("Welcome to the Personalized Travel Planner!\n")

    user_input = "kerala, 10 days, $100, nature"
    origin = "delhi"

    preferences_prompt = f"""
Extract key travel preferences from the following user input:
"{user_input}"

Please provide the extracted information in this format:
Destination:
Activities:
Budget:
Duration (in days):
"""
    extracted_preferences = llm_call(preferences_prompt)
    print("\nExtracted Preferences:")
    print(extracted_preferences)

    preferences = {}
    try:
        for line in extracted_preferences.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                preferences[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error parsing preferences: {e}")
        return

    required_keys = ["Destination", "Activities", "Budget", "Duration (in days)"]
    if not all(key in preferences for key in required_keys):
        print("\nCould not extract all required preferences. Please try again.")
        return

    try:
        budget_amount = float(preferences["Budget"].replace("$", "").replace(",", ""))
        duration_days = int(preferences["Duration (in days)"])
    except ValueError:
        print("\nInvalid format for Budget or Duration. Please try again.")
        return

    flight_price = flight_price_estimator_tool(origin, preferences["Destination"])
    print(flight_price)

    itinerary_agent = ItineraryAgent()
    itinerary = itinerary_agent.plan_itinerary(
        extracted_preferences, duration_days
    )
    print("\nPlanned Itinerary:")
    print(itinerary)

    summary_prompt = f"""
Summarize the following travel plan:

Destination: {preferences['Destination']}
Activities: {preferences['Activities']}
Budget: {preferences['Budget']}
Duration: {preferences['Duration (in days)']} days
Itinerary: {itinerary}
Flight Price: {flight_price}

Travel Summary:
"""
    travel_summary = llm_call(summary_prompt)
    print("\nTravel Summary:")
    print(travel_summary)

def main():
    travel_agent()

if __name__ == "__main__":
    main()
    tracer.stop()
    neo_session.launch_dashboard()