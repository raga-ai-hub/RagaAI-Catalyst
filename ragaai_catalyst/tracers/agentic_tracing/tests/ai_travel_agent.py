import os
import json
from openai import OpenAI
import requests
from datetime import datetime
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tracer
from tracer import Tracer
tracer = Tracer(
    project_name="travel_agent_demo",
    output_dir="./traces"
)

# Start tracing
tracer.start()

@tracer.trace_tool(
    name="llm_call",
    tool_type="llm",
    version="1.0.0"
)
def llm_call(prompt, max_tokens=512, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@tracer.trace_tool(
    name="weather_tool",
    tool_type="api",
    version="1.0.0"
)
def weather_tool(destination):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": destination, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return f"{weather_description.capitalize()}, {temperature:.1f}°C"
    except requests.RequestException:
        return "Weather data not available."

@tracer.trace_tool(
    name="currency_converter_tool",
    tool_type="api",
    version="1.0.0"
)
def currency_converter_tool(amount, from_currency, to_currency):
    api_key = os.environ.get("EXCHANGERATE_API_KEY")
    base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        if data["result"] == "success":
            rate = data["conversion_rate"]
            return amount * rate
        return None
    except requests.RequestException:
        return None

@tracer.trace_tool(
    name="flight_price_estimator_tool",
    tool_type="mock",
    version="1.0.0"
)
def flight_price_estimator_tool(origin, destination):
    return f"Estimated price from {origin} to {destination}: $500-$1000"

@tracer.trace_agent(
    name="itinerary_agent",
    agent_type="planner",
    capabilities=["itinerary_planning", "llm_interaction"]
)
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
            return llm_call(itinerary_prompt, max_tokens=512)

@tracer.trace_agent(
    name="travel_agent",
    agent_type="orchestrator",
    capabilities=["preference_extraction", "travel_planning", "information_gathering"]
)
def travel_agent():
    print("Welcome to the Personalized Travel Planner!\n")

    # Get user input
    user_input = "karela, 10 days, $100, nature"

    # Extract preferences
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

    # Parse extracted preferences
    preferences = {}
    for line in extracted_preferences.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            preferences[key.strip()] = value.strip()

    # Validate extracted preferences
    required_keys = ["Destination", "Activities", "Budget", "Duration (in days)"]
    if not all(key in preferences for key in required_keys):
        print("\nCould not extract all required preferences. Please try again.")
        return

    # Fetch additional information
    weather = weather_tool(preferences["Destination"])
    print(f"\nWeather in {preferences['Destination']}: {weather}")

    origin = "delhi"
    flight_price = flight_price_estimator_tool(origin, preferences["Destination"])
    print(flight_price)

    # Plan itinerary
    itinerary_agent = ItineraryAgent()
    itinerary = itinerary_agent.plan_itinerary(
        extracted_preferences, int(preferences["Duration (in days)"])
    )
    print("\nPlanned Itinerary:")
    print(itinerary)

    # Currency conversion
    budget_amount = float(preferences["Budget"].replace("$", "").replace(",", ""))
    converted_budget = currency_converter_tool(budget_amount, "USD", "INR")
    if converted_budget:
        print(f"\nBudget in INR: {converted_budget:.2f} INR")
    else:
        print("\nCurrency conversion not available.")

    # Generate travel summary
    summary_prompt = f"""
Summarize the following travel plan:

Destination: {preferences['Destination']}
Activities: {preferences['Activities']}
Budget: {preferences['Budget']}
Duration: {preferences['Duration (in days)']} days
Itinerary: {itinerary}
Weather: {weather}
Flight Price: {flight_price}

Travel Summary:
"""
    travel_summary = llm_call(summary_prompt, max_tokens=2048)
    print("\nTravel Summary:")
    print(travel_summary)

def main():
    try:
        travel_agent()
    finally:
        # Stop tracing and save results
        tracer.stop()

if __name__ == "__main__":
    main()