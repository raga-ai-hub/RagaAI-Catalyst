import os
import requests
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI


load_dotenv()


import sys

sys.path.append(".")
from agentneo import AgentNeo, Tracer, Evaluation, launch_dashboard

neo_session = AgentNeo(session_name="ai_travel_agent_session22")

project_name = "ai_travel_agent_demo22"

try:
    neo_session.create_project(project_name=project_name)
    print("Project created successfully")
except:
    neo_session.connect_project(project_name=project_name)
    print("Project connected successfully")

tracer = Tracer(session=neo_session)
tracer.start()


# Shared LLM call function
# def llm_call(prompt, max_tokens=512, model="gpt-4o-mini"):
#     response = completion(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=max_tokens,
#         temperature=0.7,
#     )

#     return response.choices[0].message.content.strip()


@tracer.trace_llm(name="llm_call")
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# Tools outside agents
@tracer.trace_tool(name="weather_tool")
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


@tracer.trace_tool(name="currency_converter_tool")
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
        else:
            return None
    except requests.RequestException:
        return None


@tracer.trace_tool(name="flight_price_estimator_tool")
def flight_price_estimator_tool(origin, destination):
    # This is a mock function. In a real scenario, you'd integrate with a flight API.
    api_key = os.environ.get("FLIGHT_API_KEY")
    # Implement actual API call here
    return f"Estimated price from {origin} to {destination}: $500-$1000"


# Agent with persona
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
        return llm_call(itinerary_prompt, max_tokens=512)


# Main function
@tracer.trace_agent(name="travel_agent")
def travel_agent():
    print("Welcome to the Personalized Travel Planner!\n")

    # Get user input
    user_input = input("Please describe your ideal vacation: ")

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

    origin = input("Please enter your departure city: ")
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
    travel_summary = llm_call(summary_prompt, max_tokens=2248)
    print("\nTravel Summary:")
    print(travel_summary)


if __name__ == "__main__":
    travel_agent()
    tracer.stop()


    # Evaluate the performance
    exe = Evaluation(session=neo_session, trace_id=tracer.trace_id)
    # exe.evaluate(metric_list=['goal_decomposition_efficiency', 
    #                      'goal_fulfillment_rate', 
    #                      'tool_call_correctness_rate', 
    #                      'tool_call_success_rate'])
    # print the performance result


    CUSTOM_RUBRIC = {
    "criteria": {
        "problem_solving": {
            "description": "How effectively does the AI solve the user's problem?",
            "weight": 0.45
        },
        "clarity": {
            "description": "How clear and understandable are the AI's responses?",
            "weight": 0.45
        },
        "useful":{
            "description":"Is the response from AI useful to the user",
            "weight":0.1
        }
    },
    "scoring_guidelines": {
        "0.0-0.5":"Poor performance, significant improvements needed",
        "0.5-1.0": "Excellent performance, meets or exceeds expectations"
        }
    }
    
    # Example 1: Complete criteria with weights
    prompt1 = """
    Evaluate the conversation based on:
    1. Technical accuracy (40%) - How accurate is the technical information
    2. Response time (30%) - How quickly does the AI respond
    3. Completeness (30%) - How thorough are the responses
    
    Use very strict scoring criteria where excellent means perfect execution.
    """
    
    # Example 2: Criteria without weights
    prompt2 = """
    Please evaluate based on:
    - Code quality - How well-structured is the code
    - Documentation - How well is the code documented
    - Error handling - How robust is the error handling
    """
    
    # Example 3: Minimal prompt
    prompt3 = """
    Evaluate based on user satisfaction and response accuracy
    """

    # exe.evaluate(metric_list=['custom_evaluation_metric',])

    # exe.evaluate(metric_list=['custom_evaluation_metric',],
    #              custom_criteria=CUSTOM_RUBRIC)
    
    exe.evaluate(metric_list=['custom_evaluation_metric',],
                 custom_criteria=prompt1)
    
    # exe.evaluate(metric_list=['custom_evaluation_metric',],
    #              custom_criteria=prompt3)

    metric_results = exe.get_results()
    print(metric_results)
    

    # Launch dashboard
    launch_dashboard(port=3000)


