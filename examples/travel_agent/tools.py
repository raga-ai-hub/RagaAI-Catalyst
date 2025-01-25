import os
import random
import requests
from dotenv import load_dotenv
from openai import OpenAI

from config import initialize_tracer

# Initialize tracer
tracer = initialize_tracer()

@tracer.trace_llm(name="llm_call")
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini", name="default"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tracer.span("llm_call").add_metrics(
            name=f"Q/A_v3_{random.randint(1, 10000)}", 
            score=0.3, 
            reasoning="Some Reason 1", 
            cost=0.0003, 
            latency=0.002
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

@tracer.trace_tool(name="weather_tool")
def weather_tool(destination):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    tracer.span("weather_tool").add_metrics(
            name="Q/A_v2",
            score=0.3,
            reasoning="Some Reason 2",
            cost=0.00036,
            latency=0.0021,
        )
    params = {"q": destination, "appid": api_key, "units": "metric"}
    print("Calculating weather for:", destination)
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"{data['weather'][0]['description'].capitalize()}, {data['main']['temp']:.1f}°C"
    except requests.RequestException:
        return "Weather data not available."

@tracer.trace_tool(name="currency_converter_tool")
def currency_converter_tool(amount, from_currency, to_currency):
    tracer.span("currency_converter_tool").add_metrics(
            name="Q/A_v2",
            score=0.11,
            reasoning="Some Reason 4",
            cost=0.0009,
            latency=0.0089,
        )
    api_key = os.environ.get("EXCHANGERATE_API_KEY")
    base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        return amount * data["conversion_rate"] if data["result"] == "success" else None
    except requests.RequestException:
        return None

@tracer.trace_tool(name="flight_price_estimator_tool")
def flight_price_estimator_tool(origin, destination):
    tracer.span("flight_price_estimator_tool").add_metrics(
            name="Q/A_v1",
            score=0.67,
            reasoning="Some Reason 3",
            cost=0.0067,
            latency=0.0011,
        )
    return f"Estimated price from {origin} to {destination}: $500-$1000"
