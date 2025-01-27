from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst
import asyncio
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from litellm import completion
import openai
from openai import OpenAI
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from ragaai_catalyst.tracers.agentic_tracing.tracers.main_tracer import AgenticTracing
import json
from typing import Dict, List, Any
import time


catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)
# Initialize tracer
tracer = Tracer(
    project_name="cost_testing",
    dataset_name="sync_sample_llm_testing_openai",
    tracer_type="anything",
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
load_dotenv()

tracer.start()


@tracer.trace_custom(name="process_data", custom_type="data_processor", trace_variables=False)
def process_data(data):
    """Example function showing custom function tracing with line traces"""
    processed = []
    total = 0
    print('my name is khan1')
    for i, item in enumerate(data):
        value = item * 2
        total += value
        processed.append(value)
        if i == len(data) - 1:
            average = total / len(data)
            print('average is', average)
    
    return processed

@tracer.trace_custom(name="calculate_statistics", custom_type="data_processor", trace_variables=False)
def calculate_statistics(numbers):
    """Example function using the trace_custom decorator without line traces"""
    stats = {}
    
    # Calculate mean
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    stats['mean'] = mean
    
    # Calculate range
    min_val = min(numbers)
    max_val = max(numbers)
    diff = max_val - min_val
    stats['range'] = diff

    print('my name is khan2')
    print('stats are', stats)
    
    return stats

@tracer.trace_custom(name="network_call", custom_type="network_call", trace_variables=True)
def weather_tool(destination='kerela'):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {"q": destination, "appid": api_key, "units": "metric"}
    print("Calculating weather for:", destination)
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        
        actual_result = f"{weather_description.capitalize()}, {temperature:.1f}°C"
        
        return actual_result
    except requests.RequestException:
        return "Weather data not available."


def main():
    try:
        # Example data processing
        data = [1, 2, 3, 4, 5]
        processed_data = process_data(data)
        print("Processed Data:", processed_data)
        stats = calculate_statistics(processed_data)
        print("Statistics:", stats)
        weather_result = weather_tool()
        print("Weather Result:", weather_result)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
    tracer.stop()