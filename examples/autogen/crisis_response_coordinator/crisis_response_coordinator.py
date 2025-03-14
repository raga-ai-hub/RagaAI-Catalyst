import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import datetime
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer


catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
# Initialize tracer
tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/autogen",
)
load_dotenv()

init_tracing(catalyst=catalyst, tracer=tracer)

config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    file_location="examples/autogen/crisis_response_coordinator"
)

assistant = AssistantAgent(
    name="Crisis_Coordinator",
    system_message="You are a crisis response coordinator. Coordinate between news monitoring, resource mapping, and volunteer notification agents.",
    llm_config={"config_list": config_list}
)

user_proxy = UserProxyAgent(
    name="Admin",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "crisis"}, 
    max_consecutive_auto_reply=20,
)

def extract_location(text: str) -> str:
    """Use LLM to extract location from text through dedicated sub-agents"""
    if not text:
        return "Unknown location"

    location_assistant = AssistantAgent(
        name="Location_Expert",
        system_message="You are a geography expert. Extract ONLY the primary location name from text. Respond ONLY with the location.",
        llm_config={"config_list": config_list}
    )

    location_user = UserProxyAgent(
        name="Location_User",
        human_input_mode="NEVER",
        code_execution_config=False, 
        max_consecutive_auto_reply=0
    )

    try:
        location_user.initiate_chat(
            location_assistant,
            message=f"Extract the primary geographic location from this news content. Respond ONLY with the location name:\n{text}"
        )
        
        last_message = location_assistant.last_message()["content"]
        return last_message.strip(" .").split(",")[0]  # Take primary location
    except Exception as e:
        print(f"Location extraction error: {str(e)}")
        return "Unknown location"

@assistant.register_for_llm(name="monitor_disasters")
@user_proxy.register_for_execution(name="monitor_disasters")
def monitor_disasters(keywords: list = ["earthquake", "flood", "wildfire"]):
    """Monitor news for disaster alerts using NewsAPI"""
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": " OR ".join(keywords),
                "apiKey": os.getenv("NEWSAPI_KEY"),
                "sortBy": "publishedAt",
                "pageSize": 10
            }
        )
        articles = response.json().get("articles", [])
        
        disasters = []
        for article in articles:
            content = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            print(f"Extracting location from: {content}")
            disasters.append({
                "title": article["title"],
                "description": article["description"],
                "location": extract_location(content),
                "publishedAt": article["publishedAt"]
            })
        return disasters
    except Exception as e:
        return f"Error monitoring disasters: {str(e)}"


@assistant.register_for_llm(name="map_resources")
@user_proxy.register_for_execution(name="map_resources")
def map_resources(location: str):
    """Find nearby emergency resources using OpenStreetMap"""
    try:
        geo_response = requests.get(
            "https://api.opencagedata.com/geocode/v1/json",
            params={"q": location, "key": os.getenv("OPENCAGE_KEY")}
        )
        coords = geo_response.json()["results"][0]["geometry"]
        
        overpass_query = f"""
        [out:json];
        node[emergency](around:10000,{coords['lat']},{coords['lng']});
        out body;
        """
        osm_response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=overpass_query
        )
        
        resources = []
        for item in osm_response.json().get("elements", []):
            resources.append({
                "name": item.get("tags", {}).get("name", "Unnamed Facility"),
                "type": item["tags"].get("emergency", "unknown"),
                "address": item["tags"].get("addr:street", "Address not available"),
                "coordinates": (item["lat"], item["lon"])
            })
        return resources
    except Exception as e:
        return f"Error mapping resources: {str(e)}"

@assistant.register_for_llm(name="notify_volunteers")
@user_proxy.register_for_execution(name="notify_volunteers")
def notify_volunteers(disaster_info: dict, resources: list):
    """Send emergency alerts via email using SMTP"""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"Crisis Alert: {disaster_info['title']}"
        msg["From"] = os.getenv("SMTP_EMAIL")
        msg["To"] = ", ".join(os.getenv("VOLUNTEER_EMAILS").split(","))
        
        html = f"""<h2>{disaster_info['title']}</h2>
        <p><strong>Time:</strong> {disaster_info['publishedAt']}</p>
        <p>{disaster_info['description']}</p>
        <h3>Nearby Resources:</h3>
        <ul>"""
        
        for resource in resources[:5]:  # Limit to top 5 results
            map_link = f"https://www.openstreetmap.org/?mlat={resource['coordinates'][0]}&mlon={resource['coordinates'][1]}#map=16/{resource['coordinates'][0]}/{resource['coordinates'][1]}"
            html += f"""
            <li>
                <strong>{resource['name']}</strong> ({resource['type']})<br>
                {resource['address']}<br>
                <a href="{map_link}">View on Map</a>
            </li>"""
        html += "</ul>"
        
        msg.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
            server.send_message(msg)
        
        return "Alert sent successfully"
    except Exception as e:
        return f"Error sending notification: {str(e)}"


def main():
    user_proxy.initiate_chat(
        assistant,
        message="""Monitor for disasters, map resources, and notify volunteers.
        Use the functions in order: monitor_disasters -> map_resources -> notify_volunteers."""
    )

if __name__ == "__main__":
    try:
        with tracer:
            main()
    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        print(f"Crisis response cycle completed at {datetime.datetime.now()}")