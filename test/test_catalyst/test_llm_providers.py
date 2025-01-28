import os
import re
import json
import subprocess
import logging
from typing import Dict, Optional
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, cwd: Optional[str] = None):
    command_in_debug = 'DEBUG=1 {}'.format(command)
    cwd = cwd or os.getcwd()
    logger.info(f"Running command: {command_in_debug} in cwd: {cwd}")
    try:
        result = subprocess.run(
            command_in_debug,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command run successfully")
        output = result.stdout + '\n' + result.stderr
        return output
    except Exception as e:
        logger.error(f"Command failed: {e}")
        raise

def extract_information(logs: str) -> Dict:
    logger.info("Extracting information from logs")
    pattern = re.compile(r"Trace saved to (.*)$")

    # Split the text into lines to process them individually
    lines = logs.splitlines()
    locations = []

    # Search each line for the pattern
    for line in lines:
        match = pattern.search(line)
        if match:
            # The captured group (.*) will contain the file path
            locations.append(match.group(1).strip())
    return locations[0]

@pytest.mark.parametrize("model, provider, async_llm, syntax", [
    ("gpt-4o-mini", "openai", False, "chat"),
    ("gpt-4o-mini", "openai", True, "chat"),
    ("gpt-4o-mini", "azure", False, "chat"),
    ("gpt-4o-mini", "chat", True, "chat"),
    ("gemini-1.5-pro", "google", False, ""),
    ("gemini-1.5-pro", "google", True, ""),
    ("gpt-4o-mini", "litellm", False, ""),
    ("gpt-4o-mini", "litellm", True, ""),
])
def test_llm_providers(model: str, provider: str, async_llm: bool, syntax: str):
    command = f'python test/test_catalyst/autonomous_research_agent/research_script.py --model {model} --provider {provider} --async_llm {async_llm} --syntax {syntax}'
    cwd = ''
    output = run_command(command)
    location = extract_information(output)

    with open(location, 'r') as f:
        data = json.load(f)
    span_data = data.get('data', [{}])[0]
    
    process_discovery_children = []
    process_synthesis_children = []
    for span in span_data['spans']:
        if span['name'] == 'Main':
            main_children = span['data']['children']
            for child in main_children:
                if child['name'] == 'Conduct Research':
                    conduct_research_children = child['data']['children']
                    for child in conduct_research_children:
                        if child['name'] == 'Process Research':
                            process_research_children = child['data']['children']
                            for child in process_research_children:
                                if child['name'] == 'Coordinate Research':
                                    coordinate_research_children = child['data']['children']
                                    for child in coordinate_research_children:
                                        if child['name'] == 'Process Discovery':
                                            process_discovery_children = child['data']['children']
                                        if child['name'] == 'Process Synthesis':
                                            process_synthesis_children = child['data']['children']
                                    break
                            break
                    break
            break
    assert len([child for child in process_discovery_children if child['name'] == 'llm response']) == 2, f"Expected 2 llm response children, got {len([child for child in process_discovery_children if child['name'] == 'llm response'])}"
    assert len([child for child in process_synthesis_children if child['name'] == 'llm response']) == 5, f"Expected 5 llm response children, got {len([child for child in process_synthesis_children if child['name'] == 'llm response'])}"

# if __name__ == '__main__':
#     command = 'python test/test_catalyst/autonomous_research_agent/research_script.py --provider azure --async_llm True'
#     cwd = ''
#     output = run_command(command)
#     print(output)