import requests

def fetch_analysis_trace(base_url, trace_id):
    """
    Fetches the analysis trace data from the server.

    :param base_url: The base URL of the server (e.g., "http://localhost:3000").
    :param trace_id: The ID of the trace to fetch.
    :return: The JSON response from the server if successful, otherwise None.
    """
    try:
        url = f"{base_url}/api/analysis_traces/{trace_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching analysis trace: {e}")
        return None
