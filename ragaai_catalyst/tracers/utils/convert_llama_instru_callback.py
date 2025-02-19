def convert_llamaindex_instrumentation_to_callback(data):
    data = data[0]
    initial_struc = [{
        "trace_id": data["trace_id"],
        "project_id": data["project_id"],
        "session_id": data["session_id"],
        "trace_type": data["trace_type"],
        "metadata" : data["metadata"],
        "pipeline" : data["pipeline"],
        "traces" : []
    }]

    traces_data = []

    prompt = data["data"]["prompt"]
    response = data["data"]["response"]
    context = data["data"]["context"]
    system_prompt = data["data"]["system_prompt"]

    prompt_structured_data = {
        "event_type": "query",
        "payload": {
            "query_str": prompt
        }
    }
    traces_data.append(prompt_structured_data)

    response_structured_data = {
        "event_type": "llm",
        "payload": {
            "response": {
                "message": {
                    "content": response,
                }
            }
        }
    }
    traces_data.append(response_structured_data)

    context_structured_data = {
        "event_type": "retrieve",
        "payload": {
            "nodes": [
                {
                    "node": {
                        "text": context
                    }
                }
            ]
        }
    }
    traces_data.append(context_structured_data)

    system_prompt_structured_data = {
        "event_type": "llm",
        "payload": {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        }
    }
    traces_data.append(system_prompt_structured_data)

    initial_struc[0]["traces"] = traces_data    

    return initial_struc