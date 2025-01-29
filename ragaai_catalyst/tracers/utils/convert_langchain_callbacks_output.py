import json

def convert_langchain_callbacks_output(result, project_name="", metadata="", pipeline=""):
    initial_struc = [{
        "project_name": project_name,
        "trace_id": "NA",
        "session_id": "NA",
        "metadata" : metadata,
        "pipeline" : pipeline,
        "traces" : []
    }]
    traces_data = []

    prompt = result["data"]["prompt"]
    response = result["data"]["response"]
    context = result["data"]["context"]
    final_prompt = ""

    prompt_structured_data = {
        "traceloop.entity.input": json.dumps({
            "kwargs": {
                "input": prompt,
            }
        })
    }    
    prompt_data = {
        "name": "retrieve_documents.langchain.workflow",
        "attributes": prompt_structured_data,
    }

    traces_data.append(prompt_data)

    context_structured_data = {
        "traceloop.entity.input": json.dumps({
            "kwargs": {
                "context": context
            }
        }),
        "traceloop.entity.output": json.dumps({
            "kwargs": {
                "text": prompt
            }
        })
    }
    context_data = {
        "name": "PromptTemplate.langchain.task",
        "attributes": context_structured_data,
    }
    traces_data.append(context_data)

    response_structured_data = {"gen_ai.completion.0.content": response,
                                "gen_ai.prompt.0.content": prompt}
    response_data = {
        "name": "ChatOpenAI.langchain.task",
        "attributes" : response_structured_data
    }
    traces_data.append(response_data)

    initial_struc[0]["traces"] = traces_data

    return initial_struc