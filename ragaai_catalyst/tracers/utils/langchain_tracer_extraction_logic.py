import json
import uuid

def langchain_tracer_extraction(data, user_context=""):
    trace_aggregate = {}
    import uuid

    def generate_trace_id():
        """
        Generate a random trace ID using UUID4.
        Returns a string representation of the UUID with no hyphens.
        """
        return '0x'+str(uuid.uuid4()).replace('-', '')

    trace_aggregate["tracer_type"] = "langchain"
    trace_aggregate['trace_id'] = generate_trace_id()
    trace_aggregate['session_id'] = None
    trace_aggregate["pipeline"] = {
        'llm_model': 'gpt-3.5-turbo', 
        'vector_store': 'faiss',
        'embed_model': 'text-embedding-ada-002'
        }
    trace_aggregate["metadata"] = {
        'key1': 'value1',
        'key2': 'value2',
        'log_source': 'langchain_tracer',
        'recorded_on': '2024-06-14 08:57:27.324410'
        }
    trace_aggregate["prompt_length"] = 0
    trace_aggregate["data"] = {}

    def get_prompt(data):
        # if "chain_starts" in data and data["chain_starts"] != []:
        #     for item in data["chain_starts"]:

        if "chat_model_calls" in data and data["chat_model_calls"] != []:
            for item in data["chat_model_calls"]:
                messages = item["messages"][0]
                for message in messages:
                    if message["type"]=="human":
                        human_messages = message["content"].strip()
                        return human_messages
        if  "llm_calls" in data and data["llm_calls"] != []:
            if "llm_start" in data["llm_calls"][0]["event"]:
                for item in data["llm_calls"]:
                    prompt = item["prompts"]
                    return prompt[0].strip()

    def get_response(data):
        for item in data["llm_calls"]:
            if item["event"] == "llm_end":
                llm_end_responses = item["response"]["generations"][0]
                for llm_end_response in llm_end_responses:
                    response = llm_end_response["text"]
                return response.strip()

    def get_context(data, user_context):
        if user_context:
            return user_context
        if "retriever_actions" in data and data["retriever_actions"] != []:
            for item in data["retriever_actions"]:
                if item["event"] == "retriever_end":
                    context = item["documents"][0]["page_content"].replace('\n', ' ')
                    return context
        # if "chat_model_calls" in data and data["chat_model_calls"] != []:
        #     for item in data["chat_model_calls"]:
        #         messages = item["messages"][0]
        #         for message in messages:
        #             if message["type"]=="system":
        #                 content = message["content"].strip().replace('\n', ' ')
        #                 return content


    prompt = get_prompt(data)
    response = get_response(data)
    context = get_context(data, user_context)

    trace_aggregate["data"]["prompt"]=prompt
    trace_aggregate["data"]["response"]=response
    trace_aggregate["data"]["context"]=context

    return trace_aggregate
