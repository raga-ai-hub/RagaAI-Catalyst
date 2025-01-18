import litellm
import json
import requests
import os
import logging
logger = logging.getLogger('LiteLLM')
logger.setLevel(logging.ERROR)

class GuardExecutor:

    def __init__(self,id,guard_manager,field_map={}):
        self.deployment_id = id
        self.field_map = field_map
        self.guard_manager = guard_manager
        self.deployment_details = self.guard_manager.get_deployment(id)
        if not self.deployment_details:
            raise ValueError('Error in getting deployment details')
        self.base_url = guard_manager.base_url
        for key in field_map.keys():
            if key not in ['prompt','context','response','instruction']:
                print('Keys in field map should be in ["prompt","context","response","instruction"]')

    def execute_deployment(self,payload):
        api = self.base_url + f'/guardrail/deployment/{self.deployment_id}/ingest'

        payload = json.dumps(payload)
        headers = {
            'x-project-id': str(self.guard_manager.project_id),
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
        }
        try:
            response = requests.request("POST", api, headers=headers, data=payload,timeout=self.guard_manager.timeout)
        except Exception as e:
            print('Failed running guardrail: ',str(e))
            return None
        if response.status_code!=200:
            print('Error in running deployment ',response.json()['message'])
        if response.json()['success']:
            return response.json()
        else:
            print(response.json()['message'])
            return None

    def llm_executor(self,messages,model_params,llm_caller):
        if llm_caller == 'litellm':
            model_params['messages'] = messages
            response = litellm.completion(**model_params)
            return response
        else:
            print(f"{llm_caller} not supported currently, use litellm as llm caller")

    
    def __call__(self,messages,prompt_params,model_params,llm_caller='litellm'):
        for key in self.field_map:
            if key not in ['prompt','response']:
                if self.field_map[key] not in prompt_params:
                    raise ValueError(f'{key} added as field map but not passed as prompt parameter')
        context_var = self.field_map.get('context',None)
        prompt = None
        for msg in messages:
            if 'role' in msg:
                if msg['role'] == 'user':
                    prompt = msg['content']
                    if not context_var:
                        msg['content'] += '\n' + prompt_params[context_var]
        doc = dict()
        doc['prompt'] = prompt
        doc['context'] = prompt_params[context_var]
        
        # inactive the guardrails that needs Response variable
        #deployment_response = self.execute_deployment(doc)
        
        # activate only guardrails that require response
        try:
            llm_response = self.llm_executor(messages,model_params,llm_caller)
        except Exception as e:
            print('Error in running llm:',str(e))
            return None
        doc['response'] = llm_response['choices'][0].message.content
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
            doc['instruction'] = instruction
        response = self.execute_deployment(doc)
        if response and response['data']['status'] == 'FAIL':
            print('Guardrail deployment run retured failed status, replacing with alternate response')
            return response['data']['alternateResponse'],llm_response,response
        else:
            return None,llm_response,response




        



