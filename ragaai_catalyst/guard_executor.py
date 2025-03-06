import litellm
import json
import requests
import os
from google import genai
from typing import Optional, List, Dict, Any
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
        self.current_doc_id = None
        self.id_2_doc = {}

    def execute_deployment(self, payload, is_output=False, executionId: Optional[str] = None):
        api = self.base_url + f'/guardrail/deployment/{self.deployment_id}/ingest'

        payload['input_guardrail'] = not is_output

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
            return response['choices'][0].message.content
        elif llm_caller == 'genai':
            genai_client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))
            model_params[messages] = messages
            response = genai_client.models.generate(**model_params)
            return response.text
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
        deployment_response = self.execute_deployment(doc)
        if deployment_response and deployment_response['data']['status'] == 'FAIL':
            print('Guardrail deployment run returned failed status on inputs, replacing with alternate response')
            return deployment_response['data']['alternate_response'], None, deployment_response
        
        # activate only guardrails that require response
        try:
            llm_response = self.llm_executor(messages,model_params,llm_caller)
        except Exception as e:
            print('Error in running llm:',str(e))
            return None, None, deployment_response
        doc['response'] = llm_response
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
            doc['instruction'] = instruction
        response = self.execute_deployment(doc)
        if response and response['data']['status'] == 'FAIL':
            print('Guardrail deployment run retured failed status, replacing with alternate response')
            return response['data']['alternateResponse'],llm_response,response
        else:
            return None,llm_response,response

    def execute_input_guardrails(self, messages, prompt_params):
        for key in self.field_map:
            if key not in ['prompt', 'response']:
                if self.field_map[key] not in prompt_params:
                    raise ValueError(f'{key} added as field map but not passed as prompt parameter')
        context_var = self.field_map.get('context', None)
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
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
            doc['instruction'] = instruction
        deployment_response = self.execute_deployment(doc)
        pass

    def some_method_2(
            self, 
            llm_response: str, 
            doc_id: Optional[str]=None,
            messages: Optional[List[dict]] = None, 
            prompt_params: Optional[Dict[str, Any]] = None, 
            ) -> None:
        doc_id = doc_id or self.current_doc_id
        if not doc_id:
            for key in self.field_map:
                if key not in ['prompt', 'response']:
                    if not prompt_params or self.field_map[key] not in prompt_params:
                        raise ValueError(f'{key} added as field map but not passed as prompt parameter')
                elif key == 'prompt':
                    if not messages:
                        raise ValueError('messages must be provided when prompt is used as field map')
            # raise Exception(f'\'doc_id\' not provided and there is no doc_id currently available. Either run \'execute_input_guardrails\' or pass a valid \'doc_id\'')
            deployment_details = self.guard_manager.get_deployment(self.deployment_id)
            deployed_guardrails = deployment_details['data']['guardrailsResponse']
            for guardrail in deployed_guardrails:
                metric_spec_mappings = guardrail['metricSpec']['config']['mappings']
                var_names = [mapping['variableNmae'].lower() for mapping in metric_spec_mappings]
                for var_name in var_names:
                    if var_name not in ['prompt', 'response']:
                        if var_name not in self.field_map:
                            raise ValueError(f'{var_name} requrired for deployment {self.deployment_id} but not added as field map')
                        if not prompt_params or self.field_map[var_name] not in prompt_params:
                            raise ValueError(f'{var_name} added as field map but not passed as prompt parameter')
                        elif var_name == 'prompt':
                            if not messages:
                                raise ValueError('messages must be provided if doc_id is not provided')
            
            prompt = None
            for msg in messages:
                if 'role' in msg:
                    if msg['role'] == 'user':
                        prompt = msg['content']
            context_var = self.field_map.get('context', None)
            doc = dict()
            doc['prompt'] = prompt
            doc['context'] = prompt_params[self.field_map[context_var]] if context_var else ''
            if 'instruction' in self.field_map:
                instruction = prompt_params[self.field_map['instruction']]
                doc['instruction'] = instruction
        else:
            doc = self.id_2_doc[doc_id]
        doc['response'] = llm_response
        response = self.execute_deployment(doc)
        if response and response['data']['status'] == 'FAIL':
            print('Guardrail deployment run retured failed status, replacing with alternate response')
            return response['data']['alternateResponse'], llm_response, response
        else:
            return None, llm_response, response

    def some_method_3(self):
        pass


        



