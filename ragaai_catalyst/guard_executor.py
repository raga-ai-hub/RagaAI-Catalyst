import litellm
import json
import requests
import os
from google import genai
from google.genai.types import GenerateContentConfig
from typing import Optional, List, Dict, Any
import logging
logger = logging.getLogger('LiteLLM')
logger.setLevel(logging.ERROR)

class GuardExecutor:

    def __init__(self,guard_manager,input_deployment_id = None,output_deployment_id=None,field_map={}):
        self.field_map = field_map
        self.guard_manager = guard_manager
        try:
            if input_deployment_id:
                self.input_deployment_id = input_deployment_id
                self.input_deployment_details = self.guard_manager.get_deployment(input_deployment_id)
            if output_deployment_id:
                self.output_deployment_id = output_deployment_id
                self.output_deployment_details = self.guard_manager.get_deployment(output_deployment_id)
            if input_deployment_id and output_deployment_id:
                # check if 2 deployments are mapped to same dataset
                if self.input_deployment_details['data']['datasetId'] != self.output_deployment_details['data']['datasetId']:
                    raise ValueError('Input deployment and output deployment should be mapped to same dataset')
            for guardrail in self.input_deployment_details['data']['guardrailsResponse']:
                maps = guardrail['metricSpec']['config']['mappings']
                for _map in maps:
                    if _map['schemaName']=='Response':
                        raise ValueError('Response field should be mapped only in output guardrails')
        except Exception as e:
            raise ValueError(str(e))
        self.base_url = guard_manager.base_url
        for key in field_map.keys():
            if key not in ['prompt','context','response','instruction']:
                print('Keys in field map should be in ["prompt","context","response","instruction"]')
        self.current_trace_id = None
        self.id_2_doc = {}

    def execute_deployment(self, deployment_id, payload):
        api = self.base_url + f'/guardrail/deployment/{deployment_id}/ingest'
        if self.current_trace_id:
            payload['traceId'] = self.current_trace_id
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

    def llm_executor(self,prompt,model_params,llm_caller):
        messages = [{
                    'role':'user',
                    'content':prompt
                    }]
        if self.current_trace_id:
            doc = self.id_2_doc[self.current_trace_id]
            messages[0]['content'] = messages[0]['content'] + '\n' + doc.get('context','')
        if llm_caller == 'litellm':
            model_params['messages'] = messages
            response = litellm.completion(**model_params)
            return response['choices'][0].message.content
        elif llm_caller == 'genai':
            genai_client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))
            model_params['messages'] = messages
            response = genai_client.models.generate(**model_params)
            return response.text
        else:
            print(f"{llm_caller} not supported currently, use litellm as llm caller")
        '''
        elif llm_caller == 'anthropic':
            response = anthropic.completion(prompt=messages, **model_params)
            return response['completion']
        elif llm_caller == 'langchain':
            response = langchain.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'azure_openai':
            response = azure_openai.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'aws_bedrock':
            response = aws_bedrock.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'meta':
            response = meta.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_csller == 'llamaindex':
            response = llamaindex.completion(prompt=messages, **model_params)
            return response['choices'][0].text'''

    def set_input_params(self, prompt: None, context: None, instruction: None,  **kwargs):
        if 'latest' not in self.id_2_doc:
            self.id_2_doc['latest'] = {}
        if prompt:
            self.id_2_doc['latest']['prompt'] = prompt
        if context:
            self.id_2_doc['latest']['context'] = context
        if instruction:
            self.id_2_doc['latest']['instruction'] = instruction

    
    def __call__(self,prompt,prompt_params,model_params,llm_caller='litellm'):
        '''for key in self.field_map:
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
        doc['context'] = prompt_params[context_var]'''
        
        # Run the input guardrails
        alternate_response,input_deployment_response = self.execute_input_guardrails(prompt,prompt_params)
        if input_deployment_response and input_deployment_response['data']['status'].lower() == 'fail':
            return alternate_response, None, input_deployment_response
        
        # activate only guardrails that require response
        try:
            llm_response = self.llm_executor(prompt,model_params,llm_caller)
        except Exception as e:
            print('Error in running llm:',str(e))
            return None, None, input_deployment_response
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
        alternate_op_response,output_deployment_response = self.execute_output_guardrails(llm_response)
        if output_deployment_response and output_deployment_response['data']['status'].lower() == 'fail':
            return alternate_op_response,llm_response,output_deployment_response
        else:
            return None,llm_response,output_deployment_response

    def set_variables(self,prompt,prompt_params):
        for key in self.field_map:
            if key not in ['prompt', 'response']:
                if self.field_map[key] not in prompt_params:
                    raise ValueError(f'{key} added as field map but not passed as prompt parameter')
        context_var = self.field_map.get('context', None)
        
        doc = dict()
        doc['prompt'] = prompt
        doc['context'] = prompt_params[context_var]
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
            doc['instruction'] = instruction
        return doc

    def execute_input_guardrails(self, prompt, prompt_params):
        doc = self.set_variables(prompt,prompt_params)
        deployment_response = self.execute_deployment(self.input_deployment_id,doc)
        self.current_trace_id = deployment_response['data']['results'][0]['executionId']
        self.id_2_doc[self.current_trace_id] = doc
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response

    def execute_output_guardrails(self, llm_response: str, prompt=None, prompt_params=None) -> None:
        if not prompt: # user has not passed input
            if self.current_trace_id not in self.id_2_doc:
                raise Exception(f'No input doc found for trace_id: {self.current_trace_id}')
            else:
                doc = self.id_2_doc[self.current_trace_id]
                doc['response'] = llm_response
        else:
            doc = self.set_variables(prompt,prompt_params)
        deployment_response = self.execute_deployment(self.output_deployment_id,doc)
        del self.id_2_doc[self.current_trace_id]
        self.current_trace_id = None
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response


        '''
        # doc = dict()
        # doc['response'] = llm_response
        # if trace_id:
        #     doc['trace_id'] = trace_id
        trace_id = self.current_trace_id
        if not trace_id:
            for key in self.field_map:
                if key not in ['prompt', 'response']:
                    if not prompt_params or self.field_map[key] not in prompt_params:
                        if key not in self.id_2_doc.get('latest', {}):
                            raise ValueError(f'{key} added as field map but not passed as prompt parameter or set in executor')
                elif key == 'prompt':
                    if not messages:
                        if key not in self.id_2_doc.get('latest', {}):
                            raise ValueError('messages should be provided when prompt is used as field or prompt should be set in executor')
            # raise Exception(f'\'doc_id\' not provided and there is no doc_id currently available. Either run \'execute_input_guardrails\' or pass a valid \'doc_id\'')
            #deployment_details = self.guard_manager.get_deployment(self.output_deployment_id)
            #deployed_guardrails = deployment_details['data']['guardrailsResponse']
            
            for guardrail in deployed_guardrails:
                metric_spec_mappings = guardrail['metricSpec']['config']['mappings']
                var_names = [mapping['variableNmae'].lower() for mapping in metric_spec_mappings]
                for var_name in var_names:
                    if var_name not in ['prompt', 'response']:
                        if var_name not in self.field_map:
                            raise ValueError(f'{var_name} requrired for {guardrail} guardrail in deployment {self.deployment_id} but not added as field map')
                        if not prompt_params or (self.field_map[var_name] not in prompt_params):
                            if var_name not in self.id_2_doc.get('latest', {}):
                                raise ValueError(f'{var_name} added as field map but not passed as prompt parameter')
                        elif var_name == 'prompt':
                            if not messages:
                                if var_name not in self.id_2_doc.get('latest', {}):
                                    raise ValueError('messages must be provided if doc_id is not provided')
            
            prompt = None
            if messages:
                for msg in messages:
                    if 'role' in msg:
                        if msg['role'] == 'user':
                            prompt = msg['content']
            else:
                prompt = self.id_2_doc['latest']['prompt']
            context_var = self.field_map.get('context', None)
            doc = dict()
            doc['prompt'] = prompt
            if context_var and prompt_params and context_var in prompt_params:
                doc['context'] = prompt_params[self.field_map[context_var]]
            elif context_var:
                doc['context'] = self.id_2_doc['latest']['context']
            elif 'latest' in self.id_2_doc and 'context' in self.id_2_doc['latest'] and self.id_2_doc['latest']['context']:
                doc['context'] = self.id_2_doc['latest']['context']
            else:
                doc['context'] = ''
            if 'instruction' in self.field_map:
                if prompt_params and 'instruction' in prompt_params:
                    instruction = prompt_params[self.field_map['instruction']]
                elif 'latest' in self.id_2_doc and 'instruction' in self.id_2_doc['latest'] and self.id_2_doc['latest']['instruction']:
                    instruction = self.id_2_doc['latest']['instruction']
                else:
                    raise ValueError('instruction added as field map but not passed as prompt parameter or set in executor')
                doc['instruction'] = instruction
        elif trace_id not in self.id_2_doc:
            raise Exception(f'trace_id {trace_id} is not valid. Please run \'execute_input_guardrails\' first')
        else:
            doc = self.id_2_doc[trace_id]
        doc['response'] = llm_response
        response = self.execute_deployment(doc)
        if response and response['data']['status'] == 'FAIL':
            print('Guardrail deployment run retured failed status, replacing with alternate response')
            return response['data']['alternateResponse'], llm_response, response
        else:
            self.current_trace_id = None
            return None, llm_response, response
            '''



