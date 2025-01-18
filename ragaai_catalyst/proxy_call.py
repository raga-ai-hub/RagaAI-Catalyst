import requests
import json
import subprocess
import logging
import traceback

logger = logging.getLogger(__name__)

def api_completion(model,messages, api_base='http://127.0.0.1:8000',
                    api_key='',model_config=dict()):
    whoami = get_username()
    all_response = list()
    job_id = model_config.get('job_id',-1)
    converted_message = convert_input(messages,model,model_config)
    payload = json.dumps(converted_message)
    response = payload
    headers = {
        'Content-Type': 'application/json',
        'Wd-PCA-Feature-Key':f'your_feature_key, $(whoami)'
    }
    try:
        response = requests.request("POST", api_base, headers=headers, data=payload, verify=False)
        if model_config.get('log_level','')=='debug':
            logger.info(f'Model response Job ID {job_id} {response.text}')
        if response.status_code!=200:
            # logger.error(f'Error in model response Job ID {job_id}:',str(response.text))
            raise ValueError(str(response.text))
    except Exception as e:
        logger.error(f'Error in calling api Job ID {job_id}:',str(e))
        raise ValueError(str(e))
    try:
        response = response.json()
        if 'error' in response:
            logger.error(f'Invalid response from API Job ID {job_id}:'+str(response))
            raise ValueError(str(response.get('error')))
        all_response.append(convert_output(response,job_id))
    except ValueError as e1:
        logger.error(f'Invalid json response from API Job ID {job_id}:'+response)
        raise ValueError(str(e1))
    except Exception as e1:
        if model_config.get('log_level','')=='debug':
            logger.info(f"Error trace Job ID: {job_id} {traceback.print_exc()}")
        logger.error(f"Exception in parsing model response Job ID:{job_id} {str(e1)}")
        logger.error(f"Model response Job ID: {job_id} {response.text}")
        all_response.append(None)
    return all_response

def get_username():
    result = subprocess.run(['whoami'], capture_output=True, text=True)
    result = result.stdout
    return result

def convert_output(response,job_id):
    try:
        if response.get('prediction',{}).get('type','')=='generic-text-generation-v1':
            return response['prediction']['output']
        elif response.get('prediction',{}).get('type','')=='gcp-multimodal-v1':
            full_response = ''
            for chunk in response['prediction']['output']['chunks']:
                candidate = chunk['candidates'][0]
                if candidate['finishReason'] and candidate['finishReason'] not in ['STOP']:
                    raise ValueError(candidate['finishReason'])
                part = candidate['content']['parts'][0]
                full_response += part['text']
            return full_response
        else:
            raise ValueError('Invalid prediction type passed in config')
    except ValueError as e1:
        raise ValueError(str(e1))
    except Exception as e:
        logger.warning(f'Exception in formatting model response Job ID {job_id}:'+str(e))
        return None


def convert_input(prompt,model,model_config):
    doc_input = {
        "target": {
            "provider": "echo",
            "model": "echo"
        },
        "task": {
            "type": "gcp-multimodal-v1",
            "prediction_type": "gcp-multimodal-v1",
            "input": {
            "contents": [
                {
                "role": "user",
                "parts": [
                    {
                    "text": "Give me a recipe for banana bread."
                    }
                ]
                }
            ],
            "safetySettings": 
                [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8000,
                "topK": 40,
                "topP": 0.95,
                "stopSequences": [],
                "candidateCount": 1
            }
            }
        }
    }
    if 'provider' not in model_config:
        doc_input['target']['provider'] = 'gcp'
    else:
        doc_input['target']['provider'] = model_config['provider']
    doc_input['task']['type'] = model_config.get('task_type','gcp-multimodal-v1')
    doc_input['task']['prediction_type'] = model_config.get('prediction_type','generic-text-generation-v1')
    if 'safetySettings' in model_config:
        doc_input['task']['input']['safetySettings'] = model_config.get('safetySettings')
    if 'generationConfig' in model_config:
        doc_input['task']['input']['generationConfig'] = model_config.get('generationConfig')
    doc_input['target']['model'] = model
    if model_config.get('log_level','')=='debug':
        logger.info(f"Using model configs Job ID {model_config.get('job_id',-1)}{doc_input}")
    doc_input['task']['input']['contents'][0]['parts'] = [{"text":prompt[0]['content']}]
    return doc_input



if __name__=='__main__':
    message_list = ["Hi How are you","I am good","How are you"]
    response = batch_completion('gemini/gemini-1.5-flash',message_list,0,1,100,api_base='http://127.0.0.1:5000')
    print(response)
