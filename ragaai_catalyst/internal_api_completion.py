import requests
import json
import subprocess
import logging
import traceback
import pandas as pd

logger = logging.getLogger(__name__)

def api_completion(messages, model_config, kwargs):
    attempts = 0
    while attempts < 3:

        user_id = kwargs.get('user_id', '1')
        internal_llm_proxy = kwargs.get('internal_llm_proxy', -1)
            
            
        job_id = model_config.get('job_id',-1)
        converted_message = convert_input(messages,model_config, user_id)
        payload = json.dumps(converted_message)
        headers = {
            'Content-Type': 'application/json',
            # 'Wd-PCA-Feature-Key':f'your_feature_key, $(whoami)'
        }
        try:
            response = requests.request("POST", internal_llm_proxy, headers=headers, data=payload)
            if model_config.get('log_level','')=='debug':
                logger.info(f'Model response Job ID {job_id} {response.text}')
            if response.status_code!=200:
                # logger.error(f'Error in model response Job ID {job_id}:',str(response.text))
                raise ValueError(str(response.text))
            
            if response.status_code==200:
                response = response.json()                
                if "error" in response:
                    raise ValueError(response["error"]["message"])
                else:
                    result=  response["choices"][0]["message"]["content"]
                    response1 = result.replace('\n', '').replace('```json','').replace('```', '').strip()
                    try:
                        json_data = json.loads(response1)
                        df = pd.DataFrame(json_data)
                        return(df)
                    except json.JSONDecodeError:
                        attempts += 1  # Increment attempts if JSON parsing fails
                        if attempts == 3:
                            raise Exception("Failed to generate a valid response after multiple attempts.")

        except Exception as e:
            raise ValueError(f"{e}")


def get_username():
    result = subprocess.run(['whoami'], capture_output=True, text=True)
    result = result.stdout
    return result


def convert_input(messages, model_config, user_id):
    doc_input = {
      "model": model_config.get('model'),
      **model_config,
      "messages": messages,
      "user_id": user_id
    }
    return doc_input


if __name__=='__main__':
    messages = [
        {
            "role": "system",
            "content": "you are a poet well versed in shakespeare literature"
        },
        {
          "role": "user",
          "content": "write a poem on pirates and penguins"
        }
      ]
    kwargs = {"internal_llm_proxy": "http://13.200.11.66:4000/chat/completions", "user_id": 1}
    model_config = {"model": "workday_gateway", "provider":"openai", "max_tokens": 10}
    answer = api_completion(messages, model_config, kwargs)
    print(answer)