from typing import Dict, Any, Optional, Literal
import os
import json
import litellm
from openai import OpenAI

class LLMGenerator:
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-1106-preview", temperature: float = 0.7, 
                 provider: str = "openai"):
        """
        Initialize the LLM generator with specified provider client.
        
        Args:
            model_name: The model to use (e.g., "gpt-4-1106-preview" for OpenAI, "grok-2-latest" for X.AI)
            temperature: The sampling temperature to use for generation (default: 0.7)
            provider: The LLM provider to use (default: "openai"), can be any provider supported by LiteLLM
            api_key: The API key for the provider
        """
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key

        self._validate_provider_api()

    
    def _validate_provider_api(self):
        if self.provider.lower() == 'azure':
            os.environ["AZURE_API_KEY"] = self.api_key
            azure_api_base = os.getenv("AZURE_API_BASE", None)
            if azure_api_base is None:
                raise ValueError("AZURE_API_BASE must be provided in the environment for Azure.")
            os.environ["AZURE_API_BASE"] = azure_api_base
            
            azure_api_version = os.getenv("AZURE_API_VERSION", None)
            if azure_api_version is None:
                raise ValueError("AZURE_API_VERSION must be provided in the environment for Azure.")
            os.environ["AZURE_API_VERSION"] = azure_api_version
        
    def get_xai_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        try:
            # Configure API call
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": max_tokens
            }
            
            # Add response_format for JSON-capable models
            kwargs["response_format"] = {"type": "json_object"}
            
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if isinstance(content, str):
                # Remove code block markers if present
                content = content.strip()
                if content.startswith("```"):
                    # Remove language identifier if present (e.g., ```json)
                    content = content.split("\n", 1)[1] if content.startswith("```json") else content[3:]
                    # Find the last code block marker and remove everything after it
                    if "```" in content:
                        content = content[:content.rfind("```")].strip()
                    else:
                        # If no closing marker is found, just use the content as is
                        content = content.strip()
                
                content = json.loads(content)

            return content
            
        except Exception as e:
            raise Exception(f"Error generating LLM response: {str(e)}")

        
    
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using LiteLLM.
        
        Args:
            system_prompt: The system prompt to guide the model's behavior
            user_prompt: The user's input prompt
            max_tokens: The maximum number of tokens to generate (default: 1000)
            
        Returns:
            Dict containing the generated response
        """
        if self.provider.lower() == "xai":
            return self.get_xai_response(system_prompt, user_prompt, max_tokens)

        try:
            kwargs = {
                "model": f"{self.provider}/{self.model_name}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "api_key": self.api_key,
            }
            
            response = litellm.completion(**kwargs)
            content = response["choices"][0]["message"]["content"]
            
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if content.startswith("```json") else content[3:]
                    if "```" in content:
                        content = content[:content.rfind("```")].strip()
                    else:
                        content = content.strip()
                
                content = json.loads(content)
            
            return content
            
        except Exception as e:
            raise Exception(f"Error generating LLM response: {str(e)}")
