from typing import Dict, Any, Optional, Literal
import os
import json
import litellm
from openai import OpenAI

class LLMGenerator:
    
    def __init__(self, api_key: str, api_base: str = '', api_version: str = '', model_name: str = "gpt-4-1106-preview", temperature: float = 0.7, 
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
        self.api_base = api_base
        self.api_version = api_version

        self._validate_api_key()
        self._validate_provider()

    def _validate_api_key(self):
        if self.api_key == '' or self.api_key is None:
            raise ValueError("Api Key is required")

    def _validate_azure_keys(self):
        if self.api_base == '' or self.api_base is None:
            raise ValueError("Azure Api Base is required")
        if self.api_version == '' or self.api_version is None:
            raise ValueError("Azure Api Version is required")

    def _validate_provider(self):
        if self.provider.lower() == 'azure':
            self._validate_azure_keys()
            os.environ["AZURE_API_KEY"] = self.api_key
            os.environ["AZURE_API_BASE"] = self.api_base
            os.environ["AZURE_API_VERSION"] = self.api_version
        
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
