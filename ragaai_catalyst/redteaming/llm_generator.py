from typing import Dict, Any, Optional, Literal
import os
import json
from openai import OpenAI

class LLMGenerator:
    # Models that support JSON mode
    JSON_MODELS = {"gpt-4-1106-preview", "gpt-3.5-turbo-1106"}
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-1106-preview", temperature: float = 0.7, 
                 provider: Literal["openai", "xai"] = "openai"):
        """
        Initialize the LLM generator with specified provider client.
        
        Args:
            model_name: The model to use (e.g., "gpt-4-1106-preview" for OpenAI, "grok-2-latest" for X.AI)
            temperature: The sampling temperature to use for generation (default: 0.7)
            provider: The LLM provider to use, either "openai" or "xai" (default: "openai")
            api_key: The API key for the provider
        """
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
        
        # Initialize client based on provider
        if provider.lower() == "openai":
            self.client = OpenAI(api_key=self.api_key)       
        elif provider.lower() == "xai":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI API.
        
        Args:
            system_prompt: The system prompt to guide the model's behavior
            user_prompt: The user's input prompt
            
        Returns:
            Dict containing the generated requirements
        """
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
            if self.model_name in self.JSON_MODELS:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
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
