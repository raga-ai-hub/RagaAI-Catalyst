from typing import Dict, Any, Optional
import os
import json
from openai import OpenAI

class LLMGenerator:
    # Models that support JSON mode
    JSON_MODELS = {"gpt-4-1106-preview", "gpt-3.5-turbo-1106"}
    
    def __init__(self, model_name: str = "gpt-4-1106-preview", temperature: float = 0.7):
        """
        Initialize the LLM generator with OpenAI client.
        
        Args:
            model_name: The OpenAI model to use (e.g., "gpt-4-1106-preview", "gpt-3.5-turbo-1106")
        """
        self.model_name = model_name
        self.temperature = temperature
        # Initialize OpenAI client with API key from environment
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
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
            
            # Parse the response content as JSON
            try:
                if isinstance(content, str):
                    return json.loads(content)
                return content
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {content}")
            
        except Exception as e:
            raise Exception(f"Error generating LLM response: {str(e)}")
