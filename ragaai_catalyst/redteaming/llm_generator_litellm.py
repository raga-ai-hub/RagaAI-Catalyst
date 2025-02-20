from typing import Dict, Any, Optional, Literal
import os
import json
import litellm

class LLMGenerator:
    
    def __init__(self, model_name: str = "gpt-4-1106-preview", temperature: float = 0.7, 
                 provider: str = "openai"):
        """
        Initialize the LLM generator with specified provider client.
        
        Args:
            model_name: The model to use (e.g., "gpt-4-1106-preview" for OpenAI, "grok-2-latest" for X.AI)
            temperature: The sampling temperature to use for generation (default: 0.7)
            provider: The LLM provider to use (default: "openai"), can be any provider supported by LiteLLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        
        # Set API key based on provider
        try:
            self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not self.api_key:
                raise ValueError(f"API key for provider '{provider}' is not set.\n set it as: {provider.upper()}_API_KEY")
        except Exception as e:
            raise ValueError(f"Error retrieving API key: {str(e)}")
    
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
