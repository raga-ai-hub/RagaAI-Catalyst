


import sys
import os
import json
from typing import Any, Dict, Optional
import asyncio

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
import google.generativeai as genai
from litellm import completion, acompletion
import litellm
from ragaai_catalyst import trace_llm

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Azure OpenAI setup
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

azure_openai_client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=azure_api_key, api_version=azure_api_version)
async_azure_openai_client = AsyncAzureOpenAI(azure_endpoint=azure_endpoint, api_key=azure_api_key, api_version=azure_api_version)

# Google AI setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@trace_llm('llm response')
async def get_llm_response(
    prompt,
    model, 
    provider,
    temperature,
    max_tokens,
    async_llm=False,
    **kwargs
    ):
    """
    Main interface for getting responses from various LLM providers
    """
    if 'azure' in provider.lower():
        syntax = kwargs.get("syntax", "completion")
        if async_llm:
            return await _get_async_azure_openai_response(prompt, model, temperature, max_tokens, syntax)
        else:
            return _get_azure_openai_response(prompt, model, temperature, max_tokens, syntax)
    elif 'openai' in provider.lower():
        syntax = kwargs.get("syntax", "completion")
        if async_llm:
            return await _get_async_openai_response(prompt, model, temperature, max_tokens, syntax)
        else:
            return _get_openai_response(prompt, model, temperature, max_tokens, syntax)
    elif 'google' in provider.lower():
        if async_llm:
            return await _get_async_google_generativeai_response(prompt, model, temperature, max_tokens)
        else:
            return _get_google_generativeai_response(prompt, model, temperature, max_tokens)
    elif 'litellm' in provider.lower():
        if async_llm:
            return await _get_async_litellm_response(prompt, model, temperature, max_tokens)
        else:
            return _get_litellm_response(prompt, model, temperature, max_tokens)


def _get_openai_response(
    prompt,
    model, 
    temperature,
    max_tokens, 
    syntax='completion'
    ):
    """
    Get response from OpenAI API
    """
    try:
        if syntax == 'chat':
            print("Using chat completion")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            print("Using completion")
            response = openai_client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text
    except Exception as e:
        print(f"Error with OpenAI API: {str(e)}")
        return None

async def _get_async_openai_response(
    prompt,
    model, 
    temperature,
    max_tokens, 
    syntax='completion'
    ):
    """
    Get async response from OpenAI API
    """
    try:
        if syntax == 'chat':
            print("Using async chat completion")
            response = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            print("Using async completion")
            response = await async_openai_client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text
    except Exception as e:
        print(f"Error with async OpenAI API: {str(e)}")
        return None

def _get_azure_openai_response(
    prompt,
    model, 
    temperature,
    max_tokens, 
    syntax='completion'
    ):
    """
    Get response from Azure OpenAI API
    """
    try:
        if syntax == 'chat':
            print("Using chat completion")
            response = azure_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            print("Using completion")
            response = azure_openai_client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text
    except Exception as e:
        print(f"Error with Azure OpenAI API: {str(e)}")
        return None

async def _get_async_azure_openai_response(
    prompt,
    model, 
    temperature,
    max_tokens, 
    syntax='completion'
    ):
    """
    Get async response from Azure OpenAI API
    """
    try:
        if syntax == 'chat':
            print("Using async chat completion")
            response = await async_azure_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            print("Using async completion")
            response = await async_azure_openai_client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text
    except Exception as e:
        print(f"Error with async Azure OpenAI API: {str(e)}")
        return None

def _get_litellm_response(
    prompt,
    model, 
    temperature,
    max_tokens
    ):
    """
    Get response using LiteLLM
    """
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with LiteLLM: {str(e)}")
        return None

async def _get_async_litellm_response(
    prompt,
    model, 
    temperature,
    max_tokens
    ):
    """
    Get async response using LiteLLM
    """
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with async LiteLLM: {str(e)}")
        return None

def _get_google_generativeai_response(
    prompt,
    model, 
    temperature,
    max_tokens
    ):
    """
    Get response from Google GenerativeAI
    """
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    except Exception as e:
        print(f"Error with Google GenerativeAI: {str(e)}")
        return None

async def _get_async_google_generativeai_response(
    prompt,
    model, 
    temperature,
    max_tokens
    ):
    """
    Get async response from Google GenerativeAI
    """
    try:
        model = genai.GenerativeModel(model)
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    except Exception as e:
        print(f"Error with async Google GenerativeAI: {str(e)}")
        return None

async def main():
    response = await get_llm_response(
        prompt="Hello, how are you?",
        # model="davinci-002",
        model = "gemini-1.5-pro",
        provider="google",
        temperature=0.7,
        max_tokens=100,
        async_llm=True, 
        syntax = 'chat'
    )
    print(response)

if __name__ == '__main__':
    asyncio.run(main())