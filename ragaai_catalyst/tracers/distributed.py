"""
Distributed tracing functionality for RagaAI Catalyst.
Provides simplified initialization and decorator-based tracing.
"""

import os
import threading
from typing import Optional, Dict, Any, List
from functools import wraps
from contextlib import contextmanager
import uuid
from .agentic_tracing.utils.unique_decorator import generate_unique_hash_simple
from datetime import datetime
import asyncio

from .tracer import Tracer
from ..ragaai_catalyst import RagaAICatalyst

# Global state
_global_tracer: Optional[Tracer] = None
_global_catalyst: Optional[RagaAICatalyst] = None
_tracer_lock = threading.Lock()
_active_spans = threading.local()

def get_current_tracer() -> Optional[Tracer]:
    """Get the current global tracer instance."""
    return _global_tracer

def get_current_catalyst() -> Optional[RagaAICatalyst]:
    """Get the current global catalyst instance."""
    return _global_catalyst

def init_tracing(
    project_name: str = None,
    dataset_name: str = None,
    access_key: str = None,
    secret_key: str = None,
    base_url: str = None,
    tracer: Tracer = None,
    catalyst: RagaAICatalyst = None, 
    **kwargs
) -> None:
    """Initialize distributed tracing.
    
    Args:
        project_name: Project name for new tracer
        dataset_name: Dataset name for new tracer
        access_key: RagaAI Catalyst access key
        secret_key: RagaAI Catalyst secret key  
        base_url: RagaAI Catalyst API base URL
        tracer: Existing Tracer instance
        catalyst: Existing RagaAICatalyst instance
        **kwargs: Additional tracer parameters
    """
    global _global_tracer, _global_catalyst
    
    with _tracer_lock:
        if tracer and catalyst:
            if isinstance(tracer, Tracer) and isinstance(catalyst, RagaAICatalyst):
                _global_tracer = tracer
                _global_catalyst = catalyst
            else:
                raise ValueError("Both Tracer and Catalyst objects must be instances of Tracer and RagaAICatalyst, respectively.")
        else:
            raise ValueError("Both Tracer and Catalyst objects must be provided.")


def trace_agent(name: str = None, agent_type: str = "generic", version: str = "1.0.0", **kwargs):
    """Decorator for tracing agent functions."""
    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        span_name = name or func.__name__
        # Generate hash based on the decorated function
        top_level_hash_id = generate_unique_hash_simple(func)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            # Set current agent name and store the token
            name_token = tracer.current_agent_name.set(span_name)
            
            try:
                # Use async agent tracing
                return await tracer._trace_agent_execution(
                    func,
                    span_name,
                    agent_type,
                    version,
                    None,  # capabilities
                    top_level_hash_id, 
                    *args,
                    **kwargs
                )
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_agent_name.reset(name_token)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return func(*args, **kwargs)

            # Set current agent name and store the token
            name_token = tracer.current_agent_name.set(span_name)
            
            try:
                # Use synchronous agent tracing
                return tracer._trace_sync_agent_execution(
                    func,
                    span_name,
                    agent_type,
                    version,
                    None,  # capabilities
                    top_level_hash_id,   
                    *args,
                    **kwargs
                )
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_agent_name.reset(name_token)

        return async_wrapper if is_async else sync_wrapper
    return decorator

def trace_llm(name: str = None, model: str = None, **kwargs):
    """Decorator for tracing LLM calls."""
    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        span_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            # Set current LLM name and store the token
            name_token = tracer.current_llm_call_name.set(span_name)
            
            try:
                # Just execute the function within the current span
                result = await func(*args, **kwargs)
                return result
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_llm_call_name.reset(name_token)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return func(*args, **kwargs)

            # Set current LLM name and store the token
            name_token = tracer.current_llm_call_name.set(span_name)
            
            try:
                # Just execute the function within the current span
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_llm_call_name.reset(name_token)

        return async_wrapper if is_async else sync_wrapper
    return decorator
    

def trace_tool(name: str = None, tool_type: str = "generic", version: str = "1.0.0", **kwargs):
    """Decorator for tracing tool functions."""
    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        span_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            # Set current tool name and store the token
            name_token = tracer.current_tool_name.set(span_name)
            
            try:
                # Use async tool tracing
                return await tracer._trace_tool_execution(
                    func,
                    span_name,
                    tool_type,
                    version,
                    *args,
                    **kwargs
                )
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_tool_name.reset(name_token)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return func(*args, **kwargs)

            # Set current tool name and store the token
            name_token = tracer.current_tool_name.set(span_name)
            
            try:
                # Use synchronous tool tracing
                return tracer._trace_sync_tool_execution(
                    func,
                    span_name,
                    tool_type,
                    version,
                    *args,
                    **kwargs
                )
            finally:
                # Reset using the stored token
                if name_token:
                    tracer.current_tool_name.reset(name_token)

        return async_wrapper if is_async else sync_wrapper
    return decorator



def trace_custom(name: str = None, custom_type: str = "generic", version: str = "1.0.0", trace_variables: bool = False, **kwargs):
    """Decorator for tracing custom functions."""
    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            # Use async tool tracing
            return await tracer._trace_custom_execution(
                func,
                name or func.__name__,
                custom_type,
                version,
                trace_variables,
                *args,
                **kwargs
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer:
                return func(*args, **kwargs)

            # Use synchronous tool tracing
            return tracer._trace_sync_custom_execution(
                func,
                name or func.__name__,
                custom_type,
                version,
                trace_variables,
                *args,
                **kwargs
            )

        return async_wrapper if is_async else sync_wrapper
    return decorator


def current_span():
    """Get the current active span for adding metrics."""
    tracer = get_current_tracer()
    if not tracer:
        return None
    
    # First check for LLM context
    llm_name = tracer.current_llm_call_name.get()
    if llm_name:
        return tracer.span(llm_name)
    
    # Then check for tool context
    tool_name = tracer.current_tool_name.get()
    if tool_name:
        return tracer.span(tool_name)
    
    # Finally fall back to agent context
    agent_name = tracer.current_agent_name.get()
    if not agent_name:
        raise ValueError("No active span found. Make sure you're calling this within a traced function.")
    
    return tracer.span(agent_name)
