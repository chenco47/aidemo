"""
Model factory for creating various AI model instances.
"""
import os
from typing import Dict, Any, Optional, Union, List

from aidemo.src.models.base import BaseModel
from aidemo.src.models.openai_models import OpenAIModel
from aidemo.src.models.anthropic_models import AnthropicModel
from aidemo.src.models.huggingface_models import (
    HuggingFaceModel, 
    HuggingFaceAPIModel
)
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

# Model registry for different providers
_MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "huggingface_local": HuggingFaceModel,
    "huggingface_api": HuggingFaceAPIModel,
}

# Model name mappings to provider and model-specific names
_MODEL_MAPPINGS = {
    # OpenAI models
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4-turbo": ("openai", "gpt-4-turbo"),
    "gpt-4": ("openai", "gpt-4"),
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "dall-e-3": ("openai", "dall-e-3"),
    "text-embedding-3-small": ("openai", "text-embedding-3-small"),
    "text-embedding-3-large": ("openai", "text-embedding-3-large"),
    
    # Anthropic models
    "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
    "claude-3-sonnet": ("anthropic", "claude-3-sonnet-20240229"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet-20240620"),
    
    # HuggingFace models - local
    "mistral-7b": ("huggingface_local", "mistralai/Mistral-7B-Instruct-v0.1"),
    "llama-3-8b": ("huggingface_local", "meta-llama/Meta-Llama-3-8B-Instruct"),
    
    # HuggingFace models - API
    "mistral-large": ("huggingface_api", "mistralai/Mistral-Large-2023-08-22"),
}

def create_model(
    model_name: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseModel:
    """Create a model instance based on the model name or provider.
    
    Args:
        model_name: Name of the model to create
        provider: Optional provider name (if not provided, will be inferred from model name)
        api_key: Optional API key for the provider
        **kwargs: Additional parameters for the model constructor
        
    Returns:
        A model instance that implements the BaseModel interface
        
    Raises:
        ValueError: If model name or provider is unknown
    """
    # If provider is not specified, try to infer from model name
    if provider is None:
        if model_name in _MODEL_MAPPINGS:
            provider, model_name = _MODEL_MAPPINGS[model_name]
        else:
            # Try to infer provider from model name prefixes
            if model_name.startswith(("gpt-", "text-embedding-", "dall-e-")):
                provider = "openai"
            elif model_name.startswith("claude-"):
                provider = "anthropic"
            elif "/" in model_name:  # Typical HuggingFace format: org/model
                provider = "huggingface_local"
            else:
                raise ValueError(
                    f"Could not infer provider for model '{model_name}'. "
                    f"Please specify a provider or use a known model name."
                )
    
    # Get the model class from the registry
    if provider not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")
    
    model_class = _MODEL_REGISTRY[provider]
    
    # Create and return the model instance
    return model_class(model_name=model_name, api_key=api_key, **kwargs)


def list_available_models() -> Dict[str, List[str]]:
    """List all available models grouped by provider.
    
    Returns:
        Dictionary with providers as keys and lists of model names as values
    """
    models_by_provider = {}
    
    # Add models from the mappings
    for alias, (provider, model_name) in _MODEL_MAPPINGS.items():
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append({
            "alias": alias,
            "model_name": model_name
        })
    
    return models_by_provider 