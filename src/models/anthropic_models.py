"""
Anthropic Claude model implementations.
"""
import os
import time
from typing import Dict, List, Any, Optional, Union

import anthropic
from anthropic import Anthropic

from aidemo.src.models.base import BaseModel
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

class AnthropicModel(BaseModel):
    """Implementation for Anthropic Claude models."""
    
    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None
    ):
        """Initialize the Anthropic Claude model client.
        
        Args:
            model_name: The Claude model to use, default is claude-3-opus-20240229
            api_key: Anthropic API key (if not provided, will use ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name, api_key)
        
        self.api_key = self._validate_api_key("ANTHROPIC_API_KEY")
        
        self.client = Anthropic(
            api_key=self.api_key
        )
        
        # Validate the model name
        self._validate_model_name(model_name)
    
    def _validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported.
        
        Args:
            model_name: The name of the model to validate
            
        Raises:
            ValueError: If the model name is not supported
        """
        # Claude 3 models
        claude3_models = [
            "anthropic.claude-v1",
            "anthropic.claude-v2",
            "anthropic.claude-v2:1",
            "anthropic.claude-3-sonnet-20240229",
            "anthropic.claude-3-5-sonnet-20240620",
            "anthropic.claude-3-haiku-20240307",
            "anthropic.claude-3-opus-20240229",
            "anthropic.claude-instant-v1",
            # Old-style naming for direct API access
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            # Date format model names
            "claude-3-opus@20240229",
            "claude-3-sonnet@20240229",
            "claude-3-haiku@20240307",
            "claude-3-5-sonnet@20240620",
        ]
        
        # Legacy Claude models
        legacy_models = [
            "claude-2.0",
            "claude-2.1",
            "claude-instant-1.2"
        ]
        
        all_models = claude3_models + legacy_models
        
        if model_name not in all_models:
            logger.warning(f"Model '{model_name}' may not be officially supported by Anthropic")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with Claude.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the message
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        params = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system": "You are a helpful, honest, and concise assistant."
        }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        # Extract the system prompt if provided
        system = params.pop("system", "You are a helpful, honest, and concise assistant.")
        
        try:
            # Use the new Messages API structure
            response = self.client.messages.create(
                model=params.pop("model", self.model_name),
                system=system,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            processing_time = time.time() - start_time
            
            # Extract the response content and metadata
            output = {
                "text": response.content[0].text,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "processing_time": processing_time,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Claude doesn't currently support embeddings through their native API.
        This method raises a NotImplementedError.
        
        Args:
            text: The input text or list of texts
            **kwargs: Additional parameters (unused)
            
        Raises:
            NotImplementedError: Claude doesn't currently support embeddings
        """
        raise NotImplementedError(
            "Anthropic Claude doesn't currently support embeddings through their native API. "
            "Please use another model for embeddings."
        ) 