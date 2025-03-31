"""
Base model classes for AI model implementations.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseModel(ABC):
    """Base class for all AI models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the base model.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for the model provider (if not provided, will look for environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text based on the prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings for the provided text.
        
        Args:
            text: The input text or list of texts
            **kwargs: Additional parameters for the model
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        pass
    
    def _validate_api_key(self, env_var_name: str) -> str:
        """Validate and retrieve the API key.
        
        Args:
            env_var_name: Name of the environment variable for the API key
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not provided or found in environment variables
        """
        if self.api_key is not None:
            return self.api_key
        
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            raise ValueError(
                f"API key not provided. Either pass it as an argument or set the {env_var_name} environment variable."
            )
        
        return api_key 