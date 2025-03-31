"""
Cohere model implementations.
"""
import os
import time
from typing import Dict, List, Any, Optional, Union

from aidemo.src.models.base import BaseModel
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

class CohereModel(BaseModel):
    """Implementation for Cohere models."""
    
    def __init__(
        self,
        model_name: str = "cohere.command-r-plus-v1:0",
        api_key: Optional[str] = None
    ):
        """Initialize the Cohere model client.
        
        Args:
            model_name: The Cohere model to use, default is command-r-plus-v1:0
            api_key: Cohere API key (if not provided, will use COHERE_API_KEY env var)
        """
        super().__init__(model_name, api_key)
        
        self.api_key = self._validate_api_key("COHERE_API_KEY")
        
        # Extract the actual model name from the full model identifier
        if model_name.startswith("cohere."):
            self.model_id = model_name[len("cohere."):]
        else:
            self.model_id = model_name
            
        # Import Cohere client here to avoid making it a global dependency
        try:
            from cohere import Client
            self.client = Client(api_key=self.api_key)
        except ImportError:
            logger.error("Cohere Python package not installed. Run 'pip install cohere'")
            raise
        
        # Validate the model name
        self._validate_model_name(self.model_id)
    
    def _validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported.
        
        Args:
            model_name: The name of the model to validate
        """
        # Text generation models
        text_generation_models = [
            "command-text-v1",
            "command-light-text-v1", 
            "command-r-v1:0",
            "command-r-plus-v1:0"
        ]
        
        # Embedding models
        embedding_models = [
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-multilingual-v2.0"
        ]
        
        all_models = text_generation_models + embedding_models
        
        if model_name not in all_models and not any(model_name.startswith(prefix) for prefix in ["command-", "embed-"]):
            logger.warning(f"Model '{model_name}' may not be officially supported by Cohere")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with Cohere.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        params = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7,
            "p": 0.7,
            "k": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        try:
            # Generate text
            response = self.client.generate(**params)
            
            processing_time = time.time() - start_time
            
            # Extract the response content and metadata
            output = {
                "text": response.generations[0].text,
                "model": self.model_id,
                "finish_reason": response.generations[0].finish_reason,
                "processing_time": processing_time,
                "token_count": {
                    "prompt_tokens": response.meta.billed_units.input_tokens,
                    "completion_tokens": response.meta.billed_units.output_tokens
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating text with Cohere: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings for the provided text.
        
        Args:
            text: The input text or list of texts
            **kwargs: Additional parameters for the embedding
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        start_time = time.time()
        
        # Default to multilingual embedding model if not specified
        embedding_model = kwargs.pop("model", "embed-multilingual-v3.0")
        
        # Use the model ID if it's an embedding model
        if self.model_id.startswith("embed-"):
            embedding_model = self.model_id
        
        # Prepare input format
        input_texts = text if isinstance(text, list) else [text]
        
        try:
            # Generate embeddings
            response = self.client.embed(
                texts=input_texts,
                model=embedding_model,
                input_type=kwargs.pop("input_type", "search_document"),
                **kwargs
            )
            
            embeddings = response.embeddings
            
            processing_time = time.time() - start_time
            
            # Extract embeddings and metadata
            output = {
                "embeddings": embeddings if isinstance(text, list) else embeddings[0],
                "model": embedding_model,
                "processing_time": processing_time
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Cohere: {e}")
            raise 