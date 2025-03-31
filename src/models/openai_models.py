"""
OpenAI model implementations.
"""
import os
import time
from typing import Dict, List, Any, Optional, Union

import openai
from openai import OpenAI

from aidemo.src.models.base import BaseModel
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

class OpenAIModel(BaseModel):
    """Implementation for OpenAI models."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """Initialize the OpenAI model client.
        
        Args:
            model_name: The OpenAI model to use, default is gpt-4o
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            organization: OpenAI organization ID (if not provided, will use OPENAI_ORG_ID env var)
        """
        super().__init__(model_name, api_key)
        
        self.api_key = self._validate_api_key("OPENAI_API_KEY")
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
        
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
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
        # GPT-4 models
        gpt4_models = [
            "gpt-4o", 
            "gpt-4o-2024-05-13", 
            "chatgpt-4o-latest",
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-4-turbo-preview", 
            "gpt-4-0613", 
            "gpt-4-1106-preview",
            "gpt-4-0314"
        ]
        
        # GPT-3.5 models
        gpt35_models = [
            "gpt-3.5-turbo", 
            "gpt-3.5-turbo-1106", 
            "gpt-3.5-turbo-instruct"
        ]
        
        # DALL-E models
        dalle_models = ["dall-e-2", "dall-e-3"]
        
        # Text-to-speech models
        tts_models = ["tts-1", "tts-1-hd"]
        
        # Embedding models
        embedding_models = [
            "text-embedding-3-large", 
            "text-embedding-3-small", 
            "text-embedding-ada-002"
        ]
        
        # Moderation models
        moderation_models = [
            "text-moderation-latest", 
            "text-moderation-stable", 
            "text-moderation-006"
        ]
        
        # Legacy models
        legacy_models = ["babbage-002", "davinci-002"]
        
        all_models = (
            gpt4_models + gpt35_models + dalle_models + 
            tts_models + embedding_models + moderation_models + 
            legacy_models
        )
        
        if model_name not in all_models:
            logger.warning(f"Model '{model_name}' may not be officially supported by OpenAI")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with an OpenAI model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the chat completion
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        if "messages" not in kwargs:
            # If messages not in kwargs, use the prompt to create a simple user message
            params["messages"] = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(**params)
            
            processing_time = time.time() - start_time
            
            # Extract the response content and metadata
            output = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "processing_time": processing_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
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
        
        # Use text-embedding-3-small by default for embeddings if not specified
        embedding_model = kwargs.pop("model", "text-embedding-3-small")
        
        # Prepare input format
        input_texts = text if isinstance(text, list) else [text]
        
        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=input_texts,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Extract embeddings and metadata
            output = {
                "embeddings": [item.embedding for item in response.data],
                "model": response.model,
                "processing_time": processing_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # If only a single text was provided, return a single embedding
            if not isinstance(text, list):
                output["embeddings"] = output["embeddings"][0]
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise
            
    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image with DALL-E models.
        
        Args:
            prompt: The image description
            **kwargs: Additional parameters for image generation
            
        Returns:
            Dictionary containing the image URL and metadata
        """
        # Default to DALL-E 3 if model not specified in generate_image
        image_model = kwargs.pop("model", "dall-e-3")
        
        # Default parameters
        params = {
            "model": image_model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
            "response_format": "url"
        }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        try:
            response = self.client.images.generate(**params)
            
            output = {
                "url": response.data[0].url,
                "model": image_model,
                "revised_prompt": getattr(response.data[0], "revised_prompt", None),
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating image with DALL-E: {e}")
            raise 