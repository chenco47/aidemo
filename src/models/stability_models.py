"""
Stability AI model implementations.
"""
import io
import os
import time
import base64
from typing import Dict, List, Any, Optional, Union, BinaryIO

import requests
from PIL import Image

from aidemo.src.models.base import BaseModel
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

class StabilityModel(BaseModel):
    """Implementation for Stability AI models."""
    
    def __init__(
        self,
        model_name: str = "stability.stable-diffusion-xl-v1",
        api_key: Optional[str] = None
    ):
        """Initialize the Stability AI model client.
        
        Args:
            model_name: The model to use, default is stable-diffusion-xl-v1
            api_key: Stability API key (if not provided, will use STABILITY_API_KEY env var)
        """
        super().__init__(model_name, api_key)
        
        self.api_key = self._validate_api_key("STABILITY_API_KEY")
        
        # Extract the actual model name from the full model identifier
        if model_name.startswith("stability."):
            self.model_id = model_name[len("stability."):]
        else:
            self.model_id = model_name
        
        # API endpoints
        self.api_host = os.getenv("STABILITY_API_HOST", "https://api.stability.ai")
        self.text_to_image_endpoint = f"{self.api_host}/v1/generation/{self.model_id}/text-to-image"
        
        # Validate model name
        self._validate_model_name(self.model_id)
    
    def _validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported.
        
        Args:
            model_name: The name of the model to validate
        """
        # Stable Diffusion versions
        stable_diffusion_models = [
            "stable-diffusion-xl-v1",
            "stable-diffusion-xl-v1-0", 
            "stable-diffusion-v1-5",
            "stable-diffusion-v1-6",
            "stable-diffusion-512-v2-1"
        ]
        
        if model_name not in stable_diffusion_models:
            logger.warning(f"Model '{model_name}' may not be officially supported by Stability AI")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Text-to-image generation is not supported for this model.
        Use generate_image instead.
        
        Raises:
            NotImplementedError: This model only supports image generation
        """
        raise NotImplementedError(
            "Text generation is not supported for Stability AI models. "
            "Use generate_image method instead."
        )
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Embeddings are not supported for this model.
        
        Raises:
            NotImplementedError: This model doesn't support embeddings
        """
        raise NotImplementedError(
            "Embeddings are not supported for Stability AI models."
        )
    
    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        samples: int = 1,
        steps: int = 30,
        cfg_scale: float = 7.0,
        output_format: str = "base64",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image with Stable Diffusion.
        
        Args:
            prompt: The image description
            negative_prompt: Things to exclude from the image
            width: Image width
            height: Image height
            samples: Number of images to generate
            steps: Number of diffusion steps
            cfg_scale: How closely to follow the prompt
            output_format: 'base64' or 'pil'
            **kwargs: Additional parameters for image generation
            
        Returns:
            Dictionary containing the generated images and metadata
        """
        start_time = time.time()
        
        # Prepare request headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare request body
        body = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "samples": samples,
            "steps": steps,
        }
        
        # Add negative prompt if provided
        if negative_prompt:
            body["text_prompts"].append({
                "text": negative_prompt,
                "weight": -1.0
            })
        
        # Add any additional parameters
        body.update(kwargs)
        
        try:
            # Make the API request
            response = requests.post(
                self.text_to_image_endpoint,
                headers=headers,
                json=body
            )
            
            # Check for errors
            if response.status_code != 200:
                raise Exception(f"Error generating image: {response.text}")
            
            # Parse the response
            data = response.json()
            
            # Process the images
            images = []
            for i, image_data in enumerate(data["artifacts"]):
                if output_format == "base64":
                    # Return the raw base64 image data
                    images.append(image_data["base64"])
                elif output_format == "pil":
                    # Convert base64 to PIL Image
                    image_bytes = base64.b64decode(image_data["base64"])
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                else:
                    raise ValueError(f"Unsupported output_format: {output_format}")
            
            processing_time = time.time() - start_time
            
            # Create output dictionary
            output = {
                "images": images[0] if samples == 1 else images,
                "model": self.model_id,
                "processing_time": processing_time,
                "seed": data["artifacts"][0].get("seed"),
                "finish_reason": data["artifacts"][0].get("finishReason")
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating image with Stability AI: {e}")
            raise 