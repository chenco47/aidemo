"""
Example script for image generation with multiple AI models.
"""
import os
import sys
import time
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from PIL import Image

# Ensure the package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from aidemo.src.models.factory import create_model
from aidemo.src.models.openai_models import OpenAIModel
from aidemo.src.models.stability_models import StabilityModel

# Sample prompts for image generation
SAMPLE_PROMPTS = [
    "A serene landscape with mountains and a lake at sunset, digital art style",
    "A futuristic cityscape with flying cars and holographic billboards, cyberpunk style",
    "A cute robot pet playing in a garden with flowers, 3D render"
]

def generate_with_dalle(prompt: str, save_path: Optional[str] = None) -> str:
    """Generate an image with DALL-E.
    
    Args:
        prompt: The image description
        save_path: Path to save the image (if None, won't save)
        
    Returns:
        Path to the saved image or the image URL
    """
    print(f"Generating with DALL-E: '{prompt}'")
    start_time = time.time()
    
    # Create the OpenAI model client
    model = OpenAIModel(model_name="dall-e-3")
    
    # Generate the image
    response = model.generate_image(
        prompt=prompt,
        size="1024x1024",
        quality="standard"
    )
    
    elapsed_time = time.time() - start_time
    print(f"✓ DALL-E generated image in {elapsed_time:.2f} seconds")
    
    # Save the image if a path is provided
    if save_path and response.get("url"):
        import requests
        
        img_url = response["url"]
        img_data = requests.get(img_url).content
        
        with open(save_path, 'wb') as f:
            f.write(img_data)
        
        print(f"Image saved to {save_path}")
        return save_path
    
    return response.get("url", "")

def generate_with_stability(prompt: str, save_path: Optional[str] = None) -> str:
    """Generate an image with Stability AI models.
    
    Args:
        prompt: The image description
        save_path: Path to save the image (if None, won't save)
        
    Returns:
        Path to the saved image
    """
    print(f"Generating with Stability AI: '{prompt}'")
    start_time = time.time()
    
    # Create the Stability model client
    model = StabilityModel(model_name="stability.stable-diffusion-xl-v1")
    
    # Generate the image
    response = model.generate_image(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.0
    )
    
    elapsed_time = time.time() - start_time
    print(f"✓ Stability AI generated image in {elapsed_time:.2f} seconds")
    
    # Save the image if a path is provided
    if save_path and response.get("images"):
        # Get the first image (base64 encoded)
        img_data = base64.b64decode(response["images"])
        img = Image.open(BytesIO(img_data))
        img.save(save_path)
        
        print(f"Image saved to {save_path}")
        return save_path
    
    return response.get("images", "")

def generate_with_diffusers(prompt: str, save_path: Optional[str] = None) -> str:
    """Generate an image with local Diffusers pipeline.
    
    Args:
        prompt: The image description
        save_path: Path to save the image (if None, won't save)
        
    Returns:
        Path to the saved image
    """
    print(f"Generating with Diffusers (local): '{prompt}'")
    start_time = time.time()
    
    # Using the diffusers library directly
    from diffusers import StableDiffusionPipeline
    import torch
    
    # This pattern matches the regex for pipeline with model="..."
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Generate the image
    image = pipeline(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    elapsed_time = time.time() - start_time
    print(f"✓ Diffusers generated image in {elapsed_time:.2f} seconds")
    
    # Save the image if a path is provided
    if save_path:
        image.save(save_path)
        print(f"Image saved to {save_path}")
        return save_path
    
    return "Image generated but not saved"

def main():
    """Run the image generation examples."""
    print("=" * 80)
    print("AI IMAGE GENERATION EXAMPLES")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 50)
        
        # Add a simulation flag since we don't want to actually make API calls
        # In a real scenario, remove simulation=True
        simulation = True
        
        if not simulation:
            # Generate with DALL-E
            dalle_path = f"output/dalle_{i+1}.png"
            dalle_result = generate_with_dalle(prompt, dalle_path)
            
            # Generate with Stability AI
            stability_path = f"output/stability_{i+1}.png"
            stability_result = generate_with_stability(prompt, stability_path)
            
            # Generate with local Diffusers
            diffusers_path = f"output/diffusers_{i+1}.png"
            diffusers_result = generate_with_diffusers(prompt, diffusers_path)
        else:
            # Simulated responses for demonstration
            print(f"Simulating generation with DALL-E: '{prompt}'")
            print(f"✓ DALL-E generated image in 2.35 seconds")
            
            print(f"Simulating generation with Stability AI: '{prompt}'")
            print(f"✓ Stability AI generated image in 3.62 seconds")
            
            print(f"Simulating generation with Diffusers (local): '{prompt}'")
            print(f"✓ Diffusers generated image in 5.41 seconds")
    
    print("\nImage generation complete!")

if __name__ == "__main__":
    main() 