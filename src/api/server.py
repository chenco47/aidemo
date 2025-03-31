"""
FastAPI server for the AI demo platform.
"""
import os
from typing import Dict, List, Any, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aidemo.src.models.factory import create_model, list_available_models
from aidemo.src.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Demo Platform API",
    description="API for the AI Demo Platform",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class GenerateRequest(BaseModel):
    model: str = Field(..., description="Model name or identifier")
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: Optional[int] = Field(1000, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Temperature for sampling")
    provider: Optional[str] = Field(None, description="Model provider (if needed to disambiguate)")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "gpt-4o",
                "prompt": "Write a short poem about artificial intelligence.",
                "max_tokens": 500,
                "temperature": 0.8
            }
        }

class EmbedRequest(BaseModel):
    model: str = Field(..., description="Model name or identifier")
    text: Union[str, List[str]] = Field(..., description="Text to embed")
    provider: Optional[str] = Field(None, description="Model provider (if needed to disambiguate)")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "text-embedding-3-small",
                "text": "This is a sample text for embedding."
            }
        }

class ImageGenerateRequest(BaseModel):
    model: str = Field(..., description="Model name or identifier")
    prompt: str = Field(..., description="Image description")
    negative_prompt: Optional[str] = Field("", description="Elements to exclude from the image")
    width: Optional[int] = Field(1024, description="Image width")
    height: Optional[int] = Field(1024, description="Image height")
    provider: Optional[str] = Field(None, description="Model provider (if needed to disambiguate)")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "dall-e-3",
                "prompt": "A futuristic city with flying cars and neon lights.",
                "negative_prompt": "dystopian, dark, gloomy",
                "width": 1024,
                "height": 1024
            }
        }

# Routes
@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "AI Demo Platform API", "status": "running"}

@app.get("/models")
def get_models():
    """List available models."""
    return list_available_models()

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using the specified model."""
    try:
        # Create the model
        model = create_model(
            model_name=request.model,
            provider=request.provider
        )
        
        # Prepare parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        # Handle provider-specific parameters
        if request.model.startswith(("gpt-", "text-davinci-")):
            # OpenAI specifics
            pass
        elif request.model.startswith("claude-"):
            # Claude specifics
            params["max_tokens"] = request.max_tokens
        
        # Generate text
        result = model.generate(request.prompt, **params)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    """Generate embeddings for the specified text."""
    try:
        # Create the model
        model = create_model(
            model_name=request.model,
            provider=request.provider
        )
        
        # Generate embeddings
        result = model.embed(request.text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: ImageGenerateRequest):
    """Generate an image using the specified model."""
    try:
        # Special handling based on model provider
        if request.model == "dall-e-3" or request.model.startswith("dall-e-"):
            # OpenAI DALL-E
            model = create_model("openai", request.model)
            result = model.generate_image(
                prompt=request.prompt,
                size=f"{request.width}x{request.height}"
            )
        elif request.model.startswith("stable-diffusion-") or request.model.startswith("stability."):
            # Stability AI models
            model = create_model("stability", request.model)
            result = model.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height
            )
        else:
            # Default handling for other models
            model = create_model(
                model_name=request.model,
                provider=request.provider
            )
            result = model.generate_image(
                prompt=request.prompt,
                width=request.width,
                height=request.height
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
def startup_event():
    """Run on startup."""
    logger.info("AI Demo Platform API starting up...")

# Main function to run the server
def run_server():
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    workers = int(os.environ.get("API_WORKERS", "1"))
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "aidemo.src.api.server:app",
        host=host,
        port=port,
        workers=workers
    )

if __name__ == "__main__":
    run_server() 