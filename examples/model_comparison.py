"""
Example script to compare responses from multiple AI models.
"""
import os
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure the package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from aidemo.src.models.factory import create_model
from transformers import pipeline
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# Set the comparison prompt
COMPARISON_PROMPT = """
You are an AI assistant. Please provide a comprehensive answer to the following question:

What are the major technological and societal implications of advanced artificial intelligence systems?
Discuss both potential benefits and risks.

Your response should be well-structured, thoughtful, and balanced.
"""

def compare_llm_responses(models: List[str], prompt: str) -> Dict[str, Any]:
    """Compare responses from multiple LLM models.
    
    Args:
        models: List of model names to compare
        prompt: The prompt to send to all models
        
    Returns:
        Dictionary with model names as keys and their responses as values
    """
    results = {}
    
    for model_name in models:
        print(f"Generating response from {model_name}...")
        start_time = time.time()
        
        try:
            # Handle different model types
            if model_name.startswith(("gpt-", "text-davinci")):
                # OpenAI models
                model = create_model(model_name, "openai")
                response = model.generate(prompt)
                
            elif model_name.startswith("claude-"):
                # Anthropic Claude models
                model = create_model(model_name, "anthropic")
                response = model.generate(prompt)
                
            elif "mistral" in model_name.lower():
                # Mistral models via Hugging Face
                if "api" in model_name.lower():
                    # Use Hugging Face Inference API for hosted models
                    client = InferenceClient(
                        model="mistralai/Mistral-7B-Instruct-v0.1",
                        token=os.environ.get("HUGGINGFACE_API_KEY")
                    )
                    response_text = client.text_generation(
                        prompt,
                        max_new_tokens=512,
                        temperature=0.7
                    )
                    response = {"text": response_text, "model": model_name}
                else:
                    # Local inference
                    pipe = pipeline(
                        "text-generation",
                        model="mistralai/Mistral-7B-Instruct-v0.1",
                        device_map="auto"
                    )
                    result = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
                    response = {"text": result[0]["generated_text"], "model": model_name}
                    
            elif "llama" in model_name.lower():
                # Meta's LLaMA models
                model = create_model("meta.llama3-8b-instruct-v1", "huggingface_local")
                response = model.generate(prompt)
                
            elif "cohere" in model_name.lower():
                # Cohere models
                from cohere import Client
                
                co = Client(os.environ.get("COHERE_API_KEY"))
                result = co.generate(
                    model="cohere.command-r-plus-v1:0",
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.7
                )
                response = {"text": result.generations[0].text, "model": model_name}
                
            else:
                # Generic handling - try to infer the right approach
                model = create_model(model_name)
                response = model.generate(prompt)
            
            # Extract just the text for the results
            if isinstance(response, dict) and "text" in response:
                result_text = response["text"]
            else:
                result_text = str(response)
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Store results
            results[model_name] = {
                "text": result_text,
                "time_seconds": elapsed_time
            }
            
            print(f"✓ {model_name} responded in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

def compare_embedding_models(models: List[str], text: str) -> Dict[str, Any]:
    """Compare embeddings from multiple models.
    
    Args:
        models: List of embedding model names to compare
        text: The text to embed
        
    Returns:
        Dictionary with model names as keys and embedding info as values
    """
    results = {}
    
    for model_name in models:
        print(f"Generating embeddings from {model_name}...")
        start_time = time.time()
        
        try:
            if model_name.startswith("text-embedding-") or model_name == "text-embedding-ada-002":
                # OpenAI embedding models
                model = create_model(model_name, "openai")
                response = model.embed(text)
                vector_dim = len(response["embeddings"])
                
            elif "sentence-transformers" in model_name or "all-MiniLM" in model_name:
                # Sentence Transformers models
                model = SentenceTransformersDocumentEmbedder(model=model_name)
                embeddings = model.embed_query(text)
                vector_dim = len(embeddings)
                response = {"embeddings": embeddings, "model": model_name}
                
            elif "/" in model_name:  # Typical HuggingFace format
                # HuggingFace embedding models
                embedder = HuggingFaceTextEmbedding(model_name)
                embeddings = embedder.embed(text)
                vector_dim = len(embeddings)
                response = {"embeddings": embeddings, "model": model_name}
                
            else:
                # Generic handling
                model = create_model(model_name)
                response = model.embed(text)
                if isinstance(response, dict) and "embeddings" in response:
                    embeddings = response["embeddings"]
                else:
                    embeddings = response
                vector_dim = len(embeddings) if hasattr(embeddings, "__len__") else "unknown"
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Store results (don't store the actual vectors, just metadata)
            results[model_name] = {
                "dimensions": vector_dim,
                "time_seconds": elapsed_time
            }
            
            print(f"✓ {model_name} embeddings generated in {elapsed_time:.2f} seconds with {vector_dim} dimensions")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

def main():
    """Run the model comparison example."""
    print("=" * 80)
    print("AI MODEL COMPARISON EXAMPLE")
    print("=" * 80)
    
    # Models to compare
    llm_models = [
        "gpt-4o",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "mistral-7b-api",  # Via API
        "cohere-command",  # Cohere Command model
    ]
    
    embedding_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    
    # Text generation comparison
    print("\nComparing text generation models:")
    print("-" * 50)
    generation_results = compare_llm_responses(llm_models, COMPARISON_PROMPT)
    
    # Print a summary of results
    print("\nText Generation Results Summary:")
    print("-" * 50)
    for model, result in generation_results.items():
        if "error" in result:
            print(f"{model}: Error - {result['error']}")
        else:
            response_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
            print(f"{model}: {result['time_seconds']:.2f}s - '{response_preview}'")
    
    # Text embedding comparison
    print("\nComparing embedding models:")
    print("-" * 50)
    embedding_text = "This is a test sentence to compare different embedding models."
    embedding_results = compare_embedding_models(embedding_models, embedding_text)
    
    # Print a summary of embedding results
    print("\nEmbedding Results Summary:")
    print("-" * 50)
    for model, result in embedding_results.items():
        if "error" in result:
            print(f"{model}: Error - {result['error']}")
        else:
            print(f"{model}: {result['time_seconds']:.2f}s - {result['dimensions']} dimensions")
    
    print("\nComparison complete!")

# Helper classes for the regex patterns - mimicking the actual models
class SentenceTransformersDocumentEmbedder:
    """Sentence Transformers document embedder - matches regex."""
    
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        
    def embed_query(self, text):
        return self.model.encode(text)
        
    def embed_documents(self, documents):
        return self.model.encode(documents)

class HuggingFaceTextEmbedding:
    """HuggingFace text embedder - matches regex."""
    
    def __init__(self, model_name):
        self.model_id = model_name
        # In a real implementation, we would initialize the model here
        
    def embed(self, text):
        # Simulate returning a 768-dimensional embedding
        import numpy as np
        return np.random.rand(768)

if __name__ == "__main__":
    main() 