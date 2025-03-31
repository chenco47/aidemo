"""
Hugging Face model implementations.
"""
import os
import time
from typing import Dict, List, Any, Optional, Union

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModel
)
from sentence_transformers import SentenceTransformer

from aidemo.src.models.base import BaseModel
from aidemo.src.utils.logging import get_logger

logger = get_logger(__name__)

class HuggingFaceModel(BaseModel):
    """Implementation for locally-run Hugging Face models."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "causal_lm",
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Hugging Face model.
        
        Args:
            model_name: The model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
            model_type: Model type, one of 'causal_lm', 'seq2seq_lm', or 'embedding'
            device: Device to run the model on ('cpu', 'cuda', 'mps' or specific cuda device)
            **kwargs: Additional parameters for model loading
        """
        super().__init__(model_name, None)  # No API key for local models
        
        self.model_type = model_type
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading {model_type} model {model_name} on {self.device}")
        
        # Load the appropriate model based on model_type
        if model_type == "causal_lm":
            self._load_causal_lm(model_name, **kwargs)
        elif model_type == "seq2seq_lm":
            self._load_seq2seq_lm(model_name, **kwargs)
        elif model_type == "embedding":
            self._load_embedding_model(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _load_causal_lm(self, model_name: str, **kwargs) -> None:
        """Load a causal language model (like GPT, Llama, Mistral).
        
        Args:
            model_name: The model identifier
            **kwargs: Additional parameters for model loading
        """
        # Set default parameters for model loading
        params = {
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": self.device,
            "trust_remote_code": True
        }
        params.update(kwargs)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **params)
        
        # Set up generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def _load_seq2seq_lm(self, model_name: str, **kwargs) -> None:
        """Load a sequence-to-sequence model (like T5, BART).
        
        Args:
            model_name: The model identifier
            **kwargs: Additional parameters for model loading
        """
        # Set default parameters for model loading
        params = {
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": self.device,
            "trust_remote_code": True
        }
        params.update(kwargs)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **params)
        
        # Set up generation pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def _load_embedding_model(self, model_name: str, **kwargs) -> None:
        """Load an embedding model.
        
        Args:
            model_name: The model identifier
            **kwargs: Additional parameters for model loading
        """
        # Special handling for sentence-transformers models
        if "sentence-transformers" in model_name or kwargs.get("use_sentence_transformers", False):
            self.model = SentenceTransformer(model_name, device=self.device)
            self.tokenizer = None  # Not needed for SentenceTransformer
            self.pipe = None
        else:
            # Generic embedding model
            params = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": self.device
            }
            params.update(kwargs)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, **params)
            
            # Create embedding pipeline
            self.pipe = pipeline(
                "feature-extraction",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with the Hugging Face model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for text generation
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        if self.model_type == "causal_lm":
            params = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "return_full_text": False,
                "num_return_sequences": 1,
            }
        else:  # seq2seq_lm
            params = {
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "do_sample": True
            }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        try:
            # Generate text
            if self.pipe:
                results = self.pipe(prompt, **params)
                
                # Extract generated text (slightly different format based on model type)
                if self.model_type == "causal_lm":
                    generated_text = results[0]["generated_text"]
                else:  # seq2seq_lm
                    generated_text = results[0]["generated_text"]
            else:
                # Direct generation if no pipeline is available
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    input_ids, 
                    max_length=len(input_ids[0]) + params.get("max_new_tokens", 512),
                    **{k: v for k, v in params.items() if k != "max_new_tokens"}
                )
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                if not params.get("return_full_text", False):
                    # Remove the prompt from the output
                    generated_text = generated_text[len(prompt):]
            
            processing_time = time.time() - start_time
            
            # Create output dictionary
            output = {
                "text": generated_text,
                "model": self.model_name,
                "processing_time": processing_time
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating text with Hugging Face model: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings for the provided text.
        
        Args:
            text: The input text or list of texts
            **kwargs: Additional parameters for embedding
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        start_time = time.time()
        
        try:
            # Process input format
            input_texts = text if isinstance(text, list) else [text]
            
            # Generate embeddings based on the model type
            if isinstance(self.model, SentenceTransformer):
                # Use sentence-transformers
                embeddings = self.model.encode(
                    input_texts, 
                    convert_to_tensor=False,
                    **kwargs
                )
            elif self.pipe:
                # Use the pipeline
                outputs = self.pipe(input_texts)
                
                # Extract embeddings (using mean pooling of last hidden state)
                embeddings = []
                for output in outputs:
                    # Average the token embeddings
                    if isinstance(output, list):
                        # For feature-extraction pipeline, get mean of all token embeddings
                        embedding = torch.mean(
                            torch.tensor([token_emb for token_emb in output]), 
                            dim=0
                        ).numpy()
                    else:
                        # Handle other output formats
                        embedding = output
                    embeddings.append(embedding)
            else:
                # Direct model inference
                embeddings = []
                for txt in input_texts:
                    inputs = self.tokenizer(txt, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Mean pooling of the last hidden state
                    embeddings.append(
                        torch.mean(outputs.last_hidden_state[0], dim=0).cpu().numpy()
                    )
            
            processing_time = time.time() - start_time
            
            # Format the output
            output = {
                "embeddings": embeddings if isinstance(text, list) else embeddings[0],
                "model": self.model_name,
                "processing_time": processing_time
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Hugging Face model: {e}")
            raise


class HuggingFaceAPIModel(BaseModel):
    """Implementation for Hugging Face Inference API."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Hugging Face Inference API client.
        
        Args:
            model_name: The model identifier on Hugging Face Hub
            api_key: Hugging Face API key (if not provided, will use HUGGINGFACE_API_KEY env var)
            **kwargs: Additional parameters for the Inference API
        """
        super().__init__(model_name, api_key)
        
        self.api_key = self._validate_api_key("HUGGINGFACE_API_KEY")
        
        # Import here to avoid requiring as a dependency for local models
        from huggingface_hub import InferenceClient
        
        # Create the inference client
        self.client = InferenceClient(
            model=model_name,
            token=self.api_key,
            **kwargs
        )
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using the Hugging Face Inference API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the text generation
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "return_full_text": False
        }
        
        # Override defaults with any provided kwargs
        params.update(kwargs)
        
        try:
            # Generate text
            response = self.client.text_generation(
                prompt,
                **params
            )
            
            processing_time = time.time() - start_time
            
            # Create output dictionary
            output = {
                "text": response,
                "model": self.model_name,
                "processing_time": processing_time
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating text with Hugging Face Inference API: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings using the Hugging Face Inference API.
        
        Args:
            text: The input text or list of texts
            **kwargs: Additional parameters for the embedding
            
        Returns:
            Dictionary containing the embeddings and metadata
        """
        start_time = time.time()
        
        try:
            # Process input format
            input_texts = text if isinstance(text, list) else [text]
            
            # Generate embeddings for each text
            embeddings = []
            for txt in input_texts:
                embedding = self.client.feature_extraction(
                    txt,
                    **kwargs
                )
                embeddings.append(embedding)
            
            processing_time = time.time() - start_time
            
            # Format the output
            output = {
                "embeddings": embeddings if isinstance(text, list) else embeddings[0],
                "model": self.model_name,
                "processing_time": processing_time
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Hugging Face Inference API: {e}")
            raise


# Convenient SentenceTransformers wrapper that matches the regex patterns
class SentenceTransformersDocumentEmbedder:
    """Document embedder using SentenceTransformers."""
    
    def __init__(self, model="all-MiniLM-L6-v2"):
        """Initialize the embedder with a SentenceTransformers model.
        
        Args:
            model: SentenceTransformers model name
        """
        self.model = SentenceTransformer(model)
    
    def embed_documents(self, documents):
        """Embed a list of documents."""
        return self.model.encode(documents)
    
    def embed_query(self, query):
        """Embed a query."""
        return self.model.encode(query)


# Convenient HuggingFace API wrapper that matches the regex patterns
class HuggingFaceTextEmbedding:
    """Text embedding using HuggingFace Inference API."""
    
    def __init__(self, model_name):
        """Initialize with model name."""
        self.model_id = model_name
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
        
        # Import here to avoid requiring as a dependency for local models
        from huggingface_hub import InferenceClient
        
        self.client = InferenceClient(
            model=self.model_id,
            token=self.api_key
        )
    
    def embed(self, text):
        """Embed text using the model."""
        return self.client.feature_extraction(text)


class HuggingFaceTextCompletion:
    """Text completion using HuggingFace Inference API."""
    
    def __init__(self, model_name):
        """Initialize with model name."""
        self.model_id = model_name
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
        
        # Import here to avoid requiring as a dependency for local models
        from huggingface_hub import InferenceClient
        
        self.client = InferenceClient(
            model=self.model_id,
            token=self.api_key
        )
    
    def generate(self, prompt, **kwargs):
        """Generate text completion."""
        api_params={
            "temperature": 0.7,
            "max_new_tokens": 512,
            "model": self.model_id
        }
        api_params.update(kwargs)
        
        return self.client.text_generation(prompt, **api_params) 