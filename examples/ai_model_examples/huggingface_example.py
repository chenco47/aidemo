"""
Example demonstrating HuggingFace model usage.
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModel
)
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Use the Inference API with a remote model
def use_inference_api():
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        token=os.environ.get("HUGGINGFACE_API_KEY")
    )
    
    response = client.text_generation(
        "Explain how transformers work in deep learning.",
        max_new_tokens=512,
        temperature=0.7
    )
    
    return response

# Use Hugging Face pipeline with a pretrained model
def use_text_generation_pipeline():
    # This pattern should match the regex: pipeline.*model="..."
    gen_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        device_map="auto"
    )
    
    result = gen_pipeline(
        "Artificial intelligence is",
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    
    return result[0]["generated_text"]

# Use LLaMa model
def use_llama_model():
    # This pattern should match the regex: from_pretrained("...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "Write a short essay about the future of AI."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Use Mistral model
def use_mistral_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "Explain the concept of AI alignment."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Use Falcon model
def use_falcon_model():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "Write a short story about robots."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Use T5 model for translation
def use_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    
    input_text = "translate English to German: The AI revolution is here."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(input_ids, max_new_tokens=100)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

# Use BERT for embeddings
def use_bert_embeddings():
    # Load model with model_id parameter
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id)
    
    # Generate embeddings
    sentences = [
        "This is an example sentence for embedding.",
        "Each sentence will be converted to a vector."
    ]
    
    embeddings = model.encode(sentences)
    return embeddings.shape

# Use Multiple NLP pipelines
def use_multiple_pipelines():
    # Sentiment analysis
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Named entity recognition
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english"
    )
    
    # Question answering
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )
    
    # Use the pipelines
    sentiment_result = sentiment_analyzer("I love using Hugging Face models!")
    
    ner_result = ner_pipeline("Hugging Face is a technology company based in New York City and Paris.")
    
    qa_result = qa_pipeline(
        question="What is Hugging Face?",
        context="Hugging Face is a company that provides natural language processing tools and models."
    )
    
    return {
        "sentiment": sentiment_result,
        "ner": ner_result[:2],  # Just the first two recognized entities
        "qa": qa_result
    }

# Create a custom inference client
def create_custom_inference():
    # From the regex: InferenceClient.*model="..."
    client = InferenceClient(
        model="deepseek-ai/deepseek-coder-6.7b-instruct",
        token=os.environ.get("HUGGINGFACE_API_KEY")
    )
    
    # Generate code
    code_prompt = "Write a Python function to calculate the Fibonacci sequence."
    response = client.text_generation(
        code_prompt,
        max_new_tokens=500,
        temperature=0.7
    )
    
    return response

# Using text embedders
def use_text_embedders():
    # Match regex for HuggingFaceTextEmbedding
    embedder1 = HuggingFaceTextEmbedding("sentence-transformers/all-mpnet-base-v2")
    
    # Match regex for SentenceTransformersTextEmbedder
    embedder2 = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    
    # Match regex for SentenceTransformersDocumentEmbedder
    embedder3 = SentenceTransformersDocumentEmbedder(model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    
    text = "This is a sample text for embedding comparison."
    
    return {
        "embedder1_dimensions": len(embedder1.embed(text)),
        "embedder2_dimensions": len(embedder2.embed(text)),
        "embedder3_dimensions": len(embedder3.embed_query(text))
    }

# Helper classes that match regex patterns
class HuggingFaceTextEmbedding:
    def __init__(self, model_name):
        self.model_id = model_name
        # Simulate model loading
    
    def embed(self, text):
        # Simulate 768-dimensional embedding
        return [0.1] * 768

class SentenceTransformersTextEmbedder:
    def __init__(self, model):
        self.model = model
        # Simulate model loading
    
    def embed(self, text):
        # Simulate 384-dimensional embedding
        return [0.1] * 384

class SentenceTransformersDocumentEmbedder:
    def __init__(self, model):
        self.model = model
        # Simulate model loading
    
    def embed_query(self, text):
        # Simulate 768-dimensional embedding
        return [0.1] * 768
    
    def embed_documents(self, documents):
        # Simulate embedding multiple documents
        return [[0.1] * 768 for _ in documents]

# Using various model API parameters
def use_model_parameters():
    api_params = {
        "temperature": 0.7,
        "max_new_tokens": 100,
        "model": "gpt2"
    }
    
    # Create a pipeline with model_id
    summarizer = pipeline(
        "summarization",
        model_id="facebook/bart-large-cnn",
        device_map="auto"
    )
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    natural intelligence displayed by animals including humans. Leading AI textbooks define 
    the field as the study of "intelligent agents": any system that perceives its environment 
    and takes actions that maximize its chance of achieving its goals.
    """
    
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    
    return summary[0]["summary_text"]

# Call the functions
if __name__ == "__main__":
    print("Text Generation Result:")
    print(use_text_generation_pipeline())
    
    print("\nT5 Translation Result:")
    print(use_t5_model())
    
    print("\nBERT Embeddings Shape:")
    print(use_bert_embeddings()) 