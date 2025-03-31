"""
Simple test file to trigger AI scanner detection.
"""
import openai
from openai import OpenAI

# This should be detected by the AI scanner
client = OpenAI(api_key="placeholder_key")

def generate_text():
    """Generate text using GPT-4."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is artificial intelligence?"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

# This function should also be detected
def create_embedding():
    """Create an embedding using OpenAI's embedding model."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="The quick brown fox jumps over the lazy dog."
    )
    return response.data[0].embedding 