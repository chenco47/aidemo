"""
Example demonstrating OpenAI model usage.
"""
import os
import openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your_api_key_here"),
)

# Generate text with GPT-4o
def generate_with_gpt4o():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that provides informative and accurate answers."},
            {"role": "user", "content": "Explain the concept of artificial general intelligence."}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

# Generate text with GPT-4 Turbo
def generate_with_gpt4_turbo():
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that provides concise answers."},
            {"role": "user", "content": "What are the key differences between narrow AI and general AI?"}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

# Generate text with GPT-4
def generate_with_gpt4():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain how transformer neural networks work."}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

# Generate text with GPT-3.5 Turbo
def generate_with_gpt35_turbo():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What are the ethical considerations of AI development?"}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

# Generate embeddings with text-embedding-3-large
def generate_embeddings_large():
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="This is a sample text to generate embeddings for."
    )
    return response.data[0].embedding[:5]  # Return just the first 5 values for brevity

# Generate embeddings with text-embedding-3-small
def generate_embeddings_small():
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="This is another sample text to generate embeddings for."
    )
    return response.data[0].embedding[:5]  # Return just the first 5 values for brevity

# Generate embeddings with text-embedding-ada-002
def generate_embeddings_ada():
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="This is a legacy embedding model sample."
    )
    return response.data[0].embedding[:5]  # Return just the first 5 values for brevity

# Generate image with DALL-E 3
def generate_image_dalle3():
    response = client.images.generate(
        model="dall-e-3",
        prompt="A futuristic city with flying cars and holographic billboards.",
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url

# Generate image with DALL-E 2
def generate_image_dalle2():
    response = client.images.generate(
        model="dall-e-2",
        prompt="A serene landscape with mountains and a lake at sunset.",
        size="1024x1024",
        n=1
    )
    return response.data[0].url

# Generate text-to-speech with tts-1
def generate_tts():
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello, world! This is a text-to-speech example."
    )
    speech_file_path = "speech.mp3"
    response.stream_to_file(speech_file_path)
    return speech_file_path

# Generate text-to-speech with tts-1-hd
def generate_tts_hd():
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input="This is a high-definition text-to-speech example."
    )
    speech_file_path = "speech_hd.mp3"
    response.stream_to_file(speech_file_path)
    return speech_file_path

# Use text moderation
def moderate_text():
    response = client.moderations.create(
        model="text-moderation-latest",
        input="I want to harm someone."
    )
    return response.results[0]

# Use legacy models
def generate_with_legacy():
    response = client.completions.create(
        model="davinci-002",
        prompt="Write a short story about AI.",
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text

# Call the functions
if __name__ == "__main__":
    print("GPT-4o Response:")
    print(generate_with_gpt4o())
    
    print("\nGPT-4 Turbo Response:")
    print(generate_with_gpt4_turbo())
    
    print("\nEmbeddings (first 5 values):")
    print(generate_embeddings_large()) 