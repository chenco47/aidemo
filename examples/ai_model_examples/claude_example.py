"""
Example demonstrating Anthropic Claude model usage.
"""
import os
import json
import anthropic
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here"),
)

# Generate text with Claude 3 Opus
def generate_with_claude_opus():
    response = client.messages.create(
        model="anthropic.claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="You are a helpful AI assistant that provides informative and accurate answers.",
        messages=[
            {"role": "user", "content": "Explain the concept of artificial intelligence and its potential impact on society."}
        ]
    )
    return response.content[0].text

# Generate text with Claude 3 Sonnet
def generate_with_claude_sonnet():
    response = client.messages.create(
        model="anthropic.claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="You are a helpful AI assistant that provides concise answers.",
        messages=[
            {"role": "user", "content": "Summarize the key advancements in AI over the past decade."}
        ]
    )
    return response.content[0].text

# Generate text with Claude 3 Haiku
def generate_with_claude_haiku():
    response = client.messages.create(
        model="anthropic.claude-3-haiku-20240307",
        max_tokens=500,
        temperature=0.7,
        system="You are a helpful AI assistant that provides brief answers.",
        messages=[
            {"role": "user", "content": "What are the main challenges in AI safety?"}
        ]
    )
    return response.content[0].text

# Generate text with Claude 3.5 Sonnet
def generate_with_claude_3_5_sonnet():
    response = client.messages.create(
        model="anthropic.claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        system="You are a helpful AI assistant that provides informative answers.",
        messages=[
            {"role": "user", "content": "Explain how large language models work."}
        ]
    )
    return response.content[0].text

# Legacy Claude models
def generate_with_claude_legacy():
    response = client.messages.create(
        model="claude-2.1",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": "What are the ethical considerations for AI deployment?"}
        ]
    )
    return response.content[0].text

# AWS Bedrock Claude integration
def generate_with_bedrock_claude():
    import boto3
    
    bedrock = boto3.client("bedrock-runtime")
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": "Explain the concept of AI alignment and why it matters."
            }
        ]
    })
    
    model_id = "anthropic.claude-instant-v1"
    
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body["content"][0]["text"]

# Call the functions
if __name__ == "__main__":
    print("Claude 3 Opus Response:")
    print(generate_with_claude_opus())
    
    print("\nClaude 3 Sonnet Response:")
    print(generate_with_claude_sonnet())
    
    print("\nClaude 3 Haiku Response:")
    print(generate_with_claude_haiku())
    
    print("\nClaude 3.5 Sonnet Response:")
    print(generate_with_claude_3_5_sonnet()) 