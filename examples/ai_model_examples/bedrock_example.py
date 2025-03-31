"""
Example demonstrating AWS Bedrock model usage.
"""
import json
import boto3

# Create Bedrock client
bedrock = boto3.client("bedrock-runtime")

# Define request body for Titan model
body = json.dumps({
    "inputText": "Explain how large language models work",
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
})

# Define model ID (using Amazon Titan model)
model_id = "amazon.titan-text-express-v1"

# Define HTTP headers
accept = "application/json"
content_type = "application/json"

# Invoke the model
response = bedrock.invoke_model(
    body=body, 
    modelId=model_id, 
    accept=accept, 
    contentType=content_type
)

# Process response
response_body = json.loads(response.get("body").read())
generated_text = response_body.get("results")[0].get("outputText")

# Display generated text
print(generated_text)

# Example with Amazon Titan Embedding model
embedding_body = json.dumps({
    "inputText": "This is a sample text for embedding."
})

embedding_model_id = "amazon.titan-embed-text-v1"

embedding_response = bedrock.invoke_model(
    body=embedding_body, 
    modelId=embedding_model_id, 
    accept=accept, 
    contentType=content_type
)

embedding_response_body = json.loads(embedding_response.get("body").read())
embedding_vector = embedding_response_body.get("embedding")

# Example with Amazon Titan Image Generator
image_body = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": "A beautiful mountain landscape at sunset",
        "negativeText": "blurry, distorted, low quality",
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 1024,
        "width": 1024,
        "cfgScale": 8.0
    }
})

image_model_id = "amazon.titan-image-generator-v1"

image_response = bedrock.invoke_model(
    body=image_body, 
    modelId=image_model_id, 
    accept=accept, 
    contentType=content_type
)

# Example with Amazon Titan Text Premier model
premier_body = json.dumps({
    "inputText": "Write a short essay about climate change",
    "textGenerationConfig": {
        "maxTokenCount": 1024,
        "temperature": 0.5
    }
})

premier_model_id = "amazon.titan-text-premier-v1:0"

premier_response = bedrock.invoke_model(
    body=premier_body, 
    modelId=premier_model_id, 
    accept=accept, 
    contentType=content_type
) 