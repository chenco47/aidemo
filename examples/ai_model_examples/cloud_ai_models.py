"""
Example demonstrating Google Vertex AI and Azure OpenAI models.
"""
import os
import json

# Google Vertex AI Models
def use_vertex_ai_models():
    try:
        from google.cloud import aiplatform
        from vertexai.language_models import TextGenerationModel, ChatModel, TextEmbeddingModel
        from vertexai.preview.generative_models import GenerativeModel, Part
        from vertexai.vision_models import ImageGenerationModel
        
        # Initialize Vertex AI
        aiplatform.init(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id"),
            location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        )
        
        # Use PaLM 2 Text model (legacy)
        palm_model = TextGenerationModel.from_pretrained("text-bison@002")
        palm_response = palm_model.predict(
            "Explain the concept of artificial general intelligence.",
            max_output_tokens=500,
            temperature=0.7
        )
        
        # Use Gemini Pro model for text
        gemini_pro = GenerativeModel("gemini-pro")
        gemini_pro_response = gemini_pro.generate_content(
            "What are the ethical considerations of AI development?"
        )
        
        # Use Gemini Pro Vision model for multimodal
        gemini_pro_vision = GenerativeModel("gemini-pro-vision")
        # This would use an image, but we're just showing the code pattern
        multimodal_response = gemini_pro_vision.generate_content([
            "Describe what you see in this image",
            Part.from_uri("gs://cloud-samples-data/generative-ai/image/scones.jpg")
        ])
        
        # Use Gemini 1.5 Pro model
        gemini_1_5_pro = GenerativeModel("gemini-1.5-pro")
        gemini_1_5_response = gemini_1_5_pro.generate_content(
            "How do large language models work?"
        )
        
        # Use text embeddings model
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-gecko@001")
        embeddings = embedding_model.get_embeddings(["This is a sample text for embedding."])
        
        # Use Imagen for image generation
        imagen = ImageGenerationModel.from_pretrained("imagegeneration@002")
        imagen_response = imagen.generate_images(
            prompt="A futuristic cityscape with flying cars and vertical gardens",
            # sample_count=1,
            # seed=42
        )
        
        return {
            "palm_response": palm_response.text[:100] + "...",
            "gemini_pro_response": gemini_pro_response.text[:100] + "...",
            "multimodal_response": multimodal_response.text[:100] + "...",
            "gemini_1_5_response": gemini_1_5_response.text[:100] + "...",
            "embeddings_dimensions": len(embeddings[0].values),
            "imagen_response": "Image generated successfully" if imagen_response else "No image generated"
        }
        
    except ImportError:
        return "Google Cloud packages not installed. Install with 'pip install google-cloud-aiplatform vertexai'"

# Azure OpenAI Models
def use_azure_openai_models():
    try:
        import openai
        from azure.identity import DefaultAzureCredential
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from msrest.authentication import CognitiveServicesCredentials
        
        # Azure OpenAI setup
        openai.api_type = "azure"
        openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/")
        openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "your-api-key")
        openai.api_version = "2023-05-15"
        
        # Use GPT-4 on Azure
        gpt4_response = openai.ChatCompletion.create(
            engine="gpt-4",  # Azure deployment name
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain the concept of artificial general intelligence."}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Use GPT-3.5-Turbo on Azure
        gpt35_response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",  # Azure deployment name
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What are the ethical considerations of AI development?"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Use DALL-E on Azure
        dalle_response = openai.Image.create(
            prompt="A futuristic cityscape with flying cars and vertical gardens",
            n=1,
            size="1024x1024"
        )
        
        # Use Azure's embeddings model
        embedding_response = openai.Embedding.create(
            engine="text-embedding-ada-002",  # Azure deployment name
            input="This is a sample text for embedding."
        )
        
        # Use Azure Text Analytics
        text_analytics_endpoint = os.environ.get("AZURE_TEXT_ANALYTICS_ENDPOINT", "https://your-resource.cognitiveservices.azure.com/")
        text_analytics_key = os.environ.get("AZURE_TEXT_ANALYTICS_KEY", "your-key")
        
        text_analytics_client = TextAnalyticsClient(
            endpoint=text_analytics_endpoint,
            credential=CognitiveServicesCredentials(text_analytics_key)
        )
        
        text_analytics_result = text_analytics_client.analyze_sentiment(
            documents=["I had a wonderful experience at the restaurant today."]
        )
        
        # Use Azure Computer Vision
        vision_endpoint = os.environ.get("AZURE_VISION_ENDPOINT", "https://your-resource.cognitiveservices.azure.com/")
        vision_key = os.environ.get("AZURE_VISION_KEY", "your-key")
        
        vision_client = ComputerVisionClient(
            endpoint=vision_endpoint,
            credential=CognitiveServicesCredentials(vision_key)
        )
        
        # This would analyze an image, but we're just showing the code pattern
        vision_result = "Computer Vision client initialized successfully"
        
        return {
            "gpt4_response": gpt4_response["choices"][0]["message"]["content"][:100] + "...",
            "gpt35_response": gpt35_response["choices"][0]["message"]["content"][:100] + "...",
            "dalle_response": dalle_response["data"][0]["url"] if "data" in dalle_response else "No image URL",
            "embeddings_dimension": len(embedding_response["data"][0]["embedding"]),
            "text_analytics_result": "Sentiment analysis completed" if text_analytics_result else "No sentiment analysis",
            "vision_result": vision_result
        }
        
    except ImportError:
        return "Azure packages not installed. Install with 'pip install openai azure-cognitiveservices-vision-computervision azure-ai-textanalytics azure-identity'"

# Call the functions
if __name__ == "__main__":
    print("Running simulated examples only - no API calls will be made.")
    
    # These would make actual API calls if the code was run and API keys were provided
    print("Google Vertex AI Models Example:")
    # print(use_vertex_ai_models())
    
    print("\nAzure OpenAI Models Example:")
    # print(use_azure_openai_models()) 