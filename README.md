# AI Demo Platform

A comprehensive Python-based platform for demonstrating various AI model capabilities across different providers.

## Features

- Multi-provider LLM integrations (OpenAI, Anthropic, Cohere, Mistral, etc.)
- Text generation, embeddings, and completions
- Image generation with Stable Diffusion and DALL-E
- Fine-tuning examples for various models
- API server for model serving
- Benchmarking tools for model performance comparison

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aidemo.git
cd aidemo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### Running the API Server

```bash
python -m aidemo.src.api.server
```

The API will be available at `http://localhost:8000`.

## Project Structure

```
aidemo/
├── src/
│   ├── models/       # AI model implementations
│   ├── api/          # FastAPI server
│   ├── utils/        # Helper utilities
│   ├── data/         # Data handling
│   └── config/       # Configuration management
├── tests/            # Unit and integration tests
├── examples/         # Example scripts showing usage
└── docs/             # Documentation
```

## Usage Examples

See the `examples/` directory for detailed usage examples of each model provider.

## License

MIT 