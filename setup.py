"""
Setup script for the AI Demo Platform.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aidemo",
    version="0.1.0",
    author="AI Demo Team",
    author_email="example@example.com",
    description="A comprehensive Python-based platform for demonstrating various AI model capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/aidemo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "sentence-transformers>=2.2.0",
        "openai>=1.0.0",
        "anthropic>=0.4.0",
        "cohere-api>=4.0.0",
        "langchain>=0.0.200",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=0.21.0",
        "pillow>=9.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "image": [
            "diffusers>=0.14.0",
            "stability-sdk>=0.8.0",
        ],
        "vector": [
            "faiss-cpu>=1.7.0",
            "chromadb>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aidemo-server=aidemo.src.api.server:run_server",
        ],
    },
) 