#!/usr/bin/env python3
"""
Configuration for OpenRouter API usage
"""

import os
from typing import Dict, Any

class OpenRouterConfig:
    """Configuration class for OpenRouter API settings"""
    
    # OpenRouter API settings
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Available models
    CHAT_MODELS = {
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-4-turbo": "openai/gpt-4-turbo-preview",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3-opus": "anthropic/claude-3-opus",
        "llama-3-8b": "meta-llama/llama-3-8b-instruct",
        "llama-3-70b": "meta-llama/llama-3-70b-instruct",
        "gemini-pro": "google/gemini-pro",
        "mistral-7b": "mistralai/mistral-7b-instruct",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct"
    }
    
    EMBEDDING_MODELS = {
        "ada-002": "openai/text-embedding-ada-002",
        "text-embedding-3-small": "openai/text-embedding-3-small",
        "text-embedding-3-large": "openai/text-embedding-3-large"
    }
    
    # Default settings
    DEFAULT_CHAT_MODEL = "openai/gpt-3.5-turbo"
    DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-ada-002"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TOP_K = 3
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get OpenRouter API key from environment"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.\n"
                "Get your key from: https://openrouter.ai/keys"
            )
        return api_key
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get headers for OpenRouter API requests"""
        api_key = cls.get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Chatbot Demo",
            "Content-Type": "application/json"
        }
    
    @classmethod
    def validate_model(cls, model: str, model_type: str = "chat") -> str:
        """Validate and return the correct model name"""
        if model_type == "chat":
            if model in cls.CHAT_MODELS:
                return cls.CHAT_MODELS[model]
            elif model in cls.CHAT_MODELS.values():
                return model  # Already a full model name
            else:
                print(f"Warning: Unknown chat model '{model}'. Using default.")
                return cls.DEFAULT_CHAT_MODEL
        elif model_type == "embedding":
            if model in cls.EMBEDDING_MODELS:
                return cls.EMBEDDING_MODELS[model]
            elif model in cls.EMBEDDING_MODELS.values():
                return model  # Already a full model name
            else:
                print(f"Warning: Unknown embedding model '{model}'. Using default.")
                return cls.DEFAULT_EMBEDDING_MODEL
        
        raise ValueError(f"Invalid model type: {model_type}")
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "chat_models": cls.CHAT_MODELS,
            "embedding_models": cls.EMBEDDING_MODELS,
            "defaults": {
                "chat": cls.DEFAULT_CHAT_MODEL,
                "embedding": cls.DEFAULT_EMBEDDING_MODEL
            }
        }
    
    @classmethod
    def print_model_info(cls):
        """Print available models"""
        info = cls.get_model_info()
        
        print("\n🎯 Available Chat Models:")
        for key, value in info["chat_models"].items():
            print(f"  • {key}: {value}")
        
        print("\n🎯 Available Embedding Models:")
        for key, value in info["embedding_models"].items():
            print(f"  • {key}: {value}")
        
        print(f"\n📌 Defaults:")
        print(f"  • Chat: {info['defaults']['chat']}")
        print(f"  • Embedding: {info['defaults']['embedding']}")