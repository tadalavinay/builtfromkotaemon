"""
Configuration file for the simple chatbot application
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_INDEX_DIR = BASE_DIR / "vector_index"

# Create directories
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_INDEX_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Model Configuration
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Pipeline Configuration
MAX_CONTEXT_LENGTH = 4000
MAX_INTERACTIONS = 3
TRIGGER_CONTEXT_LENGTH = 150

# Chatbot Settings
CHATBOT_NAME = "Simple Chatbot"
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based on the provided context. 
If you don't know the answer, just say that you don't know, don't try to make up an answer."""

QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

# Citation Settings
HIGHLIGHT_CITATION = "highlight"  # "highlight", "inline", or "off"
ENABLE_MINDMAP = False
ENABLE_CITATION_VIZ = False
USE_MULTIMODAL = False

# Language Settings
SUPPORTED_LANGUAGE = "English"