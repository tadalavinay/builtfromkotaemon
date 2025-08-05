#!/usr/bin/env python3
"""
Simple Chatbot Application using kotaemon's FullQAPipeline
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the libs directory to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs'))

from ktem.reasoning.simple import FullQAPipeline
from kotaemon.base import Document
from kotaemon.indices.vectorindex import VectorIndexing, VectorRetrieval
from kotaemon.embeddings import OpenAIEmbedding
from kotaemon.llms import ChatOpenAI

class SimpleChatbot:
    """A simple chatbot using kotaemon's FullQAPipeline"""
    
    def __init__(self, documents_path: str = "./documents"):
        self.documents_path = Path(documents_path)
        self.pipeline = None
        self.retriever = None
        self.conversation_history = []
        
    def setup_retriever(self, documents: List[str] = None):
        """Set up document retriever with sample documents"""
        logger.info("Setting up document retriever...")
        
        # Create documents directory if it doesn't exist
        self.documents_path.mkdir(exist_ok=True)
        
        # Create sample documents if none provided
        if not documents:
            sample_docs = [
                ("ai_basics.txt", "Artificial Intelligence (AI) is the simulation of human intelligence in machines. AI systems can perform tasks like learning, reasoning, and self-correction."),
                ("machine_learning.txt", "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."),
                ("deep_learning.txt", "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.")
            ]
            
            for filename, content in sample_docs:
                doc_path = self.documents_path / filename
                if not doc_path.exists():
                    doc_path.write_text(content)
                    logger.info(f"Created sample document: {filename}")
        
        # Initialize embedding model
        embedding = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Create vector index
        indexer = VectorIndexing(
            embedding=embedding,
            index_path="./vector_index"
        )
        
        # Index documents
        doc_paths = list(self.documents_path.glob("*.txt"))
        if doc_paths:
            logger.info(f"Indexing {len(doc_paths)} documents...")
            indexer.build_index(doc_paths)
        
        # Create retriever
        self.retriever = VectorRetrieval(
            embedding=embedding,
            index_path="./vector_index"
        )
        
        logger.info("Document retriever setup complete")
        
    def initialize_pipeline(self):
        """Initialize the FullQAPipeline"""
        logger.info("Initializing FullQAPipeline...")
        
        # Create basic settings
        settings = {
            "reasoning.max_context_length": 4000,
            "reasoning.options.simple.llm": "gpt-3.5-turbo",
            "reasoning.options.simple.highlight_citation": "highlight",
            "reasoning.options.simple.create_mindmap": False,
            "reasoning.options.simple.create_citation_viz": False,
            "reasoning.options.simple.use_multimodal": False,
            "reasoning.options.simple.system_prompt": "You are a helpful AI assistant. Answer questions based on the provided context.",
            "reasoning.options.simple.qa_prompt": """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:""",
            "reasoning.options.simple.n_last_interactions": 3,
            "reasoning.options.simple.trigger_context": 150,
            "reasoning.lang": "English"
        }
        
        # Initialize pipeline
        self.pipeline = FullQAPipeline.get_pipeline(
            settings=settings,
            states={},
            retrievers=[self.retriever]
        )
        
        logger.info("Pipeline initialized successfully")
        
    def chat(self, message: str) -> Dict[str, Any]:
        """Chat with the bot and get streaming responses"""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        logger.info(f"Processing message: {message}")
        
        response_parts = []
        citations = []
        
        # Stream response
        for document in self.pipeline.stream(
            message=message,
            conv_id="chat_session",
            history=self.conversation_history
        ):
            if document.channel == "chat":
                response_parts.append(document.content or "")
            elif document.channel == "info":
                citations.append(document.content or "")
        
        # Combine response
        full_response = "".join(response_parts)
        
        # Update conversation history
        self.conversation_history.append((message, full_response))
        
        return {
            "response": full_response,
            "citations": citations,
            "history": self.conversation_history
        }
        
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

def main():
    """Main function to run the chatbot"""
    print("🤖 Simple Chatbot using kotaemon")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    try:
        # Setup retriever with sample documents
        chatbot.setup_retriever()
        
        # Initialize pipeline
        chatbot.initialize_pipeline()
        
        print("✅ Chatbot ready! Type 'quit' to exit, 'clear' to clear history")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("History cleared!")
                continue
            elif not user_input:
                continue
                
            # Get response
            result = chatbot.chat(user_input)
            
            print(f"\nBot: {result['response']}")
            
            if result['citations']:
                print("\n📚 Sources:")
                for citation in result['citations']:
                    if citation and citation.strip():
                        print(f"  - {citation}")
                        
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()