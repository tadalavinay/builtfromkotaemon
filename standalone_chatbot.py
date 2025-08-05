#!/usr/bin/env python3
"""
Standalone Chatbot Implementation
A self-contained chatbot that doesn't depend on the ktem library
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import tiktoken
from dataclasses import dataclass
import time

@dataclass
class Document:
    """Simple document structure"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]

class SimpleVectorStore:
    """Simple vector store for document embeddings"""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None
        self.client = OpenAI()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        self.documents = documents
        
        # Create embeddings for all documents
        texts = [doc.content for doc in documents]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        self.embeddings = np.array([data.embedding for data in response.data])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.embeddings:
            return []
        
        # Get query embedding
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.7:  # Similarity threshold
                results.append({
                    "document": self.documents[idx],
                    "score": float(similarities[idx])
                })
        
        return results

class SimpleQAPipeline:
    """Simple QA pipeline using OpenAI"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.vector_store = SimpleVectorStore()
        self.client = OpenAI()
        self.conversation_history = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the pipeline"""
        self.vector_store.add_documents(documents)
    
    def generate_prompt(self, question: str, context: str) -> str:
        """Generate prompt for the LLM"""
        return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above. If the context doesn't contain enough information to answer the question, say so.

Answer:"""
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Process a question and return an answer"""
        # Search for relevant documents
        search_results = self.vector_store.search(question)
        
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            doc = result["document"]
            context_parts.append(f"[{i}] {doc.title}: {doc.content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Generate prompt
        prompt = self.generate_prompt(question, context)
        
        # Get response from OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        # Add conversation history for context
        for msg in self.conversation_history[-4:]:  # Last 4 messages
            messages.append(msg)
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Prepare citations
        citations = []
        for i, result in enumerate(search_results, 1):
            doc = result["document"]
            citations.append({
                "id": doc.id,
                "title": doc.title,
                "score": result["score"]
            })
        
        return {
            "answer": answer,
            "citations": citations,
            "context_used": len(search_results) > 0
        }

class SimpleChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        self.pipeline = SimpleQAPipeline()
        self.setup_sample_documents()
    
    def setup_sample_documents(self):
        """Create and add sample documents"""
        sample_docs = [
            Document(
                id="doc1",
                title="Machine Learning Basics",
                content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                metadata={"category": "technology", "tags": ["AI", "ML"]}
            ),
            Document(
                id="doc2",
                title="Python Programming Guide",
                content="Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a large standard library. Python is widely used in web development, data analysis, AI, and scientific computing.",
                metadata={"category": "programming", "tags": ["python", "coding"]}
            ),
            Document(
                id="doc3",
                title="Data Science Overview",
                content="Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines domain expertise, programming skills, and knowledge of mathematics and statistics.",
                metadata={"category": "data science", "tags": ["analytics", "statistics"]}
            )
        ]
        
        self.pipeline.add_documents(sample_docs)
        print(f"✅ Loaded {len(sample_docs)} sample documents")
    
    def start_chat(self):
        """Start interactive chat session"""
        print("\n🤖 Simple Chatbot - Ready to chat!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🤔 Thinking...")
                
                # Get response
                result = self.pipeline.chat(question)
                
                print(f"\nBot: {result['answer']}")
                
                # Show citations if available
                if result['citations']:
                    print("\n📚 Sources:")
                    for citation in result['citations']:
                        print(f"  - {citation['title']} (relevance: {citation['score']:.2f})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please check your OpenAI API key and try again.")

def main():
    """Main function"""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        chatbot = SimpleChatbot()
        chatbot.start_chat()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")

if __name__ == "__main__":
    main()