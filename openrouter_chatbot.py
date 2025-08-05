#!/usr/bin/env python3
"""
OpenRouter Chatbot with RAG capabilities
Uses OpenRouter API for chat completions and OpenAI for embeddings
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI

# Configure OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "OpenRouter Chatbot",
    }
)

# Configure OpenAI client for embeddings (fallback)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

@dataclass
class Document:
    """Simple document class for storing text and metadata"""
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class Embeddings(ABC):
    """Abstract base class for embeddings"""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        pass

class OpenAIEmbeddings(Embeddings):
    """OpenAI embeddings implementation"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        response = openai_client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        response = openai_client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]

class SimpleVectorStore:
    """Simple in-memory vector store for document retrieval"""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store"""
        texts = [doc.content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        
        self.documents.extend(documents)
        self.vectors.extend(vectors)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Find most similar documents to query"""
        if not self.documents:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in similarities[:k]]
        
        return [self.documents[i] for i in top_indices]

class BaseRetriever:
    """Base class for document retrievers"""
    
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents"""
        return self.vector_store.similarity_search(query, k)

class OpenRouterChat:
    """OpenRouter chat interface"""
    
    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self.model = model
    
    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Stream chat responses"""
        try:
            response = openrouter_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"

class OpenRouterQAPipeline:
    """Question answering pipeline using OpenRouter"""
    
    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self.chat = OpenRouterChat(model)
        self.retriever = None
    
    def set_retriever(self, retriever: BaseRetriever):
        """Set the document retriever"""
        self.retriever = retriever
    
    def answer(self, question: str) -> Iterator[str]:
        """Generate streaming answer with context"""
        if not self.retriever:
            yield "Error: No document retriever configured"
            return
        
        # Retrieve relevant documents
        docs = self.retriever.retrieve(question)
        
        if not docs:
            yield "No relevant documents found. I'll provide a general answer.\n\n"
            context = ""
        else:
            context = "\n\n".join([doc.content for doc in docs])
        
        # Build prompt with context
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. If no context is provided, give a general answer."},
            {"role": "user", "content": prompt}
        ]
        
        yield from self.chat.stream(messages)

class OpenRouterChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        self.pipeline = None
        self.documents_loaded = False
        
    def load_documents(self, file_path: str) -> bool:
        """Load documents from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (simple approach)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            
            # Create documents
            documents = [
                Document(content=chunk, metadata={"source": file_path, "chunk_id": i})
                for i, chunk in enumerate(chunks)
            ]
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            vector_store = SimpleVectorStore(embeddings)
            vector_store.add_documents(documents)
            
            # Create retriever and pipeline
            retriever = BaseRetriever(vector_store)
            self.pipeline = OpenRouterQAPipeline()
            self.pipeline.set_retriever(retriever)
            
            self.documents_loaded = True
            print(f"✅ Loaded {len(documents)} document chunks from {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return False
    
    def start_chat(self):
        """Start interactive chat session"""
        print("\n🤖 OpenRouter Chatbot with RAG")
        print("=" * 50)
        
        # Check for required API keys
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ Please set your OpenRouter API key:")
            print("export OPENROUTER_API_KEY='your-key-here'")
            return
            
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set your OpenAI API key for embeddings:")
            print("export OPENAI_API_KEY='your-key-here'")
            return
        
        # Load sample documents if available
        sample_files = ["sample.txt", "README.md", "requirements.txt"]
        for file_path in sample_files:
            if os.path.exists(file_path):
                print(f"\n📄 Loading sample documents from {file_path}...")
                self.load_documents(file_path)
                break
        
        if not self.documents_loaded:
            print("\n⚠️  No documents loaded. The chatbot will work without RAG.")
            print("   To enable RAG, create a 'sample.txt' file with your content.")
        
        print("\n💬 Chat started! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Answer: ", end="", flush=True)
                
                if self.pipeline:
                    for chunk in self.pipeline.answer(question):
                        print(chunk, end="", flush=True)
                else:
                    # Simple chat without RAG
                    chat = OpenRouterChat()
                    messages = [{"role": "user", "content": question}]
                    for chunk in chat.stream(messages):
                        print(chunk, end="", flush=True)
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    # Check for required API keys
    missing_keys = []
    
    if not os.getenv("OPENROUTER_API_KEY"):
        missing_keys.append("OPENROUTER_API_KEY")
    
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"export {key}='your-key-here'")
        
        print("\nGet your keys from:")
        print("  • OpenRouter: https://openrouter.ai/keys")
        print("  • OpenAI: https://platform.openai.com/api-keys")
        return
    
    try:
        chatbot = OpenRouterChatbot()
        chatbot.start_chat()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")

if __name__ == "__main__":
    main()