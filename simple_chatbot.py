"""
Simple Chatbot Application using FullQAPipeline
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from datetime import datetime

# Add the libs directory to the path
sys.path.insert(0, str(Path(__file__).parent / "libs"))

from ktem.ktem.reasoning.simple import FullQAPipeline
from ktem.ktem.index.file.index import VectorIndexing
from ktem.ktem.retrievers.dense import VectorRetriever
from ktem.ktem.embeddings.openai import OpenAIEmbeddings
from ktem.ktem.llms.openai import OpenAIChat

from config import (
    OPENAI_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    VECTOR_INDEX_DIR,
    MAX_CONTEXT_LENGTH,
    MAX_INTERACTIONS,
    TRIGGER_CONTEXT_LENGTH,
    QA_PROMPT_TEMPLATE,
    HIGHLIGHT_CITATION,
    ENABLE_MINDMAP,
    ENABLE_CITATION_VIZ,
    USE_MULTIMODAL,
    SUPPORTED_LANGUAGE
)
from document_loader import DocumentLoader

class SimpleChatbot:
    """Simple chatbot using FullQAPipeline"""
    
    def __init__(self, documents_dir: str = "documents"):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
        
        self.document_loader = DocumentLoader(self.documents_dir)
        self.index = None
        self.retriever = None
        self.pipeline = None
        
        # Initialize components
        self.setup_embeddings()
        self.setup_retriever()
        self.setup_pipeline()
        
        # Conversation history
        self.conversation_history = []
    
    def setup_embeddings(self):
        """Initialize OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    
    def setup_retriever(self):
        """Setup vector retriever with document indexing"""
        # Create vector index
        self.index = VectorIndexing(
            embedding=self.embeddings,
            index_path=str(VECTOR_INDEX_DIR)
        )
        
        # Create retriever
        self.retriever = VectorRetriever(
            index=self.index,
            top_k=5,
            reranker=None
        )
    
    def setup_pipeline(self):
        """Initialize the FullQAPipeline"""
        # Create LLM
        llm = OpenAIChat(
            model_name=DEFAULT_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create pipeline
        self.pipeline = FullQAPipeline(
            retriever=self.retriever,
            llm=llm,
            qa_prompt_template=QA_PROMPT_TEMPLATE,
            citation_template=None,
            max_context_length=MAX_CONTEXT_LENGTH,
            max_interactions=MAX_INTERACTIONS,
            trigger_context_length=TRIGGER_CONTEXT_LENGTH,
            highlight_citation=HIGHLIGHT_CITATION,
            enable_mindmap=ENABLE_MINDMAP,
            enable_citation_viz=ENABLE_CITATION_VIZ,
            use_multimodal=USE_MULTIMODAL,
            lang=SUPPORTED_LANGUAGE
        )
    
    def index_documents(self, documents: List[Dict[str, Any]] = None):
        """Index documents for retrieval"""
        if documents is None:
            # Load existing documents
            documents = self.document_loader.load_documents()
        
        if not documents:
            print("No documents found. Creating sample documents...")
            documents = self.document_loader.create_sample_documents()
        
        # Convert documents to DataFrame format expected by the index
        df_data = []
        for doc in documents:
            df_data.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                **doc.get('metadata', {})
            })
        
        df = pd.DataFrame(df_data)
        
        # Index the documents
        print(f"Indexing {len(df)} documents...")
        self.index.build_index(df)
        print("Documents indexed successfully!")
    
    def chat(self, query: str, stream: bool = True) -> Dict[str, Any]:
        """Chat with the bot"""
        try:
            # Prepare input
            input_data = {
                "question": query,
                "history": self.conversation_history[-MAX_INTERACTIONS:] if self.conversation_history else []
            }
            
            # Run pipeline
            if stream:
                # For now, we'll use non-streaming mode
                # Streaming would require more complex setup
                result = self.pipeline(input_data)
            else:
                result = self.pipeline(input_data)
            
            # Extract response
            response = {
                "answer": result.get("answer", ""),
                "evidences": result.get("evidences", []),
                "citations": result.get("citations", []),
                "metadata": result.get("metadata", {})
            }
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response["answer"]
            })
            
            return response
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "evidences": [],
                "citations": [],
                "metadata": {"error": str(e)}
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about indexed documents"""
        try:
            documents = self.document_loader.list_documents()
            return {
                "total_documents": len(documents),
                "documents": documents
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "documents": [],
                "error": str(e)
            }

def main():
    """Main function to run the chatbot"""
    print("🤖 Simple Chatbot using FullQAPipeline")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    # Index documents
    print("Setting up document index...")
    chatbot.index_documents()
    
    # Get document info
    doc_info = chatbot.get_document_info()
    print(f"\n📚 Loaded {doc_info['total_documents']} documents:")
    for doc in doc_info['documents']:
        print(f"   - {doc['title']} ({doc['category']})")
    
    print("\n" + "=" * 50)
    print("Chatbot is ready! Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if query.lower() == 'clear':
                chatbot.clear_history()
                print("Conversation history cleared!")
                continue
            
            if not query:
                continue
            
            # Get response
            print("\nBot: ", end="")
            response = chatbot.chat(query)
            
            print(response["answer"])
            
            # Show citations if available
            if response["citations"]:
                print("\n📖 Sources:")
                for citation in response["citations"]:
                    print(f"   - {citation}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()