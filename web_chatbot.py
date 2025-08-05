#!/usr/bin/env python3
"""
Web-based chatbot interface using Flask
Provides a browser interface for testing the OpenRouter chatbot
"""

import os
import json
import logging
from flask import Flask, render_template, request, jsonify, stream_template
from openrouter_chatbot import OpenRouterQAPipeline, Document, SimpleVectorStore
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for the chatbot
chatbot_pipeline = None
vector_store = None

def initialize_chatbot():
    """Initialize the chatbot with documents"""
    global chatbot_pipeline, vector_store
    
    # Check for required API keys
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for embeddings")
    
    # Initialize vector store with OpenAI embeddings
    from openrouter_chatbot import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = SimpleVectorStore(embeddings=embeddings)
    
    # Load sample documents if they exist
    documents = []
    sample_files = ["sample.txt", "README.md", "requirements.txt"]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        content=content,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
                    logger.info(f"Loaded document: {file_path}")
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
    
    # Add some default content if no documents found
    if not documents:
        default_content = """
        This is a sample document for the chatbot.
        The chatbot can answer questions about this content.
        It uses OpenRouter API for chat completions and OpenAI API for embeddings.
        """
        documents.append(Document(content=default_content, metadata={"source": "default"}))
    
    # Initialize the pipeline
    chatbot_pipeline = OpenRouterQAPipeline()
    
    # Try to create retriever and index documents
    try:
        from openrouter_chatbot import BaseRetriever
        retriever = BaseRetriever(vector_store)
        chatbot_pipeline.set_retriever(retriever)
        
        # Index documents
        if documents:
            vector_store.add_documents(documents)
            logger.info(f"Indexed {len(documents)} documents")
    except Exception as e:
        logger.warning(f"Could not initialize RAG features: {e}")
        logger.info("Chatbot will work without document retrieval")
        # Pipeline will work without retriever for general chat
    
    return True

@app.route('/')
def home():
    """Home page with chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if not chatbot_pipeline:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        # Get response from chatbot
        response_parts = []
        for chunk in chatbot_pipeline.answer(message):
            response_parts.append(chunk)
        
        response = ''.join(response_parts)
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat messages"""
    def generate():
        try:
            data = request.get_json()
            message = data.get('message', '')
            
            if not message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return
            
            if not chatbot_pipeline:
                yield f"data: {json.dumps({'error': 'Chatbot not initialized'})}\n\n"
                return
            
            # Stream response from chatbot
            for chunk in chatbot_pipeline.answer(message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        except Exception as e:
            logger.error(f"Error in chat_stream endpoint: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and index new documents"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        
        # Create document
        doc = Document(
            content=content,
            metadata={"source": file.filename}
        )
        
        # Index document
        if vector_store:
            try:
                vector_store.add_documents([doc])
                logger.info(f"Uploaded and indexed document: {file.filename}")
            except Exception as e:
                logger.warning(f"Could not index uploaded document: {e}")
        
        return jsonify({'message': f'Document {file.filename} uploaded successfully'})
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'chatbot_initialized': chatbot_pipeline is not None
    })

if __name__ == '__main__':
    try:
        # Initialize chatbot
        initialize_chatbot()
        logger.info("Chatbot initialized successfully")
        
        # Run the Flask app
        port = int(os.getenv('FLASK_PORT', 5001))
        app.run(debug=True, host='0.0.0.0', port=port)
    
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print(f"Error: {e}")
        print("Please ensure you have set the following environment variables:")
        print("  - OPENROUTER_API_KEY")
        print("  - OPENAI_API_KEY")