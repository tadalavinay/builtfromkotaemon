#!/usr/bin/env python3
"""
Demo web chatbot that works without API keys for testing the interface
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import time
from werkzeug.utils import secure_filename
import threading

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Demo responses for testing
DEMO_RESPONSES = {
    "hello": "Hello! This is a demo chatbot. I'm working without API keys to test the interface.",
    "hi": "Hi there! Welcome to the demo chatbot interface.",
    "how are you": "I'm doing great! This is just a demo, but the real chatbot would use OpenRouter and OpenAI APIs.",
    "what can you do": "In demo mode, I can respond to basic questions and show you how the interface works. The real version can chat with documents and answer questions using RAG.",
    "upload": "You can upload documents using the upload button. In demo mode, I won't process them, but the interface will show the upload functionality.",
    "help": "Try asking me questions or upload a document to test the interface. The real chatbot would use your OpenRouter API key for chat and OpenAI API key for embeddings.",
    "bye": "Goodbye! Thanks for testing the demo chatbot interface."
}

def get_demo_response(message):
    """Generate a demo response based on the message"""
    message_lower = message.lower()
    
    for key in DEMO_RESPONSES:
        if key in message_lower:
            return DEMO_RESPONSES[key]
    
    return f"You asked: '{message}'. This is a demo response. In the real chatbot, this would be processed using OpenRouter API for chat and OpenAI API for embeddings."

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages in demo mode"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Simulate processing delay
        time.sleep(0.5)
        
        response = get_demo_response(message)
        
        return jsonify({
            'response': response,
            'status': 'success',
            'demo_mode': True
        })
        
    except Exception as e:
        return jsonify({
            'response': f'Demo error: {str(e)}',
            'status': 'error',
            'demo_mode': True
        })

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat in demo mode"""
    def generate():
        try:
            data = request.get_json()
            message = data.get('message', '')
            
            response = get_demo_response(message)
            
            # Simulate streaming by sending chunks
            words = response.split()
            for i, word in enumerate(words):
                chunk = {
                    'content': word + (' ' if i < len(words) - 1 else ''),
                    'done': i == len(words) - 1
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.1)
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return app.response_class(generate(), mimetype='text/plain')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads in demo mode"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'message': f'File "{filename}" uploaded successfully (demo mode - not processed)',
                'filename': filename,
                'demo_mode': True
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'demo_mode': True,
        'message': 'Demo chatbot is running'
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🚀 Starting Demo Web Chatbot...")
    print("📱 Open http://localhost:5001 in your browser")
    print("💡 This is a demo mode - no API keys required")
    print("📝 Test the interface with sample messages")
    print("📁 Try uploading documents to test file upload")
    print("🛑 Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5001)