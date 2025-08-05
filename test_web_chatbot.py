#!/usr/bin/env python3
"""
Simple test for web chatbot without requiring API keys
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Mock the OpenAI and OpenRouter clients to avoid API calls
def test_web_chatbot_structure():
    """Test that the web chatbot can be imported and basic structure works"""
    
    # Mock environment variables
    with patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-key'
    }):
        # Mock the OpenAI client
        with patch('openrouter_chatbot.openai_client') as mock_openai:
            mock_openai.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
            )
            
            # Mock the OpenRouter client
            with patch('openrouter_chatbot.openrouter_client') as mock_openrouter:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].delta.content = "Test response"
                mock_openrouter.chat.completions.create.return_value = [mock_response]
                
                # Import and test basic functionality
                from web_chatbot import app
                
                # Test that Flask app is created
                assert app is not None
                print("✅ Flask app created successfully")
                
                # Test routes exist
                routes = [rule.rule for rule in app.url_map.iter_rules()]
                expected_routes = ['/', '/chat', '/chat_stream', '/upload', '/health']
                
                for route in expected_routes:
                    assert route in routes, f"Route {route} not found"
                print("✅ All expected routes exist")
                
                print("✅ Web chatbot structure test passed!")

if __name__ == "__main__":
    test_web_chatbot_structure()