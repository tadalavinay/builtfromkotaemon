#!/usr/bin/env python3
"""
Test script for OpenRouter chatbot functionality
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock
from openrouter_chatbot import OpenRouterChatbot, OpenRouterChat, OpenAIEmbeddings, Document

def test_openrouter_config():
    """Test OpenRouter configuration"""
    print("🔧 Testing OpenRouter configuration...")
    
    try:
        from openrouter_config import OpenRouterConfig
        
        # Test model info
        model_info = OpenRouterConfig.get_model_info()
        assert "chat_models" in model_info
        assert "embedding_models" in model_info
        print("✅ Configuration loaded successfully")
        
        # Test model validation
        validated = OpenRouterConfig.validate_model("gpt-3.5-turbo")
        assert validated == "openai/gpt-3.5-turbo"
        print("✅ Model validation working")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True

def test_embeddings_mock():
    """Test embeddings with mock responses"""
    print("🔍 Testing embeddings with mock data...")
    
    try:
        # Mock the requests.post for embeddings
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
                    {"embedding": [0.2, 0.3, 0.4, 0.5, 0.6]}
                ]
            }
            mock_post.return_value = mock_response
            
            embeddings = OpenAIEmbeddings(api_key="test-key")
            result = embeddings.create_embeddings(["test text 1", "test text 2"])
            
            assert len(result) == 2
            assert len(result[0]) == 5
            print("✅ Embeddings mock test passed")
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False
    
    return True

def test_chat_mock():
    """Test chat completion with mock responses"""
    print("💬 Testing chat completion with mock data...")
    
    try:
        # Mock the requests.post for chat
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "This is a mock response"
                    }
                }]
            }
            mock_post.return_value = mock_response
            
            chat = OpenRouterChat(api_key="test-key")
            messages = [{"role": "user", "content": "Hello"}]
            response = chat.create_completion(messages)
            
            assert response == "This is a mock response"
            print("✅ Chat completion mock test passed")
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False
    
    return True

def test_vector_store():
    """Test vector store functionality"""
    print("📊 Testing vector store...")
    
    try:
        # Create test documents
        docs = [
            Document(id="1", title="Test Doc 1", content="This is about machine learning", metadata={}),
            Document(id="2", title="Test Doc 2", content="This is about Python programming", metadata={})
        ]
        
        # Mock embeddings
        with patch('openrouter_chatbot.OpenAIEmbeddings.create_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            
            from openrouter_chatbot import SimpleVectorStore
            store = SimpleVectorStore()
            store.add_documents(docs)
            
            # Mock query embedding
            with patch('openrouter_chatbot.OpenAIEmbeddings.create_embeddings') as mock_query:
                mock_query.return_value = [[0.15, 0.25, 0.35]]
                
                results = store.search("machine learning", top_k=1)
                assert len(results) > 0
                assert results[0]["document"].title == "Test Doc 1"
                print("✅ Vector store test passed")
                
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False
    
    return True

def test_qa_pipeline():
    """Test QA pipeline with mock data"""
    print("🤖 Testing QA pipeline...")
    
    try:
        # Create test documents
        docs = [
            Document(id="1", title="Python Guide", content="Python is a high-level programming language", metadata={})
        ]
        
        # Mock all API calls
        with patch('openrouter_chatbot.OpenAIEmbeddings.create_embeddings') as mock_embed, \
             patch('openrouter_chatbot.OpenRouterChat.create_completion') as mock_chat:
            
            mock_embed.return_value = [[0.1, 0.2, 0.3]]
            mock_chat.return_value = "Python is a high-level programming language known for its simplicity."
            
            from openrouter_chatbot import OpenRouterQAPipeline
            pipeline = OpenRouterQAPipeline()
            pipeline.add_documents(docs)
            
            result = pipeline.process_question("What is Python?")
            
            assert "answer" in result
            assert "citations" in result
            assert result["context_used"] == True
            print("✅ QA pipeline test passed")
            
    except Exception as e:
        print(f"❌ QA pipeline test failed: {e}")
        return False
    
    return True

def test_full_chatbot():
    """Test complete chatbot initialization"""
    print("🎯 Testing full chatbot...")
    
    try:
        # Mock all API calls
        with patch('openrouter_chatbot.OpenAIEmbeddings.create_embeddings') as mock_embed, \
             patch('openrouter_chatbot.OpenRouterChat.create_completion') as mock_chat:
            
            mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_chat.return_value = "This is a test response"
            
            chatbot = OpenRouterChatbot()
            assert len(chatbot.pipeline.vector_store.documents) > 0
            print("✅ Full chatbot test passed")
            
    except Exception as e:
        print(f"❌ Full chatbot test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🧪 Running OpenRouter Chatbot Tests...\n")
    
    tests = [
        test_openrouter_config,
        test_embeddings_mock,
        test_chat_mock,
        test_vector_store,
        test_qa_pipeline,
        test_full_chatbot
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your OpenRouter chatbot is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    # Check if running in CI/CD or testing environment
    if os.getenv("CI") or os.getenv("TESTING"):
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Interactive test mode
        print("🔍 OpenRouter Chatbot Test Suite")
        print("=" * 40)
        
        # Check API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            print("✅ OpenRouter API key found")
            # Skip interactive prompt for mock tests
            if len(sys.argv) > 1 and sys.argv[1] == "--mock-only":
                print("Running mock tests...")
                run_all_tests()
            else:
                choice = input("\nRun live tests with real API? (y/N): ").strip().lower()
                if choice == 'y':
                    print("⚠️  Live tests will use API credits")
                    # TODO: Add live API tests
                else:
                    print("Running mock tests...")
                    run_all_tests()
        else:
            print("❌ OpenRouter API key not found - running mock tests")
            run_all_tests()