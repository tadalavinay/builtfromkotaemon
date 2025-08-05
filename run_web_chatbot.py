#!/usr/bin/env python3
"""
Simple script to run the web-based chatbot interface
"""

import os
import sys
import subprocess

def check_environment():
    """Check if required environment variables are set"""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY environment variable is not set")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return False
    
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("✅ Environment variables are set")
    return True

def install_requirements():
    """Install required packages"""
    try:
        import flask
        print("✅ Flask is already installed")
    except ImportError:
        print("📦 Installing Flask...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("✅ Flask installed successfully")

def main():
    """Main function to run the web chatbot"""
    print("🚀 Starting OpenRouter Web Chatbot...")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\nPlease set the required environment variables and try again.")
        sys.exit(1)
    
    # Install requirements
    install_requirements()
    
    print("\n🌐 Starting web server...")
    print("   The chatbot will be available at: http://localhost:5001")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the web chatbot
    try:
        subprocess.run([sys.executable, "web_chatbot.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()