#!/usr/bin/env python3
"""
Simple launcher for the demo web chatbot
"""

import os
import sys
import subprocess

def main():
    """Run the demo web chatbot"""
    print("🚀 Starting Demo Web Chatbot...")
    print("📱 Open http://localhost:5001 in your browser")
    print("💡 This is a demo mode - no API keys required")
    print("📝 Test the interface with sample messages")
    print("📁 Try uploading documents to test file upload")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        # Run the demo web chatbot
        subprocess.run([sys.executable, 'demo_web_chatbot.py'])
    except KeyboardInterrupt:
        print("\n👋 Demo chatbot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()