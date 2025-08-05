#!/usr/bin/env python3
"""
Entry point for OpenRouter chatbot
"""

import os
import sys
from openrouter_chatbot import OpenRouterChatbot

def check_required_keys():
    """Check if required API keys are set"""
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
        return False
    
    return True

def main():
    """Main entry point"""
    print("🚀 Starting OpenRouter Chatbot...")
    
    if not check_required_keys():
        sys.exit(1)
    
    try:
        chatbot = OpenRouterChatbot()
        chatbot.start_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting chatbot: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your API keys are valid")
        print("2. Ensure you have internet connectivity")
        print("3. Verify the selected models are available")
        print("4. Check OpenRouter/OpenAI status")

if __name__ == "__main__":
    main()