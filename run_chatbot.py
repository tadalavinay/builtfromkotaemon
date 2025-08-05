#!/usr/bin/env python3
"""
Entry point for the simple chatbot application
"""

import os
import sys
from pathlib import Path

# Ensure we're in the correct directory
os.chdir(Path(__file__).parent)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is not set")
    print("Please set it using: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Import and run the chatbot
try:
    from simple_chatbot import main
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)