#!/usr/bin/env python3
"""Run the Irys RLM Chat UI.

Multi-turn conversation interface for legal document analysis.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from irys.ui.chat_app import create_chat_app, main

if __name__ == "__main__":
    main()
