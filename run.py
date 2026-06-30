#!/usr/bin/env python3
"""Run the Irys RLM application."""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from irys.ui.app import create_app

if __name__ == "__main__":
    # Validate API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable required")
        print("Set it with: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    app = create_app()
    app.launch(server_port=7860)
