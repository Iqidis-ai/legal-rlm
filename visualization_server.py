#!/usr/bin/env python3
"""
Visualization server for the Knowledge Graph.

This is a thin wrapper that:
1. Serves static visualization files (HTML/JS)
2. Delegates all API calls to src/api/server

Usage:
    python visualization_server.py
    python visualization_server.py --matter my_case --port 8080
"""
from src.core.config import GEMINI_API_KEY
from src.api.server import api, init_matter
from flask_cors import CORS
from flask import Flask, send_from_directory
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def create_visualization_app(matter_name: Optional[str] = None, api_key: str = GEMINI_API_KEY) -> Flask:
    """
    Create the visualization server application.

    Args:
        matter_name: Name of the matter to load
        api_key: Gemini API key

    Returns:
        Configured Flask application with static file serving and API
    """
    app = Flask(__name__, static_folder='visualization')
    CORS(app, origins=['*'], allow_headers=['Content-Type', 'X-Matter-Id'])

    # Initialize the API for this matter (skip if None - for Iqidis mode)
    if matter_name:
        init_matter(matter_name, api_key)

    # Register API blueprint
    app.register_blueprint(api)

    # Static file routes
    @app.route('/')
    def index():
        return send_from_directory('visualization', 'index.html')

    @app.route('/<path:path>')
    def static_files(path):
        return send_from_directory('visualization', path)

    return app


def main():
    parser = argparse.ArgumentParser(
        description='Start the Knowledge Graph Visualization Server'
    )
    parser.add_argument(
        '--matter', '-m',
        default=None,
        help='Matter name. Omit for Iqidis mode (matter_id sent per request)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to run on (default: 8000)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Run in debug mode (default: True)'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (default: from environment)'
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Knowledge Graph Visualization Server")
    print("=" * 50)
    if args.matter:
        db_path = Path(__file__).parent / "matters" / args.matter / "graph.db"
        print(f"Matter:   {args.matter}")
        print(f"Database: {db_path}")
    else:
        print("Mode:     Iqidis (matter_id per request)")
        print("API:      /api/matters, /api/extract-from-iqidis, /api/graph?matter_id=...")
    print(f"URL:      http://localhost:{args.port}")
    print("=" * 50)

    api_key = args.api_key or GEMINI_API_KEY
    app = create_visualization_app(args.matter, api_key)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
