#!/usr/bin/env python3
"""
Start the Knowledge Graph API server for Next.js frontend integration.

This server runs in "Iqidis mode" where matter_id is sent per request,
allowing your Next.js frontend to work with multiple matters dynamically.

Usage:
    python3 start_server.py
    python3 start_server.py --port 8000
    python3 start_server.py --host 0.0.0.0 --port 8000
"""
from visualization_server import create_visualization_app
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Start the Knowledge Graph API Server for Next.js Frontend'
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
        help='Host to bind to (default: 0.0.0.0 - accessible from other machines)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Run in debug mode (default: True)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Knowledge Graph API Server - Next.js Integration Mode")
    print("=" * 70)
    print(f"Mode:     Iqidis (matter_id sent per request)")
    print(f"Host:     {args.host}")
    print(f"Port:     {args.port}")
    print(f"URL:      http://localhost:{args.port}")
    print(f"CORS:     Enabled for all origins")
    print("=" * 70)
    print("\n📡 Available API Endpoints:")
    print("  • GET  /api/matters                    - List all matters")
    print("  • POST /api/extract-from-iqidis        - Extract KG from documents")
    print("  • GET  /api/graph?matter_id=X          - Get graph data")
    print("  • GET  /api/stats?matter_id=X          - Get statistics")
    print("  • POST /api/query                      - Natural language query")
    print("  • GET  /api/search?q=term&matter_id=X  - Search entities")
    print("  • GET  /api/timeline?matter_id=X       - Get timeline")
    print("  • GET  /api/analytics?matter_id=X      - Graph analytics")
    print("  • GET  /api/schema?matter_id=X         - Get graph schema")
    print("  • ... and 30+ more endpoints")
    print("\n💡 Usage from Next.js:")
    print("  - Send matter_id as query param: ?matter_id=123")
    print("  - Or use X-Matter-Id header")
    print("=" * 70)
    print("\n🚀 Starting server...\n")

    # Create app in Iqidis mode (no default matter)
    app = create_visualization_app(matter_name=None)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
