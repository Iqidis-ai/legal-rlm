#!/usr/bin/env python3
"""Run Irys RLM API server.

Usage:
    python run_server.py             # Run with defaults
    python run_server.py --port 8080 # Custom port
    python run_server.py --reload    # Dev mode with auto-reload
    python run_server.py --workers 4 # Production with multiple workers
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
except ImportError:
    pass


def setup_logging(level: str = "INFO", format: str = "text"):
    """Configure logging."""
    if format == "json":
        import json

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.root.handlers = [handler]
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    logging.getLogger().setLevel(level)


def main():
    parser = argparse.ArgumentParser(description="Run Irys RLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate environment
    api_key = os.environ.get("GEMINI_API_KEY")
    s3_bucket = os.environ.get("S3_BUCKET")

    print("\n" + "=" * 60)
    print("IRYS RLM API SERVER")
    print("=" * 60)
    print(f"  Host:       {args.host}")
    print(f"  Port:       {args.port}")
    print(f"  Workers:    {args.workers}")
    print(f"  Reload:     {args.reload}")
    print(f"  Log Level:  {args.log_level}")
    print()
    print(f"  Gemini API: {'✓ configured' if api_key else '✗ NOT SET'}")
    print(f"  S3 Bucket:  {s3_bucket or '✗ NOT SET'}")
    print("=" * 60)
    print()

    if not api_key:
        print("⚠️  Warning: GEMINI_API_KEY not set")
        print("   Set it in .env file or environment")

    # Run with uvicorn
    import uvicorn

    uvicorn.run(
        "irys.service.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
