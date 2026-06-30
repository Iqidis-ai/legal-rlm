"""Launch the RLM UI."""
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded environment variables from {env_path}")
    else:
        print(f"[WARN] No .env file found at {env_path}")
        print("   You can create one from .env.example")
except ImportError:
    print("[WARN] python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Falling back to system environment variables...")

# Now import after env is loaded - use multi-turn chat UI with stop button
from irys.ui.chat_app import create_chat_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Irys RLM UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  GEMINI_API_KEY       Required. Your Gemini API key.
  IRYS_STORAGE_MODE    Storage mode: "local" or "s3" (default: local)
  S3_BUCKET            S3 bucket name (required if storage_mode=s3)
  S3_REGION            AWS region (default: us-east-1)
  IRYS_TEMP_DIR        Temp directory path (default: /tmp/irys)

Examples:
  # Run locally with folder browser
  python run_ui.py

  # Run on custom port with all interfaces
  python run_ui.py --port 8080 --server-name 0.0.0.0

  # Run with S3 storage (set IRYS_STORAGE_MODE=s3 in .env)
  IRYS_STORAGE_MODE=s3 python run_ui.py

  # Run with SSL for custom domain
  python run_ui.py --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem

  # Run behind reverse proxy (e.g., nginx)
  python run_ui.py --root-path /irys
        """
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port to run on (default: 7862)",
    )
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Server hostname to bind to (default: 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Path to SSL certificate file for HTTPS",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Path to SSL key file for HTTPS",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="Root path for reverse proxy setups (e.g., /app)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check for API key in environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY environment variable not set!")
        print("\nOptions:")
        print("  1. Create a .env file with: GEMINI_API_KEY=your-key-here")
        print("  2. Or export it: export GEMINI_API_KEY='your-key-here'")
        print("  3. Or copy .env.example to .env and edit it")
        sys.exit(1)

    print(f"[OK] API key loaded: {api_key[:20]}...")

    # Create multi-turn chat app with stop button
    app = create_chat_app(api_key=api_key)

    # Build launch kwargs
    launch_kwargs = {
        "server_port": args.port,
        "server_name": args.server_name,
        "share": args.share,
    }

    # Add SSL if provided
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
        print(f"[INFO] SSL enabled with cert: {args.ssl_certfile}")

    # Add root path for reverse proxy
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path
        print(f"[INFO] Root path: {args.root_path}")

    # Determine URL to display
    protocol = "https" if args.ssl_certfile else "http"
    host_display = "localhost" if args.server_name == "0.0.0.0" else args.server_name
    url = f"{protocol}://{host_display}:{args.port}"
    if args.root_path:
        url += args.root_path

    print(f"\n[START] Launching Irys RLM Multi-Turn Chat UI on {url}")
    print("[MODE] Local mode - multi-turn chat with stop button")

    app.launch(**launch_kwargs)
