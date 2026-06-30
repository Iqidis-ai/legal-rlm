#!/usr/bin/env python3
"""
Simple GitHub webhook listener for auto-deployment.
Listens for push events and triggers deploy.sh
"""

import hashlib
import hmac
import os
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
DEPLOY_SCRIPT = "/home/ubuntu/legal-rlm/scripts/deploy.sh"
PORT = 9000


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/webhook":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(content_length)

        # Verify signature if secret is set
        if WEBHOOK_SECRET:
            signature = self.headers.get("X-Hub-Signature-256", "")
            expected = "sha256=" + hmac.new(
                WEBHOOK_SECRET.encode(), payload, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                print("Invalid signature!")
                self.send_response(403)
                self.end_headers()
                return

        # Check if it's a push event
        event = self.headers.get("X-GitHub-Event", "")
        if event == "push":
            try:
                data = json.loads(payload)
                branch = data.get("ref", "").replace("refs/heads/", "")
                print(f"Push to {branch} - triggering deployment...")

                # Run deploy script asynchronously
                subprocess.Popen(
                    ["bash", DEPLOY_SCRIPT],
                    stdout=open("/var/log/irys-deploy.log", "a"),
                    stderr=subprocess.STDOUT,
                )

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Deployment triggered")
                return
            except Exception as e:
                print(f"Error: {e}")
                self.send_response(500)
                self.end_headers()
                return

        # For ping or other events
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    print(f"Starting webhook listener on port {PORT}...")
    server = HTTPServer(("0.0.0.0", PORT), WebhookHandler)
    server.serve_forever()
