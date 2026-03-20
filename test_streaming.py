"""Test script to verify real-time streaming from /investigate/urls/stream endpoint.

This script calls the streaming endpoint and displays events as they arrive,
with timestamps to verify they're streaming in real-time (not buffered).

USAGE:
    # With S3 URLs (production):
    python test_streaming.py

    # With local files (testing):
    python test_streaming.py --local path/to/doc1.pdf path/to/doc2.pdf
"""

import httpx
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import os


def format_timestamp(ts: Optional[str] = None) -> str:
    """Format timestamp for display."""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            pass
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def create_output_folder() -> Path:
    """Create output folder for test results if it doesn't exist."""
    output_dir = Path("test_streaming_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_output_file_path() -> tuple[Path, Path]:
    """Generate unique output file paths with timestamp (txt and json)."""
    output_dir = create_output_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = output_dir / f"streaming_test_{timestamp}.txt"
    json_path = output_dir / f"streaming_test_{timestamp}.json"
    return txt_path, json_path


async def test_streaming_with_urls():
    """Test the streaming endpoint with S3 URLs."""

    url = "http://localhost:8000/investigate/urls/stream"

    payload = {
        # "query": "What are the main topics discussed in these documents? point out the key points from these documents",
        "query": "2 questiosn -> Summarize the provided documents - and do websearch and caselaw research on 'permissibility of lowest pricing claims for a texas rv'",
        "s3_urls": [
            {
                "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/af/cb/d02e2e7120dfeedb137f4daffd2656b39e92383167fd163e98e2ae03831b",
                "name": "paper1.pdf",
                "mime": "application/pdf"
            },
            # {
            #     "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/6f/dc/30f6d23b37ee4d2c6a1c32aa0af60fa4ca37f45d86a9ae3c4b105286f1e6",
            #     "name": "paper 2.pdf",
            #     "mime": "application/pdf"
            # },
            # {
            #     "url": "https://iqidis-uploads-production.s3.us-east-1.amazonaws.com/uploads/cc697e13-8cb5-411d-b859-caf474a377fb.pdf",
            #     "name": "paper 3.pdf",
            #     "mime": "application/pdf"
            # },
            # {
            #     "mime": "image/png",
            #     "name": "Screenshot 2025-10-07 172516.png",
            #     "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/94/bf/11de44d4856dff0120adcc0c5ba9dbc947329d8e1c6ba4f4ffc913a7276d",
            # },
            # {
            #     "mime": "application/pdf",
            #     "name": "Handwritten Nurse Notes.pdf",
            #     "url": "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/f0/e0/57fa26ce61de53db5980deb2df6455f5965f4d0954ffb7c0cefad432a985",
            # }
        ]
    }
    # payload = {
    #     "query": "Do web research into the legal permissibility of 'lowest pricing' advertising claims for a texas RV ??",
    #     "s3_urls": [
    #     ]
    # }    # payload = {
    #     "query": "What are the main topics discussed in these documents?",
    #     "s3_urls": [
    #         "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/af/cb/d02e2e7120dfeedb137f4daffd2656b39e92383167fd163e98e2ae03831b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAU6GDYBI7XV6KE4UW%2F20260218%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260218T074313Z&X-Amz-Expires=3600&X-Amz-Signature=ab6d8cb0b339783357c6ee8d517e58befa9e542723b2cf425e0fa35344e099eb&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject"
    #     ]
    # }

    await run_streaming_test(url, payload)


async def test_streaming_with_local_files(file_paths: list[str]):
    """Test streaming with local file uploads.

    Note: There's no streaming endpoint for file uploads yet,
    so this uses the synchronous endpoint for now.
    """

    url = "http://localhost:8000/upload/investigate/sync"

    # Validate files exist
    files_to_upload = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            return
        if not path.is_file():
            print(f"ERROR: Not a file: {path}")
            return
        files_to_upload.append(path)

    print(f"Uploading {len(files_to_upload)} files...")
    for f in files_to_upload:
        print(f"  - {f.name}")
    print()

    # Prepare multipart form data
    files = []
    for path in files_to_upload:
        files.append(("files", (path.name, open(path, "rb"), "application/octet-stream")))

    data = {
        "query": "What are the main topics discussed in these documents?"
    }

    await run_upload_test(url, data, files)


async def run_upload_test(url: str, data: dict, files: list):
    """Run test with file upload (non-streaming for now)."""

    print("=" * 80)
    print("FILE UPLOAD TEST - Synchronous Investigation")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Query: {data['query']}")
    print(f"Files: {len(files)}")
    print("=" * 80)
    print()

    start_time = asyncio.get_event_loop().time()

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            print(f"[{format_timestamp()}] Uploading files and starting investigation...")

            response = await client.post(url, data=data, files=files)

            elapsed = asyncio.get_event_loop().time() - start_time

            print(f"[{format_timestamp()}] Response received! ({elapsed:.2f}s)")
            print(f"Status: {response.status_code}\n")

            if response.status_code == 200:
                result = response.json()
                print("=" * 80)
                print("INVESTIGATION COMPLETE")
                print("=" * 80)
                print(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                print(f"Documents processed: {result.get('documents_processed', 0)}")
                print(f"Citations: {len(result.get('citations', []))}")
                print()
                print("ANALYSIS:")
                print("-" * 80)
                print(result.get('analysis', 'No analysis'))
                print("=" * 80)
            else:
                print(f"ERROR: {response.status_code}")
                print(response.text)

    except httpx.ConnectError:
        print("\n[ERROR] Could not connect to server. Is it running on localhost:8000?")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()


async def run_streaming_test(url: str, payload: dict):
    """Run streaming test with SSE endpoint."""

    # Create output files (txt for logs, json for complete response)
    output_file, json_file = get_output_file_path()

    # Store complete response data for JSON output
    response_data = {
        "request": payload,
        "events": [],
        "result": None,
    }

    def log(msg: str):
        """Print to console and write to file."""
        print(msg)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log("=" * 80)
    log("STREAMING TEST - Real-time Event Monitor")
    log("=" * 80)
    log(f"URL: {url}")
    log(f"Query: {payload['query']}")
    log(f"Documents: {len(payload.get('s3_urls', []))}")
    log(f"Output file (txt): {output_file}")
    log(f"Output file (json): {json_file}")
    log("=" * 80)
    log("")

    start_time = asyncio.get_event_loop().time()
    event_count = 0
    last_event_time = start_time

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            log(f"[{format_timestamp()}] Connecting to stream...")

            async with client.stream("POST", url, json=payload) as response:
                log(f"[{format_timestamp()}] Connected! Status: {response.status_code}")

                if response.status_code != 200:
                    log(f"ERROR: {response.status_code}")
                    text = await response.aread()
                    log(text.decode())
                    return

                log(f"[{format_timestamp()}] Waiting for events...\n")

                current_event_type = None

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time
                    gap = current_time - last_event_time
                    last_event_time = current_time

                    # Parse SSE format
                    if line.startswith("event: "):
                        current_event_type = line[7:].strip()
                        event_count += 1

                        # Color-code by event type
                        if current_event_type == "step":
                            icon = "🔄"
                        elif current_event_type == "citation":
                            icon = "📄"
                        elif current_event_type == "fact":
                            icon = "💡"
                        elif current_event_type == "progress":
                            icon = "📊"
                        elif current_event_type == "complete":
                            icon = "✅"
                        elif current_event_type == "error":
                            icon = "❌"
                        else:
                            icon = "❓"

                        log(f"[{elapsed:6.2f}s | gap: {gap:5.2f}s] {icon} Event #{event_count}: {current_event_type.upper()}")

                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])

                            # Store event in response_data
                            current_time_iso = datetime.now().isoformat()
                            response_data["events"].append({
                                "event_type": current_event_type,
                                "event_number": event_count,
                                "elapsed_seconds": round(elapsed, 3),
                                "timestamp": current_time_iso,
                                "data": data,
                            })

                            # Display relevant info based on event type
                            if current_event_type == "step":
                                content = data.get("content", "")[:100]
                                step_type = data.get("step_type", "")
                                log(f"    └─ [{step_type}] {content}")

                            elif current_event_type == "citation":
                                doc = data.get("document", "")
                                page = data.get("page")
                                page_str = f", p.{page}" if page else ""
                                log(f"    └─ {doc}{page_str}")

                            elif current_event_type == "fact":
                                fact = data.get("fact", "")[:100]
                                log(f"    └─ {fact}")

                            elif current_event_type == "progress":
                                status = data.get("status", "")
                                docs = data.get("documents_read", 0)
                                cites = data.get("citations", 0)
                                facts = data.get("facts_accumulated", 0)
                                log(f"    └─ {status} | Docs: {docs}, Citations: {cites}, Facts: {facts}")

                            elif current_event_type == "complete":
                                duration = data.get("duration_seconds", 0)
                                docs = data.get("documents_processed", 0)
                                log(f"    └─ Investigation complete in {duration:.1f}s ({docs} docs)")
                                log(f"\n{'=' * 80}")
                                log(f"FINAL ANALYSIS:")
                                log(f"{'=' * 80}")
                                analysis = data.get("analysis", "")
                                log(analysis)

                                # Store the complete result
                                response_data["result"] = data

                            elif current_event_type == "error":
                                error = data.get("error", "")
                                log(f"    └─ ERROR: {error}")
                                response_data["error"] = error

                        except json.JSONDecodeError as e:
                            log(f"    └─ [JSON Error: {e}]")
                        except Exception as e:
                            log(f"    └─ [Parse Error: {e}]")

                    log("")  # Blank line between events

                total_time = asyncio.get_event_loop().time() - start_time
                log(f"\n{'=' * 80}")
                log(f"Stream ended. Total events: {event_count}, Duration: {total_time:.2f}s")
                log(f"{'=' * 80}")

    except httpx.ReadTimeout:
        log("\n[TIMEOUT] Stream timed out after 600 seconds")
        response_data["error"] = "Timeout after 600 seconds"
    except httpx.ConnectError:
        log("\n[ERROR] Could not connect to server. Is it running on localhost:8000?")
        response_data["error"] = "Could not connect to server"
    except Exception as e:
        log(f"\n[ERROR] {type(e).__name__}: {e}")
        response_data["error"] = f"{type(e).__name__}: {e}"
        import traceback
        # Print traceback to console only
        traceback.print_exc()
        # Write traceback to file
        with open(output_file, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
    finally:
        # Save JSON response file
        response_data["total_events"] = event_count
        response_data["total_duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Output saved to:")
        print(f"   - Log file: {output_file.absolute()}")
        print(f"   - JSON file: {json_file.absolute()}")


if __name__ == "__main__":
    print("\nStarting streaming test...")
    print("Press Ctrl+C to stop\n")

    # Parse command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Local file mode
        if len(sys.argv) < 3:
            print("ERROR: --local requires file paths")
            print("\nUsage:")
            print("  python test_streaming.py --local path/to/doc1.pdf path/to/doc2.pdf")
            sys.exit(1)

        file_paths = sys.argv[2:]
        print(f"Testing with {len(file_paths)} local files\n")

        try:
            asyncio.run(test_streaming_with_local_files(file_paths))
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
    else:
        # S3 URL mode (default)
        print("Testing with S3 URLs (production)\n")
        print("To test with local files, use:")
        print("  python test_streaming.py --local path/to/file1.pdf path/to/file2.pdf\n")

        try:
            asyncio.run(test_streaming_with_urls())
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

