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


def format_timestamp(ts: Optional[str] = None) -> str:
    """Format timestamp for display."""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            pass
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def save_fixture(query: str, s3_urls: list, events: list, output_path: str = "tests/fixtures/rlm_streaming_events_captured.json"):
    """Save captured streaming events to a JSON fixture file.
    
    Args:
        query: The query string used
        s3_urls: List of S3 URLs (or empty list for local mode)
        events: List of captured events in format {"event": "<type>", "data": {...}}
        output_path: Path to save the fixture file
    """
    from datetime import timezone
    from collections import defaultdict
    
    # Count events by type
    event_counts = defaultdict(int)
    for event in events:
        event_counts[event["event"]] += 1
    
    # Build fixture structure
    fixture = {
        "query": query,
        "s3_urls": s3_urls,
        "events": events,
        "event_counts": dict(event_counts),
        "total_events": len(events),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Ensure directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    
    return output_file, event_counts


async def test_streaming_with_urls(output_fixture: Optional[str] = None):
    """Test the streaming endpoint with S3 URLs."""

    url = "http://localhost:8000/investigate/urls/stream"

    payload = {
        "query": "Reconstruct the complete argumentative structure used by the authors to justify HRM as a solution to the computational depth limitations of standard Transformers and CoT-based models. Integrate the theoretical critique of shallow architectures, the hierarchical convergence mechanism, the one-step gradient approximation and its DEQ grounding, empirical benchmark results across ARC, Sudoku, and Maze, and the neuroscientific dimensionality hierarchy analogy. Conclude by evaluating whether the empirical evidence adequately supports the theoretical claims.",
        "s3_urls": [
            "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/af/cb/d02e2e7120dfeedb137f4daffd2656b39e92383167fd163e98e2ae03831b",
            "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/6f/dc/30f6d23b37ee4d2c6a1c32aa0af60fa4ca37f45d86a9ae3c4b105286f1e6"
        ]
    }
    # payload = {
    #     "query": "What are the main topics discussed in these documents?",
    #     "s3_urls": [
    #         "https://iqidis-artifact.s3.us-east-1.amazonaws.com/default/preview/af/cb/d02e2e7120dfeedb137f4daffd2656b39e92383167fd163e98e2ae03831b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAU6GDYBI7XV6KE4UW%2F20260218%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260218T074313Z&X-Amz-Expires=3600&X-Amz-Signature=ab6d8cb0b339783357c6ee8d517e58befa9e542723b2cf425e0fa35344e099eb&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject"
    #     ]
    # }

    await run_streaming_test(url, payload, output_fixture)


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


async def run_streaming_test(url: str, payload: dict, output_fixture: Optional[str] = None):
    """Run streaming test with SSE endpoint and optionally save events to fixture.

    Args:
        url: The streaming endpoint URL
        payload: The request payload
        output_fixture: Optional path to save captured events as JSON fixture
    """

    print("=" * 80)
    print("STREAMING TEST - Real-time Event Monitor")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Query: {payload['query']}")
    print(f"Documents: {len(payload.get('s3_urls', []))}")
    if output_fixture:
        print(f"Capturing events to: {output_fixture}")
    print("=" * 80)
    print()

    start_time = asyncio.get_event_loop().time()
    event_count = 0
    last_event_time = start_time

    # Event capture for fixture
    captured_events = []

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            print(f"[{format_timestamp()}] Connecting to stream...")

            async with client.stream("POST", url, json=payload) as response:
                print(f"[{format_timestamp()}] Connected! Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"ERROR: {response.status_code}")
                    text = await response.aread()
                    print(text.decode())
                    return
                
                print(f"[{format_timestamp()}] Waiting for events...\n")
                
                current_event_type = None
                current_event_data = None
                
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
                        
                        print(f"[{elapsed:6.2f}s | gap: {gap:5.2f}s] {icon} Event #{event_count}: {current_event_type.upper()}")
                    
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            current_event_data = data
                            
                            # Capture event for fixture
                            if output_fixture and current_event_type:
                                captured_events.append({
                                    "event": current_event_type,
                                    "data": data
                                })
                            
                            # Display relevant info based on event type
                            if current_event_type == "step":
                                content = data.get("content", "")[:100]
                                step_type = data.get("step_type", "")
                                print(f"    └─ [{step_type}] {content}")
                            
                            elif current_event_type == "citation":
                                doc = data.get("document", "")
                                page = data.get("page")
                                page_str = f", p.{page}" if page else ""
                                print(f"    └─ {doc}{page_str}")
                            
                            elif current_event_type == "fact":
                                fact = data.get("fact", "")[:100]
                                print(f"    └─ {fact}")
                            
                            elif current_event_type == "progress":
                                status = data.get("status", "")
                                docs = data.get("documents_read", 0)
                                cites = data.get("citations", 0)
                                facts = data.get("facts_accumulated", 0)
                                print(f"    └─ {status} | Docs: {docs}, Citations: {cites}, Facts: {facts}")
                            
                            elif current_event_type == "complete":
                                duration = data.get("duration_seconds", 0)
                                docs = data.get("documents_processed", 0)
                                print(f"    └─ Investigation complete in {duration:.1f}s ({docs} docs)")
                                print(f"\n{'=' * 80}")
                                print(f"FINAL ANALYSIS:")
                                print(f"{'=' * 80}")
                                analysis = data.get("analysis", "")
                                print(analysis[:500] + ("..." if len(analysis) > 500 else ""))
                            
                            elif current_event_type == "error":
                                error = data.get("error", "")
                                print(f"    └─ ERROR: {error}")
                        
                        except json.JSONDecodeError as e:
                            print(f"    └─ [JSON Error: {e}]")
                        except Exception as e:
                            print(f"    └─ [Parse Error: {e}]")
                    
                    print()  # Blank line between events
                
                total_time = asyncio.get_event_loop().time() - start_time
                print(f"\n{'=' * 80}")
                print(f"Stream ended. Total events: {event_count}, Duration: {total_time:.2f}s")
                print(f"{'=' * 80}")
                
                # Save fixture if requested
                if output_fixture and captured_events:
                    print(f"\n{'=' * 80}")
                    print("SAVING FIXTURE")
                    print(f"{'=' * 80}")
                    output_file, event_counts = save_fixture(
                        query=payload['query'],
                        s3_urls=payload.get('s3_urls', []),
                        events=captured_events,
                        output_path=output_fixture
                    )
                    print(f"✅ Fixture saved to: {output_file}")
                    print(f"\nEvent counts:")
                    for event_type, count in sorted(event_counts.items()):
                        print(f"  {event_type}: {count}")
                    print(f"\nTotal events captured: {len(captured_events)}")
                    print(f"{'=' * 80}")
    
    except httpx.ReadTimeout:
        print("\n[TIMEOUT] Stream timed out after 600 seconds")
    except httpx.ConnectError:
        print("\n[ERROR] Could not connect to server. Is it running on localhost:8000?")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting streaming test...")
    print("Press Ctrl+C to stop\n")

    # Parse command-line arguments
    output_fixture = None
    args = sys.argv[1:]
    
    # Check for --output flag
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_fixture = args[idx + 1]
            args.pop(idx)  # Remove --output
            args.pop(idx)  # Remove the path
        else:
            print("ERROR: --output requires a file path")
            sys.exit(1)
    
    # Default output path if not specified
    if output_fixture is None:
        output_fixture = "tests/fixtures/rlm_streaming_events_captured.json"
    
    if len(args) > 0 and args[0] == "--local":
        # Local file mode
        if len(args) < 2:
            print("ERROR: --local requires file paths")
            print("\nUsage:")
            print("  python test_streaming.py --local path/to/doc1.pdf path/to/doc2.pdf")
            print("  python test_streaming.py --local path/to/doc1.pdf --output my_fixture.json")
            sys.exit(1)

        file_paths = args[1:]
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
        print(f"Events will be captured to: {output_fixture}\n")

        try:
            asyncio.run(test_streaming_with_urls(output_fixture))
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
