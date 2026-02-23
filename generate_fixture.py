#!/usr/bin/env python3
"""
Generate RLM fixture response for testing.

This script calls the local RLM service with a sample query and document,
then saves the response as a reusable fixture file in multiple formats.

Usage:
    python scripts/generate_fixture.py              # Generate sync response
    python scripts/generate_fixture.py --streaming  # Generate streaming events

Arguments:
    --streaming    Generate streaming API output (SSE events from /investigate/urls/stream)

Requirements:
    - RLM service running on http://localhost:8000
    - cw-procedure-manual.pdf in the project root
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx


# Configuration
SERVICE_URL = "http://localhost:8000"
SAMPLE_QUERY = "Through which transmission channels could global growth downgrades affect India’s financial stability, and how strong are the existing buffers along those channels?"
SAMPLE_DOCUMENT = r"D:\legal-rlm\rbi-fsr.PDF"
FIXTURE_PATH = "tests/fixtures/rlm_sample_response.json"
STREAMING_FIXTURE_PATH = "tests/fixtures/rlm_streaming_events.json"


async def generate_fixture():
    """Call RLM service and save response as fixture."""
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    doc_path = project_root / SAMPLE_DOCUMENT
    fixture_path = project_root / FIXTURE_PATH
    
    # Validate document exists
    if not doc_path.exists():
        print(f"❌ Error: Document not found: {doc_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("RLM FIXTURE GENERATOR")
    print("=" * 60)
    print(f"Service URL:  {SERVICE_URL}")
    print(f"Query:        {SAMPLE_QUERY}")
    print(f"Document:     {doc_path.name}")
    print(f"Fixture Path: {fixture_path}")
    print("=" * 60)
    print()
    
    # Check service health
    print("1. Checking service health...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URL}/health")
            response.raise_for_status()
            health = response.json()
            print(f"   ✓ Service status: {health['status']}")
            print(f"   ✓ Version: {health['version']}")
        except Exception as e:
            print(f"   ❌ Service not available: {e}")
            print(f"\n   Start the service with: python run_server.py")
            sys.exit(1)
    
    # Upload document and run investigation (synchronous endpoint)
    print("\n2. Uploading document and running investigation...")
    print(f"   (This may take 30-120 seconds...)")
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            with open(doc_path, "rb") as f:
                files = {"files": (doc_path.name, f, "application/pdf")}
                data = {
                    "query": SAMPLE_QUERY,
                    "keep_files": "false",
                }
                
                response = await client.post(
                    f"{SERVICE_URL}/upload/investigate/sync",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()
            
            print(f"   ✓ Investigation completed")
            print(f"   ✓ Duration: {result['duration_seconds']:.1f}s")
            print(f"   ✓ Documents processed: {result['documents_processed']}")
            print(f"   ✓ Citations: {len(result.get('citations', []))}")
            
        except httpx.TimeoutException:
            print(f"   ❌ Request timed out (investigation took too long)")
            sys.exit(1)
        except Exception as e:
            print(f"   ❌ Investigation failed: {e}")
            sys.exit(1)
    
    # Save fixture
    print("\n3. Saving fixture...")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"   ✓ Fixture saved to: {fixture_path}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ FIXTURE GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nAnalysis preview:")
    print(f"{result['analysis'][:200]}...")
    print(f"\nTo regenerate: python scripts/generate_fixture.py")
    print("=" * 60)


async def generate_streaming_fixture():
    """Call RLM streaming service and save SSE events as fixture.

    This function handles both S3 and local storage modes:
    - S3 mode: Uploads file, gets S3 URL, uses streaming endpoint
    - Local mode: Falls back to sync endpoint, simulates streaming events from result
    """

    # Resolve paths
    project_root = Path(__file__).parent.parent
    doc_path = project_root / SAMPLE_DOCUMENT
    fixture_path = project_root / STREAMING_FIXTURE_PATH

    # Validate document exists
    if not doc_path.exists():
        print(f"❌ Error: Document not found: {doc_path}")
        sys.exit(1)

    print("=" * 60)
    print("RLM STREAMING FIXTURE GENERATOR")
    print("=" * 60)
    print(f"Service URL:  {SERVICE_URL}")
    print(f"Query:        {SAMPLE_QUERY}")
    print(f"Document:     {doc_path.name}")
    print(f"Fixture Path: {fixture_path}")
    print("=" * 60)
    print()

    # Check service health and detect storage mode
    print("1. Checking service health and detecting storage mode...")
    storage_mode = "unknown"
    s3_connected = False

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URL}/health")
            response.raise_for_status()
            health = response.json()
            print(f"   ✓ Service status: {health['status']}")
            print(f"   ✓ Version: {health['version']}")

            # Detect storage mode from S3 connection status
            s3_connected = health.get("s3_connected", False)
            if s3_connected:
                storage_mode = "s3"
                print(f"   ✓ Storage mode: S3 (s3_connected=true)")
            else:
                # S3 not connected - could be local mode OR S3 mode with connection issues
                print(f"   ⚠ S3 connection: FAILED (s3_connected=false)")
                print(f"   ℹ This could mean:")
                print(f"      - Service is configured for local mode (IRYS_STORAGE_MODE=local)")
                print(f"      - OR S3 is configured but credentials/bucket are invalid")
                print(f"   ℹ Will attempt upload to determine actual mode...")
                storage_mode = "unknown"  # Will be determined by upload attempt

        except Exception as e:
            print(f"   ❌ Service not available: {e}")
            print(f"\n   Start the service with: python run_server.py")
            sys.exit(1)

    # Upload document and attempt to get S3 URL
    print("\n2. Uploading document...")
    document_urls = []
    s3_prefix = None
    sync_result = None

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            with open(doc_path, "rb") as f:
                files = {"files": (doc_path.name, f, "application/pdf")}
                data = {
                    "query": SAMPLE_QUERY,
                    "keep_files": "true",  # Keep file for streaming request (S3 mode only)
                }

                response = await client.post(
                    f"{SERVICE_URL}/upload/investigate/sync",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                sync_result = response.json()

                # Extract S3 prefix from result (only present in S3 mode)
                s3_prefix = sync_result.get("s3_prefix")

                if s3_prefix:
                    print(f"   ✓ Document uploaded to S3: {s3_prefix}")
                    # Construct S3 URL - use bucket from environment or default
                    s3_bucket = "rlm-irys"  # Default bucket name
                    s3_url = f"s3://{s3_bucket}/{s3_prefix}/{doc_path.name}"
                    document_urls.append(s3_url)
                    print(f"   ✓ S3 URL: {s3_url}")
                    storage_mode = "s3"
                else:
                    print(f"   ✓ Document uploaded (local mode)")
                    print(f"   ℹ No S3 prefix returned - service is in local storage mode")
                    storage_mode = "local"

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500:
                print(f"   ❌ Upload failed with server error (500)")
                print(f"\n   Common causes:")
                print(f"   1. Invalid AWS credentials (check server logs for 'InvalidAccessKeyId')")
                print(f"   2. S3 bucket doesn't exist or is inaccessible")
                print(f"   3. Insufficient S3 permissions")
                print(f"\n   Solutions:")
                print(f"   A. Switch to local mode (recommended for development):")
                print(f"      - Set IRYS_STORAGE_MODE=local in .env")
                print(f"      - Restart the service")
                print(f"      - Re-run this script")
                print(f"\n   B. Fix S3 credentials:")
                print(f"      - Set valid AWS_ACCESS_KEY_ID in .env")
                print(f"      - Set valid AWS_SECRET_ACCESS_KEY in .env")
                print(f"      - Ensure S3_BUCKET exists and is accessible")
                print(f"      - Restart the service")
                print(f"\n   Check server logs for detailed error message")
                sys.exit(1)
            else:
                print(f"   ❌ Upload failed: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Stream investigation events (or simulate from sync result in local mode)
    print("\n3. Capturing investigation events...")
    events = []
    event_counts = {
        "step": 0,
        "citation": 0,
        "fact": 0,
        "progress": 0,
        "complete": 0,
        "error": 0,
    }

    if storage_mode == "s3" and document_urls:
        # S3 MODE: Use actual streaming endpoint
        print(f"   ℹ Using streaming endpoint (S3 mode)")
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                # Make streaming request
                request_data = {
                    "query": SAMPLE_QUERY,
                    "s3_urls": document_urls,
                }

                async with client.stream(
                    "POST",
                    f"{SERVICE_URL}/investigate/urls/stream",
                    json=request_data,
                ) as response:
                    response.raise_for_status()

                    # Parse SSE events
                    current_event_type = None
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        # Parse SSE format: "event: <type>" and "data: <json>"
                        if line.startswith("event:"):
                            current_event_type = line[6:].strip()
                        elif line.startswith("data:") and current_event_type:
                            data_json = line[5:].strip()
                            try:
                                data = json.loads(data_json)
                                event = {
                                    "event": current_event_type,
                                    "data": data,
                                }
                                events.append(event)
                                event_counts[current_event_type] = event_counts.get(current_event_type, 0) + 1

                                # Print progress
                                if current_event_type == "step":
                                    step_type = data.get("step_type", "unknown")
                                    content = data.get("content", "")[:60]
                                    print(f"   • {step_type}: {content}...")
                                elif current_event_type == "citation":
                                    doc = data.get("document", "unknown")
                                    print(f"   • Citation from: {doc}")
                                elif current_event_type == "complete":
                                    print(f"   ✓ Investigation completed")
                                elif current_event_type == "error":
                                    print(f"   ❌ Error: {data.get('error', 'unknown')}")

                            except json.JSONDecodeError as e:
                                print(f"   ⚠ Failed to parse event data: {e}")

                print(f"\n   ✓ Streaming completed")
                print(f"   ✓ Total events: {len(events)}")

            except httpx.TimeoutException:
                print(f"   ❌ Request timed out (investigation took too long)")
                sys.exit(1)
            except Exception as e:
                print(f"   ❌ Streaming failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    else:
        # LOCAL MODE: Simulate streaming events from sync result
        print(f"   ℹ Using sync result simulation (local mode)")
        print(f"   ℹ Generating synthetic streaming events from completed investigation")

        if not sync_result:
            print(f"   ❌ No sync result available to simulate streaming")
            sys.exit(1)

        # Simulate a complete event with the sync result
        complete_event = {
            "event": "complete",
            "data": {
                "query": sync_result.get("query"),
                "analysis": sync_result.get("analysis"),
                "citations": sync_result.get("citations", []),
                "entities": sync_result.get("entities", {}),
                "facts": sync_result.get("facts", []),
                "documents_processed": sync_result.get("documents_processed", 0),
                "duration_seconds": sync_result.get("duration_seconds", 0),
            }
        }
        events.append(complete_event)
        event_counts["complete"] = 1

        # Simulate citation events
        for citation in sync_result.get("citations", []):
            citation_event = {
                "event": "citation",
                "data": citation
            }
            events.append(citation_event)
            event_counts["citation"] = event_counts.get("citation", 0) + 1

        # Simulate fact events
        for fact in sync_result.get("facts", []):
            fact_event = {
                "event": "fact",
                "data": {"fact": fact}
            }
            events.append(fact_event)
            event_counts["fact"] = event_counts.get("fact", 0) + 1

        print(f"   ✓ Generated {len(events)} synthetic events from sync result")
        print(f"   ✓ Citations: {event_counts.get('citation', 0)}")
        print(f"   ✓ Facts: {event_counts.get('fact', 0)}")

    # Save fixture
    print("\n4. Saving streaming fixture...")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    fixture_data = {
        "query": SAMPLE_QUERY,
        "document": doc_path.name,
        "storage_mode": storage_mode,
        "s3_prefix": s3_prefix,
        "s3_urls": document_urls if document_urls else [],
        "events": events,
        "event_counts": event_counts,
        "total_events": len(events),
        "note": "Events captured from streaming endpoint" if storage_mode == "s3" else "Events simulated from sync result (local mode)",
    }

    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture_data, f, indent=2, ensure_ascii=False)

    print(f"   ✓ Streaming fixture saved to: {fixture_path}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ STREAMING FIXTURE GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nStorage mode: {storage_mode.upper()}")
    if storage_mode == "s3":
        print(f"Source: Real streaming endpoint (/investigate/urls/stream)")
    else:
        print(f"Source: Simulated from sync result (local mode fallback)")
    print(f"\nEvent counts:")
    for event_type, count in event_counts.items():
        if count > 0:
            print(f"  {event_type}: {count}")
    print(f"\nTotal events: {len(events)}")
    print(f"\nTo regenerate: python scripts/generate_fixture.py --streaming")
    if storage_mode == "local":
        print(f"\nℹ To capture real streaming events, configure S3 mode:")
        print(f"  1. Set S3_BUCKET in .env")
        print(f"  2. Set IRYS_STORAGE_MODE=s3 in .env")
        print(f"  3. Restart the service")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RLM fixture responses for testing"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Generate streaming API output (SSE events)",
    )
    args = parser.parse_args()

    if args.streaming:
        asyncio.run(generate_streaming_fixture())
    else:
        asyncio.run(generate_fixture())

