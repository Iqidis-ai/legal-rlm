"""Caselaw Benchmark Test Script

Tests V1 and V2 caselaw queries from the AR Video Review (April 16, 2026).
Uses LLM-as-judge (Gemini Flash) to evaluate citation binding accuracy.

USAGE:
    python benchmark_caselaw.py                  # run all queries
    python benchmark_caselaw.py --query V1       # run V1 only
    python benchmark_caselaw.py --query V1 V2    # run both
    python benchmark_caselaw.py --url http://localhost:8000/

OUTPUT:
    Console: per-case pass/fail, score, issues
    File:    caselaw_benchmark_output/benchmark_YYYYMMDD_HHMMSS.json
"""

import httpx
import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file (same pattern as run_ui.py / run_server.py)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()  # fall back to cwd .env if present
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

DEFAULT_API_URL = "https://rlm.iryslegal.com/"
QUERIES_FILE = Path(__file__).parent / "caselaw_benchmark_queries.json"
OUTPUT_DIR = Path("caselaw_benchmark_output")


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(query_ids: list[str] | None = None) -> list[dict]:
    with open(QUERIES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    if query_ids:
        queries = [q for q in queries if q["id"] in query_ids]
    return queries


# ---------------------------------------------------------------------------
# Streaming API call (mirrors test_streaming.py pattern)
# ---------------------------------------------------------------------------

async def run_investigation(base_url: str, payload: dict) -> dict:
    """Call /investigate/urls/stream, capture all events, return full result.

    Mirrors test_streaming.py: every SSE event is stored in result["events"]
    so the LLM judge (or a later replay) has full visibility into the
    investigation trail — leads, facts, searches, citations, checkpoints, etc.
    """
    endpoint = base_url.rstrip("/") + "/investigate/urls/stream"
    result: dict = {
        "analysis": None,
        "citations": [],
        "facts": [],
        "duration_seconds": 0.0,
        "events": [],          # all SSE events, same shape as test_streaming.py
        "total_events": 0,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            current_event_type: str | None = None
            event_count = 0
            start_time = asyncio.get_event_loop().time()

            async with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    result["error"] = f"HTTP {response.status_code}: {body.decode()[:500]}"
                    return result

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("event: "):
                        current_event_type = line[7:].strip()
                        event_count += 1
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        elapsed = asyncio.get_event_loop().time() - start_time

                        # Store every event (mirrors test_streaming.py response_data["events"])
                        result["events"].append({
                            "event_type": current_event_type,
                            "event_number": event_count,
                            "elapsed_seconds": round(elapsed, 3),
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        })

                        if current_event_type == "complete":
                            result["analysis"] = data.get("analysis", "")
                            result["citations"] = data.get("citations", [])
                            result["facts"] = data.get("facts", [])
                            result["duration_seconds"] = data.get("duration_seconds", 0.0)
                            break
                        elif current_event_type == "error":
                            result["error"] = data.get("error", "Unknown error from server")
                            break

            result["total_events"] = event_count

    except httpx.ReadTimeout:
        result["error"] = "Stream timed out after 600s"
    except httpx.ConnectError:
        result["error"] = f"Could not connect to {base_url} — is the server running?"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# LLM Judge helpers
# ---------------------------------------------------------------------------

def format_citations_for_judge(citations: list[dict]) -> str:
    if not citations:
        return "(No citations retrieved)"
    lines = []
    for i, c in enumerate(citations, 1):
        doc = c.get("document", "?")
        ctx = c.get("context", "")
        url = c.get("url", "")
        src = c.get("source_type", "")
        lines.append(f"[{i}] {doc} | {ctx} | {src} | {url}")
    return "\n".join(lines)


def build_judge_prompt(query: str, analysis: str, citations: list[dict], ji: dict) -> str:
    """Build structured judge prompt from query result + judge instructions."""
    citations_text = format_citations_for_judge(citations)
    expected_json = json.dumps(ji["expected_cases"], indent=2)
    pass_text = "\n".join(f"  - {c}" for c in ji["pass_criteria"])
    fail_text = "\n".join(f"  - {c}" for c in ji["fail_indicators"])
    judge_notes = ji.get("judge_notes", "")

    return f"""You are a strict legal benchmark judge evaluating an AI legal research system.

## ORIGINAL QUERY
{query}

## AI RESPONSE — ANALYSIS / MEMO
{analysis}

## AI RESPONSE — CITATIONS / SOURCES LIST
{citations_text}

## BENCHMARK GROUND TRUTH

### Expected Cases (correct answers)
{expected_json}

### Pass Criteria (ALL must be satisfied for PASS)
{pass_text}

### Fail Indicators (ANY single indicator = FAIL)
{fail_text}

### Judge Notes
{judge_notes}

## EVALUATION INSTRUCTIONS
Examine the analysis memo AND the citations list carefully.
For each expected case, determine:
  1. Is the case addressed in the analysis?
  2. Is the correct citation number bound to the correct case name in the memo?
  3. Does the citations list contain an entry for the correct case (fuzzy name match ok)?
  4. Are there any known wrong substitute cases present (Glattly, BMC Software, Reata Construction)?

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{
  "verdict": "PASS" or "FAIL",
  "overall_score": <integer 0-10>,
  "per_case_results": [
    {{
      "case_name_or_citation": "<name or citation string>",
      "found_in_analysis": <true/false>,
      "correct_case_bound": <true/false>,
      "found_in_sources_list": <true/false>,
      "pass": <true/false>,
      "note": "<concise note — what was right or wrong>"
    }}
  ],
  "critical_issues": ["<issue description>"],
  "positive_findings": ["<positive observation>"],
  "reasoning": "<2-4 sentence overall assessment explaining the verdict>"
}}"""


# ---------------------------------------------------------------------------
# Run the LLM judge
# ---------------------------------------------------------------------------

async def run_judge(query_id: str, query: str, result: dict, ji: dict, api_key: str) -> dict:
    """Call Gemini Flash to evaluate the investigation result."""
    from google import genai  # same library used by src/irys/core/models.py

    client = genai.Client(api_key=api_key)
    prompt = build_judge_prompt(
        query,
        result.get("analysis") or "(no analysis returned)",
        result.get("citations", []),
        ji,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = (response.text or "").strip()

        # Strip accidental markdown code fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:] if lines[-1].strip() != "```" else lines[1:-1])

        verdict = json.loads(raw)
        verdict["query_id"] = query_id
        return verdict

    except json.JSONDecodeError as exc:
        return {
            "query_id": query_id,
            "verdict": "ERROR",
            "error": f"Judge returned invalid JSON: {exc}",
            "raw_response": raw[:800] if "raw" in dir() else "",
        }
    except Exception as exc:
        return {
            "query_id": query_id,
            "verdict": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_result(query_id: str, label: str, result: dict, verdict: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"QUERY {query_id}: {label}")
    print(sep)

    if result.get("error"):
        print(f"❌  INVESTIGATION ERROR: {result['error']}")
        return

    print(f"⏱   Duration: {result.get('duration_seconds', 0):.1f}s")
    print(f"📚  Citations retrieved: {len(result.get('citations', []))}")

    if verdict.get("verdict") == "ERROR":
        print(f"❌  JUDGE ERROR: {verdict.get('error', '?')}")
        raw = verdict.get("raw_response", "")
        if raw:
            print(f"    Raw: {raw[:300]}")
        return

    v = verdict.get("verdict", "?")
    score = verdict.get("overall_score", 0)
    icon = "✅" if v == "PASS" else "❌"
    print(f"\n{icon}  VERDICT: {v}  |  Score: {score}/10")
    print(f"\n📝  Reasoning: {verdict.get('reasoning', '')}")

    per_case = verdict.get("per_case_results", [])
    if per_case:
        print("\n📋  Per-Case Results:")
        for case in per_case:
            status = "✅" if case.get("pass") else "❌"
            name = case.get("case_name_or_citation", "?")
            note = case.get("note", "")
            print(f"    {status}  {name}")
            if note:
                print(f"         → {note}")

    issues = verdict.get("critical_issues", [])
    if issues:
        print("\n🚨  Critical Issues:")
        for issue in issues:
            print(f"    • {issue}")

    positives = verdict.get("positive_findings", [])
    if positives:
        print("\n✨  Positive Findings:")
        for pos in positives:
            print(f"    • {pos}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Caselaw citation binding benchmark")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="RLM API base URL")
    parser.add_argument("--query", nargs="+", metavar="ID", help="Query IDs to run (e.g. V1 V2). Default: all.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory for results JSON")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required for the LLM judge.")
        sys.exit(1)

    queries = load_queries(args.query)
    if not queries:
        ids = args.query or []
        print(f"No queries found for IDs: {ids}. Available: V1, V2")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"

    all_results: dict = {
        "timestamp": datetime.now().isoformat(),
        "api_url": args.url,
        "queries_run": [q["id"] for q in queries],
        "results": [],
        "summary": {},
    }

    passed = failed = errors = 0

    for q in queries:
        qid = q["id"]
        label = q.get("label", qid)
        query_text = q["query"]
        s3_urls = q.get("s3_urls", [])
        ji = q["judge_instructions"]

        print(f"\n{'=' * 70}")
        print(f"Running Query {qid}: {label}")
        print(f"{'=' * 70}")
        print(f"Query: {query_text[:120]}{'...' if len(query_text) > 120 else ''}")
        print("Calling streaming API…")

        payload = {
            "query": query_text,
            "message_id": f"benchmark_{qid}_{timestamp}",
            "user_id": "benchmark_runner",
            "s3_urls": s3_urls,
        }

        result = await run_investigation(args.url, payload)

        if result.get("error"):
            print(f"❌  Error: {result['error']}")
            errors += 1
            verdict: dict = {"verdict": "ERROR", "error": result["error"]}
        else:
            dur = result.get("duration_seconds", 0)
            cites = len(result.get("citations", []))
            evts = result.get("total_events", 0)
            print(f"✓  Investigation complete ({dur:.1f}s, {cites} citations, {evts} events). Running LLM judge…")
            verdict = await run_judge(qid, query_text, result, ji, api_key)

        print_result(qid, label, result, verdict)

        v = verdict.get("verdict", "ERROR")
        if v == "PASS":
            passed += 1
        elif v == "FAIL":
            failed += 1
        else:
            errors += 1

        all_results["results"].append({
            "query_id": qid,
            "label": label,
            "query": query_text,
            "investigation": {
                "analysis": result.get("analysis"),
                "citations": result.get("citations", []),
                "facts": result.get("facts", []),
                "duration_seconds": result.get("duration_seconds"),
                "total_events": result.get("total_events", 0),
                "events": result.get("events", []),   # full SSE trail for replay/re-judging
                "error": result.get("error"),
            },
            "verdict": verdict,
        })

    total = len(queries)
    all_results["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": f"{passed}/{total}",
    }

    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total:   {total}")
    print(f"  ✅ Pass: {passed}")
    print(f"  ❌ Fail: {failed}")
    print(f"  ⚠️  Err:  {errors}")
    print(f"  Rate:    {passed}/{total}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Results saved to: {output_file.absolute()}")


if __name__ == "__main__":
    print("\nCaselaw Benchmark — AR Video Review queries (V1 / V2)")
    print("Press Ctrl+C to abort\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
