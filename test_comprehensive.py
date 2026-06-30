"""Comprehensive test suite for the refactored RLM system.

Tests various query complexities with extensive logging:
- Simple lookups (single fact retrieval)
- Analytical queries (reasoning required)
- Multi-document queries (cross-referencing)
- Complex reasoning (synthesis across sources)
"""

import asyncio
import sys
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import InvestigationState, StepType
from irys.core.models import GeminiClient

# Configure extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s',
    datefmt='%H:%M:%S'
)

# Create loggers for different modules
logger = logging.getLogger("test_comprehensive")
engine_logger = logging.getLogger("irys.rlm.engine")
decisions_logger = logging.getLogger("irys.rlm.decisions")
search_logger = logging.getLogger("irys.core.search")

# Set all to DEBUG
for log in [logger, engine_logger, decisions_logger, search_logger]:
    log.setLevel(logging.DEBUG)


class TestMetrics:
    """Track metrics for each test."""
    def __init__(self, query: str, complexity: str):
        self.query = query
        self.complexity = complexity
        self.start_time = None
        self.end_time = None
        self.steps = []
        self.citations = []
        self.api_calls = 0
        self.documents_read = 0
        self.searches_performed = 0
        self.facts_found = 0
        self.status = "pending"
        self.error = None
        self.final_output = None

    def start(self):
        self.start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING TEST: {self.complexity.upper()}")
        logger.info(f"Query: {self.query}")
        logger.info(f"{'='*80}")

    def end(self, state: InvestigationState):
        self.end_time = time.time()
        self.status = state.status
        self.api_calls = state.api_calls
        self.documents_read = state.documents_read
        self.searches_performed = state.searches_performed
        self.facts_found = len(state.findings.get("accumulated_facts", []))
        self.final_output = state.findings.get("final_output", "")

        # Log summary
        duration = self.end_time - self.start_time
        logger.info(f"\n{'-'*80}")
        logger.info(f"TEST COMPLETED: {self.status.upper()}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"API Calls: {self.api_calls}")
        logger.info(f"Documents Read: {self.documents_read}")
        logger.info(f"Searches: {self.searches_performed}")
        logger.info(f"Facts Found: {self.facts_found}")
        logger.info(f"Citations: {len(state.citations)}")
        logger.info(f"Steps: {len(state.thinking_steps)}")
        logger.info(f"{'-'*80}\n")

    def fail(self, error: str):
        self.end_time = time.time()
        self.status = "failed"
        self.error = error
        logger.error(f"TEST FAILED: {error}")

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def to_dict(self):
        return {
            "query": self.query,
            "complexity": self.complexity,
            "status": self.status,
            "duration_seconds": round(self.duration, 2),
            "api_calls": self.api_calls,
            "documents_read": self.documents_read,
            "searches_performed": self.searches_performed,
            "facts_found": self.facts_found,
            "error": self.error,
        }


def on_step(step):
    """Callback for thinking steps - log each step."""
    indent = "  " * step.depth
    step_icon = {
        StepType.THINKING: "💭",
        StepType.SEARCH: "🔍",
        StepType.READING: "📖",
        StepType.FINDING: "✨",
        StepType.REPLAN: "🔄",
        StepType.VERIFY: "✅",
        StepType.SYNTHESIS: "📝",
        StepType.ERROR: "❌",
    }.get(step.step_type, "•")

    logger.info(f"{indent}{step_icon} [{step.step_type.value:10}] {step.content[:100]}")

    if step.details:
        for key, value in step.details.items():
            if isinstance(value, list):
                logger.debug(f"{indent}   {key}: {value[:3]}{'...' if len(value) > 3 else ''}")
            elif isinstance(value, str) and len(value) > 100:
                logger.debug(f"{indent}   {key}: {value[:100]}...")
            else:
                logger.debug(f"{indent}   {key}: {value}")


def on_progress(progress: dict):
    """Callback for progress updates."""
    logger.debug(
        f"Progress: docs={progress['documents_read']}, "
        f"searches={progress['searches_performed']}, "
        f"citations={progress['citations']}, "
        f"facts={progress['facts_accumulated']}"
    )


# Define test queries by complexity
TEST_QUERIES = {
    "simple": [
        # Direct fact lookups
        ("What is the monthly fee in the original contract?", "Should find $50,000"),
        ("When was the service agreement signed?", "Should find January 15, 2024"),
        ("Who signed the contract for TechServices?", "Should find Sarah Johnson"),
    ],
    "analytical": [
        # Requires understanding and context
        ("What services are included in the agreement?", "Should list web dev, database, API, QA"),
        ("What are the termination conditions?", "Should explain convenience and cause termination"),
        ("What is the liability cap in the agreement?", "Should explain the 12-month fee cap"),
    ],
    "multi_document": [
        # Requires looking across documents
        ("How did the monthly fee change over time?", "Should find $50k -> $65k change"),
        ("What new services were added in the amendment?", "Should find mobile, 24/7 support, cloud"),
        ("What is the total contract term after the amendment?", "Should find 18 months"),
    ],
    "complex": [
        # Requires synthesis and reasoning
        ("What performance issues has TechServices had?", "Should find downtime, SLA misses, delays"),
        ("What is the risk of contract termination?", "Should discuss Q2 deadline, SLA concerns"),
        ("Summarize the key contractual obligations and their current status", "Comprehensive summary"),
    ]
}


async def run_single_test(
    engine: RLMEngine,
    query: str,
    complexity: str,
    expected: str,
    repo_path: str,
) -> TestMetrics:
    """Run a single test query."""
    metrics = TestMetrics(query, complexity)
    metrics.start()

    try:
        state = await engine.investigate(query, repo_path)
        metrics.end(state)

        # Log the final output (truncated)
        if metrics.final_output:
            logger.info("FINAL OUTPUT (first 500 chars):")
            logger.info(metrics.final_output[:500])
            if len(metrics.final_output) > 500:
                logger.info("...")

        # Log all citations
        if state.citations:
            logger.info(f"\nCITATIONS ({len(state.citations)}):")
            for i, c in enumerate(state.citations[:5], 1):
                logger.info(f"  [{i}] {c.document}, p.{c.page}: {c.text[:80]}...")

        # Log accumulated facts
        facts = state.findings.get("accumulated_facts", [])
        if facts:
            logger.info(f"\nFACTS ({len(facts)}):")
            for i, fact in enumerate(facts[:5], 1):
                logger.info(f"  {i}. {fact[:100]}...")

        # Log thinking trace summary
        logger.info(f"\nTHINKING TRACE ({len(state.thinking_steps)} steps):")
        step_types = {}
        for step in state.thinking_steps:
            step_types[step.step_type.value] = step_types.get(step.step_type.value, 0) + 1
        for stype, count in sorted(step_types.items()):
            logger.info(f"  {stype}: {count}")

    except Exception as e:
        metrics.fail(str(e))
        logger.exception(f"Test failed with exception: {e}")

    return metrics


async def run_all_tests(repo_path: str):
    """Run all tests and generate report."""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE RLM TEST SUITE")
    logger.info(f"Repository: {repo_path}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*80 + "\n")

    # Initialize engine with logging callbacks
    client = GeminiClient()
    config = RLMConfig(
        max_depth=3,
        max_leads_per_level=3,
        max_iterations=10,
        excerpt_chars=6000,
    )
    engine = RLMEngine(
        gemini_client=client,
        config=config,
        on_step=on_step,
        on_progress=on_progress,
    )

    all_results = []

    # Run tests by complexity
    for complexity, queries in TEST_QUERIES.items():
        logger.info(f"\n{'#'*80}")
        logger.info(f"# RUNNING {complexity.upper()} TESTS ({len(queries)} queries)")
        logger.info(f"{'#'*80}\n")

        for query, expected in queries:
            result = await run_single_test(engine, query, complexity, expected, repo_path)
            all_results.append(result)

            # Small delay between tests
            await asyncio.sleep(1)

    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY REPORT")
    logger.info("="*80)

    passed = sum(1 for r in all_results if r.status == "completed")
    failed = sum(1 for r in all_results if r.status != "completed")
    total_duration = sum(r.duration for r in all_results)

    logger.info(f"\nTotal Tests: {len(all_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total Duration: {total_duration:.2f}s")
    logger.info(f"Average Duration: {total_duration/len(all_results):.2f}s per test")

    # Results by complexity
    logger.info("\nResults by Complexity:")
    for complexity in TEST_QUERIES.keys():
        comp_results = [r for r in all_results if r.complexity == complexity]
        comp_passed = sum(1 for r in comp_results if r.status == "completed")
        comp_duration = sum(r.duration for r in comp_results)
        logger.info(f"  {complexity}: {comp_passed}/{len(comp_results)} passed, {comp_duration:.2f}s total")

    # Detailed results table
    logger.info("\nDetailed Results:")
    logger.info(f"{'Query'[:50]:50} | {'Status':10} | {'Duration':10} | {'Facts':6} | {'Docs':5}")
    logger.info("-" * 90)
    for r in all_results:
        logger.info(
            f"{r.query[:50]:50} | {r.status:10} | {r.duration:8.2f}s | {r.facts_found:6} | {r.documents_read:5}"
        )

    # Save detailed results to JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(all_results),
            "passed": passed,
            "failed": failed,
            "total_duration_seconds": round(total_duration, 2),
        },
        "tests": [r.to_dict() for r in all_results],
    }

    report_path = Path(__file__).parent / "test_results_comprehensive.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nDetailed report saved to: {report_path}")

    return all_results


async def run_quick_test(repo_path: str):
    """Run a quick single test for debugging."""
    logger.info("\n" + "="*80)
    logger.info("QUICK SINGLE TEST")
    logger.info("="*80 + "\n")

    client = GeminiClient()
    config = RLMConfig(
        max_depth=2,
        max_leads_per_level=2,
        max_iterations=5,
    )
    engine = RLMEngine(
        gemini_client=client,
        config=config,
        on_step=on_step,
        on_progress=on_progress,
    )

    query = "What is the monthly fee in the contract?"
    result = await run_single_test(engine, query, "simple", "Should find $50,000", repo_path)

    return result


if __name__ == "__main__":
    repo_path = str(Path(__file__).parent / "test_documents")

    # Check args for quick vs full test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(run_quick_test(repo_path))
    else:
        asyncio.run(run_all_tests(repo_path))
