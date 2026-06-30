"""Quality comparison framework: RLM vs Claude vs Codex.

Runs the same complex legal query through all three systems and compares outputs.
"""

import asyncio
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path
import os

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig


@dataclass
class ComparisonResult:
    """Result from a single system."""
    system: str
    query: str
    output: str
    duration_seconds: float
    tokens_used: Optional[int] = None
    citations_count: int = 0
    error: Optional[str] = None


@dataclass
class QualityComparison:
    """Full comparison across all systems."""
    query: str
    repository_path: str
    timestamp: datetime
    results: dict[str, ComparisonResult] = field(default_factory=dict)
    evaluation: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "repository_path": self.repository_path,
            "timestamp": self.timestamp.isoformat(),
            "results": {
                k: {
                    "system": v.system,
                    "output": v.output,
                    "duration_seconds": v.duration_seconds,
                    "citations_count": v.citations_count,
                    "error": v.error,
                }
                for k, v in self.results.items()
            },
            "evaluation": self.evaluation,
        }

    def save(self, path: Path):
        """Save comparison to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class QualityTester:
    """
    Compare RLM output against Claude and Codex baselines.

    Usage:
        tester = QualityTester(repo_path, gemini_key)
        comparison = await tester.compare(query)
        comparison.save("comparison_result.json")
    """

    def __init__(
        self,
        repository_path: str,
        gemini_api_key: str,
        anthropic_api_key: Optional[str] = None,
    ):
        self.repository_path = repository_path
        self.gemini_api_key = gemini_api_key
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def compare(self, query: str) -> QualityComparison:
        """Run comparison across all systems."""
        comparison = QualityComparison(
            query=query,
            repository_path=self.repository_path,
            timestamp=datetime.now(),
        )

        # Run all systems (could parallelize, but sequential for debugging)
        comparison.results["rlm"] = await self._run_rlm(query)
        comparison.results["claude"] = await self._run_claude(query)
        comparison.results["codex"] = await self._run_codex(query)

        return comparison

    async def _run_rlm(self, query: str) -> ComparisonResult:
        """Run query through RLM system."""
        start = datetime.now()

        try:
            client = GeminiClient(api_key=self.gemini_api_key)
            engine = RLMEngine(gemini_client=client, config=RLMConfig())

            state = await engine.investigate(query, self.repository_path)

            output = state.findings.get("final_output", "")
            duration = (datetime.now() - start).total_seconds()

            return ComparisonResult(
                system="rlm",
                query=query,
                output=output,
                duration_seconds=duration,
                citations_count=len(state.citations),
            )

        except Exception as e:
            return ComparisonResult(
                system="rlm",
                query=query,
                output="",
                duration_seconds=(datetime.now() - start).total_seconds(),
                error=str(e),
            )

    async def _run_claude(self, query: str) -> ComparisonResult:
        """Run query through Claude sub-agent."""
        start = datetime.now()

        # Build prompt with repository context
        prompt = self._build_claude_prompt(query)

        try:
            # Use Claude Code sub-agent via Task tool pattern
            # For now, we'll use the Anthropic API directly if available
            if self.anthropic_api_key:
                import anthropic

                client = anthropic.Anthropic(api_key=self.anthropic_api_key)

                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )

                output = message.content[0].text
                duration = (datetime.now() - start).total_seconds()

                return ComparisonResult(
                    system="claude",
                    query=query,
                    output=output,
                    duration_seconds=duration,
                )
            else:
                return ComparisonResult(
                    system="claude",
                    query=query,
                    output="",
                    duration_seconds=0,
                    error="No Anthropic API key provided",
                )

        except Exception as e:
            return ComparisonResult(
                system="claude",
                query=query,
                output="",
                duration_seconds=(datetime.now() - start).total_seconds(),
                error=str(e),
            )

    async def _run_codex(self, query: str) -> ComparisonResult:
        """Run query through Codex (GPT-5.2 via codex exec)."""
        start = datetime.now()

        prompt = self._build_codex_prompt(query)

        try:
            # Run codex exec command
            result = subprocess.run(
                ["codex", "exec", prompt],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nSTDERR:\n{result.stderr}"

            duration = (datetime.now() - start).total_seconds()

            return ComparisonResult(
                system="codex",
                query=query,
                output=output,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return ComparisonResult(
                system="codex",
                query=query,
                output="",
                duration_seconds=300,
                error="Timeout after 5 minutes",
            )
        except FileNotFoundError:
            return ComparisonResult(
                system="codex",
                query=query,
                output="",
                duration_seconds=0,
                error="Codex CLI not found",
            )
        except Exception as e:
            return ComparisonResult(
                system="codex",
                query=query,
                output="",
                duration_seconds=(datetime.now() - start).total_seconds(),
                error=str(e),
            )

    def _build_claude_prompt(self, query: str) -> str:
        """Build prompt for Claude with repository context."""
        # Get file listing
        from irys.core.repository import MatterRepository

        repo = MatterRepository(self.repository_path)
        stats = repo.get_stats()
        structure = repo.get_structure()

        structure_str = "\n".join(f"  {k}: {v} files" for k, v in structure.items())

        return f"""You are a senior legal analyst reviewing a matter repository.

Repository: {self.repository_path}
Total files: {stats.total_files}
Size: {stats.size_mb:.1f} MB

Structure:
{structure_str}

Query: {query}

Analyze the available documents and provide a comprehensive legal analysis.
Include specific citations where possible in format [Document, p. X].
Focus on accuracy and thoroughness.
"""

    def _build_codex_prompt(self, query: str) -> str:
        """Build prompt for Codex."""
        return f"""You are a senior legal analyst. Analyze the following legal query:

Repository path: {self.repository_path}

Query: {query}

Provide a thorough legal analysis with citations to specific documents where relevant.
Use first-principles reasoning and be brutally honest about any limitations.
"""


async def run_comparison(
    query: str,
    repo_path: str,
    output_path: Optional[str] = None,
) -> QualityComparison:
    """
    Run a quality comparison.

    Args:
        query: Legal query to test
        repo_path: Path to matter repository
        output_path: Optional path to save results

    Returns:
        QualityComparison with results from all systems
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable required")

    tester = QualityTester(repo_path, gemini_key)
    comparison = await tester.compare(query)

    if output_path:
        comparison.save(Path(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY COMPARISON RESULTS")
    print("=" * 60)
    print(f"Query: {query}")
    print()

    for system, result in comparison.results.items():
        print(f"\n--- {system.upper()} ---")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Output length: {len(result.output)} chars")
            print(f"Citations: {result.citations_count}")
            print(f"Preview: {result.output[:500]}...")

    return comparison


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What are the key claims in this dispute?"
    repo = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\devan\Downloads\CITIOM v Gulfstream"

    asyncio.run(run_comparison(query, repo, "comparison_result.json"))
