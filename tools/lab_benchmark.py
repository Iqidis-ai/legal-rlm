"""Run Irys RLM against Harvey's Legal Agent Benchmark (LAB).

Bypasses LAB's agent harness and uses Irys as a standalone investigation
engine, then writes results in LAB-compatible format for scoring with
LAB's evaluation pipeline.

Usage:
    # Single task
    python tools/lab_benchmark.py --task corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts

    # All tasks in a practice area
    python tools/lab_benchmark.py --practice-area corporate-ma

    # All tasks (full benchmark)
    python tools/lab_benchmark.py --all

    # Score results after a run
    python tools/lab_benchmark.py --score --run-id <run-id>

    # Run + auto-score
    python tools/lab_benchmark.py --task corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts --auto-score

Requires:
    - GEMINI_API_KEY environment variable (or .env file in project root)
    - Harvey LAB repo cloned at ../harvey-labs (or set --lab-root)
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irys.api import Irys, IrysConfig

logger = logging.getLogger("lab_benchmark")


def _markdown_to_docx(md_text: str, output_path: Path):
    """Convert markdown text to a .docx file using python-docx."""
    from docx import Document
    from docx.shared import Pt, Inches
    import re

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)
    style.font.name = "Calibri"

    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            doc.add_paragraph("")
            i += 1
            continue
        # Detect markdown table (line with | separators)
        if "|" in stripped and stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                # Skip separator rows (|---|---|)
                if re.match(r'^\|[\s\-:|]+\|$', row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.split("|")[1:-1]]
                table_lines.append(cells)
                i += 1
            if table_lines:
                n_cols = max(len(r) for r in table_lines)
                table = doc.add_table(rows=len(table_lines), cols=n_cols)
                table.style = "Table Grid"
                for ri, row_cells in enumerate(table_lines):
                    for ci, cell_text in enumerate(row_cells):
                        if ci < n_cols:
                            table.rows[ri].cells[ci].text = cell_text
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=4)
        elif stripped.startswith("- "):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        elif re.match(r'^\d+\.\s', stripped):
            doc.add_paragraph(re.sub(r'^\d+\.\s', '', stripped), style="List Number")
        elif stripped.startswith("**") and stripped.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(stripped.strip("*"))
            run.bold = True
        elif stripped.startswith("> "):
            p = doc.add_paragraph(stripped[2:])
            p.style = doc.styles.get("Quote", doc.styles["Normal"])
        else:
            doc.add_paragraph(stripped)
        i += 1

    doc.save(str(output_path))

DEFAULT_LAB_ROOT = PROJECT_ROOT.parent / "harvey-labs"
RESULTS_SUBDIR = "results"


def _load_dotenv():
    """Load .env from project root if it exists."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)


def discover_tasks(lab_root: Path, practice_area: str | None = None) -> list[str]:
    """Discover all task IDs under the LAB tasks/ directory."""
    tasks_dir = lab_root / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"LAB tasks directory not found: {tasks_dir}")

    task_ids = []
    areas = [practice_area] if practice_area else sorted(
        p.name for p in tasks_dir.iterdir() if p.is_dir()
    )

    for area in areas:
        area_dir = tasks_dir / area
        if not area_dir.exists():
            logger.warning("Practice area not found: %s", area)
            continue
        for task_dir in sorted(area_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_json = task_dir / "task.json"
            if task_json.exists():
                # Check for scenario subdirectories
                scenarios = [
                    d for d in task_dir.iterdir()
                    if d.is_dir() and (d / "task.json").exists()
                ]
                if scenarios:
                    for s in sorted(scenarios):
                        task_ids.append(f"{area}/{task_dir.name}/{s.name}")
                else:
                    task_ids.append(f"{area}/{task_dir.name}")

    return task_ids


def load_task(lab_root: Path, task_id: str) -> dict:
    """Load a LAB task by ID."""
    parts = task_id.split("/")
    task_dir = lab_root / "tasks" / Path(*parts)
    task_json = task_dir / "task.json"

    if not task_json.exists():
        raise FileNotFoundError(f"task.json not found: {task_json}")

    config = json.loads(task_json.read_text(encoding="utf-8"))
    docs_dir = task_dir / "documents"
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    instructions = config.get("instructions", "")
    if not instructions:
        instructions_path = task_dir / "instructions.md"
        if instructions_path.exists():
            instructions = instructions_path.read_text(encoding="utf-8")

    return {
        "task_id": task_id,
        "task_dir": task_dir,
        "docs_dir": docs_dir,
        "instructions": instructions,
        "config": config,
        "deliverables": config.get("deliverables", {}),
        "criteria": config.get("criteria", []),
    }


async def run_task(
    task: dict,
    lab_root: Path,
    research_mode: str = "deep",
    api_key: str | None = None,
) -> dict:
    """Run Irys investigation on a single LAB task.

    Returns a dict with run_id, output path, timing, and result metadata.
    """
    task_id = task["task_id"]
    docs_dir = task["docs_dir"]
    instructions = task["instructions"]
    deliverables = task["deliverables"]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{task_id}/irys-rlm-{research_mode}/{ts}"

    results_dir = lab_root / RESULTS_SUBDIR / run_id
    output_dir = results_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config = {
        "model": "irys-rlm",
        "task": task_id,
        "run_id": run_id,
        "research_mode": research_mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Copy documents to a temp directory so matter model artifacts
    # don't pollute LAB's repo and each run starts fresh.
    logger.info("Running task: %s", task_id)
    start = time.monotonic()

    work_dir = results_dir / "workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = work_dir / "documents"
    shutil.copytree(docs_dir, repo_dir, dirs_exist_ok=True)

    irys = Irys(IrysConfig(
        api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        max_depth=8,
        max_leads_per_level=12,
        output_format="markdown",
        enable_matter_model=True,
    ))

    try:
        result = await irys.investigate(
            query=instructions,
            repository=str(repo_dir),
            research_mode=research_mode,
        )
        elapsed = time.monotonic() - start

        output_text = result.output or ""
        status = result.status
        success = result.success

        # Write output for each expected deliverable in the expected format
        if deliverables:
            for name, filename in deliverables.items():
                ext = Path(filename).suffix.lower()
                if ext == ".docx":
                    docx_path = output_dir / filename
                    _markdown_to_docx(output_text, docx_path)
                else:
                    out_path = output_dir / filename
                    out_path.write_text(output_text, encoding="utf-8")
        else:
            (output_dir / "output.md").write_text(output_text, encoding="utf-8")

        # Also write raw output for inspection
        (results_dir / "irys_raw_output.md").write_text(output_text, encoding="utf-8")

        # Write metrics
        usage = getattr(result.state, "llm_usage", {}) or {}
        metrics = {
            "model": "irys-rlm",
            "task": task_id,
            "run_id": run_id,
            "wall_clock_seconds": elapsed,
            "finished_cleanly": success,
            "status": status,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_documents": len(list(docs_dir.iterdir())),
            "documents_read": len(list(docs_dir.iterdir())),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        progress = result.state.get_progress()
        if progress:
            metrics["irys_progress"] = progress

        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Write transcript stub (LAB eval doesn't require it, but useful for debugging)
        transcript = {
            "task_id": task_id,
            "query": instructions,
            "research_mode": research_mode,
            "status": status,
            "route": result.state.findings.get("route", {}),
            "output_length": len(output_text),
            "citations_count": len(result.citations),
            "elapsed_seconds": elapsed,
        }
        (results_dir / "transcript.json").write_text(json.dumps(transcript, indent=2))

        logger.info(
            "Completed %s in %.1fs — status=%s, output=%d chars",
            task_id, elapsed, status, len(output_text),
        )

        return {
            "run_id": run_id,
            "task_id": task_id,
            "results_dir": str(results_dir),
            "status": status,
            "success": success,
            "elapsed": elapsed,
            "output_length": len(output_text),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("Failed %s after %.1fs: %s", task_id, elapsed, e)
        metrics = {
            "model": "irys-rlm",
            "task": task_id,
            "run_id": run_id,
            "wall_clock_seconds": elapsed,
            "finished_cleanly": False,
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return {
            "run_id": run_id,
            "task_id": task_id,
            "results_dir": str(results_dir),
            "status": "error",
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
        }
    finally:
        irys.close_all_matter_models()


def score_run(lab_root: Path, run_id: str, task_id: str, judge_model: str = "claude-sonnet-4-6"):
    """Score a completed run using LAB's evaluation pipeline."""
    eval_script = lab_root / "evaluation" / "run_eval.py"
    if not eval_script.exists():
        logger.error("LAB evaluation script not found: %s", eval_script)
        return None

    cmd = [
        sys.executable, "-m", "evaluation.run_eval",
        "--run-id", run_id,
        "--task", task_id,
        "--judge-model", judge_model,
    ]
    logger.info("Scoring: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=str(lab_root), capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        logger.error("Scoring failed:\n%s\n%s", result.stdout, result.stderr)
        return None

    print(result.stdout)
    scores_path = lab_root / RESULTS_SUBDIR / run_id / "scores.json"
    if scores_path.exists():
        return json.loads(scores_path.read_text())
    return None


async def run_benchmark(args):
    """Main benchmark runner."""
    lab_root = Path(args.lab_root).resolve()
    if not (lab_root / "tasks").exists():
        print(f"Error: LAB repo not found at {lab_root}")
        print("Clone it with: git clone https://github.com/harveyai/harvey-labs.git")
        sys.exit(1)

    # Discover tasks
    if args.task:
        task_ids = [args.task]
    elif args.practice_area:
        task_ids = discover_tasks(lab_root, practice_area=args.practice_area)
    elif args.all:
        task_ids = discover_tasks(lab_root)
    else:
        print("Error: specify --task, --practice-area, or --all")
        sys.exit(1)

    print(f"LAB root: {lab_root}")
    print(f"Tasks to run: {len(task_ids)}")
    print(f"Research mode: {args.research_mode}")
    print()

    results = []
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        try:
            task = load_task(lab_root, task_id)
            result = await run_task(
                task=task,
                lab_root=lab_root,
                research_mode=args.research_mode,
                api_key=args.api_key,
            )
            results.append(result)

            status_icon = "OK" if result["success"] else "FAIL"
            elapsed = result.get("elapsed", 0)
            output_len = result.get("output_length", 0)
            print(f"  {status_icon} — {elapsed:.1f}s, {output_len} chars")

            if args.auto_score and result["success"]:
                print("  Scoring...")
                scores = score_run(
                    lab_root, result["run_id"], task_id,
                    judge_model=args.judge_model,
                )
                if scores:
                    result["scores"] = scores
                    print(f"  Score: {scores['n_passed']}/{scores['n_criteria']} criteria passed")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "task_id": task_id,
                "status": "error",
                "success": False,
                "error": str(e),
            })

        print()

    # Summary
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    total = len(results)
    succeeded = sum(1 for r in results if r.get("success"))
    failed = total - succeeded
    print(f"  Total tasks:  {total}")
    print(f"  Succeeded:    {succeeded}")
    print(f"  Failed:       {failed}")

    if any("scores" in r for r in results):
        scored = [r for r in results if "scores" in r]
        all_pass = sum(1 for r in scored if r["scores"].get("all_pass"))
        total_criteria = sum(r["scores"]["n_criteria"] for r in scored)
        total_passed = sum(r["scores"]["n_passed"] for r in scored)
        print(f"  Tasks all-pass: {all_pass}/{len(scored)}")
        print(f"  Criteria passed: {total_passed}/{total_criteria} ({100*total_passed/total_criteria:.1f}%)")

    avg_time = sum(r.get("elapsed", 0) for r in results) / max(total, 1)
    print(f"  Avg time/task: {avg_time:.1f}s")

    # Write aggregate results
    summary_path = lab_root / RESULTS_SUBDIR / "irys_benchmark_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tasks": total,
        "succeeded": succeeded,
        "failed": failed,
        "research_mode": args.research_mode,
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Irys RLM against Harvey's Legal Agent Benchmark"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", help="Single task ID (e.g., corporate-ma/analyze-change-of-control-...)")
    group.add_argument("--practice-area", help="Run all tasks in a practice area")
    group.add_argument("--all", action="store_true", help="Run the full benchmark")
    group.add_argument("--score", action="store_true", help="Score an existing run (use with --run-id)")

    parser.add_argument("--lab-root", default=str(DEFAULT_LAB_ROOT),
                        help="Path to harvey-labs repo (default: ../harvey-labs)")
    parser.add_argument("--research-mode", default="deep", choices=["simple", "deep", "sebih_special"],
                        help="Irys research depth (default: deep)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    parser.add_argument("--auto-score", action="store_true", help="Score each task after running")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6", help="LLM judge for scoring")
    parser.add_argument("--run-id", help="Run ID to score (with --score)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _load_dotenv()

    if args.score:
        if not args.run_id or not args.task:
            print("Error: --score requires --run-id and --task")
            sys.exit(1)
        lab_root = Path(args.lab_root).resolve()
        scores = score_run(lab_root, args.run_id, args.task, args.judge_model)
        if scores:
            print(json.dumps(scores, indent=2))
        sys.exit(0 if scores else 1)

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
