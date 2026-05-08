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
    from docx.shared import Pt
    import re

    doc = Document()
    try:
        style = doc.styles["Normal"]
        style.font.size = Pt(11)
        style.font.name = "Calibri"
    except (KeyError, AttributeError):
        pass

    def _add_styled_para(text, style_name=None):
        try:
            return doc.add_paragraph(text, style=style_name)
        except (KeyError, AttributeError):
            return doc.add_paragraph(text)

    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            doc.add_paragraph("")
            i += 1
            continue
        if "|" in stripped and stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                if re.match(r'^\|[\s\-:|]+\|$', row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.split("|")[1:-1]]
                table_lines.append(cells)
                i += 1
            if table_lines:
                n_cols = max(len(r) for r in table_lines)
                table = doc.add_table(rows=len(table_lines), cols=n_cols)
                try:
                    table.style = "Table Grid"
                except (KeyError, AttributeError):
                    pass
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
            _add_styled_para(stripped[2:], "List Bullet")
        elif re.match(r'^\d+\.\s', stripped):
            _add_styled_para(re.sub(r'^\d+\.\s', '', stripped), "List Number")
        elif stripped.startswith("**") and stripped.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(stripped.strip("*"))
            run.bold = True
        elif stripped.startswith("> "):
            _add_styled_para(stripped[2:], "Quote")
        else:
            doc.add_paragraph(stripped)
        i += 1

    doc.save(str(output_path))


def _markdown_to_xlsx(md_text: str, output_path: Path):
    """Convert markdown tables to an Excel workbook.

    Extracts all markdown tables from the text and writes each as a sheet.
    Uses the nearest preceding heading as the sheet name.
    Non-table text goes into a 'Summary' sheet.
    """
    from openpyxl import Workbook
    import re

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary"

    lines = md_text.split("\n")
    summary_lines: list[str] = []
    table_count = 0
    last_heading = ""
    used_names: set[str] = set()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        heading_match = re.match(r'^#{1,4}\s+(.+)', stripped)
        if heading_match:
            last_heading = heading_match.group(1).strip().rstrip("#").strip()
            summary_lines.append(stripped)
            i += 1
            continue
        if "|" in stripped and stripped.startswith("|"):
            table_rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                if re.match(r'^\|[\s\-:|]+\|$', row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.split("|")[1:-1]]
                table_rows.append(cells)
                i += 1
            if table_rows:
                table_count += 1
                raw_name = last_heading if last_heading else f"Table {table_count}"
                sheet_name = re.sub(r'[\\/*?\[\]:]', '', raw_name)[:31]
                if sheet_name in used_names:
                    sheet_name = f"{sheet_name[:27]} ({table_count})"
                used_names.add(sheet_name)
                if table_count == 1:
                    ws = ws_summary
                    ws.title = sheet_name
                else:
                    ws = wb.create_sheet(title=sheet_name)
                for ri, row_cells in enumerate(table_rows, 1):
                    for ci, cell_text in enumerate(row_cells, 1):
                        cell = ws.cell(row=ri, column=ci, value=cell_text)
                        try:
                            num = float(cell_text.replace(",", "").replace("$", "").replace("%", "").strip())
                            cell.value = num
                        except (ValueError, AttributeError):
                            pass
        else:
            summary_lines.append(stripped)
            i += 1

    if not table_count:
        for ri, line in enumerate(summary_lines, 1):
            ws_summary.cell(row=ri, column=1, value=line)
    elif summary_lines:
        if table_count >= 1:
            ws_text = wb.create_sheet(title="Summary", index=0)
        else:
            ws_text = ws_summary
        for ri, line in enumerate(summary_lines, 1):
            ws_text.cell(row=ri, column=1, value=line)

    wb.save(str(output_path))


def _split_output_by_deliverable(
    output_text: str, deliverables: dict[str, str],
) -> dict[str, str]:
    """Split a single synthesis output into per-deliverable sections.

    Looks for markdown headings that match deliverable names/keys and splits
    the output accordingly. Falls back to the full output for any deliverable
    without a matching section.
    """
    import re as _re
    if len(deliverables) <= 1:
        return {name: output_text for name in deliverables}

    result: dict[str, str] = {}
    stem_map: dict[str, str] = {}
    for name, filename in deliverables.items():
        stem = Path(filename).stem.lower().replace("-", " ").replace("_", " ")
        stem_map[name] = stem

    headings = list(_re.finditer(r'^(#{1,3})\s+(.+)$', output_text, _re.MULTILINE))
    if not headings:
        return {name: output_text for name in deliverables}

    def _fuzzy_match(heading_text: str, stem: str) -> bool:
        ht = heading_text.lower().replace("-", " ").replace("_", " ")
        stem_words = stem.split()
        return sum(1 for w in stem_words if w in ht) >= max(1, len(stem_words) // 2)

    # A deliverable section runs from its boundary heading to the NEXT
    # heading at the same or shallower depth. Sub-headings (deeper levels)
    # are part of the same section. Without this, "# Title" followed by
    # "## Subhead" closes the section after only the title line.
    sections: list[tuple[str, int, int, int]] = []
    for i, match in enumerate(headings):
        level = len(match.group(1))
        start = match.start()
        end = len(output_text)
        for j in range(i + 1, len(headings)):
            next_level = len(headings[j].group(1))
            if next_level <= level:
                end = headings[j].start()
                break
        sections.append((match.group(2).strip(), start, end, level))

    for name, stem in stem_map.items():
        best_section = None
        # Prefer the shallowest matching heading (a deliverable boundary
        # is typically `#` level-1, not a buried subsection).
        candidates = [
            (level, start, end)
            for heading_text, start, end, level in sections
            if _fuzzy_match(heading_text, stem)
        ]
        if candidates:
            candidates.sort(key=lambda c: (c[0], c[1]))
            _, s, e = candidates[0]
            best_section = output_text[s:e].strip()
        result[name] = best_section if best_section else output_text

    return result


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

    # Append deliverable metadata so synthesis knows what to produce
    if deliverables and len(deliverables) > 1:
        deliverable_desc = ", ".join(
            f"`{fn}`" for fn in deliverables.values()
        )
        xlsx_files = [fn for fn in deliverables.values() if fn.endswith((".xlsx", ".xls"))]
        instructions += (
            f"\n\nYou must produce content for {len(deliverables)} separate deliverables: "
            f"{deliverable_desc}. Structure your output with a clear top-level "
            f"heading (# or ##) for each deliverable so they can be separated."
        )
        if xlsx_files:
            instructions += (
                "\n\nCRITICAL — XLSX WORKBOOK REQUIREMENTS:\n"
                "For each .xlsx deliverable, you must produce MULTIPLE markdown tables, "
                "each preceded by a ### heading that names the worksheet tab. "
                "Each table must use | Column | Header | format with data rows below.\n"
            )
            for xf in xlsx_files:
                stem = Path(xf).stem.replace("-", " ").replace("_", " ").title()
                instructions += (
                    f"- `{xf}`: Under the `# {stem}` section, produce at least 3 tables "
                    f"with ### headings for each tab. Each table needs 5+ data rows with "
                    f"actual numbers from the source documents.\n"
                )

    # Inject criteria as quality requirements so the engine knows what
    # evaluators expect. Each criterion title becomes a coverage target.
    criteria = task.get("criteria", [])
    if criteria:
        criteria_lines = []
        for c in criteria:
            title = c.get("title", "")
            if title:
                criteria_lines.append(f"- {title}")
        if criteria_lines:
            instructions += (
                "\n\nQUALITY REQUIREMENTS — your output will be evaluated on "
                f"these {len(criteria_lines)} criteria. Ensure your analysis "
                "explicitly addresses EACH one:\n"
                + "\n".join(criteria_lines)
            )

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

        # Extract the raw synthesis output (final_output) for deliverables.
        # The formatted output wraps synthesis with investigation metadata
        # (Key Findings, Citations, Entities) that shouldn't be in deliverables.
        synthesis_text = ""
        if hasattr(result, "state") and result.state:
            synthesis_text = result.state.findings.get("final_output", "")
        if not synthesis_text:
            synthesis_text = output_text

        # Write output for each expected deliverable in the expected format.
        # Use synthesis_text (raw synthesis) for deliverables, not the full
        # investigation report wrapper which contains metadata sections.
        if deliverables:
            per_deliverable = _split_output_by_deliverable(synthesis_text, deliverables)
            for name, filename in deliverables.items():
                section_text = per_deliverable.get(name, synthesis_text)
                ext = Path(filename).suffix.lower()
                if ext == ".docx":
                    docx_path = output_dir / filename
                    _markdown_to_docx(section_text, docx_path)
                elif ext in (".xlsx", ".xls"):
                    xlsx_path = output_dir / filename
                    _markdown_to_xlsx(section_text, xlsx_path)
                else:
                    out_path = output_dir / filename
                    out_path.write_text(section_text, encoding="utf-8")
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
    elif getattr(args, "tasks_file", None):
        tf_path = Path(args.tasks_file)
        if not tf_path.exists():
            print(f"Error: tasks file not found: {tf_path}")
            sys.exit(1)
        task_ids = [
            line.strip() for line in tf_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not task_ids:
            print(f"Error: tasks file empty: {tf_path}")
            sys.exit(1)
    else:
        print("Error: specify --task, --practice-area, --all, or --tasks-file")
        sys.exit(1)

    print(f"LAB root: {lab_root}")
    print(f"Tasks to run: {len(task_ids)}")
    print(f"Research mode: {args.research_mode}")
    print()

    results = []
    completed_count = 0
    sem = asyncio.Semaphore(getattr(args, "concurrency", 1))

    async def _process(idx: int, task_id: str):
        nonlocal completed_count
        async with sem:
            try:
                task = load_task(lab_root, task_id)
                result = await run_task(
                    task=task,
                    lab_root=lab_root,
                    research_mode=args.research_mode,
                    api_key=args.api_key,
                )
                if args.auto_score and result["success"]:
                    scores = score_run(
                        lab_root, result["run_id"], task_id,
                        judge_model=args.judge_model,
                    )
                    if scores:
                        result["scores"] = scores
            except Exception as e:
                result = {
                    "task_id": task_id,
                    "status": "error",
                    "success": False,
                    "error": str(e),
                }
        completed_count += 1
        results.append(result)
        status_icon = "OK" if result.get("success") else ("ERR" if result.get("status") == "error" else "FAIL")
        elapsed = result.get("elapsed", 0)
        score_str = ""
        if "scores" in result:
            s = result["scores"]
            score_str = f"  score={s['n_passed']}/{s['n_criteria']}"
        print(f"[{completed_count}/{len(task_ids)}] {task_id}  {status_icon}  {elapsed:.1f}s{score_str}")

    await asyncio.gather(*(_process(i, t) for i, t in enumerate(task_ids, 1)))

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
    group.add_argument("--tasks-file", help="Path to a file with one task_id per line")

    parser.add_argument("--lab-root", default=str(DEFAULT_LAB_ROOT),
                        help="Path to harvey-labs repo (default: ../harvey-labs)")
    parser.add_argument("--research-mode", default="deep", choices=["simple", "deep", "sebih_special"],
                        help="Irys research depth (default: deep)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    parser.add_argument("--auto-score", action="store_true", help="Score each task after running")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6", help="LLM judge for scoring")
    parser.add_argument("--run-id", help="Run ID to score (with --score)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of tasks to run in parallel (default: 1)")
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
