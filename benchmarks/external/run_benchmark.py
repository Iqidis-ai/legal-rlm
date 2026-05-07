"""Run external benchmarks against Irys RLM.

Loads downloaded JSONL data, runs each example through Irys, scores outputs,
and reports aggregate metrics.

Usage:
    python benchmarks/external/run_benchmark.py --benchmark cuad --split test --limit 50
    python benchmarks/external/run_benchmark.py --benchmark longbench_v2 --split train
    python benchmarks/external/run_benchmark.py --benchmark docfinqa --split test --limit 100
    python benchmarks/external/run_benchmark.py --benchmark facts_grounding --split public
    python benchmarks/external/run_benchmark.py --all --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irys.api import Irys, IrysConfig

logger = logging.getLogger("ext_benchmark")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_dotenv():
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


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_benchmark(name: str, split: str, limit: int | None = None) -> list[dict]:
    data_dir = DATA_DIR / name
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Benchmark data not found: {data_dir}. "
            f"Run: python benchmarks/external/download_benchmarks.py --benchmark {name}"
        )
    candidates = list(data_dir.glob(f"{split}*.jsonl"))
    if not candidates:
        candidates = list(data_dir.glob(f"*_{split}.jsonl"))
    if not candidates:
        available = sorted(set(f.stem for f in data_dir.glob("*.jsonl")))
        raise FileNotFoundError(
            f"Split '{split}' not found for {name}. Available: {available}"
        )
    rows = []
    for f in sorted(candidates):
        rows.extend(_load_jsonl(f, limit=limit - len(rows) if limit else None))
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Benchmark adapters — each converts raw data → (query, context_text, expected)
# ---------------------------------------------------------------------------

def _adapt_cuad(row: dict) -> tuple[str, str, str]:
    question = row["question"]
    context = row["context"]
    answers = row["answers"]
    if isinstance(answers, str):
        answers = eval(answers)  # stored as repr'd dict
    expected_texts = answers.get("text", [])
    expected = expected_texts[0] if expected_texts else ""
    return question, context, expected


def _adapt_docfinqa(row: dict) -> tuple[str, str, str]:
    question = row["Question"]
    context = row["Context"]
    expected = str(row["Answer"])
    return question, context, expected


def _adapt_longbench_v2(row: dict) -> tuple[str, str, str]:
    context = row.get("context", "")
    question = row["question"]
    choices = []
    for letter in ("A", "B", "C", "D"):
        choice = row.get(f"choice_{letter}", "")
        if choice:
            choices.append(f"{letter}. {choice}")
    if choices:
        question = question + "\n\nChoices:\n" + "\n".join(choices)
    expected = row["answer"]
    return question, context, expected


def _adapt_facts_grounding(row: dict) -> tuple[str, str, str]:
    question = row.get("user_request", "")
    context = row.get("context_document", "")
    expected = ""  # no reference answer — scored by LLM judge
    return question, context, expected


def _adapt_legalbench(row: dict) -> tuple[str, str, str]:
    question = row.get("text", "") or row.get("question", "") or row.get("input", "")
    context = row.get("passage", "") or row.get("context", "")
    expected = row.get("answer", "") or row.get("label", "")
    return question, context, str(expected)


ADAPTERS = {
    "cuad": _adapt_cuad,
    "docfinqa": _adapt_docfinqa,
    "longbench_v2": _adapt_longbench_v2,
    "facts_grounding": _adapt_facts_grounding,
    "legalbench": _adapt_legalbench,
}


# ---------------------------------------------------------------------------
# Scorers — each returns (score: float, detail: str) for a single example
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _f1_token(pred: str, gold: str) -> float:
    pred_tokens = set(_normalize(pred).split())
    gold_tokens = set(_normalize(gold).split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_cuad(output: str, expected: str) -> tuple[float, str]:
    if not expected:
        has_content = len(output.strip()) > 10
        return (0.0 if has_content else 1.0), "no-answer check"
    norm_expected = _normalize(expected)
    norm_output = _normalize(output)
    if norm_expected in norm_output:
        return 1.0, "contains_expected"
    exp_tokens = norm_expected.split()
    if len(exp_tokens) >= 2:
        best_f1 = 0.0
        words = norm_output.split()
        window = len(exp_tokens) * 3
        for start in range(0, max(1, len(words) - window + 1), max(1, len(exp_tokens))):
            chunk = " ".join(words[start:start + window])
            f1 = _f1_token(chunk, expected)
            best_f1 = max(best_f1, f1)
        if best_f1 >= 0.5:
            return best_f1, f"window_f1={best_f1:.3f}"
    f1 = _f1_token(output, expected)
    return f1, f"f1={f1:.3f}"


def score_docfinqa(output: str, expected: str) -> tuple[float, str]:
    numbers_in_output = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", output)
    expected_clean = expected.strip().replace(",", "")
    for num_str in numbers_in_output:
        num_clean = num_str.replace(",", "")
        try:
            if abs(float(num_clean) - float(expected_clean)) < 0.01:
                return 1.0, f"exact_match: {num_str}"
        except ValueError:
            continue
    if expected_clean.lower() in _normalize(output):
        return 1.0, "string_match"
    return 0.0, "no_match"


def score_longbench_v2(output: str, expected: str) -> tuple[float, str]:
    expected = expected.strip().upper()
    output_upper = output.strip().upper()
    if output_upper.startswith(expected):
        return 1.0, "exact"
    if re.search(rf"\b{expected}\b", output_upper):
        return 1.0, "found_letter"
    for line in output.split("\n"):
        line = line.strip()
        if line.upper().startswith(f"{expected}.") or line.upper().startswith(f"({expected})"):
            return 1.0, "choice_prefix"
    return 0.0, "no_match"


def score_facts_grounding(output: str, expected: str) -> tuple[float, str]:
    return -1.0, "requires_llm_judge"


def score_legalbench(output: str, expected: str) -> tuple[float, str]:
    output_norm = _normalize(output)
    expected_norm = _normalize(expected)
    if expected_norm in output_norm:
        return 1.0, "contains_expected"
    if output_norm.startswith(expected_norm):
        return 1.0, "starts_with"
    return 0.0, "no_match"


SCORERS = {
    "cuad": score_cuad,
    "docfinqa": score_docfinqa,
    "longbench_v2": score_longbench_v2,
    "facts_grounding": score_facts_grounding,
    "legalbench": score_legalbench,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_single(
    irys: Irys,
    query: str,
    context: str,
    tmp_base: Path,
    idx: int,
) -> str:
    repo_dir = tmp_base / f"example_{idx}"
    repo_dir.mkdir(parents=True, exist_ok=True)
    if context:
        ctx_file = repo_dir / "context.txt"
        ctx_file.write_text(context, encoding="utf-8")
    try:
        result = await irys.investigate(
            query=query,
            repository=str(repo_dir),
            research_mode="simple",
        )
        return result.output or ""
    except Exception as e:
        logger.error("Example %d failed: %s", idx, e)
        return f"[ERROR] {e}"
    finally:
        irys.close_matter_model(str(repo_dir))


async def run_benchmark(
    name: str,
    split: str,
    limit: int | None = None,
    api_key: str | None = None,
) -> dict:
    logger.info("Loading %s/%s (limit=%s)...", name, split, limit)
    rows = load_benchmark(name, split, limit)
    logger.info("Loaded %d examples", len(rows))

    adapter = ADAPTERS[name]
    scorer = SCORERS[name]

    irys = Irys(IrysConfig(
        api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        max_depth=8,
        max_leads_per_level=12,
        output_format="markdown",
        enable_matter_model=True,
    ))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / name / f"{split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_score = 0.0
    scored_count = 0

    with tempfile.TemporaryDirectory(prefix=f"irys_bench_{name}_") as tmp:
        tmp_base = Path(tmp)
        for i, row in enumerate(rows):
            query, context, expected = adapter(row)
            if not query:
                logger.warning("Example %d has no query, skipping", i)
                continue

            t0 = time.monotonic()
            output = await run_single(irys, query, context, tmp_base, i)
            elapsed = time.monotonic() - t0

            score, detail = scorer(output, expected)
            entry = {
                "idx": i,
                "query": query[:200],
                "expected": expected[:200],
                "output": output[:500],
                "score": score,
                "detail": detail,
                "elapsed": round(elapsed, 1),
            }
            results.append(entry)

            if score >= 0:
                total_score += score
                scored_count += 1

            status = "PASS" if score >= 0.5 else ("SKIP" if score < 0 else "FAIL")
            logger.info(
                "[%d/%d] %s  score=%.2f  (%.1fs)  %s",
                i + 1, len(rows), status, score, elapsed, detail,
            )

            # Write incremental results
            with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    irys.close_all_matter_models()

    avg_score = total_score / scored_count if scored_count > 0 else 0.0
    pass_count = sum(1 for r in results if r["score"] >= 0.5)

    summary = {
        "benchmark": name,
        "split": split,
        "timestamp": ts,
        "total_examples": len(rows),
        "scored": scored_count,
        "avg_score": round(avg_score, 4),
        "pass_rate": round(pass_count / scored_count, 4) if scored_count else 0,
        "pass_count": pass_count,
        "score_distribution": dict(Counter(
            "pass" if r["score"] >= 0.5 else ("skip" if r["score"] < 0 else "fail")
            for r in results
        )),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "\n%s/%s — Score: %.1f%%  Pass: %d/%d  (results: %s)",
        name, split, avg_score * 100, pass_count, scored_count, out_dir,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run external benchmarks against Irys")
    parser.add_argument("--benchmark", choices=list(ADAPTERS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _load_dotenv()

    if args.all:
        benchmarks = list(ADAPTERS.keys())
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        parser.print_help()
        return

    summaries = []
    for name in benchmarks:
        split = args.split
        # FACTS Grounding uses "public" not "test"
        if name == "facts_grounding" and split == "test":
            split = "public"
        # LongBench v2 only has "train" split
        if name == "longbench_v2" and split == "test":
            split = "train"

        try:
            summary = asyncio.run(run_benchmark(
                name=name,
                split=split,
                limit=args.limit,
                api_key=args.api_key,
            ))
            summaries.append(summary)
        except FileNotFoundError as e:
            logger.error("SKIP %s: %s", name, e)
        except Exception as e:
            logger.error("FAIL %s: %s", name, e)

    if summaries:
        print("\n" + "=" * 60)
        print("EXTERNAL BENCHMARK SUMMARY")
        print("=" * 60)
        for s in summaries:
            print(f"  {s['benchmark']}/{s['split']}: "
                  f"{s['avg_score']*100:.1f}% avg, "
                  f"{s['pass_count']}/{s['scored']} pass "
                  f"({s['pass_rate']*100:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
