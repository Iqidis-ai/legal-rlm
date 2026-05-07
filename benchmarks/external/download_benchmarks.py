"""Download external benchmark datasets from HuggingFace.

Downloads to benchmarks/external/data/<name>/ as JSONL for uniform loading.

Usage:
    python benchmarks/external/download_benchmarks.py [--benchmark NAME]
    python benchmarks/external/download_benchmarks.py --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"


BENCHMARKS = {
    "legalbench": {
        "hf_id": "nguha/legalbench",
        "description": "162 legal reasoning tasks (NeurIPS 2023)",
        "metric": "balanced_accuracy",
    },
    "docfinqa": {
        "hf_id": "kensho/DocFinQA",
        "description": "7,437 expert QA pairs on SEC filings (ACL 2024)",
        "metric": "execution_accuracy",
    },
    "longbench_v2": {
        "hf_id": "THUDM/LongBench-v2",
        "description": "503 expert MC questions, 8K-2M word contexts (ACL 2025)",
        "metric": "accuracy",
    },
    "cuad": {
        "hf_id": "theatticusproject/cuad-qa",
        "description": "4,128 contract clause extraction across 41 risk categories",
        "metric": "f1",
    },
    "facts_grounding": {
        "hf_id": "google/FACTS-grounding-public",
        "description": "860 source-grounding factuality tests (Google DeepMind)",
        "metric": "llm_judge_factuality",
    },
}


def download_legalbench(out_dir: Path) -> int:
    from datasets import load_dataset, get_dataset_config_names
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        configs = get_dataset_config_names("nguha/legalbench", trust_remote_code=True)
    except Exception:
        configs = ["default"]
    for config in configs:
        try:
            ds = load_dataset("nguha/legalbench", config, trust_remote_code=True)
            for split_name, split_data in ds.items():
                outfile = out_dir / f"{config}_{split_name}.jsonl"
                with outfile.open("w", encoding="utf-8") as f:
                    for row in split_data:
                        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
                        count += 1
        except Exception as e:
            print(f"    WARN: legalbench/{config}: {e}")
    return count


def download_docfinqa(out_dir: Path) -> int:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("kensho/DocFinQA")
    count = 0
    for split_name, split_data in ds.items():
        outfile = out_dir / f"{split_name}.jsonl"
        with outfile.open("w", encoding="utf-8") as f:
            for row in split_data:
                row_dict = dict(row)
                f.write(json.dumps(row_dict, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count


def download_longbench_v2(out_dir: Path) -> int:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("THUDM/LongBench-v2")
    count = 0
    for split_name, split_data in ds.items():
        outfile = out_dir / f"{split_name}.jsonl"
        with outfile.open("w", encoding="utf-8") as f:
            for row in split_data:
                row_dict = dict(row)
                f.write(json.dumps(row_dict, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count


def download_cuad(out_dir: Path) -> int:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    count = 0
    for split_name, split_data in ds.items():
        outfile = out_dir / f"{split_name}.jsonl"
        with outfile.open("w", encoding="utf-8") as f:
            for row in split_data:
                row_dict = dict(row)
                f.write(json.dumps(row_dict, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count


def download_facts_grounding(out_dir: Path) -> int:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/FACTS-grounding-public")
    count = 0
    for split_name, split_data in ds.items():
        outfile = out_dir / f"{split_name}.jsonl"
        with outfile.open("w", encoding="utf-8") as f:
            for row in split_data:
                row_dict = dict(row)
                f.write(json.dumps(row_dict, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count


_DOWNLOADERS = {
    "legalbench": download_legalbench,
    "docfinqa": download_docfinqa,
    "longbench_v2": download_longbench_v2,
    "cuad": download_cuad,
    "facts_grounding": download_facts_grounding,
}


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if "--all" in args:
        names = list(_DOWNLOADERS.keys())
    elif "--benchmark" in args:
        idx = args.index("--benchmark")
        if idx + 1 >= len(args):
            print("ERROR: --benchmark requires a name", file=sys.stderr)
            return 2
        name = args[idx + 1]
        if name not in _DOWNLOADERS:
            print(f"ERROR: unknown benchmark '{name}'. Available: {', '.join(_DOWNLOADERS)}", file=sys.stderr)
            return 2
        names = [name]
    else:
        print("Usage: download_benchmarks.py --all | --benchmark NAME")
        print(f"Available: {', '.join(_DOWNLOADERS)}")
        return 0

    for name in names:
        out_dir = DATA_DIR / name
        if out_dir.exists() and any(out_dir.glob("*.jsonl")):
            count = sum(1 for f in out_dir.glob("*.jsonl") for _ in f.open())
            print(f"  SKIP {name}: already downloaded ({count} lines in {out_dir})")
            continue
        print(f"  Downloading {name} ({BENCHMARKS[name]['description']})...")
        try:
            count = _DOWNLOADERS[name](out_dir)
            print(f"  OK {name}: {count} examples -> {out_dir}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
