#!/usr/bin/env python3
"""One-shot migration: facts.jsonl -> facts.db.

Usage:
    python scripts/migrate_fact_store.py /path/to/matter_directory
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.fact_store import FactStore


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <matter_directory>", file=sys.stderr)
        sys.exit(1)

    matter_dir = Path(sys.argv[1]).resolve()
    jsonl_path = matter_dir / ".irys" / "facts.jsonl"

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    store = FactStore(matter_dir)
    count = store.migrate_from_jsonl(jsonl_path)
    print(f"Migration complete: {count} facts written to {matter_dir / '.irys' / 'facts.db'}")
    s = store.stats()
    print(f"Store stats: {s.total_facts} total, {s.draft_facts} draft, "
          f"{s.validated_facts} validated, {s.core_facts} core")


if __name__ == "__main__":
    main()
