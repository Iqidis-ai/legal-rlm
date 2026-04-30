"""List inference-call sites for broker migration planning.

The memory broker cannot be enforced until model/external-content calls are
visible. This script gives a deterministic first inventory of Python call sites
that invoke `.complete(...)`, including literal `usage_label` and `tier` values
when they can be read from the AST.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def literal_keyword(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        value = keyword.value
        if isinstance(value, ast.Constant):
            return repr(value.value)
        dotted = dotted_name(value)
        if dotted:
            return dotted
        return ast.unparse(value)
    return None


def is_inference_complete_call(path: Path, func: str, usage_label: str | None, tier: str | None) -> bool:
    if func in {"self.client.complete", "self._client.complete", "client.complete"}:
        return True
    if path.name == "models.py" and func == "self.complete":
        return True
    return bool(usage_label or tier)


def iter_calls(path: Path) -> list[tuple[int, str, str | None, str | None]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []

    found: list[tuple[int, str, str | None, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = dotted_name(node.func)
        if not func or not func.endswith(".complete"):
            continue
        usage_label = literal_keyword(node, "usage_label")
        tier = literal_keyword(node, "tier")
        if not is_inference_complete_call(path, func, usage_label, tier):
            continue
        found.append(
            (
                node.lineno,
                func,
                usage_label,
                tier,
            )
        )
    return sorted(found)


def main() -> int:
    rows: list[tuple[str, int, str, str | None, str | None]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for lineno, func, usage_label, tier in iter_calls(path):
            rows.append((rel, lineno, func, usage_label, tier))

    print("| file | line | call | usage_label | tier |")
    print("| --- | ---: | --- | --- | --- |")
    for rel, lineno, func, usage_label, tier in rows:
        print(
            f"| `{rel}` | {lineno} | `{func}` | "
            f"{usage_label or ''} | {tier or ''} |"
        )
    print(f"\nTotal `.complete(...)` call sites: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
