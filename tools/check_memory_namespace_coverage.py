"""Check Memory Coordination Protocol namespace coverage.

This intentionally stays small: it verifies that every table declared in the
MatterModel schema appears in the protocol namespace matrix, and that every
namespace used by the matrix is listed in the required namespace list.
"""

from __future__ import annotations

import re
import sys
import json
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "src" / "irys" / "matter" / "schema.py"
PROTOCOL_PATH = ROOT / "MEMORY_COORDINATION_PROTOCOL.md"
MANIFEST_PATH = ROOT / "memory_surface_manifest.json"


TABLE_RE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)
ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
CODE_RE = re.compile(r"`([^`]+)`")
CACHE_STAGE_RE = re.compile(
    r"_BROKER_REQUIRED_STAGES\s*=\s*frozenset\(\s*\{([^}]+)\}\s*\)",
    re.S,
)
STRING_RE = re.compile(r"['\"]([^'\"]+)['\"]")
MUTATION_PREFIXES = (
    "add",
    "answer",
    "apply",
    "bump",
    "candidate",
    "correct",
    "delete",
    "enqueue",
    "link",
    "mark",
    "merge",
    "record",
    "reject",
    "resolve",
    "set",
    "stage",
    "update",
    "upsert",
    "verify",
)
WRITER_CLASS_NAMES = {"MatterModel", "MatterRuntimeAdapter"}
EXCLUDED_WRITER_SURFACES = {
    "MatterModel.open",
    "MatterModel.open_in_memory",
}
MUTATING_SQL_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|REPLACE|ALTER|CREATE|DROP)\b", re.I
)
INSERT_TABLE_RE = re.compile(
    r"\b(?:INSERT|REPLACE)\s+(?:OR\s+[A-Z_]+\s+)?INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.I,
)
UPDATE_TABLE_RE = re.compile(
    r"\bUPDATE\s+(?:OR\s+[A-Z_]+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.I,
)
DELETE_TABLE_RE = re.compile(
    r"\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.I,
)
CREATE_TABLE_RE = re.compile(
    r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.I,
)
MUTATING_EXACT_CALLS = {
    "answer_question",
    "compute_all",
    "compute_and_store",
    "compute_thresholds",
}
WRITER_SOURCE_PATHS = (
    ROOT / "src" / "irys" / "matter" / "belief_revision.py",
    ROOT / "src" / "irys" / "matter" / "graph.py",
    ROOT / "src" / "irys" / "matter" / "matter.py",
    ROOT / "src" / "irys" / "matter" / "runtime.py",
)


def schema_tables(schema_text: str) -> set[str]:
    return set(TABLE_RE.findall(schema_text))


def required_namespaces(protocol_text: str) -> set[str]:
    try:
        required_block = protocol_text.split("Required namespaces:", 1)[1].split(
            "Every canonical table", 1
        )[0]
    except IndexError as exc:
        raise ValueError("Could not find Required namespaces block") from exc

    namespaces: set[str] = set()
    for line in required_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            namespaces.add(stripped[2:].strip())
    return namespaces


def matrix_tables_and_namespaces(protocol_text: str) -> tuple[set[str], set[str]]:
    try:
        matrix_block = protocol_text.split("### Namespace Coverage Matrix", 1)[1].split(
            "Legacy canonical writes", 1
        )[0]
    except IndexError as exc:
        raise ValueError("Could not find Namespace Coverage Matrix block") from exc

    tables: set[str] = set()
    namespaces: set[str] = set()

    for line in matrix_block.splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        left, right = match.groups()
        if left.strip() in {"Current table/read surface", "---"}:
            continue
        tables.update(CODE_RE.findall(left))
        for code in CODE_RE.findall(right):
            if ":" in code:
                namespaces.add(code.split(":", 1)[0])

    return tables, namespaces


def matrix_table_namespace_map(protocol_text: str) -> dict[str, set[str]]:
    try:
        matrix_block = protocol_text.split("### Namespace Coverage Matrix", 1)[1].split(
            "Legacy canonical writes", 1
        )[0]
    except IndexError as exc:
        raise ValueError("Could not find Namespace Coverage Matrix block") from exc

    table_namespaces: dict[str, set[str]] = {}
    for line in matrix_block.splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        left, right = match.groups()
        if left.strip() in {"Current table/read surface", "---"}:
            continue
        namespaces = {
            code.split(":", 1)[0]
            for code in CODE_RE.findall(right)
            if ":" in code
        }
        for table in CODE_RE.findall(left):
            table_namespaces.setdefault(table, set()).update(namespaces)
    return table_namespaces


def object_taint_kinds(protocol_text: str) -> set[str]:
    try:
        taint_block = protocol_text.split("Add object-taint tracking for:", 1)[1].split(
            "Object taint records:", 1
        )[0]
    except IndexError as exc:
        raise ValueError("Could not find object-taint tracking block") from exc

    kinds: set[str] = set()
    for line in taint_block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        kinds.add(stripped[2:].strip().rstrip(";."))
    return kinds


def broker_required_cache_stages() -> set[str]:
    graph_text = (ROOT / "src" / "irys" / "matter" / "graph.py").read_text(
        encoding="utf-8"
    )
    match = CACHE_STAGE_RE.search(graph_text)
    if not match:
        return set()
    return set(STRING_RE.findall(match.group(1)))


def actual_inference_usage_labels() -> set[str]:
    sys.path.insert(0, str(ROOT / "tools"))
    from list_inference_calls import SRC, iter_calls  # type: ignore

    labels: set[str] = set()
    for path in SRC.rglob("*.py"):
        for _lineno, _func, usage_label, _tier in iter_calls(path):
            if usage_label and usage_label.startswith("'") and usage_label.endswith("'"):
                labels.add(usage_label.strip("'"))
    return labels


def iter_surface_entries(manifest: dict, surface_key: str, group_key: str) -> list[dict]:
    entries = list(manifest.get(surface_key, []))
    for group in manifest.get(group_key, []):
        for name in group.get("surfaces", []):
            expanded = dict(group)
            expanded.pop("surfaces", None)
            expanded["surface"] = name
            entries.append(expanded)
    return entries


def missing_surface_sources(manifest: dict) -> list[str]:
    missing: list[str] = []
    for label, entries in (
        ("read_surfaces", iter_surface_entries(manifest, "read_surfaces", "read_surface_groups")),
        ("write_surfaces", iter_surface_entries(manifest, "write_surfaces", "write_surface_groups")),
    ):
        for surface in entries:
            source = surface.get("source")
            name = str(surface.get("surface") or "")
            path = ROOT / source if source else None
            if path is None or not path.exists():
                missing.append(f"{label}:{name}:missing_source:{source}")
                continue
            if not surface_symbol_exists(path, name):
                missing.append(f"{label}:{name}:missing_symbol:{source}")
    return missing


def surface_symbol_exists(path: Path, name: str) -> bool:
    parts = name.split(".")
    if len(parts) == 1:
        return any(
            isinstance(node, ast.ClassDef) and node.name == parts[0]
            for node in _parse_ast(path).body
        )
    if len(parts) != 2:
        return False
    class_name, method_name = parts
    for node in _parse_ast(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        return any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == method_name
            for child in node.body
        )
    return False


def missing_legacy_metadata(manifest: dict) -> list[str]:
    missing: list[str] = []
    defaults = manifest.get("legacy_surface_defaults", {})
    for surface in iter_surface_entries(manifest, "write_surfaces", "write_surface_groups"):
        if surface.get("broker_status") != "legacy_direct_writer":
            continue
        name = str(surface.get("surface") or "")
        for field in ("migration_owner", "hard_expiry", "taint_behavior", "profile_behavior"):
            if not (surface.get(field) or defaults.get(field)):
                missing.append(f"{name}:missing_{field}")
    return missing


def _parse_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def writer_method_index() -> dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    index: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for path in WRITER_SOURCE_PATHS:
        tree = _parse_ast(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[child.name] = child
            index[node.name] = methods
    return index


def writer_attribute_type_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in WRITER_SOURCE_PATHS:
        tree = _parse_ast(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            attr_types: dict[str, str] = {}
            init = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__init__"
                ),
                None,
            )
            if init is None:
                index[node.name] = attr_types
                continue
            arg_types = {
                arg.arg: annotation_name(arg.annotation)
                for arg in init.args.args
                if arg.annotation is not None
            }
            for child in ast.walk(init):
                if not isinstance(child, ast.Assign):
                    continue
                targets = [
                    target
                    for target in child.targets
                    if isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ]
                if not targets:
                    continue
                type_name: str | None = None
                if isinstance(child.value, ast.Call):
                    type_name = call_name(child.value.func)
                elif isinstance(child.value, ast.Name):
                    type_name = arg_types.get(child.value.id)
                if not type_name:
                    continue
                for target in targets:
                    attr_types[target.attr] = type_name
            index[node.name] = attr_types
    return index


def annotation_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        return annotation_name(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def method_written_tables(
    class_name: str,
    method_name: str,
    *,
    method_index: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None = None,
    attr_index: dict[str, dict[str, str]] | None = None,
    seen: set[tuple[str, str]] | None = None,
) -> set[str]:
    method_index = method_index or writer_method_index()
    attr_index = attr_index or writer_attribute_type_index()
    seen = seen or set()
    key = (class_name, method_name)
    if key in seen:
        return set()
    seen.add(key)
    method = method_index.get(class_name, {}).get(method_name)
    if method is None:
        return set()

    tables = method_direct_mutation_tables(method)
    for owner_class, delegated_method in delegated_method_calls(method, class_name, attr_index):
        tables.update(
            method_written_tables(
                owner_class,
                delegated_method,
                method_index=method_index,
                attr_index=attr_index,
                seen=seen,
            )
        )
    return tables


def method_direct_mutation_tables(node: ast.AST) -> set[str]:
    tables: set[str] = set()
    string_env = method_string_assignments(node)
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr in {"execute", "executemany", "executescript"}:
                sql = call_sql_text(child, string_env)
                if sql:
                    tables.update(sql_mutation_tables(sql))
    return tables


def sql_mutation_tables(sql: str) -> set[str]:
    stripped = sql.strip()
    if not MUTATING_SQL_RE.match(stripped):
        return set()
    if INSERT_TABLE_RE.match(stripped):
        return set(INSERT_TABLE_RE.findall(stripped))
    if UPDATE_TABLE_RE.match(stripped):
        return set(UPDATE_TABLE_RE.findall(stripped))
    if DELETE_TABLE_RE.match(stripped):
        return set(DELETE_TABLE_RE.findall(stripped))
    if CREATE_TABLE_RE.match(stripped):
        return set(CREATE_TABLE_RE.findall(stripped))
    return set()


def delegated_method_calls(
    node: ast.AST,
    current_class: str,
    attr_index: dict[str, dict[str, str]],
) -> set[tuple[str, str]]:
    calls: set[tuple[str, str]] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        owner_class = resolve_self_attribute_class(
            child.func.value,
            current_class,
            attr_index,
        )
        if owner_class is None:
            continue
        calls.add((owner_class, child.func.attr))
    return calls


def resolve_self_attribute_class(
    node: ast.AST,
    current_class: str,
    attr_index: dict[str, dict[str, str]],
) -> str | None:
    chain: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "self":
        return None
    if not chain:
        return current_class
    owner = current_class
    for attr in reversed(chain):
        next_owner = attr_index.get(owner, {}).get(attr)
        if next_owner is None:
            return None
        owner = next_owner
    return owner


def discovered_family_read_surfaces() -> set[str]:
    tree = _parse_ast(ROOT / "src" / "irys" / "rlm" / "governance.py")
    surfaces: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("FamilyHandler"):
            continue
        if any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "run"
            for child in node.body
        ):
            surfaces.add(f"{node.name}.run")
    return surfaces


def discovered_writer_surfaces() -> set[str]:
    surfaces: set[str] = set()
    for path in (
        ROOT / "src" / "irys" / "matter" / "graph.py",
        ROOT / "src" / "irys" / "matter" / "matter.py",
        ROOT / "src" / "irys" / "matter" / "runtime.py",
    ):
        tree = _parse_ast(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not (node.name.endswith("Store") or node.name in WRITER_CLASS_NAMES):
                continue
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_"):
                    continue
                if not method_has_mutation_evidence(child):
                    continue
                surface = f"{node.name}.{child.name}"
                if surface in EXCLUDED_WRITER_SURFACES:
                    continue
                surfaces.add(surface)
    return surfaces


def method_has_mutation_evidence(node: ast.AST) -> bool:
    name = getattr(node, "name", "")
    name_suggests_mutation = (
        name in MUTATING_EXACT_CALLS or name.startswith(MUTATION_PREFIXES)
    )
    string_env = method_string_assignments(node)
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            attr = child.func.attr
            if attr in {"execute", "executemany", "executescript"}:
                sql = call_sql_text(child, string_env)
                if sql and MUTATING_SQL_RE.search(sql):
                    return True
            if (
                name_suggests_mutation
                and call_is_self_delegation(child)
                and (
                attr in MUTATING_EXACT_CALLS or attr.startswith(MUTATION_PREFIXES)
                )
            ):
                return True
    return False


def call_is_self_delegation(call: ast.Call) -> bool:
    value = call.func.value if isinstance(call.func, ast.Attribute) else None
    while isinstance(value, ast.Attribute):
        value = value.value
    return isinstance(value, ast.Name) and value.id == "self"


def method_string_assignments(node: ast.AST) -> dict[str, str]:
    env: dict[str, str] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        text = literal_string(child.value)
        if text is None:
            continue
        for target in child.targets:
            if isinstance(target, ast.Name):
                env[target.id] = text
    return env


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            item.value
            for item in node.values
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = literal_string(node.left)
        right = literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return literal_string(node.func.value)
    return None


def call_sql_text(call: ast.Call, string_env: dict[str, str] | None = None) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    literal = literal_string(first)
    if literal is not None:
        return literal
    if isinstance(first, ast.Name) and string_env and first.id in string_env:
        return string_env[first.id]
    return None


def missing_discovered_surfaces(manifest: dict) -> tuple[list[str], list[str]]:
    manifest_reads = {
        str(surface.get("surface") or "")
        for surface in iter_surface_entries(manifest, "read_surfaces", "read_surface_groups")
    }
    manifest_writes = {
        str(surface.get("surface") or "")
        for surface in iter_surface_entries(manifest, "write_surfaces", "write_surface_groups")
    }
    return (
        sorted(discovered_family_read_surfaces() - manifest_reads),
        sorted(discovered_writer_surfaces() - manifest_writes),
    )


def writer_surface_tables(surface: str) -> set[str]:
    parts = surface.split(".")
    if len(parts) != 2:
        return set()
    method_index = writer_method_index()
    attr_index = writer_attribute_type_index()
    return method_written_tables(
        parts[0],
        parts[1],
        method_index=method_index,
        attr_index=attr_index,
    )


def writer_surface_namespaces(
    surface: str,
    table_namespaces: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    tables = writer_surface_tables(surface)
    namespaces: set[str] = set()
    unmapped_tables: set[str] = set()
    for table in tables:
        mapped = table_namespaces.get(table)
        if mapped:
            namespaces.update(mapped)
        else:
            unmapped_tables.add(table)
    return namespaces, unmapped_tables


def writer_namespace_mismatches(manifest: dict, protocol_text: str) -> list[str]:
    table_namespaces = matrix_table_namespace_map(protocol_text)
    issues: list[str] = []
    for surface in iter_surface_entries(manifest, "write_surfaces", "write_surface_groups"):
        name = str(surface.get("surface") or "")
        if not name:
            continue
        expected, unmapped_tables = writer_surface_namespaces(name, table_namespaces)
        declared = set(surface.get("writes", []))
        if unmapped_tables:
            issues.append(f"{name}:unmapped_tables:{','.join(sorted(unmapped_tables))}")
        if not expected:
            issues.append(f"{name}:no_detected_write_tables")
            continue
        extra = sorted(declared - expected)
        missing = sorted(expected - declared)
        if extra:
            issues.append(f"{name}:extra:{','.join(extra)}")
        if missing:
            issues.append(f"{name}:missing:{','.join(missing)}")
    return sorted(issues)


def main() -> int:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    tables = schema_tables(schema_text)
    matrix_tables, matrix_namespaces = matrix_tables_and_namespaces(protocol_text)
    required = required_namespaces(protocol_text)
    protocol_taint = object_taint_kinds(protocol_text)
    broker_cache_stages = broker_required_cache_stages()
    actual_usage_labels = actual_inference_usage_labels()

    missing_tables = sorted(tables - matrix_tables)
    unknown_namespaces = sorted(matrix_namespaces - required)
    unused_required = sorted(required - matrix_namespaces)
    manifest_namespaces = set()
    for surface in iter_surface_entries(manifest, "read_surfaces", "read_surface_groups"):
        manifest_namespaces.update(surface.get("dependencies", []))
    for surface in iter_surface_entries(manifest, "write_surfaces", "write_surface_groups"):
        manifest_namespaces.update(surface.get("writes", []))
    manifest_namespaces.update(manifest.get("domain_profile_namespaces", []))
    missing_manifest_namespaces = sorted(manifest_namespaces - required)
    missing_sources = missing_surface_sources(manifest)
    missing_legacy = missing_legacy_metadata(manifest)
    missing_discovered_reads, missing_discovered_writes = missing_discovered_surfaces(manifest)
    writer_mismatches = writer_namespace_mismatches(manifest, protocol_text)

    manifest_taint = set(manifest.get("object_taint_kinds", []))
    missing_taint_kinds = sorted(manifest_taint - protocol_taint)

    manifest_semantic_stages = {
        item["stage"] for item in manifest.get("semantic_cache_stages", [])
    }
    missing_cache_guards = sorted(manifest_semantic_stages - broker_cache_stages)

    registry_labels = set(manifest.get("inference_call_registry", []))
    missing_registry_entries = sorted(actual_usage_labels - registry_labels)
    stale_registry_entries = sorted(registry_labels - actual_usage_labels)

    if (
        missing_tables
        or unknown_namespaces
        or missing_manifest_namespaces
        or missing_taint_kinds
        or missing_cache_guards
        or missing_registry_entries
        or missing_sources
        or missing_legacy
        or missing_discovered_reads
        or missing_discovered_writes
        or writer_mismatches
    ):
        if missing_tables:
            print("Missing schema tables in namespace matrix:")
            for table in missing_tables:
                print(f"  - {table}")
        if unknown_namespaces:
            print("Namespaces used by matrix but absent from required list:")
            for namespace in unknown_namespaces:
                print(f"  - {namespace}")
        if unused_required:
            print("Required namespaces not currently used by matrix:")
            for namespace in unused_required:
                print(f"  - {namespace}")
        if missing_manifest_namespaces:
            print("Manifest namespaces absent from required list:")
            for namespace in missing_manifest_namespaces:
                print(f"  - {namespace}")
        if missing_taint_kinds:
            print("Manifest object-taint kinds absent from protocol:")
            for kind in missing_taint_kinds:
                print(f"  - {kind}")
        if missing_cache_guards:
            print("Semantic cache stages missing broker guards:")
            for stage in missing_cache_guards:
                print(f"  - {stage}")
        if missing_registry_entries:
            print("Inference usage labels missing registry entries:")
            for label in missing_registry_entries:
                print(f"  - {label}")
        if missing_sources:
            print("Manifest surfaces missing source files or symbols:")
            for missing in missing_sources:
                print(f"  - {missing}")
        if missing_legacy:
            print("Legacy write surfaces missing migration metadata:")
            for missing in missing_legacy:
                print(f"  - {missing}")
        if missing_discovered_reads:
            print("Discovered user-facing read surfaces absent from manifest:")
            for surface in missing_discovered_reads:
                print(f"  - {surface}")
        if missing_discovered_writes:
            print("Discovered canonical write surfaces absent from manifest:")
            for surface in missing_discovered_writes:
                print(f"  - {surface}")
        if writer_mismatches:
            print("Manifest writer namespaces do not match derived write tables:")
            for mismatch in writer_mismatches:
                print(f"  - {mismatch}")
        if stale_registry_entries:
            print("Registry entries without current call sites:")
            for label in stale_registry_entries:
                print(f"  - {label}")
        return 1

    print(
        "OK: "
        f"{len(tables)} schema tables covered; "
        f"{len(matrix_namespaces)} matrix namespaces are required; "
        f"{len(iter_surface_entries(manifest, 'read_surfaces', 'read_surface_groups'))} read surfaces checked; "
        f"{len(iter_surface_entries(manifest, 'write_surfaces', 'write_surface_groups'))} write surfaces checked; "
        f"{len(actual_usage_labels)} inference labels registered."
    )
    if unused_required:
        print("Unused required namespaces:")
        for namespace in unused_required:
            print(f"  - {namespace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
