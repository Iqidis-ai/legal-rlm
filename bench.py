"""
bench.py -- MRR + Retrieval Benchmark for feat/context-store-v2

Usage:
  python bench.py                  # run all three phases
  python bench.py --phase mrr      # offline MRR only
  python bench.py --phase cold     # cold live run only
  python bench.py --phase warm     # warm live run only (requires facts.db from cold)
  python bench.py --no-wipe        # skip .irys/ wipe before cold (re-run warm)
"""

import sys, os, asyncio, time, sqlite3, shutil, re, logging, argparse
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

WAYMO_DIR  = Path(__file__).parent / "v2-dataset" / "waymo_dataset"
MANIFEST   = WAYMO_DIR / "dataset_manifest.json"

DELE004_REPO = Path(r"D:\legal-rlm\DELE-004 formatted (1)\documents")
IRYS_DIR     = DELE004_REPO / ".irys"
BASELINE_DIR = DELE004_REPO / ".irys_baseline"

QUERY = (
    "Compare the conditions precedent (Article 2) in all three agreements. "
    "For each condition: state the condition, identify which agreement(s) impose it, "
    "and call out any condition unique to one agreement."
)

MRR_BASELINE   = 0.164
MRR_PASS_FLOOR = 0.160

# ── PhaseResult ────────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ── Logging capture for warm phase ─────────────────────────────────────────────

revisit_log: list[str] = []

class _RevisitCapture(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "Skipping already extracted" in msg or "on_source_revisited" in msg.lower():
            revisit_log.append(msg)


def _setup_logging():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    handler = _RevisitCapture()
    for name in ("irys.rlm.engine", "irys.core.fact_store"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)


# ── Shared helpers (from bench_dele004.py) ─────────────────────────────────────

def read_db(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    synopses = conn.execute("SELECT source, synopsis FROM source_synopses").fetchall()
    synopsis_lines = {Path(r["source"]).stem[:30]: r["synopsis"].count("\n  - ") for r in synopses}

    total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    tiers = dict(conn.execute("SELECT tier, COUNT(*) FROM facts GROUP BY tier").fetchall())

    elevated = conn.execute(
        "SELECT source, COUNT(*) as n, AVG(importance) as avg_imp "
        "FROM facts WHERE importance > 50 GROUP BY source"
    ).fetchall()
    elevated_info = {
        Path(r["source"]).stem[:30]: {"count": r["n"], "avg_imp": round(r["avg_imp"], 1)}
        for r in elevated
    }

    conn.close()
    return {"synopsis_lines": synopsis_lines, "total_facts": total_facts,
            "tiers": tiers, "elevated_importance": elevated_info}


def snapshot_importance(db_path: Path) -> dict:
    """Return {source: avg_importance} for warm-phase delta calculation."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT source, AVG(importance) as avg_imp FROM facts GROUP BY source").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def extract_evidence_chars(state) -> int:
    prompt = state.findings.get("synthesis_prompt", "")
    if prompt:
        return len(prompt)
    evidence = state.findings.get("evidence_bundle", "")
    if evidence:
        return len(evidence)
    facts = state.findings.get("accumulated_facts", [])
    texts = [f[0] if isinstance(f, (list, tuple)) else f for f in facts]
    return sum(len(t) for t in texts)


def on_step_cb(step):
    t = step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)
    if t in ("reading", "searching", "planning", "thinking"):
        label = str(step.display or step.content or "")[:80]
        print(f"    [{t}] {label}", flush=True)


# ── Phase 1: MRR regression (offline) ─────────────────────────────────────────

_CHARS_PER_PAGE = 3000  # matches search/density-ranking-upgrade constant


def _density_score(match_count: int, page_count: int, total_chars: int) -> float:
    """Replicate engine density scoring formula from search/density-ranking-upgrade."""
    if page_count > 1:
        divisor = float(page_count)
    else:
        divisor = max(1.0, total_chars / _CHARS_PER_PAGE)
    return match_count / divisor


def phase_mrr() -> PhaseResult:
    import json

    print("\n  [mrr] Building Waymo index ...")

    if not MANIFEST.exists():
        return PhaseResult("MRR regression", False, notes=["dataset_manifest.json not found"])

    with open(MANIFEST) as f:
        manifest = json.load(f)

    from irys.core.reader import DocumentReader
    from irys.core.search import DocumentSearch

    reader = DocumentReader()
    search = DocumentSearch(reader=reader)

    # Build index: doc_id -> {file, page_count, total_chars}
    index: dict = {}
    for doc_meta in manifest["documents"]:
        doc_id = doc_meta["doc_id"]
        pdf_path = WAYMO_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue
        try:
            content = reader.read(pdf_path)
            index[doc_id] = {
                "file": pdf_path,
                "page_count": content.page_count,
                "total_chars": content.total_chars,
            }
        except Exception as exc:
            print(f"  [mrr] WARN {doc_id}: {exc}")

    print(f"  [mrr] Indexed {len(index)} PDFs. Running 20 queries ...")

    QUERIES = [
        {"qid": "Q01", "query": "What did the court order regarding Levandowski Fifth Amendment privilege?", "ground_truth": "doc_013"},
        {"qid": "Q02", "query": "What discovery did the court compel defendants to produce?", "ground_truth": "doc_002"},
        {"qid": "Q03", "query": "What did the court decide about the Stroz Friedberg subpoena?", "ground_truth": "doc_004"},
        {"qid": "Q04", "query": "Why did the court issue an order to show cause regarding withheld evidence?", "ground_truth": "doc_023"},
        {"qid": "Q05", "query": "Why did the court deny Otto Trucking permission to move for summary judgment?", "ground_truth": "doc_030"},
        {"qid": "Q06", "query": "What technical analysis did Waymo's expert provide about LiDAR sensor design?", "ground_truth": "doc_017"},
        {"qid": "Q07", "query": "How did defendants rebut the LiDAR technology claims in their expert declaration?", "ground_truth": "doc_020"},
        {"qid": "Q08", "query": "What trade secrets were identified in the LiDAR system design?", "ground_truth": "doc_017"},
        {"qid": "Q09", "query": "What were the specific technical differences in the LiDAR designs between Waymo and Uber?", "ground_truth": "doc_017"},
        {"qid": "Q10", "query": "What arguments did Waymo present for preliminary injunction relief?", "ground_truth": "doc_005"},
        {"qid": "Q11", "query": "How did Uber defend against Waymo's motion for preliminary injunction?", "ground_truth": "doc_009"},
        {"qid": "Q12", "query": "Why did Waymo seek to exclude evidence related to its public road testing?", "ground_truth": "doc_007"},
        {"qid": "Q13", "query": "What discovery issues did defendants raise in their letter brief?", "ground_truth": "doc_033"},
        {"qid": "Q14", "query": "How did Waymo respond to defendants' discovery arguments?", "ground_truth": "doc_034"},
        {"qid": "Q15", "query": "Why did Waymo request expedited discovery in the early phase of litigation?", "ground_truth": "doc_035"},
        {"qid": "Q16", "query": "What scheduling did the court set for preliminary injunction proceedings?", "ground_truth": "doc_028"},
        {"qid": "Q17", "query": "What were the initial trade secret allegations in the original complaint?", "ground_truth": "doc_021"},
        {"qid": "Q18", "query": "What changes were made in the amended complaint compared to the original?", "ground_truth": "doc_022"},
        {"qid": "Q19", "query": "What procedures were established for accessing sealed materials?", "ground_truth": "doc_026"},
        {"qid": "Q20", "query": "What expedited briefing schedule did the court order for Waymo's motion for relief?", "ground_truth": "doc_031"},
    ]

    files = [v["file"] for v in index.values()]
    reciprocal_ranks: list[float] = []
    misses = 0

    for q in QUERIES:
        results = search.smart_search(q["query"], files)

        # Aggregate match_count per doc_id then apply density scoring
        doc_matches: dict[str, int] = {}
        for hit in results.hits:
            doc_id = Path(hit.file_path).stem
            doc_matches[doc_id] = doc_matches.get(doc_id, 0) + hit.match_count

        if not doc_matches:
            reciprocal_ranks.append(0.0)
            misses += 1
            continue

        # Apply density scoring and rank
        scored = []
        for doc_id, mc in doc_matches.items():
            meta = index.get(doc_id, {})
            score = _density_score(mc, meta.get("page_count", 1), meta.get("total_chars", 3000))
            scored.append((score, doc_id))
        scored.sort(reverse=True)

        ranked_ids = [doc_id for _, doc_id in scored]
        gt = q["ground_truth"]
        if gt in ranked_ids:
            rank = ranked_ids.index(gt) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses += 1

    mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    passed = mrr_score >= MRR_PASS_FLOOR

    return PhaseResult(
        name="MRR regression",
        passed=passed,
        metrics={"mrr": round(mrr_score, 4), "baseline": MRR_BASELINE, "misses": misses},
        notes=[f"MRR {mrr_score:.3f} {'>='+str(MRR_PASS_FLOOR) if passed else '<'+str(MRR_PASS_FLOOR)+' FAIL'}"],
    )


# ── Phase 2: Cold live run ─────────────────────────────────────────────────────

async def phase_cold(no_wipe: bool = False) -> PhaseResult:
    if not os.environ.get("GEMINI_API_KEY"):
        return PhaseResult("Cold run", False, notes=["GEMINI_API_KEY not set"])

    if not DELE004_REPO.exists():
        return PhaseResult("Cold run", False, notes=[f"Repo not found: {DELE004_REPO}"])

    # Preserve baseline once
    if IRYS_DIR.exists() and not BASELINE_DIR.exists():
        shutil.copytree(IRYS_DIR, BASELINE_DIR)
        print(f"  [cold] Saved baseline DB -> {BASELINE_DIR}")

    # Wipe for fresh run
    if not no_wipe and IRYS_DIR.exists():
        shutil.rmtree(IRYS_DIR)
        print("  [cold] Wiped .irys/ for fresh run")

    from irys.api import Irys, IrysConfig

    irys = Irys(config=IrysConfig(
        api_key=os.environ["GEMINI_API_KEY"],
        max_depth=3,
        max_leads_per_level=3,
        log_level="WARNING",
        enable_inline_citations=False,
    ))
    irys.on_step(on_step_cb)

    print("  [cold] Running investigation ...\n")
    t0 = time.time()
    result = await irys.investigate(QUERY, DELE004_REPO)
    elapsed = time.time() - t0
    print(f"\n  [cold] Done in {elapsed:.1f}s")

    db_path = IRYS_DIR / "facts.db"
    if not db_path.exists():
        return PhaseResult("Cold run", False,
                           metrics={"elapsed": round(elapsed, 1)},
                           notes=["facts.db not found after run"])

    db = read_db(db_path)
    synopsis_lines = db["synopsis_lines"]
    total_facts    = db["total_facts"]
    evidence_chars = extract_evidence_chars(result.state)

    avg_synopsis = (sum(synopsis_lines.values()) / len(synopsis_lines)) if synopsis_lines else 0.0

    passed = avg_synopsis >= 8 and total_facts >= 30 and evidence_chars >= 15000

    return PhaseResult(
        name="Cold run",
        passed=passed,
        metrics={
            "synopsis_avg_lines": round(avg_synopsis, 1),
            "synopsis_per_source": synopsis_lines,
            "total_facts": total_facts,
            "tiers": db["tiers"],
            "evidence_chars": evidence_chars,
            "elapsed": round(elapsed, 1),
        },
        notes=[
            f"synopsis avg {avg_synopsis:.1f} {'>=8 OK' if avg_synopsis >= 8 else '<8 FAIL'}",
            f"total facts {total_facts} {'>=30 OK' if total_facts >= 30 else '<30 FAIL'}",
            f"evidence chars {evidence_chars} {'>=15000 OK' if evidence_chars >= 15000 else '<15000 FAIL'}",
        ],
    )


# ── Phase 3: Warm live run ─────────────────────────────────────────────────────

async def phase_warm() -> PhaseResult:
    if not os.environ.get("GEMINI_API_KEY"):
        return PhaseResult("Warm run", False, notes=["GEMINI_API_KEY not set"])

    db_path = IRYS_DIR / "facts.db"
    if not db_path.exists():
        return PhaseResult("Warm run", False,
                           notes=["facts.db missing — run cold phase first"])

    # Snapshot importance before warm run
    pre_snap = snapshot_importance(db_path)

    revisit_log.clear()

    from irys.api import Irys, IrysConfig

    irys = Irys(config=IrysConfig(
        api_key=os.environ["GEMINI_API_KEY"],
        max_depth=3,
        max_leads_per_level=3,
        log_level="WARNING",
        enable_inline_citations=False,
    ))
    irys.on_step(on_step_cb)

    print("  [warm] Running investigation (warm facts.db) ...\n")
    t0 = time.time()
    result = await irys.investigate(QUERY, DELE004_REPO)
    elapsed = time.time() - t0
    print(f"\n  [warm] Done in {elapsed:.1f}s")

    post_snap = snapshot_importance(db_path)
    evidence_chars = extract_evidence_chars(result.state)

    # Compute per-source importance delta
    deltas = []
    for source, pre_imp in pre_snap.items():
        post_imp = post_snap.get(source, pre_imp)
        delta = post_imp - pre_imp
        if delta > 0:
            deltas.append(delta)

    revisit_fires = len(revisit_log)
    avg_delta     = round(sum(deltas) / len(deltas), 2) if deltas else 0.0

    # Detect pure cache-answer path: engine answered without reading any docs.
    # state.documents_read == 0 and answered_from_cache flag set.
    answered_from_cache = bool(result.state.findings.get("answered_from_cache"))

    # Budget utilization: token_budget default is excerpt_chars_complex = 40000
    # char_budget = token_budget * 4 chars/token
    char_budget = 40000 * 4
    budget_pct  = round(evidence_chars / char_budget * 100, 1)

    # Pass if: re-read path (revisit fired + delta) OR cache-answer path (no reads needed)
    revisit_ok = revisit_fires >= 1 and avg_delta >= 4.0
    cache_ok   = answered_from_cache
    passed     = revisit_ok or cache_ok

    mode = "cache-answer (no doc reads)" if cache_ok else "re-read"

    return PhaseResult(
        name="Warm run",
        passed=passed,
        metrics={
            "mode": mode,
            "revisit_fires": revisit_fires,
            "avg_importance_delta": avg_delta,
            "sources_with_delta": len(deltas),
            "answered_from_cache": answered_from_cache,
            "evidence_chars": evidence_chars,
            "budget_pct": budget_pct,
            "elapsed": round(elapsed, 1),
        },
        notes=[
            f"mode: {mode}",
            f"revisit fires {revisit_fires} {'>=1 OK' if revisit_fires >= 1 else 'n/a (cache-answer)' if cache_ok else '<1 FAIL'}",
            f"avg importance delta +{avg_delta} {'>=4.0 OK' if avg_delta >= 4.0 else 'n/a (cache-answer)' if cache_ok else '<4.0 FAIL'}",
            f"budget utilization {budget_pct}% (informational)",
        ],
    )


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list[PhaseResult]):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  bench.py -- feat/context-store-v2")
    print(SEP)

    for pr in results:
        print()
        print(f"  Phase: {pr.name}")
        m = pr.metrics

        if pr.name == "MRR regression":
            score    = m.get("mrr", 0.0)
            baseline = m.get("baseline", MRR_BASELINE)
            diff     = score - baseline
            sign     = "+" if diff >= 0 else ""
            eq       = "=" if abs(diff) < 0.001 else (("^" if diff > 0 else "v"))
            print(f"  MRR score:          {score:.3f}    baseline={baseline:.3f}    {eq} ({sign}{diff:.3f})")
            print(f"  Pass: {'YES' if pr.passed else 'NO  <-- FAIL'}")

        elif pr.name == "Cold run":
            avg  = m.get("synopsis_avg_lines", 0)
            tot  = m.get("total_facts", 0)
            ech  = m.get("evidence_chars", 0)
            ela  = m.get("elapsed", 0)
            per  = m.get("synopsis_per_source", {})
            trs  = m.get("tiers", {})
            print(f"  Synopsis per source: {per}")
            print(f"  Synopsis avg lines: {avg:<8} target>=8         {'PASS' if avg >= 8 else 'FAIL'}")
            print(f"  Total facts:        {tot:<8} target>=30        {'PASS' if tot >= 30 else 'FAIL'}")
            print(f"  Evidence chars:     {ech:<8} target>=15000     {'PASS' if ech >= 15000 else 'FAIL'}")
            print(f"  Tier distribution:  {trs}")
            print(f"  Elapsed:            {ela}s")

        elif pr.name == "Warm run":
            fires  = m.get("revisit_fires", 0)
            delta  = m.get("avg_importance_delta", 0.0)
            bpct   = m.get("budget_pct", 0.0)
            ela    = m.get("elapsed", 0)
            mode   = m.get("mode", "re-read")
            cache  = m.get("answered_from_cache", False)
            print(f"  Mode:               {mode}")
            if cache:
                print(f"  Answered from cache (no doc reads)        PASS")
            else:
                print(f"  Revisit fires:      {fires:<8} target>=1         {'PASS' if fires >= 1 else 'FAIL'}")
                print(f"  Importance delta:   +{delta:<7} target>=4.0       {'PASS' if delta >= 4.0 else 'FAIL'}")
            print(f"  Budget utilization: {bpct}%      (informational)")
            print(f"  Elapsed:            {ela}s")

    print()
    print(SEP)
    total   = len(results)
    passing = sum(1 for r in results if r.passed)
    all_ok  = passing == total
    verdict = "ALL PASS" if all_ok else f"PARTIAL ({passing}/{total} phases pass)"
    print(f"  VERDICT: {verdict}")
    print(SEP)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

async def _run(phases: list[str], no_wipe: bool):
    results: list[PhaseResult] = []

    if "mrr" in phases:
        print("\n[Phase 1] MRR regression (offline, no API calls)")
        r = phase_mrr()
        results.append(r)

    if "cold" in phases:
        print("\n[Phase 2] Cold live run (DELE-004, fresh facts.db)")
        r = await phase_cold(no_wipe=no_wipe)
        results.append(r)

    if "warm" in phases:
        print("\n[Phase 3] Warm live run (DELE-004, warm facts.db)")
        r = await phase_warm()
        results.append(r)

    print_report(results)

    if any(not r.passed for r in results):
        sys.exit(1)


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(description="bench.py -- feat/context-store-v2 benchmark")
    parser.add_argument("--phase", choices=["mrr", "cold", "warm", "all"], default="all")
    parser.add_argument("--no-wipe", action="store_true", help="Skip .irys/ wipe before cold run")
    args = parser.parse_args()

    if args.phase == "all":
        phases = ["mrr", "cold", "warm"]
    else:
        phases = [args.phase]

    asyncio.run(_run(phases, args.no_wipe))


if __name__ == "__main__":
    main()
