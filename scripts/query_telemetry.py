"""Query telemetry data for a specific investigation run from the production DB.

Usage:
    python scripts/query_telemetry.py inv_1a8ad77b
    python scripts/query_telemetry.py inv_1a8ad77b --json
    python scripts/query_telemetry.py --latest
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.irys.db.session import session_scope
from src.irys.db.models.investigation_log import (
    InvestigationLog,
    InvestigationStepModel,
    InvestigationOperation,
)


def _fmt_dt(dt):
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_ms(ms):
    if ms is None:
        return "N/A"
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms}ms"


def query_investigation(investigation_id: str, as_json: bool = False):
    with session_scope() as session:
        # 1. Investigation log
        log = session.query(InvestigationLog).filter_by(id=investigation_id).first()
        if not log:
            print(f"❌ No investigation found with id='{investigation_id}'")
            # Show recent investigations
            recent = session.query(InvestigationLog).order_by(
                InvestigationLog.started_at.desc()
            ).limit(5).all()
            if recent:
                print(f"\nRecent investigations:")
                for r in recent:
                    print(f"  {r.id}  {_fmt_dt(r.started_at)}  {r.status}  {_fmt_ms(r.total_duration_ms)}")
            return

        # 2. Steps
        steps = session.query(InvestigationStepModel).filter_by(
            investigation_id=investigation_id
        ).order_by(InvestigationStepModel.seq).all()

        # 3. Operations
        operations = session.query(InvestigationOperation).filter_by(
            investigation_id=investigation_id
        ).order_by(InvestigationOperation.started_at).all()

        # Group operations by step_id
        ops_by_step = {}
        for op in operations:
            ops_by_step.setdefault(op.step_id, []).append(op)

        if as_json:
            _print_json(log, steps, ops_by_step)
        else:
            _print_pretty(log, steps, ops_by_step)


def _print_json(log, steps, ops_by_step):
    data = {
        "investigation": {
            "id": log.id,
            "message_id": log.message_id,
            "user_id": getattr(log, "user_id", None),
            "started_at": _fmt_dt(log.started_at),
            "completed_at": _fmt_dt(log.completed_at),
            "status": log.status,
            "total_duration_ms": log.total_duration_ms,
            "total_cost_usd": float(log.total_cost_usd) if log.total_cost_usd else None,
            "total_steps": log.total_steps,
            "phase_breakdown": log.phase_breakdown,
        },
        "steps": [],
    }
    for step in steps:
        step_data = {
            "seq": step.seq,
            "step_name": step.step_name,
            "phase": step.phase,
            "started_at": _fmt_dt(step.started_at),
            "latency_ms": step.step_latency_ms,
            "operations": [],
        }
        for op in ops_by_step.get(step.id, []):
            step_data["operations"].append({
                "type": op.type,
                "started_at": _fmt_dt(op.started_at),
                "latency_ms": op.latency_ms,
                "details": op.details,
            })
        data["steps"].append(step_data)
    print(json.dumps(data, indent=2, default=str))


def _print_pretty(log, steps, ops_by_step):
    print(f"\n{'═' * 80}")
    print(f"  INVESTIGATION: {log.id}")
    print(f"{'═' * 80}")
    print(f"  Message ID:    {log.message_id or 'N/A'}")
    print(f"  User ID:       {getattr(log, 'user_id', 'N/A') or 'N/A'}")
    print(f"  Status:        {log.status}")
    print(f"  Started:       {_fmt_dt(log.started_at)}")
    print(f"  Completed:     {_fmt_dt(log.completed_at)}")
    print(f"  Duration:      {_fmt_ms(log.total_duration_ms)}")
    cost = float(log.total_cost_usd) if log.total_cost_usd else 0
    print(f"  Total Cost:    ${cost:.6f}")
    print(f"  Total Steps:   {log.total_steps}")

    if log.phase_breakdown:
        print(f"\n  Phase Breakdown:")
        for phase, info in log.phase_breakdown.items():
            dur = info.get("duration_ms", info.get("total_duration_ms", "?"))
            cnt = info.get("step_count", "?")
            print(f"    {phase:<25} {_fmt_ms(dur):>10}  ({cnt} steps)")

    print(f"\n{'─' * 80}")
    print(f"  {'#':<4} {'Step Name':<30} {'Phase':<20} {'Latency':>10}  Operations")
    print(f"{'─' * 80}")

    for step in steps:
        step_ops = ops_by_step.get(step.id, [])
        op_summary = f"{len(step_ops)} ops"
        print(f"  {step.seq:<4} {step.step_name:<30} {step.phase:<20} {_fmt_ms(step.step_latency_ms):>10}  {op_summary}")

        for op in step_ops:
            details_keys = list((op.details or {}).keys())
            detail_str = ", ".join(details_keys[:4])
            if len(details_keys) > 4:
                detail_str += f", +{len(details_keys)-4} more"
            print(f"       └─ {op.type:<12} {_fmt_ms(op.latency_ms):>10}  [{detail_str}]")

            # Show key details for LLM ops
            if op.type == "llm_call" and op.details:
                model = op.details.get("model", "")
                tier = op.details.get("tier", "")
                tokens_in = op.details.get("input_tokens", "?")
                tokens_out = op.details.get("output_tokens", "?")
                cost = op.details.get("cost_usd", 0)
                print(f"              model={model} tier={tier} in={tokens_in} out={tokens_out} cost=${cost:.4f}" if cost else
                      f"              model={model} tier={tier} in={tokens_in} out={tokens_out}")

    # Summary
    total_ops = sum(len(v) for v in ops_by_step.values())
    print(f"\n{'═' * 80}")
    print(f"  Summary: {len(steps)} steps, {total_ops} operations")
    llm_ops = [op for ops in ops_by_step.values() for op in ops if op.type == "llm_call"]
    if llm_ops:
        total_llm_ms = sum(op.latency_ms or 0 for op in llm_ops)
        print(f"  LLM calls: {len(llm_ops)}, total LLM time: {_fmt_ms(total_llm_ms)}")
    print(f"{'═' * 80}\n")


def query_latest(as_json: bool = False):
    with session_scope() as session:
        log = session.query(InvestigationLog).order_by(
            InvestigationLog.started_at.desc()
        ).first()
        if not log:
            print("❌ No investigations found in database")
            return
        print(f"Latest investigation: {log.id}")
        query_investigation(log.id, as_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Query telemetry for an investigation run")
    p.add_argument("investigation_id", nargs="?", help="Investigation ID (e.g. inv_1a8ad77b)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--latest", action="store_true", help="Query the most recent investigation")
    args = p.parse_args()

    if args.latest:
        query_latest(args.json)
    elif args.investigation_id:
        query_investigation(args.investigation_id, args.json)
    else:
        p.print_help()
