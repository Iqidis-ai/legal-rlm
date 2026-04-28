"""Verify that citation injection no longer blocks the event loop.

Boots `irys.service.api.app` (the production FastAPI app, the production
`/health` endpoint, the production middleware stack) and exercises the
production `InlineCitationService.inject` → `_call_gemini_lite` path.

Only one external dependency is faked: `GeminiClient.complete` is monkey-patched
to `await asyncio.sleep(N)` instead of calling Gemini.  Everything else —
`InlineCitationService.inject`, the async `_call_gemini_lite`, the FastAPI
app, uvicorn config — runs unchanged.

A debug route is attached at startup:
  POST /debug/inject-citations  – calls await InlineCitationService.inject()

The probe loop hits the real /health every 250 ms while the debug route
is executing a slow (simulated) LLM call.

PASS criteria: all /health probes return in < 2s while injection is running.
If the event loop were blocked, probes would queue up and report multi-second
latency matching the simulated LLM duration.

Run:
    python scripts/repro_event_loop_freeze.py
    python scripts/repro_event_loop_freeze.py --slow-seconds 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI

# Tunables — overridden by CLI flags.
HOST = "127.0.0.1"
PORT = 8765
DEFAULT_SLOW_SECONDS = 5.0    # quick local test; use --slow-seconds 30 for prod-like
PROBE_INTERVAL_S = 0.25
WARMUP_S = 2.0
COOLDOWN_S = 3.0


# ── Real-app bootstrap ──────────────────────────────────────────────────────


def _patch_gemini_complete(slow_seconds: float) -> None:
    """Replace GeminiClient.complete with a sleep stub.

    Returns a string containing every citation marker the prompt mentions so
    the inject() validator has a chance to pass — though we don't actually
    care about the returned annotated text, only the timing.
    """
    import re
    from irys.core import models as _models

    async def _fake_complete(self, prompt: str, *args, **kwargs) -> str:
        await asyncio.sleep(slow_seconds)
        # Echo back any UUID-like markers the prompt references so validation
        # has something to chew on (failures fall back to original answer,
        # which is fine — we only care about latency on /health).
        ids = re.findall(r"\[([a-f0-9]{8})\]", prompt)
        return " ".join(f"sentence [{i}]." for i in ids) or "stub."

    _models.GeminiClient.complete = _fake_complete  # type: ignore[assignment]


def _add_debug_routes(app: FastAPI) -> None:
    """Attach a debug route that drives InlineCitationService.inject."""
    from irys.api import IrysConfig
    from irys.rlm.state import Citation
    from irys.service.inline_citation_service import InlineCitationService

    def _make_payload():
        cfg = IrysConfig(api_key="fake-key", enable_inline_citations=True)
        # Single stub citation – inject still issues the LLM call regardless
        # of citation count, so one is enough to exercise the bug path.
        cit = Citation(
            id="aaaaaaaa", document="stub.pdf", page=1,
            text="Stub citation text.", context="ctx",
            relevance="high", source_type="document",
        )
        answer = "This is a synthesized answer that needs citations."
        return answer, [cit], cfg

    @app.post("/debug/inject-citations", include_in_schema=False)
    async def debug_inject():
        answer, cits, cfg = _make_payload()
        # OLD (sync — blocks event loop): uncomment to test with pre-fix code
        # out, _, _ = InlineCitationService.inject(answer=answer, citations=cits, config=cfg)
        # NEW (async — non-blocking): uncomment after applying the async fix
        out, _, _ = await InlineCitationService.inject(answer=answer, citations=cits, config=cfg)
        return {"len": len(out)}


def bootstrap_real_app(slow_seconds: float) -> FastAPI:
    """Import the production app, patch the LLM leaf, attach debug routes."""
    os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-repro")
    # Service config insists certain things exist; provide harmless defaults.
    os.environ.setdefault("IRYS_DEBUG", "1")
    os.environ.setdefault("S3_BUCKET", "fake-bucket")

    _patch_gemini_complete(slow_seconds)

    from irys.service.api import app as real_app
    _add_debug_routes(real_app)
    return real_app


def start_server_in_thread(slow_seconds: float) -> uvicorn.Server:
    config = uvicorn.Config(
        bootstrap_real_app(slow_seconds),
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    return server


async def wait_for_server(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url, timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"Server at {url} did not become ready")


# ── Probe / scenario runner ─────────────────────────────────────────────────


@dataclass
class Probe:
    t_offset: float       # seconds since scenario start
    latency_ms: float
    status: int | None
    error: str | None = None


@dataclass
class ScenarioResult:
    label: str
    probes: list[Probe] = field(default_factory=list)
    investigate_duration_s: float = 0.0


async def _single_probe(client: httpx.AsyncClient, base_url: str, t0: float, out: list[Probe]) -> None:
    """Fire one /health request and record the result."""
    ts = time.monotonic() - t0
    t_send = time.monotonic()
    try:
        r = await client.get(f"{base_url}/health", timeout=60.0)
        out.append(Probe(ts, (time.monotonic() - t_send) * 1000, r.status_code))
    except Exception as e:
        out.append(Probe(ts, (time.monotonic() - t_send) * 1000, None, type(e).__name__))


async def probe_health_loop(base_url: str, t0: float, until: float, out: list[Probe]) -> None:
    """Fire concurrent probes every PROBE_INTERVAL_S until `until`.

    Uses a single shared client with a large connection pool so probes that
    arrive during a server freeze can pile up on independent connections,
    matching what an external monitor (ALB, Pingdom, etc.) would do.
    """
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=20)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warm one connection so first probe doesn't pay handshake cost.
        await client.get(f"{base_url}/health", timeout=5.0)

        tasks: list[asyncio.Task] = []
        while time.monotonic() < until:
            tasks.append(asyncio.create_task(_single_probe(client, base_url, t0, out)))
            await asyncio.sleep(PROBE_INTERVAL_S)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    out.sort(key=lambda p: p.t_offset)


async def trigger_investigate(base_url: str, endpoint: str, slow_seconds: float) -> float:
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        await client.post(f"{base_url}{endpoint}", timeout=slow_seconds + 30)
    return time.monotonic() - t0


async def run_scenario(label: str, base_url: str, endpoint: str, slow_seconds: float) -> ScenarioResult:
    result = ScenarioResult(label=label)
    total_window = WARMUP_S + slow_seconds + COOLDOWN_S

    print(f"\n── Scenario: {label}  ({endpoint},  slow={slow_seconds}s,  window={total_window:.0f}s)")
    print("   Probing /health every 250 ms; firing slow call after 2 s warm-up …")

    t0 = time.monotonic()
    probe_task = asyncio.create_task(
        probe_health_loop(base_url, t0, t0 + total_window, result.probes)
    )

    await asyncio.sleep(WARMUP_S)
    investigate_task = asyncio.create_task(
        trigger_investigate(base_url, endpoint, slow_seconds)
    )

    await probe_task
    try:
        result.investigate_duration_s = await asyncio.wait_for(investigate_task, timeout=5.0)
    except asyncio.TimeoutError:
        investigate_task.cancel()
        result.investigate_duration_s = -1.0

    return result


# ── Reporting ───────────────────────────────────────────────────────────────


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "p50": s[n // 2],
        "p90": s[min(n - 1, int(n * 0.9))],
        "p99": s[min(n - 1, int(n * 0.99))],
        "max": s[-1],
    }


def print_scenario(result: ScenarioResult) -> None:
    print(f"\n── Result: {result.label} ──────────────────────────────────")
    print(f"   investigate call took: {result.investigate_duration_s:.2f} s")
    print(f"   probes sent:           {len(result.probes)}")
    timed_out = [p for p in result.probes if p.error]
    over_2s = [p for p in result.probes if p.error is None and p.latency_ms > 2000]
    print(f"   probes timed-out:      {len(timed_out)}")
    print(f"   probes > 2 000 ms:     {len(over_2s)}")

    lats = [p.latency_ms for p in result.probes if p.error is None]
    s = _stats(lats)
    print(f"   /health latency  min={s['min']:.1f}  p50={s['p50']:.1f}  "
          f"p90={s['p90']:.1f}  p99={s['p99']:.1f}  max={s['max']:.1f}  (ms)")

    print("\n   timeline (t=offset s,  one row per probe):")
    print(f"   {'t (s)':>7}  {'lat (ms)':>9}  {'status':>6}  bar")
    cap = max((p.latency_ms for p in result.probes if p.error is None), default=1.0)
    for p in result.probes:
        if p.error:
            print(f"   {p.t_offset:>7.2f}  {'TIMEOUT':>9}  {'-':>6}  {p.error}")
            continue
        bar_len = int((p.latency_ms / cap) * 40) if cap else 0
        marker = "  !!!" if p.latency_ms > 2000 else ("  !" if p.latency_ms > 500 else "")
        print(f"   {p.t_offset:>7.2f}  {p.latency_ms:>9.1f}  {p.status:>6}  {'█'*bar_len}{marker}")


# ── Main ────────────────────────────────────────────────────────────────────

HEALTH_LATENCY_THRESHOLD_MS = 2000  # probes above this = FAIL


async def main_async(args: argparse.Namespace) -> int:
    server = start_server_in_thread(args.slow_seconds)
    base_url = f"http://{HOST}:{PORT}"
    await wait_for_server(f"{base_url}/health")

    try:
        result = await run_scenario(
            "async-inject", base_url, "/debug/inject-citations", args.slow_seconds,
        )
        print_scenario(result)

        worst = max(
            (p.latency_ms for p in result.probes if p.error is None),
            default=0.0,
        )
        n_bad = sum(1 for p in result.probes if p.error or p.latency_ms > HEALTH_LATENCY_THRESHOLD_MS)

        print(f"\n{'═' * 50}")
        if n_bad == 0:
            print(f"  ✅ PASS — all {len(result.probes)} health probes < {HEALTH_LATENCY_THRESHOLD_MS} ms")
            print(f"           max latency: {worst:.1f} ms")
            print(f"           Event loop was NOT blocked during {args.slow_seconds}s LLM call")
        else:
            print(f"  ❌ FAIL — {n_bad}/{len(result.probes)} probes exceeded {HEALTH_LATENCY_THRESHOLD_MS} ms")
            print(f"           max latency: {worst:.1f} ms")
            print(f"           Event loop WAS blocked during citation injection")
        print(f"{'═' * 50}\n")

        return 1 if n_bad > 0 else 0
    finally:
        server.should_exit = True
        await asyncio.sleep(0.5)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Verify citation injection does not block the event loop")
    p.add_argument("--slow-seconds", type=float, default=DEFAULT_SLOW_SECONDS,
                   help=f"Simulated LLM call duration (default {DEFAULT_SLOW_SECONDS}s)")
    args = p.parse_args()

    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
