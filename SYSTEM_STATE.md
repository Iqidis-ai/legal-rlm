# Irys RLM System State

Last updated: 2026-04-10 22:37:00 -04:00
Snapshot basis: local worktree on branch `SebihSpecial`
HEAD at capture: `9bc03de` (`Clean repo entropy and stale artifacts`)
Worktree note: the worktree was still dirty at capture time (24 modified/untracked paths). Always run `git status --short` before assuming this snapshot exactly matches your checkout.

Purpose: this is the operational snapshot for coding agents. Use this first for "what exists now." Use [`.claude/CLAUDE.md`](./.claude/CLAUDE.md) for mission/governance, [`DECISIONS.md`](./DECISIONS.md) for historical decisions, and [`ASSUMPTIONS.md`](./ASSUMPTIONS.md) for explicit assumptions.

## Canonical Entry Points

- UI: `python -m irys.ui.app`
- Service: `uvicorn irys.service.api:app --host 0.0.0.0 --port 8000`
- Python API: `from irys import Irys`
- FastAPI app factory: `src/irys/service/api.py`
- Gradio app factory: `src/irys/ui/app.py`

Do not reintroduce `run_ui.py` or `run_server.py`. Those wrappers were removed because the module entrypoints above are the canonical launch paths.

## Current Repo Shape

- `src/irys/core`: Gemini client, pricing, usage accounting, reader, repository, search
- `src/irys/rlm`: investigation engine, checkpoints, state, research-mode budget control
- `src/irys/matter`: SQLite schema, canonical stores, rollups, timeline/evidence/communication views, reasoning ledger
- `src/irys/service`: FastAPI service layer and S3-backed repository support
- `src/irys/ui`: Gradio app plus in-process and HTTP backends
- `tests`: matter-model, service, interruptibility, and usage coverage

Repo hygiene was tightened on 2026-04-10. Stale `clients/`, `experiments/`, wrapper launchers, old roundtable docs, sample `test_documents/`, and machine-local deploy files were removed.

## What Exists Today

### Durable Matter Model

- SQLite-backed matter model with schema version 47
- Persistent `run_session` tracking plus `llm_call` per-request telemetry
- Canonical stores for assertions, issues, actors, gaps, proof state, quantitative facts, and reasoning ledger
- Timeline, evidence matrix, communication map, and damages waterfall exposed as structured matter-level views

Relevant files:
- `src/irys/matter/schema.py`
- `src/irys/matter/matter.py`
- `src/irys/matter/graph.py`
- `src/irys/matter/reasoning.py`

### Investigation Engine

- Recursive investigation loop: orient -> investigate -> verify -> synthesize
- Checkpointed state with interrupt, redirect, and resume support
- Resume works from checkpoint boundaries, not from the middle of an in-flight Gemini request
- Research budget is user-selected via `simple`, `deep`, or `sebih_special`
- Research mode changes investigation budget and stopping behavior only; it does not change model routing

Relevant files:
- `src/irys/rlm/engine.py`
- `src/irys/rlm/state.py`
- `src/irys/api.py`

### LLM Telemetry and Cost Tracking

- Every Gemini request is persisted to `llm_call`
- `run_session` stores cheap rollup totals for UI and service queries
- Tracked fields include model tier, model id, input tokens, cache-read tokens, output tokens, total prompt tokens, estimated cost, latency, success, and error kind
- Matter model exposes both aggregate usage summaries and recent call listings
- UI surfaces matter-level LLM usage and cost analytics

Relevant files:
- `src/irys/core/models.py`
- `src/irys/matter/matter.py`
- `src/irys/matter/schema.py`
- `src/irys/service/api.py`
- `src/irys/ui/app.py`

### UI

- Gradio dashboard with live run/output plus matter intelligence panels
- Live run status includes current research mode and live LLM call/cost totals
- Matter visualizations include:
  - overview cards and SO-style summary metrics
  - issues and assertions surfaces
  - timeline
  - evidence matrix
  - communication graph
  - LLM analytics
  - quant / damages surfaces
- Resume and steering actions are wired through the UI backends

Relevant files:
- `src/irys/ui/app.py`
- `src/irys/ui/backends/base.py`
- `src/irys/ui/backends/in_process.py`
- `src/irys/ui/backends/http.py`

## Current Model Tiering

Model tiers and research modes are separate concepts.

| Tier | Current model | Primary role |
| --- | --- | --- |
| `NANO` | `gemini-2.5-flash-lite` | Short-output triage and classification |
| `LITE` | `gemini-2.5-flash-lite` | Bulk document reading and extraction |
| `FLASH` | `gemini-3.1-flash-lite-preview` | Search analysis, routing, planning |
| `PRO` | `gemini-2.5-pro` | Final synthesis and heavier reasoning |

Pricing snapshot encoded in `src/irys/core/models.py` and verified against Google pricing docs on 2026-04-10:

| Tier | Input / 1M | Cache read / 1M | Output / 1M |
| --- | --- | --- | --- |
| `NANO` / `LITE` | `$0.10` | `$0.01` | `$0.40` |
| `FLASH` | `$0.25` | `$0.025` | `$1.50` |
| `PRO` | `$1.25` up to 200k prompt tokens, then higher large-context rate | `10%` of active input rate | `$10.00` up to 200k prompt tokens, then higher large-context rate |

## Recent Checkpoints

- `9bc03de` - repo hygiene cleanup: stale artifacts removed, ignore rules tightened
- `6e84bcb` - research-mode budget controls (`simple`, `deep`, `sebih_special`)

## Known Constraints and Caveats

- OCR is still not implemented. PDFs must contain machine-readable text.
- Gemini is the only active production provider path.
- Research mode does not currently alter model routing.
- Some historical docs are intentionally preserved as history, not live truth. Treat this file as the current implementation snapshot.
- In this OneDrive-backed environment, full `pytest` runs can hit temp-directory cleanup permission issues. Targeted tests and direct sanity checks have been more reliable.
- The local worktree may still contain in-progress edits beyond `HEAD`. Check git state before writing code or making assumptions about exact file contents.

## Agent Workflow

1. Read this file first.
2. Read [`.claude/CLAUDE.md`](./.claude/CLAUDE.md) for mission, sacred outcomes, and governance rules.
3. Run `git status --short` before changing anything.
4. Verify live code before updating model ids, pricing, schema version, or API surfaces.
5. Do not recreate deleted wrapper/client/experiment surfaces unless there is a concrete product reason.
