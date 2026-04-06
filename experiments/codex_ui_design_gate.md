I reviewed [.claude/CLAUDE.md](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/.claude/CLAUDE.md), [STATUS.md](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/STATUS.md), the current Gradio UI in [src/irys/ui/app.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py), the in-process facade in [src/irys/api.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py), and the REST layer in [src/irys/service/api.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py), plus the matter/ledger substrate in [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py), and [src/irys/matter/reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py). I did not find a repo-root `CLAUDE.md`; the project file present is the one under `.claude`.

**Recommendation**
- Canonical UI architecture should be `HTTP -> FastAPI service`, not direct coupling to the in-process `Irys` class.
- Keep an in-process adapter only as a dev fallback while the service gets 3 missing UI endpoints: `overview`, `steering-actions`, and incremental event streaming.
- Reason: the current UI in [src/irys/ui/app.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py) only gets a final `InvestigationState`, shows a thin stats box, does not keep `matter_id` / `run_id`, and its `request_stop()` is not wired to the real stop path. By contrast, the service already exposes runs, ledger events, stop, redirect, assertions, issues, gaps, proof-state, reconciliation, timeline, evidence matrix, annotations, and trust overrides.
- Tradeoff at this stage: in-process is faster to hack on and gives immediate `on_step` callbacks, but it will force the UI to reach into private engine state or re-implement a second backend surface. HTTP costs a bit more now, but it gives a stable contract, deployable separation, restart recovery, and future non-Gradio clients.
- Streaming recommendation: keep the current queue only for temporary in-process dev mode. Service mode should use `SSE`, not WebSockets. This is a server-to-client stream; stop/redirect/correct remain ordinary REST writes. SSE also maps cleanly onto the durable ledger `seq_no` model.

**Panel Priority**
1. `Overview / Command Center`
Shows matter identity, latest run, true reuse rate, assertion count, issue coverage average, proof-gap count, open-gap count, advocacy-only warnings, top pending clarifications, and top recommended steering actions. This is what the user should see first, not a blank answer pane.
2. `Run / Output`
Shows the final memo, citations, current run status, and the live reasoning ledger. Use durable `ledger_event` rows as the primary live trace; raw `ThinkingStep` output should be a dev-only toggle.
3. `Issues / Proof`
Shows the issue table with `coverage_fraction`, `proof_status`, `supporting_count`, `attacking_count`, predicate counts, `advocacy_only`, linked authorities, and proof-gap indicators. This is the backbone view for SO-4.
4. `Assertions / Evidence`
Shows the typed assertion table with `belief_state`, confidence, SPO fields, source roles, speech acts, documents, issue links, and dependency neighbors. Corrections happen here.
5. `Gaps / Clarifications / Steering`
Shows missing documents, proof gaps, unresolved contradictions, pending clarification questions, document annotations, and the actionable steering inbox.
6. `Quant / Timeline`
Shows payment reconciliation, invoice chain, damages waterfall, numeric conflicts, and timeline. Ship this in the first demo only if the demo corpus is payment/damages-heavy; otherwise make it the first panel after the core five.

**Priority By SO**
- First UI integrations should be `SO-1`, `SO-4`, `SO-2`, `SO-5`, and `SO-7`.
- Then add `SO-3` controls on top of visible state. Steering without visible assertions/issues/gaps is not testable.
- `SO-6` comes after that unless the demo matter is explicitly a payments/damages matter.
- Treat `SO-5` as cross-cutting, not a standalone afterthought. Source-role and trust badges belong inside issues, assertions, and quant views from day one.

**UX Decisions**
- `Stop`: global run control in the run header. One click, immediate `stop_requested` state, no hidden behavior.
- `Redirect`: inline action on issue rows and on the overview “weakest issues” list. Redirect is issue-scoped, so it belongs next to issues.
- `Correct assertion`: inline in an assertion drawer or row action, not in a separate detached form. The user needs the proposition, source roles, issue links, and dependencies in view while correcting it.
- `Trust override` and `annotate document`: inline in source/document detail, with a separate list page only for audit and cleanup.
- `Steering panel`: yes, but as an inbox of suggested actions. The actual edit surfaces stay contextual.

**Matter State Presentation**
- Use `table-first progressive disclosure`.
- Primary surfaces should be issue and assertion tables with sorting, filtering, and row drawers.
- Add graph visualization only for a selected assertion neighborhood or a selected issue slice. A full matter-wide node graph is not the primary legal UX.
- Use heatmaps and summary bars before force-directed graphs. The evidence matrix is more useful than a general graph on day one.

**File Structure Changes**
- Slim [src/irys/ui/app.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py) down to Gradio composition and shared UI state.
- Add `src/irys/ui/backends/base.py` for a UI backend interface.
- Add `src/irys/ui/backends/http.py` as the canonical implementation.
- Add `src/irys/ui/backends/in_process.py` only as a dev fallback.
- Add `src/irys/ui/viewmodels.py` to normalize service payloads into panel-ready shapes.
- Add panel modules:
  - `src/irys/ui/panels/overview.py`
  - `src/irys/ui/panels/run_output.py`
  - `src/irys/ui/panels/issues.py`
  - `src/irys/ui/panels/assertions.py`
  - `src/irys/ui/panels/gaps_steering.py`
  - `src/irys/ui/panels/quant.py`

**Backend Changes Needed**
- Add `GET /matter/{matter_id}/overview`.
This should aggregate stats, metrics, latest run, top issues, proof summary, top gaps, pending clarifications, source summary, and latest quant summary so the landing page is one request, not ten.
- Add `GET /matter/{matter_id}/steering-actions`.
This should expose `MatterModel.get_ledger_steering_surface()` directly.
- Add incremental ledger reads: `GET /matter/{matter_id}/runs/{run_id}/events?after_seq=...`.
- Add `SSE /matter/{matter_id}/runs/{run_id}/events/stream`.
Use ledger `seq_no` for reconnect/resume.
- Add run-scoped stop: `POST /matter/{matter_id}/runs/{run_id}/stop`.
The current matter-level stop is workable, but the UI should target a specific run, just like redirect already does.
- Add assertion graph detail endpoints.
Minimum: `GET /matter/{matter_id}/assertions/{assertion_id}` and `GET /matter/{matter_id}/assertions/{assertion_id}/neighbors`. Better: one `GET /matter/{matter_id}/assertions/graph`.
- Add a source summary surface.
Either `GET /matter/{matter_id}/sources/summary` or include it in `overview`.
- Optional for local dev if you insist on repo-path UX: add a dev-only local filesystem investigate endpoint. Otherwise switch the UI to upload-based corpus selection and reuse the existing sync upload endpoints.

**First Demo Quality Bar**
- A second run over the same corpus clearly shows the same matter, a higher reuse rate, and a run delta: what was reused, what new assertions were added, what issues improved, and what gaps were closed or opened.
- The landing page immediately exposes the weakest issues, proof gaps, top missing documents, and any advocacy-only support warnings.
- A user can stop a live run, redirect it to a weak issue, and see those actions reflected in the ledger.
- A user can correct an assertion inline and see downstream issue/proof state change without restarting from scratch.
- If the corpus is numeric, the quant panel shows invoice total, payments, exposure, and conflicts; if not, the panel degrades cleanly instead of showing empty chrome.
- No dead controls. Every visible button must be wired to a real backend mutation and must update the relevant panel state.

If you want, I can turn this into a step-by-step implementation sequence against the exact files next.