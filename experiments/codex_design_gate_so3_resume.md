**Current Code**
1. Checkpoints are only written from [engine.py:1699](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1699) via [engine.py:4716](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4716). The interval is 5 at [engine.py:94](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L94).
- `_save_checkpoint()` writes two files under `self.config.checkpoint_dir`:
  - `checkpoint_<state.id>_iter<iteration>.json` at [engine.py:4721](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4721)
  - `latest_<state.id>.json` at [engine.py:4726](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4726)
- Format is plain JSON from `InvestigationState.to_dict()` at [state.py:1651](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/state.py#L1651), written by [state.py:1868](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/state.py#L1868), loaded by [state.py:1876](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/state.py#L1876).
- The checkpoint includes `repository_path`, findings/leads/citations/entities, counters, `status`, timestamps, `reasoning_trail`, and `pending_clarifications`. It does not persist live runtime bindings like `_matter_adapter` / `_run_id`, and it also does not serialize `llm_calls_avoided` / `llm_calls_required` even though those fields exist at [state.py:702](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/state.py#L702) and [state.py:703](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/state.py#L703).

2. `resume_investigation()` is at [engine.py:4729](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4729).
- It takes one explicit input: `checkpoint_path`.
- It loads the checkpoint at [engine.py:4742](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4742), rebuilds `MatterRepository(state.repository_path)` at [engine.py:4743](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4743), starts a new run session with `Resume: ...` at [engine.py:4751](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4751), then runs `_investigate_loop()`, `_verify_citations()`, `_synthesize()`, and `complete_run()` at [engine.py:4758](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4758), [engine.py:4762](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4762), [engine.py:4764](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4764).
- Implicit requirements:
  - checkpoint file exists
  - `state.repository_path` still exists on disk
  - `self._matter_model` is already wired if you want ledger writes / stop propagation
- Gaps in current resume path:
  - it does not set `state._run_id` the way normal investigate does at [engine.py:1040](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1040)
  - it does not mirror the normal stop/interrupted branch at [engine.py:1061](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1061)
  - it does not mirror normal post-run packaging: clarifications at [engine.py:1111](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1111) and reasoning trail at [engine.py:1127](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1127)

3. `checkpoint_dir` belongs in service storage config next to `temp_dir` / `matter_db_dir` in [service/config.py:32](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L32) and [service/config.py:33](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L33).
- Add env loading in `from_env()` next to [service/config.py:72](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L72) and [service/config.py:73](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L73).
- Add validation in [service/config.py:90](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L90) if you want non-empty `IRYS_CHECKPOINT_DIR`.

4. `run_session.next_action` already exists.
- Base fields are in [schema.py:151](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L151) through [schema.py:161](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L161): `id`, `matter_id`, `query`, `objective`, `active_branch_issue_id`, `status`, `stop_requested`, `redirect_requested`, `next_action`, `started_at`, `completed_at`.
- Later migrations add `assertions_at_start` / `reuse_rate` at [schema.py:1244](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L1244) and [schema.py:1247](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L1247), plus `llm_calls_avoided` / `llm_calls_required` at [schema.py:1479](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L1479) and [schema.py:1480](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L1480).
- `RunSessionRecord` already exposes `next_action` at [models.py:148](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/models.py#L148), and `get_run()` already reads it at [reasoning.py:268](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L268) and [reasoning.py:285](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L285).

5. The run-scoped stop endpoint at [service/api.py:2706](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2706) does not trigger a checkpoint.
- It only validates the run, calls `model.ledger.request_stop(run_id)` at [service/api.py:2720](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2720), and appends a `USER_INTERRUPTED` event at [service/api.py:2724](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2724).
- Same for the matter-level stop endpoint at [service/api.py:1496](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1496). The actual interrupt happens later inside the engine.

**Implementation Plan**
1. Add checkpoint config in [service/config.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py).
- Add `checkpoint_dir: str = "/tmp/irys/checkpoints"` after `matter_db_dir` near [service/config.py:33](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L33).
- Load `IRYS_CHECKPOINT_DIR` in `from_env()` next to [service/config.py:72](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L72)-[service/config.py:73](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L73).
- Validate non-empty alongside the matter DB check near [service/config.py:101](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/config.py#L101).

2. Thread `checkpoint_dir` into every service-created `Irys` in [service/api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py).
- Add a small helper next to [_wire_matter_model()]( /C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L239) that returns `str(Path(config.checkpoint_dir) / corpus_key)`.
- Pass that into all five existing constructors at [service/api.py:468](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L468), [service/api.py:761](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L761), [service/api.py:1049](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1049), [service/api.py:1226](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1226), [service/api.py:1376](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1376).
- For the sites where `corpus_key` is computed after the constructor today, move key computation before `Irys(...)`.

3. Persist the latest checkpoint path into `run_session.next_action`.
- Add `set_next_action(run_id, next_action)` to [reasoning.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py) next to [reasoning.py:201](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L201) / [reasoning.py:268](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L268).
- Add thin pass-throughs on [matter.py:230](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L230)-[matter.py:282](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L282), or call `ledger` directly from the engine.
- In [engine.py:4716](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4716), change `_save_checkpoint()` to also update `next_action` to `str(latest_path)` whenever `state._run_id` and `_matter_model` exist.
- In [reasoning.py:140](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L140) and [reasoning.py:173](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L173), clear `next_action` on completed/failed runs so only interrupted runs remain resumable. Leave [reasoning.py:187](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L187) preserving it.

4. Force a checkpoint on the actual stop path in [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py).
- Keep the periodic save at [engine.py:1699](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1699).
- Also call `_save_checkpoint()` in the interrupted branch at [engine.py:1061](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1061) before `state.interrupt()` / `interrupt_run(run_id)`.
- Change `_save_checkpoint()` to accept `iteration: int | None` so the stop branch can write `latest_<state.id>.json` even when there is no clean “iteration boundary” value.

5. Fix `resume_investigation()` so it behaves like normal `investigate()`.
- In [engine.py:4751](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4751)-[engine.py:4752](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4752), add `state._run_id = run_id`.
- After `_investigate_loop()`, add the same stop/interrupted branch used by normal investigate at [engine.py:1061](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1061), including forced checkpoint + `interrupt_run(run_id)`.
- After `complete_run()` in the resume path, mirror the normal completion tail from [engine.py:1111](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1111) and [engine.py:1127](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1127): generate clarifications, attach `pending_clarifications`, attach `reasoning_trail`.

6. Expose a public resume wrapper on [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py).
- Add `async def resume_investigation(self, checkpoint_path)` next to [api.py:123](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py#L123).
- It should `_ensure_initialized()`, call `_engine.resume_investigation(checkpoint_path)`, then format and return `InvestigationResult` just like normal investigate.
- This keeps the service route off the private `_engine` API.

7. Add the new route in [service/api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py), next to the other UI run routes around [service/api.py:2628](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2628) and [service/api.py:2706](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2706).
- Path: `POST /matter/{matter_id}/runs/{run_id}/resume`
- Response model: reuse [SyncInvestigateResponse]( /C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/models.py#L217)
- Validation:
  - run exists in this matter
  - run status is `interrupted`
  - run is not `manual_flush` / `background_flush`
  - `run.next_action` is non-empty
  - checkpoint file exists
  - no other non-utility `status='running'` row exists for this matter
  - `InvestigationState.load_checkpoint(run.next_action).repository_path` exists
- Handler steps:
  - get `model = await _get_matter_model_or_404(matter_id)`
  - build `Irys(..., checkpoint_dir=str(Path(run.next_action).parent))`
  - inject the already-open matter model so resume uses the same DB, not `MatterModel.open(repo_key)`
  - call `await irys.resume_investigation(run.next_action)`
  - serialize the returned result exactly the way sync handlers do at [service/api.py:1096](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1096) and [service/api.py:1410](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1410)
  - return the new `run_id` from `result.state._run_id`

**Blocking Risks**
- Checkpoint may not exist if stop lands between periodic saves. This is real today and is fixed only by engine-side forced checkpointing, not by the API stop handler.
- The bigger blocker is repository lifetime: resume rebuilds `MatterRepository(state.repository_path)` at [engine.py:4743](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4743), but service investigation handlers delete temp repos in `finally` at [service/api.py:516](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L516), [service/api.py:815](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L815), [service/api.py:819](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L819), [service/api.py:1070](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1070), [service/api.py:1073](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1073), [service/api.py:1272](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1272), [service/api.py:1392](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1392).
- That means the minimal route works for the shared-filesystem/local UI case, but not reliably for service-managed S3/upload/url runs unless you add one more layer:
  - either persist enough source metadata to redownload on resume
  - or stop cleaning temp repos for interrupted runs and add later cleanup
- Service async handlers also currently mark interrupted results as completed at [service/api.py:484](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L484), [service/api.py:782](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L782), [service/api.py:1242](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1242). If you want resume for service-owned runs, add `INTERRUPTED` to [service/models.py:9](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/models.py#L9) and branch on `result.state.status`.
- Shared matter DB root is still a prerequisite for the current UI hybrid path: local `Irys.investigate()` opens `MatterModel.open(repo_key)` at [api.py:153](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py#L153)-[api.py:157](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py#L157), while the service opens `config.matter_db_dir/corpus_key` at [service/api.py:255](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L255)-[service/api.py:260](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L260). Without that earlier fix, the service resume endpoint will look at a different matter DB than the in-process run.