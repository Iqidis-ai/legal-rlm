SO-3 end-to-end is `FAIL` on `fef9152` for the audited UI workflow. Start and stop are real. The workflow breaks after stop: redirecting the stopped run is rejected once it becomes `interrupted`, and resume is broken in the default in-process UI because no checkpoint directory is configured.

1. `STEP 1 — Start: PASS`
`stream_investigation()` calls the in-process thread runner, which reaches `irys.investigate()`. The engine creates the run with `self._matter_model.start_run(query)` and does not pass an objective, so `objective` stays `None` all the way into the `run_session` insert. That makes the run steerable rather than a utility run.
[src/irys/ui/app.py:357](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L357)
[src/irys/ui/backends/in_process.py:395](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L395)
[src/irys/rlm/engine.py:1031](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1031)
[src/irys/matter/matter.py:230](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L230)
[src/irys/matter/reasoning.py:40](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L40)

2. `STEP 2 — Stop: PASS`
The stop request is real. UI stop submits `backend().stop_run()`, `request_stop()` sets `stop_requested=1`, `MatterRuntimeAdapter.is_stop_requested()` reads and caches that flag, and the engine checks it before each iteration and again before verify/synthesis. When seen, the engine calls `state.interrupt()` and `self._matter_model.interrupt_run(run_id)`, which moves the run to `interrupted`.
[src/irys/ui/app.py:490](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L490)
[src/irys/ui/backends/in_process.py:130](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L130)
[src/irys/matter/runtime.py:729](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py#L729)
[src/irys/rlm/engine.py:1500](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1500)
[src/irys/rlm/engine.py:1058](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1058)
[src/irys/matter/reasoning.py:190](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L190)

3. `STEP 3 — Post-stop state: PARTIAL (MEDIUM)`
`current_run_id` is intentionally preserved, and the UI will happily stuff it into the redirect form. So the user can submit `do_redirect()` with that stopped run ID. But this is only a dead handle unless the redirect wins a race before the run leaves `running`.
[src/irys/ui/app.py:497](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L497)
[src/irys/ui/app.py:878](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L878)
[src/irys/ui/app.py:654](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L654)
[src/irys/ui/backends/in_process.py:341](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L341)
[src/irys/matter/reasoning.py:243](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L243)

4. `STEP 4 — Redirect after stop: FAIL (HIGH)`
This breaks the sacred SO-3 workflow. Both the service route and in-process backend require the run to still be `running`, and the ledger update itself also requires `status='running'`. After a real stop completes, the run is `interrupted`, so redirecting that stopped run does not work. In the current UI it is not silent; `do_redirect()` will surface an error. But it still means “stop, then redirect without restarting from scratch” is not implemented.
[src/irys/service/api.py:1534](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1534)
[src/irys/service/api.py:1544](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1544)
[src/irys/ui/backends/in_process.py:349](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L349)
[src/irys/matter/reasoning.py:250](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L250)
[src/irys/ui/app.py:661](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L661)

5. `STEP 5 — Resume: FAIL (HIGH)`
`interrupt_run()` itself is correct: it does not clear `next_action`, so an existing checkpoint would survive interruption. But the audited path is the Gradio in-process path, and that backend creates `IrysConfig` without `checkpoint_dir`. The engine therefore no-ops `_save_checkpoint()`, `run.next_action` stays empty, and `resume_run()` rejects the run. Worse, `do_resume()` swallows that validation failure and still tells the user resume was “launched in background.”
[src/irys/ui/backends/in_process.py:83](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L83)
[src/irys/api.py:35](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/api.py#L35)
[src/irys/rlm/engine.py:1063](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1063)
[src/irys/rlm/engine.py:4729](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4729)
[src/irys/matter/reasoning.py:190](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L190)
[src/irys/ui/backends/in_process.py:155](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L155)
[src/irys/ui/app.py:640](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L640)

Bottom line: `STEP 1 PASS`, `STEP 2 PASS`, `STEP 3 PARTIAL (MEDIUM)`, `STEP 4 FAIL (HIGH)`, `STEP 5 FAIL (HIGH)`. The current HEAD does not satisfy adversarial #033’s stop-then-redirect or stop-then-resume workflow in the actual in-process UI path.