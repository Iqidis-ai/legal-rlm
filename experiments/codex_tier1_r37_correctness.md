Tier 1 is not clean.

**Findings**
1. Medium: async investigation responses still synthesize `job.run_id` with `recent_runs(1)` instead of the exact engine-owned `state._run_id`. [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L1033), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L421), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L732), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1128). With overlapping runs on one matter, the client can be handed a valid-but-wrong `run_id`, and that wrong `run_id` will pass the new r36 validation later because it is real, running, and on the same matter.

2. Medium: `set_trust_override` still lacks exact-run attribution. [models.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/models.py#L129), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1437), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1446), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1458), [matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L427). It does not accept a client `run_id`; it always binds trust-override-triggered revisions to the latest running row for the matter. If two runs overlap on the same matter, attribution can still land on the wrong run.

3. Medium: the in-process correction path bypasses the new validation entirely. [in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L218), [in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L232), [matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L303), [schema.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L107), [schema.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L128). Service `correct_assertion` is fixed, but Gradio/in-process callers can still pass any `run_id` straight into audit writes.

**Answers**
1. `correct_assertion` service validation: yes for the narrow r36 bug. [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1753) normalizes `""` to `None`; [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1758) validates `id + matter_id + status='running'`; stale/foreign/non-running ids are discarded before use. Nuance: it does not 4xx on an invalid explicit id; it silently falls back to the same latest-running lookup used for omission.

2. `set_trust_override`: no equivalent exact-run validation. It does not accept client `run_id`, so it is not vulnerable in the same way, but it still cannot target a specific run and still uses latest-running attribution.

3. Remaining correctness issues in the full run-id attribution chain: yes, the three findings above remain.

4. Tier 1 CLEAN: no.

I did not run tests; this was a static review. I also did not find service coverage for invalid/foreign/empty `run_id` correction cases or concurrent-run trust-override attribution in [test_matter_api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/tests/service/test_matter_api.py#L310) and [test_matter_api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/tests/service/test_matter_api.py#L353).