**Findings**

- MEDIUM: [`src/irys/service/api.py:120`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L120), [`src/irys/service/api.py:124`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L124), [`src/irys/service/api.py:284`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L284), [`src/irys/service/api.py:2057`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2057), [`src/irys/service/api.py:2276`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2276): the r50b fixes make job-backed eviction correct, but `_cleanup_loop` still only treats `_jobs` as “live.” Matter-only work such as `flush_pending_propagation`, `compute_proof_state`, or `_background_flush` can hold a model with no `_jobs` entry; those paths only bump `_matter_model_last_used` once on entry. If that work runs longer than `cleanup_after_seconds`, the orphan pass can evict the model from `_active_matter_models` while it is still in active use, allowing a concurrent request to rehydrate/open a second model for the same matter. That is still a premature-eviction gap.

- LOW: [`src/irys/service/api.py:101`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L101): expired-job detection still uses `timedelta.seconds` instead of elapsed `total_seconds()`. With the default `300s` cleanup window this is usually masked, but if `IRYS_CLEANUP_SECONDS` is configured above 24 hours, completed jobs and their models will never expire.

**Summary**

The three specific `766b98c` fixes do verify:

- The helper that now wires models (`_wire_matter_model`, not `_initialize_matter_model` in current source) stamps last-used on registration at [`src/irys/service/api.py:243`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L243).
- Expired-job eviction now guards on `_other_live` before deleting the model at [`src/irys/service/api.py:103`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L103).
- The orphan pass now runs after expired jobs are removed from `_jobs` at [`src/irys/service/api.py:116`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L116) and [`src/irys/service/api.py:120`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L120).

The `set_trust_override` comment fix is correct at [`src/irys/matter/matter.py:689`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L689): when `affected_ids` is empty, the proof-state recompute block is skipped, so the behavior is now accurately described as a no-op.

Overall rating: MEDIUM. The job-eviction edge cases from `766b98c` are fixed, but eviction semantics are still not fully correct for non-job matter activity. Static review only; I did not run tests.