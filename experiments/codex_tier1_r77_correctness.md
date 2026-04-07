CLEAN.

I walked the resume paths in [src/irys/rlm/engine.py#L4875](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L4875) against the ledger semantics in [src/irys/matter/reasoning.py#L204](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L204). The restructured `try`/`except` now handles the requested cases correctly:
- (a) CAS loser: `clear_next_action()` returns `False`, `_claimed` stays `False`, `ConcurrentResumeError` is re-raised through the dedicated handler, and no restore runs.
- (b) `start_run()` fails after CAS: `_claimed=True`, `run_id=None`; the generic `except` restores `next_action`, and redirect restoration is ordered after `set_next_action()`, which matches the interrupted-run guard.
- (c) investigation work fails after `run_id` is set: the new run is failed, then the original interrupted run gets `next_action` and any captured redirect restored.

The service-side conflict mapping in [src/irys/service/api.py#L2844](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2844) is also correct. The deferred `from ..rlm.engine import ConcurrentResumeError` resolves the same class object the engine raises, so `isinstance(exc, ConcurrentResumeError)` will match and return HTTP 409.

Residual risk: I did not find targeted tests for the concurrent-resume loser path or the “CAS succeeded, `start_run()` failed” restore path, so this is clean by inspection rather than by demonstrated coverage.