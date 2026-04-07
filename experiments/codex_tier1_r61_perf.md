**Findings**
- `LOW` Not fully CLEAN. The canonical stop path is now fixed, but the early-stop fallback still has an uncapped per-click daemon thread.

- Canonical path: yes, the old double-thread overhead is eliminated on the normal `matter_id/run_id` path. `stop_investigation()` now does a direct bounded-pool submit at [app.py:517](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L517), instead of waiting through `_run_async(...).result(...)` at [app.py:36](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L36) and [app.py:39](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L39). So the r60 MEDIUM about “OS waiter thread + executor thread” is closed for the canonical path.

- Early-stop fallback: no, that concern is not fully resolved. If `current_run_id` is still unset, the fallback still launches a raw daemon thread at [app.py:518](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L518) and [app.py:539](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L539). There is no cap or dedupe on that path in this code, so it remains unbounded in the narrow pre-run-id race window.

- Blocking in `stop_investigation()`: I do not see material handler blocking anymore. The method sets local stop flags at [app.py:505](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L505) and [app.py:506](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L506), schedules background work, and returns at [app.py:541](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L541). The SQLite `busy_timeout=5000` wait at [db.py:67](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/db.py#L67) is now off the click-handler path.

- Overall: `LOW`, not `CLEAN`. `c5009a8` resolves the prior r60 MEDIUM on the canonical stop path, but the early-stop daemon-thread fallback still leaves a smaller residual perf concern.

Static review only.