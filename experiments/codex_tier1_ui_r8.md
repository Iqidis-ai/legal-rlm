FAIL

1. HIGH: Issue IDs truncated to 12 chars in Issues table; backend requires exact ID for redirect.
   do_redirect() showed success even on error (pre-8c05d21).
2. MEDIUM: Assertion IDs same truncation problem.
3. MEDIUM: get_steering_surface() + get_quant_summary() in InProcessBackend still had try/except (pre-8c05d21).
4. MEDIUM: HttpBackend called /steering and /quant/summary — routes that don't exist in service.
   Service has /reconciliation and /damages-waterfall separately.

**Verified fixes from prior rounds:**
- redirect_run() validation in in_process.py ✓
- UIBackend abstract get_steering_surface/get_quant_summary ✓

**All four findings fixed:**
- Commit 8c05d21: do_redirect() error check + panel method exception propagation
- Commit 15e8c9c: full IDs in tables + HttpBackend correct routes + /steering-surface endpoint added to service

Static review only; no tests run by auditor.
