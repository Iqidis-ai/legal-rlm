FAIL

1. `HIGH` The Run flow still bypasses `UIBackend` — `_run_thread()` calls `_get_irys()` directly.
   The `hasattr` guard only converts the crash to an error message.

2. `MEDIUM` `do_redirect()` still shows ✅ success even when backend returns `{"status": "error", ...}`.

3. `MEDIUM` Panel methods (list_issues, list_assertions, list_gaps, list_clarifications,
   get_steering_surface, get_quant_summary) catch exceptions and return empty payloads.
   UI sees "No open issues." / "No assertions." instead of "Error loading: <detail>".

**All three fixed in commit 8c05d21:**
- HIGH: _run_thread delegates to InProcessBackend.run_investigation_thread() via isinstance + callbacks
- MEDIUM: do_redirect() checks result.get("status")=="error"
- MEDIUM: panel methods now propagate exceptions to AppState loaders

Static review only; no tests were run by auditor.
