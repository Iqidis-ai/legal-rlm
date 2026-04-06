PASS

r10 correctness — Tier 1 UI CLEAN declared.

Three MEDIUM findings from r9 correctness re-run (hook-triggered Codex session), all fixed:

**MEDIUM 1: Quant panel key mismatch**
- `_fmt_quant()` checked `payment_recon.get("total_invoiced")`, `"total_paid"`, `"net_exposure"`, `"conflict_count"`
- `reconcile_payment_chain()` returns `"invoiced"`, `"paid"`, `"disputed"`, `"exposure"`, `"currency"`
- Result: payment reconciliation section never rendered even when data existed
- Fix: Updated `_fmt_quant()` in app.py to use correct keys; added `disputed` row when non-zero

**MEDIUM 2: issue_id assertion filter false contract**
- `UIBackend.list_assertions(issue_id=)` advertised issue-scoped filtering
- `HttpBackend` sent `issue_id` param to service; service ignored it; `InProcessBackend` ignored it
- Callers got full unfiltered list silently
- Fix: Removed `issue_id` param from `list_assertions` in base.py, in_process.py, and http.py
  (false contract removed; no UI code actually passed issue_id anyway)

**MEDIUM 3: HttpBackend.get_quant_summary() exception masking**
- Both `/reconciliation` and `/damages-waterfall` fetches were wrapped in `try/except: return {}/[]`
- Broken routes or service failures showed as "no quant data" instead of propagating error
- Fix: Removed exception wrapping; errors now propagate to AppState loader for display

All 3 fixes committed. 725/725 tests passing.
