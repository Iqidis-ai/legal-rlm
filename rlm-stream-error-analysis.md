ubuntu@ip-172-31-26-16:~$ cat rlm-stream-error-analysis.md 
# RLM Stream Error Analysis
Generated: 2026-04-15 | Based on logs from last 48 hours + nginx error log

---

## Two Distinct Error Types

### Type A: `UND_ERR_SOCKET: terminated`
The Vercel function's undici HTTP client had its TCP socket forcibly closed by the server mid-stream.

Observed in:
- Chat `e26acdca` (User `7538c94e`) — 7:15 AM + 8:05 AM (retry), bytesRead 362K and 229K
- Chat `1213b656` (User `a960d584`) — 10:36 AM (bytesRead 25K < bytesWritten 32K = very early kill) + 2:03 PM retry
- Chat `caf46ddc` (User `a960d584`) — 2:03 PM, bytesRead 291K

### Type B: `RLM stream ended without complete event`
The SSE stream closed cleanly at the TCP level but without emitting a `complete` event.

Observed in:
- Chat `4b043507` (User `a960d584`) — 4:37 PM
- Chat `8ae7a14e` (User `a960d584`) — 4:37 PM

---

## Confirmed Root Causes

### Root Cause 1 — Document search with huge hit sets blocks the asyncio event loop (PRIMARY)

**Confirmed from logs.** This is the dominant cause of Type A errors.

Trace from the investigation started at 01:37:31 UTC (139 documents):

```
01:37:31  Stream request received, 139 documents downloaded
01:40:35  analyze_external completed (LLM call done)
--- 5 minutes 48 seconds of total silence ---
01:45:35  nginx: "upstream timed out (110: Connection timed out)"  <-- UND_ERR_SOCKET
01:46:23  analyze_search: combined analysis of 15165 hits
01:46:23  analyze_search: combined analysis of 24507 hits
01:46:23  analyze_search: combined analysis of 9546 hits
```

The investigation resumed at 01:46:23 — **48 seconds too late**. The document text search
across 139 documents generated 15,000–24,000 hits. Processing these (ranking, scoring, 
filtering) is synchronous CPU work that blocks the asyncio event loop. While blocked:
- No heartbeats sent (`event_generator()` can't yield)
- nginx waits for 300 seconds (proxy_read_timeout), gets nothing, sends RST
- Vercel's undici receives RST → `UND_ERR_SOCKET: terminated`
- FastAPI detects disconnect → cancels streaming generator → cancels investigation task
- Investigation status stays `initialized` (never reached `completed` or `failed`)

The same pattern confirmed for two other nginx timeouts:

```
05:06:05  nginx: upstream timed out — client: 3.88.185.3 (Vercel IP)
           → matches investigation inv_880414c3 (968s, status=initialized, 30 steps)
             and inv_602582af (404s, status=initialized, 6 steps)

08:33:48  nginx: upstream timed out — clients: 100.52.230.27 + 54.211.77.15 (Vercel IPs)
           → matches inv_f6ca1fc4 (738s, status=initialized, 31 steps)
             and inv_a1b85db5 (352s, status=initialized, 10 steps)
```

All `status=initialized` investigations in the logs are investigations that were cancelled
by this exact mechanism.

**Longest investigations seen (many would hit the 300s nginx wall):**
```
968s (16 min)  status=initialized  30 steps
847s (14 min)  status=completed    36 steps
787s (13 min)  status=completed
765s (12.5m)   status=completed
752s (12.5m)   status=failed
738s (12.3m)   status=initialized  31 steps
474s (7.9m)    status=failed
456s (7.6m)    status=completed
404s (6.7m)    status=initialized
352s (5.9m)    status=initialized
```

### Root Cause 2 — Service restarts terminate active SSE connections

**Confirmed from nginx error log:**

```
02:35:39  nginx: "upstream prematurely closed connection" — client: 3.95.148.66
          → matches systemd: "Killing process 19479 with SIGKILL" at exactly 02:35:39
```

The service was restarted (`systemctl restart`). Uvicorn received SIGTERM but couldn't 
exit within TimeoutStopSec (90s default) because active streaming connections were running.
Systemd sent SIGKILL after 90 seconds. SIGKILL closes all TCP connections instantly.
The Vercel function mid-read on the SSE stream gets a RST packet → `UND_ERR_SOCKET`.

**Service restart frequency in 48 hours — 22 restarts:**
```
Apr 13: 15:30, 16:29, 22:22, 22:28, 22:42, 22:45, 23:44, 23:48, 23:50, 23:50
Apr 14: 00:12, 00:18, 00:29, 08:18, 08:30, 09:20, 11:14, 12:30, 12:34, 17:25
Apr 15: 02:35 (SIGKILL), 12:49
```

Every restart is a potential `UND_ERR_SOCKET` for any concurrent user.

### Root Cause 3 — Gemini API 503/timeout causes "RLM stream ended without complete event"

**Confirmed pattern from Apr 13 21:24:23:**

```
ERROR - Stream investigation failed: 503 UNAVAILABLE.
{'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}
```

Chain:
1. Gemini API returns 503 or times out during any LLM step
2. Exception propagates up through `irys.investigate()`
3. `run_investigation()` catches it → puts `{"event": "error", "data": {...}}` on queue
4. Puts `None` sentinel → stream closes cleanly at TCP level
5. `investigateUrlsStream` on Vercel receives the error event
6. The `onError` callback **only logs** the error — it does not throw or set a flag
7. `investigateUrlsStream` continues waiting for a `complete` event that never comes
8. When the stream ends (None sentinel), the function throws:
   "RLM stream ended without complete event"

The 4:37 PM errors on Apr 14 correlate with a service restart at 20:42 UTC (5 min after),
suggesting those errors triggered a manual redeploy.

Also seen: `gemini-2.5-flash-lite timed out` and `gemini-3-flash-preview timed out` in 
logs — both handled by fallback model logic, so those specific calls don't cause Type B
errors, but they do contribute to investigation slow-down which worsens Root Cause 1.

---

## Infrastructure Configuration Issues Found

### nginx `proxy_read_timeout 300s` — Too short for long investigations
Location: `/etc/nginx/sites-enabled/` (applies to all routes including SSE stream)

This is the 300-second hard wall. For any investigation that produces a 300+ second gap
in the event stream (because the asyncio event loop is blocked by heavy search processing),
nginx terminates the upstream connection with `Connection timed out`.

The `X-Accel-Buffering: no` header from the backend is correct and nginx respects it.
The heartbeat mechanism (`": heartbeat\n\n"` every 0.1s) is correctly designed.
The problem is that the heartbeat **cannot be sent** when the event loop is blocked.

### S3 connection pool exhaustion (contributing factor)
```
urllib3.connectionpool - WARNING - Connection pool is full, 
discarding connection: iqidis-artifact.s3.amazonaws.com. Connection pool size: 10
```
Appears dozens of times during high-load periods. During concurrent investigations with
many documents, S3 downloads saturate the boto3 connection pool. This slows document
downloads and forces new TCP connections, adding latency and CPU overhead.

---

## Potential Fixes (Discussion Only)

### Fix 1 — Unblock the event loop during document search (addresses Root Cause 1)

The core issue: `analyze_search` and similar operations that process 10,000–25,000 search
hits do so synchronously in the asyncio event loop. Wrapping CPU-heavy document processing
in `asyncio.to_thread()` would allow the event loop to keep running (and sending heartbeats)
while the CPU work happens on a thread pool thread.

The specific callsites to examine: wherever `DocumentSearch.smart_search()` or similar
produces and processes large result sets. Even a single `await asyncio.sleep(0)` injected
periodically during hit processing would yield control back to the event loop.

Additionally, any MuPDF (pymupdf/fitz) synchronous PDF reading that isn't already wrapped
in `asyncio.to_thread()` would also block the loop during large document loads.

### Fix 2 — Increase nginx `proxy_read_timeout` for the stream endpoint (addresses Root Cause 1 symptom)

Add a location-specific override in nginx for `/investigate/urls/stream`:

```nginx
location /investigate/urls/stream {
    proxy_pass http://irys_api;
    proxy_read_timeout 900s;   # 15 minutes — covers even the longest investigations
    proxy_send_timeout 900s;
    proxy_buffering off;       # Belt-and-suspenders alongside X-Accel-Buffering header
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
}
```

This alone does NOT fix the blocking — if the loop is blocked long enough (e.g., 16-minute
investigation with multiple 6-minute event loop stalls), even 900s would be hit. Fix 1 is
the real fix; this is a safety net.

### Fix 3 — Graceful shutdown during service restarts (addresses Root Cause 2)

Add to `irys-rlm.service`:
```ini
TimeoutStopSec=600
KillMode=mixed
```

This gives uvicorn 10 minutes to drain active streaming connections before systemd sends
SIGKILL. Combined with uvicorn's `--graceful-timeout` flag in `run_server.py`, active
streams would complete before the new version starts.

For zero-downtime deploys, a blue/green approach would be needed (run two instances, 
drain one, switch nginx upstream, then restart the old one).

### Fix 4 — Handle the `error` SSE event properly in `investigateUrlsStream` (addresses Root Cause 3)

The Vercel `onError` callback currently only logs:
```javascript
onError: (error) => {
  Logger.error("RLM stream error in SSE handler", { error, chatId });
},
```

When the `error` SSE event is received, `investigateUrlsStream` should throw immediately
rather than waiting for a `complete` event that will never arrive. This would convert the
"RLM stream ended without complete event" error into a proper, descriptive error message
that reaches the user instead of a silent failure.

### Fix 5 — Increase S3 connection pool size (addresses contributing factor)

In `s3_repository.py`, the httpx client (used for URL downloads) already has a 60s timeout.
But the urllib3 pool used for boto3 defaults to 10 connections. When downloading 100+
documents concurrently, this saturates and forces sequential retries. Configure boto3 with
a larger connection pool:

```python
from botocore.config import Config
config = Config(max_pool_connections=50)
s3_client = boto3.client('s3', config=config)
```

---

## Quick Diagnostic Commands

```bash
# Watch nginx error log for new upstream timeouts in real time
sudo tail -f /var/log/nginx/error.log | grep upstream

# Watch for investigations being killed (status=initialized means cancelled)
sudo journalctl -u irys-rlm.service -f | grep -E "investigation_complete|SIGKILL|Stopped|started"

# Check current event loop blockage by watching for heartbeat gaps
# (if log timestamps jump by >60s with no output, the loop is blocked)
sudo journalctl -u irys-rlm.service -f | grep -E "LLM_CALL|LLM_DONE|analyze_search|checkpoint"

# Check service restart history
sudo journalctl -u irys-rlm.service --since "7 days ago" | grep -E "Stopped|Started|SIGKILL" | wc -l
```

---

## Priority Order

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | Unblock event loop in document search (asyncio.to_thread) | Eliminates Root Cause 1 | Medium |
| 2 | Increase nginx proxy_read_timeout for /investigate/urls/stream | Reduces RC1 impact immediately | Low |
| 3 | Fix onError in investigateUrlsStream to throw on error event | Eliminates Type B errors | Low |
| 4 | Add TimeoutStopSec=600 to service file | Reduces Root Cause 2 | Low |
| 5 | Increase boto3 S3 connection pool size | Reduces investigation slowdown | Low |