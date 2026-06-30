# Telemetry Requirements — Investigation Cost & Latency Tracking

## Goal

Track cost and latency for every investigation, per step and per operation within each step.
Data is collected in-memory during execution, persisted to DB asynchronously at the end.
Never blocks the investigation. Never degrades performance.

---

## Key Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Token source | `response.usage_metadata` from Gemini SDK | Current `len(prompt) // 4` is inaccurate |
| Token fields | Store raw (`prompt_tokens`, `thinking_tokens`, `output_tokens`) | Cost is derived; raw counts survive price changes |
| Cost field | Computed per-operation from `MODEL_CONFIGS` pricing | Co-located with the operation that generated it |
| Concurrency | Telemetry object created per `investigate()` call, in local scope | No shared state; no race conditions between concurrent investigations |
| DB write | `asyncio.create_task()` fire-and-forget in `finally` block | Non-blocking; investigation result is returned before DB write completes |
| Log emit | `logger.info()` with full telemetry dict, always | Zero-dependency observability; plug in any log aggregator later |
| Existing `TelemetryCollector` in `utils.py` | Leave as-is, do not extend | Too generic; new purpose-built classes are cleaner |
| `UsageStats` on `GeminiClient` | Superseded by new telemetry; can be deprecated | Shared across investigations, estimated tokens, hardcoded Flash pricing |

---

## In-Memory Dataclass Hierarchy

Lives in `src/irys/core/telemetry.py` (new file).

```
InvestigationTelemetry
  investigation_id : str
  message_id       : str | None
  started_at       : datetime
  completed_at     : datetime | None      # set by finalize()
  status           : str | None           # "completed" | "failed"
  steps            : list[InvestigationStep]

  finalize() -> TelemetrySummary          # computes totals, returns summary dict

InvestigationStep
  seq              : int                  # 1, 2, 3... global execution order
  step_name        : str                  # "planning" | "analyze_document" | "sufficiency_check" | "synthesize" | ...
  phase            : str                  # "planning" | "investigation_loop" | "synthesis"
  started_at       : datetime
  step_latency_ms  : int                  # total wall clock for this step (I/O + API + CPU)
  operations       : list[StepOperation]

StepOperation  — discriminated by type field
  type             : str                  # "llm" | "ext_search"
  started_at       : datetime
  latency_ms       : int                  # pure API call time

  # When type == "llm":
  tier             : str                  # "LITE" | "FLASH" | "PRO"
  model_id         : str                  # actual model string e.g. "gemini-2.5-pro-preview-05-06"
  prompt_tokens    : int                  # from response.usage_metadata.prompt_token_count
  thinking_tokens  : int                  # from response.usage_metadata.thoughts_token_count (0 if N/A)
  output_tokens    : int                  # from response.usage_metadata.candidates_token_count
  cost_usd         : float                # derived: lookup MODEL_CONFIGS[tier] pricing
  cached           : bool                 # True = cache hit, no API call made

  # When type == "ext_search":
  service          : str                  # "tavily" | "courtlistener"
  query            : str
  result_count     : int
  usage_raw        : dict | None          # raw usage object returned by the API
```

`TelemetrySummary` (output of `finalize()`, what gets stored and logged):

```
investigation_id   : str
message_id         : str | None
started_at         : datetime
completed_at       : datetime
status             : str
total_duration_ms  : int
total_cost_usd     : float               # sum of all operation cost_usd
total_steps        : int
phase_breakdown    : dict                # { phase: { duration_ms, step_count } }
steps              : list[dict]          # full serialized step list (see sample below)
```

---

## Sample Serialized Output

For a real 80s investigation (1 doc, 3 web searches, FLASH planning + analysis, LITE sufficiency check, PRO synthesis):

```json
{
  "investigation_id": "inv_c2bfdf54",
  "message_id": null,
  "started_at": "2026-03-20T16:27:39.000Z",
  "completed_at": "2026-03-20T16:28:59.000Z",
  "status": "completed",
  "total_duration_ms": 79420,
  "total_cost_usd": 0.0031,
  "total_steps": 4,
  "phase_breakdown": {
    "planning":           { "duration_ms": 5700,  "step_count": 1 },
    "investigation_loop": { "duration_ms": 45500, "step_count": 2 },
    "synthesis":          { "duration_ms": 28300, "step_count": 1 }
  },
  "steps": [
    {
      "seq": 1, "step_name": "planning", "phase": "planning",
      "started_at": "2026-03-20T16:27:39.000Z", "step_latency_ms": 5700,
      "operations": [
        { "type": "llm", "started_at": "...", "latency_ms": 5650,
          "tier": "FLASH", "model_id": "gemini-2.5-flash-preview-04-17",
          "prompt_tokens": 8200, "thinking_tokens": 0, "output_tokens": 700,
          "cost_usd": 0.00073, "cached": false }
      ]
    },
    {
      "seq": 2, "step_name": "analyze_document", "phase": "investigation_loop",
      "started_at": "2026-03-20T16:27:44.700Z", "step_latency_ms": 40200,
      "operations": [
        { "type": "ext_search", "started_at": "...", "latency_ms": 1800,
          "service": "tavily", "query": "Texas RV advertising lowest price regulations",
          "result_count": 5, "usage_raw": { "credits": 1 } },
        { "type": "llm", "started_at": "...", "latency_ms": 37800,
          "tier": "FLASH", "model_id": "gemini-2.5-flash-preview-04-17",
          "prompt_tokens": 29800, "thinking_tokens": 0, "output_tokens": 1200,
          "cost_usd": 0.00071, "cached": false }
      ]
    },
    {
      "seq": 3, "step_name": "sufficiency_check", "phase": "investigation_loop",
      "started_at": "2026-03-20T16:28:25.700Z", "step_latency_ms": 1100,
      "operations": [
        { "type": "llm", "started_at": "...", "latency_ms": 1090,
          "tier": "LITE", "model_id": "gemini-2.5-flash-lite-preview-06-17",
          "prompt_tokens": 800, "thinking_tokens": 0, "output_tokens": 200,
          "cost_usd": 0.000008, "cached": false }
      ]
    },
    {
      "seq": 4, "step_name": "synthesize", "phase": "synthesis",
      "started_at": "2026-03-20T16:28:30.400Z", "step_latency_ms": 28300,
      "operations": [
        { "type": "llm", "started_at": "...", "latency_ms": 28280,
          "tier": "PRO", "model_id": "gemini-2.5-pro-preview-05-06",
          "prompt_tokens": 14000, "thinking_tokens": 3200, "output_tokens": 2000,
          "cost_usd": 0.00166, "cached": false }
      ]

---

## Database Schema

Three tables. Follows the existing SQLAlchemy `mapped_column` pattern in `src/irys/db/models/document.py`.
Table prefix matches existing convention: `irys_rlm_*`.

### Table 1: `irys_rlm_investigation_logs`
One row per investigation. Stores the summary and is the FK parent for the other two tables.

```python
class InvestigationLog(Base):
    __tablename__ = "irys_rlm_investigation_logs"

    id                : String(36)   PK                     # investigation_id
    message_id        : String(255)  nullable  index        # caller-supplied ID for cross-referencing
    started_at        : DateTime(tz) NOT NULL  index
    completed_at      : DateTime(tz) nullable
    status            : String(32)   NOT NULL               # "completed" | "failed"
    total_duration_ms : Integer      nullable
    total_cost_usd    : Numeric(10,6) nullable
    total_steps       : Integer      nullable
    phase_breakdown   : JSON         nullable               # { phase: { duration_ms, step_count } }
    created_at        : DateTime(tz) default=utcnow
```

### Table 2: `irys_rlm_investigation_steps`
One row per `InvestigationStep`. FK to `irys_rlm_investigation_logs`.

```python
class InvestigationStep(Base):
    __tablename__ = "irys_rlm_investigation_steps"

    id                : String(36)   PK                     # UUID
    investigation_id  : String(36)   NOT NULL  FK  index   # → irys_rlm_investigation_logs.id
    seq               : Integer      NOT NULL               # execution order within investigation
    step_name         : String(128)  NOT NULL  index        # "planning", "analyze_document", etc.
    phase             : String(64)   NOT NULL               # "planning" | "investigation_loop" | "synthesis"
    started_at        : DateTime(tz) NOT NULL
    step_latency_ms   : Integer      nullable
    created_at        : DateTime(tz) default=utcnow
```

### Table 3: `irys_rlm_investigation_operations`
One row per `StepOperation`. FK to step; `investigation_id` denormalized for direct queries without joins.

```python
class InvestigationOperation(Base):
    __tablename__ = "irys_rlm_investigation_operations"

    id                : String(36)   PK                     # UUID
    step_id           : String(36)   NOT NULL  FK  index   # → irys_rlm_investigation_steps.id
    investigation_id  : String(36)   NOT NULL  index        # denormalized for direct queries
    type              : String(32)   NOT NULL  index        # "llm" | "ext_search"
    started_at        : DateTime(tz) NOT NULL
    latency_ms        : Integer      nullable
    details           : JSON         NOT NULL               # all type-specific fields (see below)
    created_at        : DateTime(tz) default=utcnow
```

**`details` JSON shape by type:**

```json
// type == "llm"
{
  "tier": "PRO",
  "model_id": "gemini-2.5-pro-preview-05-06",
  "prompt_tokens": 14000,
  "thinking_tokens": 3200,
  "output_tokens": 2000,
  "cost_usd": 0.00166,
  "cached": false
}

// type == "ext_search"
{
  "service": "tavily",
  "query": "Texas RV advertising lowest price regulations",
  "result_count": 5,
  "usage_raw": { "credits": 1 }
}
```

**Why `details` as JSON:** Operation fields differ by type. Flat nullable columns would create a sparse table.
Postgres supports JSON path queries (`WHERE details->>'tier' = 'PRO'`) natively, matching the existing
`pages_json` / `metadata_json` pattern already used in `StoredDocument`.

### Example queries

```sql
-- PRO model calls this week with cost
SELECT s.step_name, o.details->>'model_id', o.details->>'cost_usd', o.latency_ms
FROM irys_rlm_investigation_operations o
JOIN irys_rlm_investigation_steps s ON o.step_id = s.id
WHERE o.type = 'llm' AND o.details->>'tier' = 'PRO'
  AND o.started_at > NOW() - INTERVAL '7 days';

-- Average synthesis latency
SELECT AVG(step_latency_ms) FROM irys_rlm_investigation_steps
WHERE step_name = 'synthesize';

-- All investigations for a specific message
SELECT * FROM irys_rlm_investigation_logs WHERE message_id = 'msg_abc123';

-- Tavily usage by investigation
SELECT investigation_id, COUNT(*), SUM((details->>'result_count')::int)
FROM irys_rlm_investigation_operations
WHERE type = 'ext_search' AND details->>'service' = 'tavily'
GROUP BY investigation_id;
```

---

## Implementation Order

### Step 1 — Fix actual token counts in `GeminiClient.complete()`
**File:** `src/irys/core/models.py`

Replace lines that do `len(prompt) // 4` with `response.usage_metadata.prompt_token_count`,
`candidates_token_count`, and `thoughts_token_count`. Also remove the hardcoded Flash pricing
from `UsageStats.estimated_cost` — this class is superseded by the new telemetry system.
One commit.

---

### Step 2 — Create `InvestigationTelemetry` dataclass
**File:** `src/irys/core/telemetry.py` (new)

Define `StepOperation`, `InvestigationStep`, `InvestigationTelemetry`, and `TelemetrySummary`
exactly as specified in the schema section above. Include a `finalize()` method on
`InvestigationTelemetry` that computes `total_duration_ms`, `total_cost_usd`, and `phase_breakdown`
by iterating over `steps`. No wiring yet — just the dataclasses. One commit.

---

### Step 3 — Thread `StepOperation` capture into `GeminiClient.complete()`
**File:** `src/irys/core/models.py`

Add two optional parameters to `complete()`:
- `active_step: Optional[InvestigationStep] = None`
- `step_operation_name: Optional[str] = None` (for labelling, may not be needed)

After the API call returns, if `active_step is not None`, append one `StepOperation` to
`active_step.operations` with actual token counts, latency, tier, model_id, and derived cost.
**Zero-overhead path when `active_step is None`** — all existing callers remain unaffected.
One commit.

---

### Step 4 — Wire `InvestigationStep` creation into `RLMEngine.investigate()`
**File:** `src/irys/rlm/engine.py`

At the top of `investigate()`, create `InvestigationTelemetry`. For each logical engine phase:
- Create an `InvestigationStep` with `started_at = datetime.now(UTC)` and the correct `step_name` / `phase`
- Pass it into the relevant `decisions.*` calls (which forward it to `client.complete()`)
- After the phase completes, record `step_latency_ms` and append to `telemetry.steps`

Phases to instrument: `_assess_and_create_plan`, each document read in `_investigate_loop`,
each external search call, `_synthesize`. Also capture external search operations (Tavily,
CourtListener) as `StepOperation(type="ext_search", ...)` on the relevant step.

In the `finally` block, call `telemetry.finalize()` and attach the summary to `InvestigationState`.
One commit.

---

### Step 5 — Add `message_id` to the call chain
**Files:** `src/irys/rlm/engine.py`, `src/irys/service/models.py`

Add `message_id: Optional[str] = None` to `RLMEngine.investigate()`. Pass it into
`InvestigationTelemetry` at construction. Add `message_id` field to `S3UrlsInvestigateRequest`
and `InvestigateRequest` in `service/models.py`, and forward it through `service/api.py`.
One commit.

---

### Step 6 — Emit structured JSON log at investigation end
**File:** `src/irys/rlm/engine.py`

In the `finally` block, after `telemetry.finalize()`:

```python
logger.info("investigation_complete", extra={"telemetry": summary.to_dict()})
```

This is the immediate observability output. Parseable by any log aggregator without code changes.
One commit.

---

### Step 7 — Add three DB models and async write
**Files:** `src/irys/db/models/investigation_log.py` (new), `src/irys/rlm/engine.py`

Add `InvestigationLog`, `InvestigationStep`, `InvestigationOperation` SQLAlchemy models
following the `mapped_column` pattern in `src/irys/db/models/document.py`.

In the engine `finally` block, after the log emit:

```python
if is_database_configured():
    asyncio.create_task(_persist_telemetry(summary))
```

Use the existing `try_get_database_config()` / `session_scope()` helpers. If the DB write fails,
log a warning — never raise. One commit.

---

## Implementation Notes for the AI

- The existing `TelemetryCollector` in `src/irys/core/utils.py` is **not** to be modified or extended.
- `GeminiClient` is shared across concurrent investigations. Do **not** store `active_step` on the
  client instance. Pass it as a parameter to `complete()`.
- `asyncio.create_task()` requires a running event loop. The engine already runs in async context.
- Implementer has freedom to adjust field names or split files differently, but the three-table DB
  structure and the nested `steps → operations` in-memory shape are fixed requirements.
- Token counts on cache hits: set all token fields to `0` and `cached=True`. Do not call
  `usage_metadata` on cache-served responses (there is no API call to get metadata from).

    }
  ]
}
```

