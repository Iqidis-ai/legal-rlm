"""FastAPI REST API for Irys RLM service."""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import json as _json

from .config import ServiceConfig, get_config
from .models import (
    InvestigateRequest,
    InvestigateResponse,
    JobResult,
    JobStatus,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    ErrorResponse,
    UploadInvestigateResponse,
    UploadSearchResponse,
    SyncInvestigateResponse,
    S3UrlsInvestigateRequest,
    S3UrlsSearchRequest,
    MatterStatsResponse,
    StopRunRequest,
    RedirectRunRequest,
    AnswerClarificationRequest,
    CorrectAssertionRequest,
    TrustOverrideRequest,
    DocumentAnnotationRequest,
)
from .s3_repository import S3Repository

logger = logging.getLogger(__name__)

# In-memory job storage (use Redis in production for multi-worker)
_jobs: dict[str, JobResult] = {}
_start_time: float = time.time()

# Matter model registry: matter_id → MatterModel (active investigations only)
_active_matter_models: dict[str, Any] = {}
# Last-access times for matter models not backed by an active job (rehydrated models).
# Used by _cleanup_loop to evict idle rehydrated models so they don't accumulate.
_matter_model_last_used: dict[str, datetime] = {}
# Per-matter asyncio locks for cold-rehydration deduplication.  Only one coroutine
# scans corpus directories per matter_id; concurrent misses wait and then re-check cache.
_rehydration_locks: dict[str, "asyncio.Lock"] = {}
# Refcount of in-flight synchronous investigation runs per matter_id.
# Non-zero entry means the cleanup loop must not evict that model mid-run.
# Refcount (not a plain set) so overlapping runs on the same corpus don't unpin each other.
_sync_running_matter_ids: dict[str, int] = {}
# Counter for concurrently active synchronous upload handlers (no job backing).
# Capped at max_concurrent_jobs to prevent multiple large request bodies in RAM.
_active_sync_requests: int = 0

# Version
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    config = get_config()

    # Validate config on startup
    errors = config.validate()
    if errors and not config.debug:
        for error in errors:
            logger.error(f"Config error: {error}")

    # Create temp directory
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Irys RLM Service v{VERSION} starting...")
    logger.info(f"S3 Bucket: {config.s3_bucket}")
    logger.info(f"Temp Dir: {config.temp_dir}")

    # Start cleanup task
    cleanup_task = asyncio.create_task(_cleanup_loop(config))

    yield

    # Cleanup on shutdown
    cleanup_task.cancel()
    logger.info("Shutting down...")


async def _cleanup_loop(config: ServiceConfig):
    """Background task to clean up expired temp files."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        try:
            # Clean old jobs from memory
            now = datetime.now()
            expired = [
                job_id for job_id, job in _jobs.items()
                if job.completed_at and
                (now - job.completed_at).total_seconds() > config.cleanup_after_seconds
            ]
            _expired_set = set(expired)
            _idle_cutoff = now - timedelta(seconds=config.cleanup_after_seconds)
            # Precompute matter_ids that still have at least one live (non-expired) job.
            # Avoids O(expired × jobs) rescans in the per-expired-job loop below.
            _live_job_matter_ids = {
                j.matter_id
                for jid, j in _jobs.items()
                if jid not in _expired_set and j.matter_id
            }
            for job_id in expired:
                job = _jobs[job_id]
                if job.matter_id and job.matter_id in _active_matter_models:
                    # Only evict the model if:
                    # (a) no other non-expired job still references this matter, AND
                    # (b) the model has not been recently accessed (last_used < idle_cutoff).
                    _last = _matter_model_last_used.get(job.matter_id, datetime.min)
                    if (job.matter_id not in _live_job_matter_ids
                            and job.matter_id not in _sync_running_matter_ids
                            and _last < _idle_cutoff):
                        _active_matter_models.pop(job.matter_id, None)
                        _matter_model_last_used.pop(job.matter_id, None)
                        # Note: SQLiteMatterDB uses threading.local so db.close() from
                        # this thread only closes this thread's handle. Background-flush
                        # threads hold their own per-thread connections released on thread exit.
                del _jobs[job_id]
                logger.debug(f"Cleaned up job {job_id}")
            # Evict matter models not backed by any active job (rehydrated models).
            # These are not associated with a job so the per-job eviction above misses them.
            # Skip models that are currently pinned by an in-flight sync run; evicting them
            # mid-investigation would cause open_gaps to return empty after the run.
            _live_matter_ids = ({j.matter_id for j in _jobs.values() if j.matter_id}
                                | _sync_running_matter_ids.keys())
            for _mid in list(_active_matter_models.keys()):
                if _mid not in _live_matter_ids:
                    if _matter_model_last_used.get(_mid, datetime.min) < _idle_cutoff:
                        _active_matter_models.pop(_mid, None)
                        _matter_model_last_used.pop(_mid, None)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""
    config = config or get_config()

    app = FastAPI(
        title="Irys RLM API",
        description="Legal document investigation service with recursive language model",
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware — permissive for local dev/research use.
    # KNOWN DEFERRED: no authentication or authorization layer is implemented.
    # This service is designed for trusted internal/local use only. Before any
    # network-exposed deployment, add an API key or OAuth2 guard via FastAPI Depends.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state
    app.state.config = config

    return app


# Create default app instance
app = create_app()


def _serialize_result(result) -> tuple[list, dict]:
    """Convert InvestigationResult citations and entities to serializable dicts."""
    # Convert citations (list of Citation dataclasses)
    citations = []
    for c in result.citations:
        try:
            d = asdict(c)
            # Convert datetime to ISO string
            if 'timestamp' in d and d['timestamp']:
                d['timestamp'] = d['timestamp'].isoformat()
            citations.append(d)
        except Exception:
            citations.append({"error": "Failed to serialize citation"})

    # Convert entities (dict of str -> Entity dataclasses)
    entities = {}
    for name, entity in result.entities.items():
        try:
            entities[name] = asdict(entity)
        except Exception:
            entities[name] = {"error": "Failed to serialize entity"}

    return citations, entities


def _url_to_str(u) -> str:
    """Normalize UrlInput (str or UrlWithMetadata) to a plain URL string."""
    if isinstance(u, str):
        return u
    return str(getattr(u, "url", u))


def _compute_corpus_key(descriptor: str) -> str:
    """Derive a stable 16-char hex key from a corpus descriptor string.

    The key is used as the matter DB directory name so the same corpus always
    opens the same persistent matter model across runs (SO-1 durable model).

    Known limitation: for S3-prefix paths the descriptor is the bucket+prefix
    location string, NOT a hash of the file contents.  If files under the same
    S3 prefix are replaced or added between runs, the corpus_key stays unchanged
    and new/modified documents will be served from the prior matter model's hot
    path rather than triggering a cold re-ingest.  The hot path will still skip
    ingestion for documents already marked complete, which is safe; only truly
    new files (not yet in document_inventory) will be cold-ingested correctly.
    A future improvement can list + hash S3 objects at job-start for full
    content-based identity.
    """
    return hashlib.sha256(descriptor.encode()).hexdigest()[:16]


def _wire_matter_model(irys_instance, temp_dir: str, corpus_key: str, config) -> Optional[str]:
    """Pre-create and register the matter model before investigation starts.

    The DB is placed in config.matter_db_dir/{corpus_key}/ — keyed to the
    document corpus, not the job — so repeated runs on the same corpus reuse
    the same persistent matter model (SO-1 durable matter model).

    Returns matter_id, or None if matter model is not enabled.
    """
    if not config.enable_matter_model:
        return None

    from irys.matter import MatterModel

    irys_instance._ensure_initialized()
    # Store DB under corpus_key so the same corpus always reopens the same DB.
    matter_db_path = Path(config.matter_db_dir) / corpus_key
    matter_model = MatterModel.open(matter_db_path)
    matter_id = matter_model.matter_id
    # repo_key maps the temp download path → the pre-registered model so the
    # engine's MatterModel.open() call reuses this instance instead of reopening.
    repo_key = str(Path(temp_dir).resolve())
    irys_instance._matter_models[repo_key] = matter_model
    irys_instance._engine._matter_model = matter_model

    _active_matter_models[matter_id] = matter_model
    _matter_model_last_used[matter_id] = datetime.now()
    return matter_id


def _try_rehydrate_matter_model(matter_id: str, config: ServiceConfig) -> Optional[Any]:
    """Scan matter_db_dir for a persistent DB containing matter_id.

    Called when a matter_id is not in the in-memory registry (service restart
    or post-cleanup request). Iterates corpus_key subdirectories of matter_db_dir
    looking for a DB where the matter row exists.  Rehydrating re-opens the
    connection without re-running ingestion.
    """
    matter_db_dir = Path(config.matter_db_dir)
    if not matter_db_dir.exists():
        return None
    try:
        from irys.matter.matter import MatterModel
        from irys.matter.db import SQLiteMatterDB
        for corpus_dir in matter_db_dir.iterdir():
            if not corpus_dir.is_dir():
                continue
            db_path = corpus_dir / ".irys" / "matter.sqlite3"
            if not db_path.exists():
                continue
            try:
                db = SQLiteMatterDB(db_path)
                row = db.execute(
                    "SELECT id FROM matter WHERE id=?", (matter_id,)
                ).fetchone()
                if row is not None:
                    return MatterModel(db, matter_id)
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"Matter model rehydration failed for {matter_id}: {e}")
    return None


async def _get_matter_model_or_404(matter_id: str):
    model = _active_matter_models.get(matter_id)
    if model is not None:
        _matter_model_last_used[matter_id] = datetime.now()
        return model
    # Try rehydrating from persistent storage (service restart recovery).
    # Runs in a thread pool to avoid blocking the event loop on filesystem/SQLite I/O.
    # Per-matter lock deduplicates concurrent cold-miss requests so only one thread
    # scans corpus directories per matter_id; waiters re-check the cache on lock acquire.
    config = get_config()
    if config.enable_matter_model and config.matter_db_dir:
        if matter_id not in _rehydration_locks:
            _rehydration_locks[matter_id] = asyncio.Lock()
        async with _rehydration_locks[matter_id]:
            # Re-check: a concurrent coroutine may have populated the cache while we waited.
            existing = _active_matter_models.get(matter_id)
            if existing is not None:
                _matter_model_last_used[matter_id] = datetime.now()
                # Pop inside the lock while asyncio is single-threaded; safe because any
                # waiter already holds a reference to the Lock object via the dict lookup
                # above, so removing the dict entry only affects future miss requests.
                _rehydration_locks.pop(matter_id, None)
                return existing
            model = await asyncio.to_thread(_try_rehydrate_matter_model, matter_id, config)
            if model is not None:
                _active_matter_models[matter_id] = model
                _matter_model_last_used[matter_id] = datetime.now()
                logger.info(f"Rehydrated matter model {matter_id} from persistent storage")
                _rehydration_locks.pop(matter_id, None)
                return model
        # Rehydration failed (404).  Remove the stale lock so long-lived processes don't
        # accumulate one lock per distinct miss (including 404 probes from callers).
        _rehydration_locks.pop(matter_id, None)
    raise HTTPException(
        status_code=404,
        detail=f"Matter model '{matter_id}' not found or no longer active",
    )


# === ENDPOINTS ===


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    config = get_config()

    # Check S3 connection
    s3_connected = False
    if config.s3_bucket:
        try:
            import boto3
            s3 = boto3.client("s3", region_name=config.s3_region)
            s3.head_bucket(Bucket=config.s3_bucket)
            s3_connected = True
        except Exception:
            pass

    # Check Gemini connection
    gemini_connected = bool(config.gemini_api_key)

    # Calculate temp storage usage
    temp_dir = Path(config.temp_dir)
    temp_size_mb = 0.0
    if temp_dir.exists():
        temp_size_mb = sum(
            f.stat().st_size for f in temp_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)

    # Count active jobs
    active_jobs = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )

    return HealthResponse(
        status="healthy",
        version=VERSION,
        gemini_connected=gemini_connected,
        s3_connected=s3_connected,
        active_jobs=active_jobs,
        temp_storage_mb=round(temp_size_mb, 2),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post(
    "/investigate",
    response_model=InvestigateResponse,
    tags=["Investigation"],
    responses={400: {"model": ErrorResponse}},
)
async def start_investigation(
    request: InvestigateRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new document investigation.

    Downloads documents from S3, runs investigation using Gemini,
    and returns results asynchronously.
    """
    config = get_config()

    # Check concurrent job limit
    active_count = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    if active_count >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent jobs. Max: {config.max_concurrent_jobs}",
        )

    # Generate job ID
    job_id = f"inv_{uuid.uuid4().hex[:12]}"

    # Create job record
    job = JobResult(
        job_id=job_id,
        status=JobStatus.PENDING,
        query=request.query,
        s3_prefix=request.s3_prefix,
        created_at=datetime.now(),
    )
    _jobs[job_id] = job

    # Start background investigation
    background_tasks.add_task(
        _run_investigation,
        job_id,
        request,
        config,
    )

    return InvestigateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Investigation started",
        estimated_seconds=60,
    )


async def _run_investigation(
    job_id: str,
    request: InvestigateRequest,
    config: ServiceConfig,
):
    """Background task to run investigation."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    s3_repo = None
    temp_dir = None
    _run_interrupted = False  # set True on user stop; temp dir kept for resume

    try:
        # Download documents from S3
        s3_repo = S3Repository(
            bucket=config.s3_bucket,
            prefix=request.s3_prefix,
            config=config,
        )
        temp_dir = await s3_repo.download_to_temp(job_id)

        # Run investigation
        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model, checkpoint_dir=config.checkpoint_dir)
        corpus_key = _compute_corpus_key(f"s3://{config.s3_bucket}/{request.s3_prefix}")
        matter_id = _wire_matter_model(irys, str(temp_dir), corpus_key, config)
        if matter_id:
            job.matter_id = matter_id
            job.corpus_key = corpus_key

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        # Extract results
        job.run_id = getattr(result.state, "_run_id", None)  # exact; no recent_runs(1) race
        job.documents_processed = result.state.documents_read
        job.completed_at = datetime.now()
        job.duration_seconds = (job.completed_at - job.created_at).total_seconds()
        job.pending_clarifications = getattr(result.state, "pending_clarifications", [])

        if getattr(result.state, "status", None) == "interrupted":
            # User stopped the run — checkpoint written; temp dir preserved for resume.
            _run_interrupted = True
            job.status = JobStatus.INTERRUPTED
        else:
            job.analysis = result.output
            job.citations, job.entities = _serialize_result(result)
            job.status = JobStatus.COMPLETED

        # Record open_gaps from the matter model (SO-7).
        if job.matter_id and job.matter_id in _active_matter_models:
            _mm = _active_matter_models[job.matter_id]
            try:
                job.open_gaps = _mm.gaps.open_gaps(min_materiality=0.3)
            except Exception:
                pass

        logger.info(f"Job {job_id} {job.status.value} in {job.duration_seconds:.1f}s")

        # Call webhook if provided
        if request.callback_url:
            await _send_callback(request.callback_url, job)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        # Preserve temp dir when interrupted — checkpoint references this path for resume.
        if not _run_interrupted and s3_repo and temp_dir:
            await s3_repo.cleanup(job_id)


async def _send_callback(url: str, job: JobResult):
    """Send results to callback URL."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                json=job.model_dump(mode="json"),
                timeout=30,
            )
            logger.info(f"Sent callback for job {job.job_id}")
    except Exception as e:
        logger.error(f"Callback failed for job {job.job_id}: {e}")


@app.get(
    "/investigate/{job_id}",
    response_model=JobResult,
    tags=["Investigation"],
    responses={404: {"model": ErrorResponse}},
)
async def get_investigation(job_id: str):
    """Get investigation status and results."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/jobs", response_model=list[JobResult], tags=["Investigation"])
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 20,
):
    """List recent investigation jobs."""
    jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
)
async def quick_search(request: SearchRequest):
    """Quick keyword search across documents."""
    config = get_config()
    job_id = f"search_{uuid.uuid4().hex[:8]}"

    try:
        # Download documents
        s3_repo = S3Repository(
            bucket=config.s3_bucket,
            prefix=request.s3_prefix,
            config=config,
        )
        temp_dir = await s3_repo.download_to_temp(job_id)

        # Run search
        from irys import quick_search as irys_search
        results = await irys_search(
            query=request.query,
            repository=str(temp_dir),
            api_key=config.gemini_api_key,
        )

        # Cleanup
        await s3_repo.cleanup(job_id)

        return SearchResponse(
            results=[SearchResult(**r) for r in results[:request.max_results]],
            total_matches=len(results),
            documents_searched=len(list(temp_dir.rglob("*"))),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === FILE UPLOAD ENDPOINTS ===



@app.post(
    "/upload/investigate",
    response_model=UploadInvestigateResponse,
    tags=["File Upload"],
    responses={400: {"model": ErrorResponse}},
)
async def upload_investigate(
    query: str = Form(..., description="Investigation query"),
    files: list[UploadFile] = File(..., description="Document files to analyze"),
    callback_url: Optional[str] = Form(None, description="Webhook URL for results"),
    keep_files: bool = Form(False, description="Keep files in S3 after processing"),
    background_tasks: BackgroundTasks = None,
):
    """Start investigation with uploaded files (async).

    Storage mode controlled by IRYS_STORAGE_MODE env var:
    - "local": Files stored on VM disk (for development)
    - "s3": Files streamed to S3 (for production, keeps VM light)

    Set keep_files=true to preserve files in S3 for later re-query (s3 mode only).
    """
    config = get_config()

    # Check concurrent job limit
    active_count = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    if active_count >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent jobs. Max: {config.max_concurrent_jobs}",
        )

    # Check file count
    if len(files) > config.max_documents_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Max: {config.max_documents_per_job}",
        )

    job_id = f"upload_{uuid.uuid4().hex[:12]}"

    try:
        # Read files into memory
        _max_bytes = config.max_document_size_mb * 1024 * 1024
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
            if len(content) > _max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {Path(file.filename).name!r} exceeds maximum size"
                           f" of {config.max_document_size_mb} MB",
                )
            filename = Path(file.filename).name
            file_data.append((filename, content))

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Compute stable corpus identity from file content hashes (SO-1)
        upload_corpus_key = _compute_corpus_key(
            "|".join(sorted(
                f"{fname}:{hashlib.sha256(content).hexdigest()}"
                for fname, content in file_data
            ))
        )

        # Branch based on storage mode
        if config.storage_mode == "local":
            # LOCAL MODE: Save to temp directory
            temp_dir = Path(config.temp_dir) / job_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in file_data:
                (temp_dir / filename).write_bytes(content)
            s3_prefix = f"local:{job_id}"
        else:
            # S3 MODE: Upload to S3
            s3_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix="",
                config=config,
            )
            s3_prefix = await s3_repo.upload_files(job_id, file_data)

        # Create job record
        job = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            query=query,
            s3_prefix=s3_prefix,
            created_at=datetime.now(),
            corpus_key=upload_corpus_key,
        )
        _jobs[job_id] = job

        # Start background investigation
        background_tasks.add_task(
            _run_upload_investigation,
            job_id,
            query,
            s3_prefix,
            callback_url,
            keep_files,
            config,
        )

        return UploadInvestigateResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Investigation started",
            files_received=len(file_data),
            estimated_seconds=60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload investigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_upload_investigation(
    job_id: str,
    query: str,
    s3_prefix: str,
    callback_url: Optional[str],
    keep_files: bool,
    config: ServiceConfig,
):
    """Background task to run investigation on uploaded files."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    s3_repo = None
    temp_dir = None
    is_local = s3_prefix.startswith("local:")
    _run_interrupted = False  # set True on user stop; temp dir kept for resume

    try:
        if is_local:
            # LOCAL MODE: Files already on disk
            temp_dir = Path(config.temp_dir) / job_id
        else:
            # S3 MODE: Download from S3
            s3_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix=s3_prefix,
                config=config,
            )
            temp_dir = await s3_repo.download_to_temp(job_id)

        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model, checkpoint_dir=config.checkpoint_dir)
        # corpus_key must have been set by the upload endpoint from file content hashes.
        # If absent (should never happen), skip matter model wiring rather than fall back
        # to a per-run job_id — a per-run key would silently fragment the matter DB (SO-1).
        corpus_key = job.corpus_key
        if corpus_key:
            matter_id = _wire_matter_model(irys, str(temp_dir), corpus_key, config)
            if matter_id:
                job.matter_id = matter_id
        else:
            logger.warning(f"Upload job {job_id}: corpus_key absent, matter model wiring skipped")

        result = await irys.investigate(
            query=query,
            repository=str(temp_dir),
        )

        # Extract results — use exact state._run_id (r39 fix: avoids concurrent-run race).
        job.run_id = getattr(result.state, "_run_id", None)
        job.documents_processed = result.state.documents_read
        job.completed_at = datetime.now()
        job.duration_seconds = (job.completed_at - job.created_at).total_seconds()
        job.pending_clarifications = getattr(result.state, "pending_clarifications", [])

        if getattr(result.state, "status", None) == "interrupted":
            _run_interrupted = True
            job.status = JobStatus.INTERRUPTED
        else:
            job.analysis = result.output
            job.citations, job.entities = _serialize_result(result)
            job.status = JobStatus.COMPLETED

        if job.matter_id and job.matter_id in _active_matter_models:
            try:
                job.open_gaps = _active_matter_models[job.matter_id].gaps.open_gaps(min_materiality=0.3)
            except Exception:
                pass
        logger.info(f"Upload job {job_id} {job.status.value} in {job.duration_seconds:.1f}s (mode={'local' if is_local else 's3'})")

        # Call webhook if provided
        if callback_url:
            await _send_callback(callback_url, job)

    except Exception as e:
        logger.error(f"Upload job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        if _run_interrupted:
            pass  # Preserve temp dir — checkpoint references this path for resume
        elif is_local:
            # LOCAL MODE: Delete temp directory
            import shutil
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        else:
            # S3 MODE: Cleanup temp and optionally S3
            if s3_repo:
                await s3_repo.cleanup(job_id)
            if not keep_files:
                try:
                    cleanup_repo = S3Repository(
                        bucket=config.s3_bucket,
                        prefix="",
                        config=config,
                    )
                    await cleanup_repo.delete_prefix(s3_prefix)
                except Exception as e:
                    logger.warning(f"Failed to cleanup S3 prefix {s3_prefix}: {e}")


@app.post(
    "/upload/search",
    response_model=UploadSearchResponse,
    tags=["File Upload"],
)
async def upload_search(
    query: str = Form(..., description="Search query"),
    files: list[UploadFile] = File(..., description="Document files to search"),
    max_results: int = Form(20, ge=1, le=100, description="Maximum results"),
):
    """Quick search across uploaded files.

    Storage mode controlled by IRYS_STORAGE_MODE env var.
    Synchronous - returns results immediately.
    """
    global _active_sync_requests
    config = get_config()
    # Check file count (same cap as other upload endpoints)
    if len(files) > config.max_documents_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Max: {config.max_documents_per_job}",
        )
    if _active_sync_requests >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent sync requests. Max: {config.max_concurrent_jobs}",
        )
    _active_sync_requests += 1
    job_id = f"uploadsearch_{uuid.uuid4().hex[:8]}"
    s3_prefix = None
    s3_repo = None
    temp_dir = None

    try:
        # Read files into memory
        _max_bytes = config.max_document_size_mb * 1024 * 1024
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
            if len(content) > _max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {Path(file.filename).name!r} exceeds maximum size"
                           f" of {config.max_document_size_mb} MB",
                )
            filename = Path(file.filename).name
            file_data.append((filename, content))

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Branch based on storage mode
        if config.storage_mode == "local":
            # LOCAL MODE: Save to temp directory
            temp_dir = Path(config.temp_dir) / job_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in file_data:
                (temp_dir / filename).write_bytes(content)
        else:
            # S3 MODE: Upload to S3, then download to temp
            s3_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix="",
                config=config,
            )
            s3_prefix = await s3_repo.upload_files(job_id, file_data)

            upload_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix=s3_prefix,
                config=config,
            )
            temp_dir = await upload_repo.download_to_temp(job_id)

        # Run search
        from irys import quick_search as irys_search
        results = await irys_search(
            query=query,
            repository=str(temp_dir),
            api_key=config.gemini_api_key,
        )

        # Cleanup
        if config.storage_mode == "local":
            import shutil
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        else:
            await s3_repo.cleanup(job_id)
            await s3_repo.delete_prefix(s3_prefix)

        return UploadSearchResponse(
            results=[SearchResult(**r) for r in results[:max_results]],
            total_matches=len(results),
            files_searched=len(file_data),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload search failed: {e}")
        # Cleanup on error
        if config.storage_mode == "local":
            import shutil
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        elif s3_repo and s3_prefix:
            try:
                await s3_repo.delete_prefix(s3_prefix)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _active_sync_requests -= 1


@app.post(
    "/upload/investigate/sync",
    response_model=SyncInvestigateResponse,
    tags=["File Upload"],
    responses={400: {"model": ErrorResponse}},
)
async def upload_investigate_sync(
    query: str = Form(..., description="Investigation query"),
    files: list[UploadFile] = File(..., description="Document files to analyze"),
    keep_files: bool = Form(False, description="Keep files in S3 after processing for re-query"),
):
    """Upload and investigate files synchronously.

    Storage mode controlled by IRYS_STORAGE_MODE env var:
    - "local": Files stored on VM disk (for development)
    - "s3": Files streamed to S3 (for production, keeps VM light)

    Set keep_files=true to preserve files in S3 for later re-query (only applies in s3 mode).

    Note: This endpoint blocks until investigation completes (may take 30-120 seconds).
    """
    global _active_sync_requests
    config = get_config()
    job_id = f"sync_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    s3_prefix = None
    s3_repo = None
    temp_dir = None
    sync_matter_id = None  # must be visible in finally for unpin

    # Cheap validation before incrementing the concurrency slot so invalid requests
    # don't briefly consume a slot.
    if len(files) > config.max_documents_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Max: {config.max_documents_per_job}",
        )
    # Concurrency cap: each sync request may buffer up to
    # max_documents_per_job × max_document_size_mb of body data in RAM simultaneously.
    if _active_sync_requests >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent sync requests. Max: {config.max_concurrent_jobs}",
        )
    _active_sync_requests += 1
    try:
        # Read files into memory
        _max_bytes = config.max_document_size_mb * 1024 * 1024
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
            if len(content) > _max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {Path(file.filename).name!r} exceeds maximum size"
                           f" of {config.max_document_size_mb} MB",
                )
            filename = Path(file.filename).name
            file_data.append((filename, content))

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Compute stable corpus identity from file content hashes (SO-1)
        sync_corpus_key = _compute_corpus_key(
            "|".join(sorted(
                f"{fname}:{hashlib.sha256(content).hexdigest()}"
                for fname, content in file_data
            ))
        )

        # Branch based on storage mode
        if config.storage_mode == "local":
            # LOCAL MODE: Save directly to temp directory
            temp_dir = Path(config.temp_dir) / job_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in file_data:
                (temp_dir / filename).write_bytes(content)
        else:
            # S3 MODE: Upload to S3, then download to temp
            s3_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix="",
                config=config,
            )
            s3_prefix = await s3_repo.upload_files(job_id, file_data)

            upload_repo = S3Repository(
                bucket=config.s3_bucket,
                prefix=s3_prefix,
                config=config,
            )
            temp_dir = await upload_repo.download_to_temp(job_id)

        # Run investigation
        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model, checkpoint_dir=config.checkpoint_dir)
        sync_matter_id = _wire_matter_model(irys, str(temp_dir), sync_corpus_key, config)

        # Pin the model: increment refcount so the orphan-eviction pass in _cleanup_loop
        # cannot evict it mid-run.  Refcount (not a plain set) so overlapping runs on the
        # same corpus don't unpin each other.  Unpin happens in the outer finally after the
        # open_gaps snapshot is captured, not immediately after investigate() returns.
        if sync_matter_id:
            _sync_running_matter_ids[sync_matter_id] = (
                _sync_running_matter_ids.get(sync_matter_id, 0) + 1
            )

        result = await irys.investigate(
            query=query,
            repository=str(temp_dir),
        )

        _sync_interrupted = (getattr(result.state, "status", None) == "interrupted")

        # Cleanup temp files — skip if interrupted; checkpoint references this path for resume.
        if not _sync_interrupted:
            if config.storage_mode == "local":
                import shutil
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            else:
                if s3_repo:
                    await s3_repo.cleanup(job_id)
                # Cleanup S3 files (unless keep_files=True)
                if not keep_files and s3_repo and s3_prefix:
                    await s3_repo.delete_prefix(s3_prefix)

        duration = time.time() - start_time
        logger.info(f"Sync investigation {job_id} completed in {duration:.1f}s (mode={config.storage_mode})")

        citations, entities = _serialize_result(result)
        _sync_open_gaps: list[dict] = []
        if sync_matter_id:
            try:
                # Refresh last_used so a long sync run doesn't cause the model to be
                # evicted between investigation end and this post-run lookup.
                _matter_model_last_used[sync_matter_id] = datetime.now()
                _sm = _active_matter_models.get(sync_matter_id)
                if _sm is not None:
                    _sync_open_gaps = _sm.gaps.open_gaps(min_materiality=0.3)
            except Exception:
                pass
        # Use exact state._run_id set by engine; reasoning_trail[0] is best-effort only (r40 fix).
        _sync_run_id = (getattr(result.state, "_run_id", None)
                        or (((getattr(result.state, "reasoning_trail", None) or []) or [{}])[0].get("run_id")))
        response = SyncInvestigateResponse(
            query=query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
            matter_id=sync_matter_id,
            run_id=_sync_run_id,
            pending_clarifications=getattr(result.state, "pending_clarifications", []),
            open_gaps=_sync_open_gaps,
        )

        # Add S3 prefix to response if files kept (s3 mode only)
        if keep_files and s3_prefix:
            response.s3_prefix = s3_prefix

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync investigation failed: {e}")
        # Cleanup on error
        if config.storage_mode == "local":
            import shutil
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        elif s3_repo and s3_prefix:
            try:
                await s3_repo.delete_prefix(s3_prefix)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _active_sync_requests -= 1
        # Unpin here (outer finally) so the pin covers cleanup and open_gaps, not just investigate().
        if sync_matter_id:
            _cnt = _sync_running_matter_ids.get(sync_matter_id, 0)
            if _cnt <= 1:
                _sync_running_matter_ids.pop(sync_matter_id, None)
            else:
                _sync_running_matter_ids[sync_matter_id] = _cnt - 1


# === S3 URL ENDPOINTS ===


@app.post(
    "/investigate/urls",
    response_model=InvestigateResponse,
    tags=["S3 URLs"],
    responses={400: {"model": ErrorResponse}},
)
async def investigate_urls(
    request: S3UrlsInvestigateRequest,
    background_tasks: BackgroundTasks,
):
    """Start investigation with document URLs.

    Provide a list of URLs to download and analyze. Supports:
    - S3 URLs: s3://bucket/key, https://bucket.s3.region.amazonaws.com/key
    - Generic HTTP(S) URLs: https://example.com/document.pdf
    """
    config = get_config()

    # Check concurrent job limit
    active_count = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    if active_count >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent jobs. Max: {config.max_concurrent_jobs}",
        )

    # Generate job ID
    job_id = f"urls_{uuid.uuid4().hex[:12]}"

    # Create job record
    job = JobResult(
        job_id=job_id,
        status=JobStatus.PENDING,
        query=request.query,
        s3_prefix=f"urls:{len(request.s3_urls)} files",
        created_at=datetime.now(),
    )
    _jobs[job_id] = job

    # Start background investigation
    background_tasks.add_task(
        _run_urls_investigation,
        job_id,
        request,
        config,
    )

    return InvestigateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Investigation started for {len(request.s3_urls)} files",
        estimated_seconds=60,
    )


async def _run_urls_investigation(
    job_id: str,
    request: S3UrlsInvestigateRequest,
    config: ServiceConfig,
):
    """Background task to run investigation on S3 URLs."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    s3_repo = None
    temp_dir = None
    _run_interrupted = False  # set True on user stop; temp dir kept for resume

    try:
        # Create S3 repository (bucket from first URL, or config default)
        s3_repo = S3Repository(
            bucket=config.s3_bucket or "placeholder",
            prefix="",
            config=config,
        )

        # Download documents from URLs
        temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)

        # Run investigation
        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model, checkpoint_dir=config.checkpoint_dir)
        corpus_key = _compute_corpus_key(",".join(sorted(_url_to_str(u) for u in request.s3_urls)))
        matter_id = _wire_matter_model(irys, str(temp_dir), corpus_key, config)
        if matter_id:
            job.matter_id = matter_id
            job.corpus_key = corpus_key

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        # Extract results
        job.run_id = getattr(result.state, "_run_id", None)  # exact; no recent_runs(1) race
        job.documents_processed = result.state.documents_read
        job.completed_at = datetime.now()
        job.duration_seconds = (job.completed_at - job.created_at).total_seconds()
        job.pending_clarifications = getattr(result.state, "pending_clarifications", [])

        if getattr(result.state, "status", None) == "interrupted":
            _run_interrupted = True
            job.status = JobStatus.INTERRUPTED
        else:
            job.analysis = result.output
            job.citations, job.entities = _serialize_result(result)
            job.status = JobStatus.COMPLETED

        if job.matter_id and job.matter_id in _active_matter_models:
            _url_mm = _active_matter_models[job.matter_id]
            try:
                job.open_gaps = _url_mm.gaps.open_gaps(min_materiality=0.3)
            except Exception:
                pass

        logger.info(f"URLs job {job_id} {job.status.value} in {job.duration_seconds:.1f}s")

        # Call webhook if provided
        if request.callback_url:
            await _send_callback(request.callback_url, job)

    except Exception as e:
        logger.error(f"URLs job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        # Preserve temp dir when interrupted — checkpoint references this path for resume.
        if not _run_interrupted and s3_repo and temp_dir:
            await s3_repo.cleanup(job_id)


@app.post(
    "/search/urls",
    response_model=SearchResponse,
    tags=["S3 URLs"],
)
async def search_urls(request: S3UrlsSearchRequest):
    """Quick keyword search across documents from URLs.

    Provide a list of URLs to download and search. Supports:
    - S3 URLs: s3://bucket/key, https://bucket.s3.region.amazonaws.com/key
    - Generic HTTP(S) URLs: https://example.com/document.pdf

    Synchronous - returns results immediately.
    """
    config = get_config()
    job_id = f"searchurls_{uuid.uuid4().hex[:8]}"
    s3_repo = None

    try:
        # Create S3 repository
        s3_repo = S3Repository(
            bucket=config.s3_bucket or "placeholder",
            prefix="",
            config=config,
        )

        # Download documents from URLs
        temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)

        # Run search
        from irys import quick_search as irys_search
        results = await irys_search(
            query=request.query,
            repository=str(temp_dir),
            api_key=config.gemini_api_key,
        )

        # Cleanup
        await s3_repo.cleanup(job_id)

        return SearchResponse(
            results=[SearchResult(**r) for r in results[:request.max_results]],
            total_matches=len(results),
            documents_searched=len(request.s3_urls),
        )

    except Exception as e:
        logger.error(f"URL search failed: {e}")
        if s3_repo:
            await s3_repo.cleanup(job_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/investigate/urls/sync",
    response_model=SyncInvestigateResponse,
    tags=["S3 URLs"],
    responses={400: {"model": ErrorResponse}},
)
async def investigate_urls_sync(request: S3UrlsInvestigateRequest):
    """Investigate documents from URLs synchronously.

    Provide a list of URLs to download and analyze. Supports:
    - S3 URLs: s3://bucket/key, https://bucket.s3.region.amazonaws.com/key
    - Generic HTTP(S) URLs: https://example.com/document.pdf

    Returns complete results in a single request (blocks until done).
    Note: This endpoint blocks until investigation completes (may take 30-120 seconds).
    """
    global _active_sync_requests
    config = get_config()
    job_id = f"urlsync_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    s3_repo = None
    temp_dir = None
    urls_matter_id = None  # must be visible in finally for unpin

    if len(request.s3_urls) > config.max_documents_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Too many URLs ({len(request.s3_urls)}). Max: {config.max_documents_per_job}",
        )
    if _active_sync_requests >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent sync requests. Max: {config.max_concurrent_jobs}",
        )
    _active_sync_requests += 1
    try:
        # Create S3 repository
        s3_repo = S3Repository(
            bucket=config.s3_bucket or "placeholder",
            prefix="",
            config=config,
        )

        # Download documents from URLs
        temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)

        # Run investigation
        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model, checkpoint_dir=config.checkpoint_dir)
        urls_corpus_key = _compute_corpus_key(",".join(sorted(_url_to_str(u) for u in request.s3_urls)))
        urls_matter_id = _wire_matter_model(irys, str(temp_dir), urls_corpus_key, config)

        # Pin with refcount; unpin happens in the outer finally after open_gaps is captured.
        if urls_matter_id:
            _sync_running_matter_ids[urls_matter_id] = (
                _sync_running_matter_ids.get(urls_matter_id, 0) + 1
            )

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        _urls_sync_interrupted = (getattr(result.state, "status", None) == "interrupted")
        # Preserve temp dir when interrupted — checkpoint references this path for resume.
        if not _urls_sync_interrupted:
            await s3_repo.cleanup(job_id)

        duration = time.time() - start_time
        logger.info(f"URL sync investigation {job_id} {'interrupted' if _urls_sync_interrupted else 'completed'} in {duration:.1f}s")

        citations, entities = _serialize_result(result)
        _urls_open_gaps: list[dict] = []
        if urls_matter_id:
            try:
                _matter_model_last_used[urls_matter_id] = datetime.now()
                _um = _active_matter_models.get(urls_matter_id)
                if _um is not None:
                    _urls_open_gaps = _um.gaps.open_gaps(min_materiality=0.3)
            except Exception:
                pass
        # Use exact state._run_id set by engine; reasoning_trail[0] is best-effort only (r40 fix).
        _urls_run_id = (getattr(result.state, "_run_id", None)
                        or (((getattr(result.state, "reasoning_trail", None) or []) or [{}])[0].get("run_id")))
        return SyncInvestigateResponse(
            query=request.query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
            matter_id=urls_matter_id,
            run_id=_urls_run_id,
            pending_clarifications=getattr(result.state, "pending_clarifications", []),
            open_gaps=_urls_open_gaps,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL sync investigation failed: {e}")
        if s3_repo:
            await s3_repo.cleanup(job_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _active_sync_requests -= 1
        if urls_matter_id:
            _cnt = _sync_running_matter_ids.get(urls_matter_id, 0)
            if _cnt <= 1:
                _sync_running_matter_ids.pop(urls_matter_id, None)
            else:
                _sync_running_matter_ids[urls_matter_id] = _cnt - 1


# === MATTER MODEL ENDPOINTS ===


@app.get(
    "/matter/{matter_id}",
    response_model=MatterStatsResponse,
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_stats(matter_id: str):
    """Return counts and summary for an active matter model."""
    model = await _get_matter_model_or_404(matter_id)
    return MatterStatsResponse(**model.stats())


@app.get(
    "/matter/{matter_id}/runs",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_runs(matter_id: str, limit: int = 10):
    """List recent investigation runs for a matter."""
    model = await _get_matter_model_or_404(matter_id)
    return model.ledger.recent_runs(limit=limit)


@app.get(
    "/matter/{matter_id}/runs/{run_id}/events",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_run_events(matter_id: str, run_id: str):
    """Return the full reasoning ledger event sequence for a run."""
    model = await _get_matter_model_or_404(matter_id)
    run = model.ledger.get_run(run_id)
    if run is None or run.matter_id != model.matter_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in this matter")
    return model.ledger.get_events(run_id)


@app.get(
    "/matter/{matter_id}/clarifications",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_pending_clarifications(matter_id: str, limit: int = 20):
    """Return pending clarification questions for a matter."""
    model = await _get_matter_model_or_404(matter_id)
    return model.clarifications.get_pending(limit=limit)


def _resolve_resumed_run_id(model: Any, run_id: str) -> str:
    """adv#036 HIGH: resolve a stale interrupted run_id to its live resumed child.

    When a client supplies the original interrupted run_id for stop/redirect but
    the run has already been resumed, the resumed child is the correct target.
    Uses the resumed_from lineage column (schema v41) for an exact proof — not a
    count heuristic. Falls back to the original run_id if no live resumed child
    is found (e.g., not yet resumed, or no matter-model path).
    """
    try:
        run = model.ledger.get_run(run_id)
        if run is not None and run.status == "interrupted":
            row = model.db.execute(
                "SELECT id FROM run_session"
                " WHERE matter_id=? AND status='running' AND resumed_from=?",
                (model.matter_id, run_id),
            ).fetchone()
            if row is not None:
                return row["id"]
    except Exception as exc:
        logger.warning(
            "_resolve_resumed_run_id(%s) failed — falling back to original id: %s",
            run_id, exc,
        )
    return run_id


@app.post(
    "/matter/{matter_id}/stop",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def stop_investigation(matter_id: str, _: StopRunRequest):
    """Signal the active investigation to stop after the current iteration.

    Sets stop_requested=1 in the run session; the engine reads this flag
    between iterations and performs a clean interrupt without losing work.
    """
    model = await _get_matter_model_or_404(matter_id)
    # Filter out utility flush runs so stop targets the real investigation, not a
    # background/manual flush that happens to be running concurrently.
    _stop_rows = model.db.execute(
        "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
        " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))"
        " ORDER BY started_at DESC LIMIT 1",
        (model.matter_id,),
    ).fetchall()
    if not _stop_rows:
        raise HTTPException(status_code=409, detail="No running investigation to stop")
    run_id = _stop_rows[0]["id"]
    applied = model.ledger.request_stop(run_id)
    if not applied:
        # Run completed between the SELECT and the UPDATE — return 409
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' completed before stop could be applied")
    # Log the steering event so the reasoning trail reflects the user action (SO-3)
    from irys.matter.enums import LedgerEventType
    model.ledger.append_event(
        run_id=run_id,
        event_type=LedgerEventType.USER_INTERRUPTED,
        summary="User requested stop via API",
        why="User-initiated stop — investigation will halt after current iteration",
    )
    return {"status": "stop_requested", "run_id": run_id}


@app.post(
    "/matter/{matter_id}/runs/{run_id}/redirect",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def redirect_investigation(matter_id: str, run_id: str, request: RedirectRunRequest):
    """Redirect the active investigation to focus on a specific issue.

    Sets redirect_requested=1 and records the target issue_id; the engine
    picks this up on the next iteration and pivots retrieval accordingly.

    If run_id points to an interrupted run that has been resumed, the resolver
    auto-discovers the live resumed child via the resumed_from lineage column
    (adv#036 HIGH fix) so redirect reliably targets the running investigation.
    If no resumed child is running yet, redirect is queued on the interrupted
    run itself and will be picked up when resume starts.
    """
    model = await _get_matter_model_or_404(matter_id)
    run_id = _resolve_resumed_run_id(model, run_id)
    run = model.ledger.get_run(run_id)
    if run is None or run.matter_id != model.matter_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in this matter")
    if run.status not in ("running", "interrupted"):
        raise HTTPException(status_code=409, detail=f"Run is not active or interrupted (status: {run.status})")
    if run.objective in ("manual_flush", "background_flush"):
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' is a utility flush run and cannot be redirected")
    issue = model.issues.get_issue(request.issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail=f"Issue '{request.issue_id}' not found")
    applied = model.ledger.request_redirect(run_id, request.issue_id)
    if not applied:
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' completed before redirect could be applied")
    # Log the steering event so the reasoning trail reflects the user action (SO-3)
    from irys.matter.enums import LedgerEventType
    model.ledger.append_event(
        run_id=run_id,
        event_type=LedgerEventType.USER_REDIRECTED,
        summary=f"User redirected to issue: {issue.get('title', request.issue_id)[:80]}",
        why="User-initiated redirect via API",
        branch_issue_id=request.issue_id,
    )
    return {
        "status": "redirect_requested",
        "run_id": run_id,
        "issue_id": request.issue_id,
        "issue_title": issue.get("title"),
    }


@app.post(
    "/matter/{matter_id}/clarifications/{question_id}/answer",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def answer_clarification(
    matter_id: str, question_id: str, request: AnswerClarificationRequest
):
    """Submit a user answer to a pending clarification question.

    The answered clarification is injected into the orientation prompt of
    subsequent investigation runs for this matter.
    """
    model = await _get_matter_model_or_404(matter_id)
    try:
        found = model.clarifications.answer_question(question_id, request.answer_text)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not found:
        raise HTTPException(status_code=404, detail=f"Clarification question '{question_id}' not found in this matter")
    return {"status": "answered", "question_id": question_id}


@app.post(
    "/matter/{matter_id}/trust-overrides",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def set_trust_override(matter_id: str, request: TrustOverrideRequest):
    """Set or update a trust override for a document pattern (SO-3 trust steering).

    Marks a document as low-trust (force ALLEGED speech act regardless of filename
    heuristics), normal (reset to auto-inference), or high-trust (promote ALLEGED
    facts to OPERATIVE). Takes effect on the next investigation run.
    """
    model = await _get_matter_model_or_404(matter_id)
    # Validated run_id attribution — same pattern as correct_assertion (r37 fix).
    _trust_run_id: "str | None" = (getattr(request, "run_id", None) or None)
    if _trust_run_id:
        try:
            _tv = model.db.execute(
                "SELECT 1 FROM run_session WHERE id=? AND matter_id=? AND status='running'"
                " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
                (_trust_run_id, model.matter_id),
            ).fetchone()
            if not _tv:
                _trust_run_id = None
        except Exception:
            _trust_run_id = None
    if not _trust_run_id:
        try:
            _trust_run_row = model.db.execute(
                "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
                " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))"
                " ORDER BY started_at DESC LIMIT 1",
                (model.matter_id,),
            ).fetchone()
            if _trust_run_row is not None:
                _trust_run_id = _trust_run_row["id"]
        except Exception:
            pass
    try:
        override_id = model.set_trust_override(
            request.document_pattern, request.trust_level, request.note, run_id=_trust_run_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "set", "override_id": override_id}


@app.get(
    "/matter/{matter_id}/trust-overrides",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def list_trust_overrides(matter_id: str):
    """List all document trust overrides for a matter."""
    model = await _get_matter_model_or_404(matter_id)
    return {"overrides": model.trust_overrides.list_all()}


@app.delete(
    "/matter/{matter_id}/trust-overrides/{document_pattern:path}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_trust_override(matter_id: str, document_pattern: str):
    """Remove a trust override for a document pattern (SO-3 trust steering).

    Restores auto-inferred trust for the matching document pattern.
    Returns 404 if the matter is not found.
    """
    model = await _get_matter_model_or_404(matter_id)
    model.delete_trust_override(document_pattern)
    return {"status": "deleted", "document_pattern": document_pattern}


@app.post(
    "/matter/{matter_id}/flush-pending",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def flush_pending_propagation(matter_id: str):
    """Drain the durable pending propagation queues and complete deferred belief revision.

    SO-2 convergence fix (adv#029 HIGH): correction and evidence pending queues are only
    drained by flush_revisions() which is called at the end of each investigation loop
    iteration.  If no investigation run occurs after a correction, trust-override, or
    conflict-detection that exhausted the BFS budget, queued work stays permanently
    pending.  This endpoint provides a standalone flush path that does not require an
    active investigation run, closing the guarantee gap.

    Returns the count of assertions whose belief state changed during this flush.
    """
    model = await _get_matter_model_or_404(matter_id)
    from irys.matter.runtime import MatterRuntimeAdapter
    # Acquire _flush_lock BEFORE opening the run session so no spurious 'running' row
    # exists while waiting for an active investigation to finish. Calls
    # _flush_revisions_locked() directly since we already hold the lock.
    with model._flush_lock:
        # Start a real run so ledger.append_event(run_id=...) satisfies the NOT NULL FK
        # constraint on ledger_event.run_id (run_id=None would violate it).
        flush_run_id = model.start_run("Standalone flush", objective="manual_flush")
        try:
            adapter = MatterRuntimeAdapter(model, run_id=flush_run_id)
            revised = adapter._flush_revisions_locked()
        except Exception as exc:
            try:
                model.fail_run(flush_run_id, str(exc))
            except Exception as fe:
                logger.warning("flush-pending fail_run failed for run %s: %s", flush_run_id, fe)
            raise
        try:
            model.complete_run(flush_run_id)
        except Exception as ce:
            try:
                model.fail_run(flush_run_id, str(ce))
            except Exception as fe:
                logger.warning("flush-pending terminal close failed for run %s: %s", flush_run_id, fe)
            raise
    return {"status": "ok", "revised_count": revised}


@app.post(
    "/matter/{matter_id}/annotations",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def add_document_annotation(matter_id: str, request: DocumentAnnotationRequest):
    """Add a strategic annotation to a document pattern (SO-3 annotation).

    Annotations are injected into the orientation prompt so the engine uses the
    user's domain knowledge about a document when planning the investigation.
    """
    model = await _get_matter_model_or_404(matter_id)
    annotation_id = model.annotations.add(
        request.document_pattern, request.annotation_text, request.annotation_type
    )
    return {"status": "added", "annotation_id": annotation_id}


@app.get(
    "/matter/{matter_id}/annotations",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def list_document_annotations(matter_id: str, document: Optional[str] = None):
    """List document annotations for a matter.

    Pass ?document=filename.pdf to filter by document, or omit for all recent annotations.
    """
    model = await _get_matter_model_or_404(matter_id)
    if document:
        return {"annotations": model.annotations.get_for_document(document)}
    return {"annotations": model.annotations.list_recent()}


@app.delete(
    "/matter/{matter_id}/annotations/{annotation_id}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_document_annotation(matter_id: str, annotation_id: str):
    """Remove a document annotation (SO-3 annotation).

    Returns 404 if the matter is not found; returns deleted=false if annotation_id
    does not exist (idempotent delete).
    """
    model = await _get_matter_model_or_404(matter_id)
    deleted = model.annotations.delete(annotation_id)
    return {"status": "deleted" if deleted else "not_found", "annotation_id": annotation_id}


@app.get(
    "/matter/{matter_id}/reconcile",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_reconciliation(matter_id: str, currency: str = "USD"):
    """Return a payment reconciliation summary grouped by subject type.

    Groups all extracted monetary amounts by subject_type (invoice, payment,
    fee, damages, etc.) and sums each bucket. Compare invoice vs payment totals
    to identify claimed exposure.
    """
    model = await _get_matter_model_or_404(matter_id)
    reconciliation = model.reconcile(currency=currency)
    conflicts = model.quant.get_conflicts()
    return {
        "currency": currency,
        "by_subject": reconciliation,
        "conflicts": conflicts,
    }


@app.get(
    "/matter/{matter_id}/assertions",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_assertions(matter_id: str, limit: int = 50, offset: int = 0):
    """Return paginated list of assertions with source metadata (SO-2).

    Each assertion includes belief_state, source_role, speech_act, and the
    document it came from — enabling clients to audit the evidence layer.
    """
    model = await _get_matter_model_or_404(matter_id)
    return {
        "total": model.assertions.count(),
        "limit": limit,
        "offset": offset,
        "assertions": model.assertions.list_recent(limit=limit, offset=offset),
    }


@app.get(
    "/matter/{matter_id}/assertions/{assertion_id}/history",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_assertion_history(matter_id: str, assertion_id: str, limit: int = 50, offset: int = 0):
    """Return field-level revision history for an assertion (SO-2 audit trail).

    Rows are ordered newest-first by ``(created_at DESC, batch_id DESC, id DESC)``
    for a deterministic total order. ``limit`` is per-row (not per batch), capped
    at 500. Use ``offset`` for pagination. If the result is truncated,
    ``truncated=true`` is set and ``next_offset`` gives the next page start.

    Note: offset-based pagination is not stable under concurrent writes. New
    revisions arriving between page requests can shift rows. This is expected for
    an audit endpoint and is documented in the response's ``history_note``.

    Pre-schema-v34 matters will have empty history by design — no backfill is
    possible. The ``history_note`` field describes this.
    """
    import json as _json

    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    model = await _get_matter_model_or_404(matter_id)
    # Verify assertion belongs to this matter (assertion_revision has no matter_id col)
    row = model.db.execute(
        "SELECT id FROM assertion WHERE id=? AND matter_id=?",
        (assertion_id, model.matter_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Assertion '{assertion_id}' not found in this matter")
    # Fetch one extra row to detect truncation without a separate COUNT query
    rev_rows = model.db.execute(
        """SELECT ar.id, ar.batch_id, ar.changed_field,
                  ar.old_value_json, ar.new_value_json,
                  ar.actor_kind, ar.actor_ref, ar.cause,
                  ar.run_id, ar.note, ar.created_at
           FROM assertion_revision ar
           WHERE ar.assertion_id=?
           ORDER BY ar.created_at DESC, ar.batch_id DESC, ar.id DESC
           LIMIT ? OFFSET ?""",
        (assertion_id, limit + 1, offset),
    ).fetchall()
    truncated = len(rev_rows) > limit
    rev_rows = rev_rows[:limit]

    def _decode(raw: str | None):
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except (_json.JSONDecodeError, TypeError):
            return raw  # return raw string rather than 500-ing

    history = [
        {
            "id": r["id"],
            "batch_id": r["batch_id"],
            "changed_field": r["changed_field"],
            "old_value": _decode(r["old_value_json"]),
            "new_value": _decode(r["new_value_json"]),
            "actor_kind": r["actor_kind"],
            "actor_ref": r["actor_ref"],
            "cause": r["cause"],
            "run_id": r["run_id"],
            "note": r["note"],
            "created_at": r["created_at"],
        }
        for r in rev_rows
    ]
    return {
        "assertion_id": assertion_id,
        "history_note": (
            "History records revisions since schema v34. Pre-v34 revisions are not available. "
            "Offset-based pagination is not stable under concurrent writes."
        ),
        "count": len(history),
        "offset": offset,
        "truncated": truncated,
        "next_offset": offset + len(history) if truncated else None,
        "history": history,
    }


@app.get(
    "/matter/{matter_id}/issues",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_issues(matter_id: str, min_materiality: float = 0.0):
    """Return open issues with assertion coverage stats ordered by weakness (SO-4).

    Uses get_issue_coverage_report() for consistent, belief-state-filtered counts
    including coverage_fraction, has_proof_gap, and gap_id — the canonical SO-4
    coverage view used internally by the investigation engine.
    """
    model = await _get_matter_model_or_404(matter_id)
    report = model.get_issue_coverage_report()
    # Filter by materiality post-hoc (get_issue_coverage_report returns all issues)
    if min_materiality > 0.0:
        report = [r for r in report if r.get("materiality", 0.0) >= min_materiality]
    return report


@app.get(
    "/matter/{matter_id}/reconciliation",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_reconciliation(matter_id: str, currency: str = "USD"):
    """Return structured payment reconciliation for a matter (SO-6).

    Shows invoiced total, paid total, disputed amount (linked to disputed assertions),
    and claimed exposure (invoiced − paid). Each figure is grounded via source_spans
    linking back to the original document spans.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.reconcile_payment_chain(currency)


@app.get(
    "/matter/{matter_id}/reconciliation/invoices",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_invoice_chain(matter_id: str, currency: str = "USD"):
    """Return per-invoice reconciliation for a matter (SO-6).

    Each row shows an individual invoice: how much was invoiced, how much
    was paid against it, and what remains outstanding.  Payments are matched
    to invoices by subject_id equality in the quant_fact store.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.reconcile_invoice_chain(currency)


@app.get(
    "/matter/{matter_id}/metrics",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_so_metrics(matter_id: str):
    """Return measurable Sacred Outcome success criteria for a matter.

    Reports assertion_structure_rate, source_role_known_rate, issue_coverage_avg,
    steerability, and belief_revision, with targets and pass/fail flags.
    Metrics requiring ground truth or run telemetry are returned as null.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_so_metrics()


@app.get(
    "/matter/{matter_id}/decision-context",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_decision_context(matter_id: str):
    """Return the decision-context overlay for a matter (Priority 1).

    Returns null when no context has been set.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.decision_context.get()


@app.put(
    "/matter/{matter_id}/decision-context",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def set_decision_context(matter_id: str, payload: dict):
    """Set or update the decision-context overlay for a matter.

    Accepted fields: decision_maker_type, decision_maker_name, objective,
    strategic_notes, scope_narrow.  Unknown/invalid type and objective values
    are coerced to 'unknown'.  Influences synthesis framing only — does not
    alter the canonical record model.
    """
    model = await _get_matter_model_or_404(matter_id)
    ctx_id = model.decision_context.set(
        decision_maker_type=payload.get("decision_maker_type"),
        decision_maker_name=payload.get("decision_maker_name"),
        objective=payload.get("objective"),
        strategic_notes=payload.get("strategic_notes"),
        scope_narrow=bool(payload.get("scope_narrow", False)),
    )
    return {"id": ctx_id, "matter_id": matter_id, "status": "ok"}


@app.delete(
    "/matter/{matter_id}/decision-context",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def clear_decision_context(matter_id: str):
    """Remove the decision-context overlay for a matter."""
    model = await _get_matter_model_or_404(matter_id)
    model.decision_context.clear()
    return {"matter_id": matter_id, "status": "cleared"}


@app.get(
    "/matter/{matter_id}/gaps",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_gaps(matter_id: str, min_materiality: float = 0.0, limit: Optional[int] = None):
    """Return open gaps for a matter (SO-7 — missingness is modeled, not ignored).

    Each gap represents something the system knows is missing: a document,
    a predicate, an unresolved contradiction, or a needed clarification.
    Filtered by materiality threshold (0.0 = all gaps, 0.5 = significant only).
    limit: cap the number returned (None = use schema default pagination).
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.gaps.open_gaps(min_materiality=min_materiality, limit=limit)


@app.post(
    "/matter/{matter_id}/assertions/{assertion_id}/correct",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def correct_assertion(
    matter_id: str, assertion_id: str, request: CorrectAssertionRequest,
    background_tasks: BackgroundTasks,
):
    """Apply a user correction to an assertion's belief state (SO-2).

    Propagates the change through the assertion dependency graph, updating
    all downstream conclusions that depended on the corrected assertion.
    Returns the revision result including which dependent assertions changed.
    """
    from irys.matter.enums import BeliefState, RevisionCause

    model = await _get_matter_model_or_404(matter_id)

    try:
        new_state = BeliefState(request.new_belief_state)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid belief_state '{request.new_belief_state}'. "
                   "Valid values: alleged/argued/admitted/operative/performed/"
                   "disputed/superseded/withdrawn/inferred/resolved/unknown",
        )

    # Single active-run lookup shared by both the correction attribution and the
    # steering injection — prevents divergence if two runs overlap (r35 fix).
    # Client-provided run_id is validated against run_session before use so stale
    # or bogus IDs cannot misattribute audit rows (r36 HIGH fix).
    # Normalize to None: empty string is treated as absent (r36 MEDIUM fix).
    _active_run_id: "str | None" = (getattr(request, "run_id", None) or None)
    if _active_run_id:
        # Validate: must be a running session for this exact matter.
        try:
            _valid = model.db.execute(
                "SELECT 1 FROM run_session WHERE id=? AND matter_id=? AND status='running'"
                " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
                (_active_run_id, model.matter_id),
            ).fetchone()
            if not _valid:
                _active_run_id = None  # stale or foreign run — fall through to lookup
        except Exception:
            _active_run_id = None
    if not _active_run_id:
        try:
            _active_run_row = model.db.execute(
                "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
                " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))"
                " ORDER BY started_at DESC LIMIT 1",
                (model.matter_id,),
            ).fetchone()
            if _active_run_row is not None:
                _active_run_id = _active_run_row["id"]
        except Exception:
            pass

    try:
        result = model.correct_assertion(
            assertion_id=assertion_id,
            new_state=new_state,
            run_id=_active_run_id,
            confidence=request.confidence,
            note=request.note,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # SO-3 active-run steering: inject correction as a synthetic clarification
    # so the currently-running investigation loop re-examines related evidence.
    try:
        assertion_record = model.assertions.get(assertion_id)
        if assertion_record is not None and _active_run_id is not None:
            prop_text = assertion_record.proposition_text[:100]
            synth_note = request.note or ""
            synth_q_id = model.clarifications.add_question(
                question_text=(
                    f"User correction: '{prop_text}' → {request.new_belief_state}"
                ),
                run_id=_active_run_id,
                why_it_matters="User directly corrected an assertion during this run",
            )
            model.clarifications.answer_question(
                synth_q_id,
                f"Assertion '{prop_text}' corrected to {request.new_belief_state}. "
                f"Re-examine evidence related to this claim. {synth_note}".strip(),
            )
    except Exception:
        pass  # steering injection is best-effort; never block the response

    # SO-2 convergence: if BFS was truncated after 3 inline rounds, deferred work
    # is in the durable pending queue but will only run on the next investigation flush.
    # Trigger a background flush so convergence is not conditional on a later run
    # (adv#030 HIGH fix).
    if result.propagation_truncated:
        background_tasks.add_task(_background_flush, matter_id, model)

    return {
        "assertion_id": assertion_id,
        "old_belief_state": result.old_belief_state.value,
        "new_belief_state": result.new_belief_state.value,
        "propagated_to": result.propagated_to or [],
        "cause": result.cause.value,
        "propagation_truncated": result.propagation_truncated,
    }


def _do_one_background_flush(matter_id: str, model) -> None:
    """Execute a single flush pass. Called from within the flush loop with _bg_flush_running held."""
    # Refresh last-used so the cleanup loop does not evict the model mid-flush.
    # Background flushes hold the model with no jobs entry; without this refresh
    # the orphan eviction could remove the model if a pass takes > cleanup_after_seconds.
    _matter_model_last_used[matter_id] = datetime.now()
    from irys.matter.runtime import MatterRuntimeAdapter
    with model._flush_lock:
        flush_run_id = model.start_run("Background flush", objective="background_flush")
        try:
            adapter = MatterRuntimeAdapter(model, run_id=flush_run_id)
            adapter._flush_revisions_locked()
        except Exception as exc:
            try:
                model.fail_run(flush_run_id, str(exc))
            except Exception as fe:
                logger.warning("background_flush fail_run failed for %s run %s: %s", matter_id, flush_run_id, fe)
            raise
        else:
            try:
                model.complete_run(flush_run_id)
            except Exception as ce:
                try:
                    model.fail_run(flush_run_id, str(ce))
                except Exception as fe:
                    logger.warning("background_flush terminal close failed for %s run %s: %s", matter_id, flush_run_id, fe)


def _background_flush_loop(matter_id: str, model) -> None:
    """Flush loop body — called with _bg_flush_running already held.

    Loops while _bg_flush_event is set, clearing it BEFORE each pass so
    corrections enqueued during a pass are caught. On exit, releases
    _bg_flush_running and spawns a new loop thread if work arrived in
    the release window.
    """
    try:
        while model._bg_flush_event.is_set():
            model._bg_flush_event.clear()
            try:
                _do_one_background_flush(matter_id, model)
            except Exception as exc:
                logger.warning("background_flush failed for matter %s: %s", matter_id, exc)
                break
    finally:
        model._bg_flush_running.release()
        # Final race: work enqueued between last event check and release
        if model._bg_flush_event.is_set():
            if model._bg_flush_running.acquire(blocking=False):
                import threading as _threading
                try:
                    _threading.Thread(
                        target=_background_flush_loop, args=(matter_id, model), daemon=False
                    ).start()
                except Exception:
                    model._bg_flush_running.release()


def _background_flush(matter_id: str, model) -> None:
    """Fire-and-forget flush of pending BFS propagation work.

    Called as a FastAPI BackgroundTask after a truncated correct_assertion() so
    deferred propagation converges without waiting for the next investigation run
    (adv#030 HIGH fix — SO-2 convergence guarantee).

    Sets _bg_flush_event then, if no loop is running, runs _background_flush_loop
    inline (already in a BackgroundTasks thread). Multiple concurrent BackgroundTasks
    callers all set the event and return; only the first one that acquires
    _bg_flush_running does actual work.
    """
    model._bg_flush_event.set()
    if not model._bg_flush_running.acquire(blocking=False):
        return  # loop already active; it will see the event and run another pass
    _background_flush_loop(matter_id, model)


# ---------------------------------------------------------------------------
# Legal Research Layer — Authority endpoints (SO-4)
# ---------------------------------------------------------------------------

@app.post(
    "/matter/{matter_id}/authorities",
    tags=["Matter Model"],
    status_code=201,
    responses={404: {"model": ErrorResponse}},
)
async def upsert_authority(matter_id: str, payload: dict):
    """Create or update a legal authority for a matter.

    Required: citation (string).
    Optional: authority_type, name, jurisdiction, decided_at, holdings (list),
    key_rules (list), weight, applicability, source_doc_id, source_span_id.

    Duplicate citations are updated in place.
    """
    model = await _get_matter_model_or_404(matter_id)
    citation = payload.get("citation", "").strip()
    if not citation:
        raise HTTPException(status_code=422, detail="citation is required")

    auth_id, is_new = model.authority.upsert(
        citation=citation,
        authority_type=payload.get("authority_type", "case"),
        name=payload.get("name"),
        jurisdiction=payload.get("jurisdiction"),
        decided_at=payload.get("decided_at"),
        holdings=payload.get("holdings"),
        key_rules=payload.get("key_rules"),
        weight=payload.get("weight", "persuasive"),
        applicability=payload.get("applicability"),
        source_doc_id=payload.get("source_doc_id"),
        source_span_id=payload.get("source_span_id"),
    )
    return {"id": auth_id, "matter_id": matter_id, "is_new": is_new}


@app.get(
    "/matter/{matter_id}/authorities",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def list_authorities(
    matter_id: str,
    authority_type: Optional[str] = None,
    weight: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
):
    """List authorities for a matter.

    Optionally filter by authority_type or weight, or provide a search query
    for substring matching on citation/name.
    """
    model = await _get_matter_model_or_404(matter_id)
    if search:
        return model.authority.search(search, limit=limit)
    return model.authority.list_all(authority_type=authority_type, weight=weight, limit=limit)


@app.get(
    "/matter/{matter_id}/authorities/{authority_id}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_authority(matter_id: str, authority_id: str):
    """Return a single authority by id."""
    model = await _get_matter_model_or_404(matter_id)
    auth = model.authority.get(authority_id)
    if auth is None:
        raise HTTPException(status_code=404, detail="Authority not found")
    return auth


@app.post(
    "/matter/{matter_id}/authorities/{authority_id}/issues/{issue_id}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def link_authority_to_issue(
    matter_id: str,
    authority_id: str,
    issue_id: str,
    relevance: str = "supporting",
):
    """Link an authority to an issue with a relevance label.

    relevance: supporting | attacking | neutral
    """
    model = await _get_matter_model_or_404(matter_id)
    model.authority.link_to_issue(authority_id, issue_id, relevance=relevance)
    return {"authority_id": authority_id, "issue_id": issue_id, "relevance": relevance, "status": "linked"}


@app.delete(
    "/matter/{matter_id}/authorities/{authority_id}/issues/{issue_id}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def unlink_authority_from_issue(matter_id: str, authority_id: str, issue_id: str):
    """Remove an authority-issue link."""
    model = await _get_matter_model_or_404(matter_id)
    model.authority.unlink_from_issue(authority_id, issue_id)
    return {"authority_id": authority_id, "issue_id": issue_id, "status": "unlinked"}


@app.get(
    "/matter/{matter_id}/issues/{issue_id}/authorities",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_issue_authorities(matter_id: str, issue_id: str):
    """Return all authorities linked to an issue, with their relevance."""
    model = await _get_matter_model_or_404(matter_id)
    return model.authority.list_for_issue(issue_id)


# ---------------------------------------------------------------------------
# Proof-Aware Reasoning — ProofState endpoints (SO-4)
# ---------------------------------------------------------------------------

@app.post(
    "/matter/{matter_id}/proof-state/compute",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def compute_proof_state(matter_id: str):
    """Recompute proof state for all open issues in the matter.

    Should be called after adding assertions, resolving predicates, or making
    other changes that affect evidence coverage.  Returns per-issue proof states.
    """
    model = await _get_matter_model_or_404(matter_id)
    states = model.proof_state.compute_all()
    return {"matter_id": matter_id, "updated_count": len(states), "states": states}


@app.post(
    "/matter/{matter_id}/issues/{issue_id}/proof-state/compute",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def compute_issue_proof_state(matter_id: str, issue_id: str):
    """Recompute proof state for a single issue."""
    model = await _get_matter_model_or_404(matter_id)
    _issue_check = model.db.execute(
        "SELECT 1 FROM issue WHERE id=? AND matter_id=?", (issue_id, model.matter_id)
    ).fetchone()
    if _issue_check is None:
        raise HTTPException(status_code=404, detail=f"Issue '{issue_id}' not found in this matter")
    state = model.proof_state.compute_and_store(issue_id)
    return state


@app.get(
    "/matter/{matter_id}/proof-state",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_proof_state_summary(matter_id: str):
    """Return proof coverage summary for the matter.

    Includes aggregate stats (avg sufficiency, count by status) and the full
    list of per-issue proof states ordered by sufficiency ascending.
    """
    model = await _get_matter_model_or_404(matter_id)
    summary = model.proof_state.get_summary()
    all_states = model.proof_state.get_all()
    return {"matter_id": matter_id, "summary": summary, "issues": all_states}


@app.get(
    "/matter/{matter_id}/issues/{issue_id}/proof-state",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_issue_proof_state(matter_id: str, issue_id: str):
    """Return stored proof state for a single issue, or null if not yet computed."""
    model = await _get_matter_model_or_404(matter_id)
    return model.proof_state.get(issue_id)


@app.get(
    "/matter/{matter_id}/proof-state/gaps",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_proof_gaps(matter_id: str, threshold: float = 0.25):
    """Return issues with sufficiency below threshold — the weakest proof points.

    threshold: float 0..1, default 0.25.  Issues with sufficiency < threshold
    are surfaced as requiring additional evidence.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.proof_state.get_gaps(threshold=threshold)


# ---------------------------------------------------------------------------
# Visual Work Product — Timeline view (Priority 2)
# ---------------------------------------------------------------------------

@app.get(
    "/matter/{matter_id}/timeline",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_timeline(matter_id: str, limit: int = 200):
    """Return a chronological event list derived from date-type quant facts and
    temporally-scoped assertions.

    Events are ordered by date ascending (undated events last).
    Each entry has: date, event, source_doc, quant_id, assertion_id, subject, kind.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_timeline(limit=limit)


@app.get(
    "/matter/{matter_id}/evidence-matrix",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_evidence_matrix(matter_id: str):
    """Return the evidence matrix: issues × source documents.

    Rows = open issues; columns = source documents; cells = supporting/attacking
    assertion counts per (issue, document) pair.  Useful for identifying which
    sources contribute evidence to which claims and which issues lack source coverage.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_evidence_matrix()


@app.get(
    "/matter/{matter_id}/communication-map",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_communication_map(matter_id: str):
    """Return the actor-document interaction graph.

    Shows which actors appeared in which documents (actor_document_edges) and
    which actors co-appear in the same documents (actor_actor_edges).
    Useful for mapping communication patterns, principal relationships, and
    identifying which parties are most active in document production.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_communication_map()


@app.get(
    "/matter/{matter_id}/damages-waterfall",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_damages_waterfall(matter_id: str, currency: str = "USD"):
    """Return a structured damages breakdown by category.

    Groups amount-type quant facts by subject_type and computes totals.
    Ordered by claimed_amount descending.  Includes conflict detection when
    multiple sources cite different amounts for the same component.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_damages_waterfall(currency=currency)


@app.get(
    "/matter/{matter_id}/steering-surface",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_steering_surface(matter_id: str, run_id: Optional[str] = None):
    """Return structured steering actions from the reasoning ledger (SO-3).

    Surfaces actionable recommendations: gaps to clarify, issues to redirect to,
    assertions to review.  Used by the UI Gaps & Steering panel.

    Args:
        run_id: When provided, redirect_focus action params will include this run_id
                so callers can invoke the redirect directly.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.get_ledger_steering_surface(run_id=run_id)


# ---------------------------------------------------------------------------
# Actor Resolution — alias matching and duplicate detection (SO-5)
# ---------------------------------------------------------------------------

@app.get(
    "/matter/{matter_id}/actors/duplicates",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_actor_duplicates(matter_id: str, min_prefix_len: int = 6):
    """Return pairs of actors whose normalized names share a common prefix.

    Use this to identify actors that should be merged (e.g. "Acme Corp" and
    "Acme Corporation").  Returns actor pairs with their shared prefix.
    """
    model = await _get_matter_model_or_404(matter_id)
    return model.actors.find_possible_duplicates(min_prefix_len=min_prefix_len)


@app.post(
    "/matter/{matter_id}/actors/{keep_id}/merge/{merge_id}",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def merge_actors(matter_id: str, keep_id: str, merge_id: str):
    """Merge merge_id into keep_id.

    Moves all aliases and assertion occurrence references from merge_id to
    keep_id, then deletes merge_id.  The keep_id actor's canonical name is
    preserved.
    """
    model = await _get_matter_model_or_404(matter_id)
    try:
        model.actors.merge_actors(keep_id=keep_id, merge_id=merge_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"keep_id": keep_id, "merged_id": merge_id, "status": "merged"}


@app.get(
    "/matter/{matter_id}/actors/resolve",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def resolve_actor_by_name(matter_id: str, name: str):
    """Resolve an actor_id from a name string using alias and substring matching.

    Returns {actor_id, actor} if found, or {actor_id: null} if not resolved.
    """
    model = await _get_matter_model_or_404(matter_id)
    actor_id = model.actors.resolve_by_name(name)
    if actor_id is None:
        return {"actor_id": None}
    actor = next(
        (a for a in model.actors.list_actors() if a["id"] == actor_id), None
    )
    return {"actor_id": actor_id, "actor": actor}


# ---------------------------------------------------------------------------
# UI Dashboard Endpoints — aggregated views for the front-end panels
# ---------------------------------------------------------------------------


@app.get(
    "/matter/{matter_id}/overview",
    tags=["UI"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_overview(matter_id: str):
    """Aggregated landing-page payload for the Overview panel.

    Combines stats, SO metrics, recent runs, weakest issues, open gaps,
    and pending clarifications into a single low-latency response.
    """
    model = await _get_matter_model_or_404(matter_id)
    stats = model.stats()

    # Fetch coverage report once — shared by both weakest_issues and get_so_metrics()
    # to avoid duplicate get_issue_coverage_report() calls.
    coverage_report: list = []
    try:
        coverage_report = model.get_issue_coverage_report()
    except Exception:
        pass

    so = {}
    try:
        so = model.get_so_metrics(_coverage_report=coverage_report)
    except Exception:
        pass

    # Weakest issues — lowest coverage_fraction first, limit 5
    weakest_issues = []
    if coverage_report:
        weakest_issues = sorted(
            coverage_report, key=lambda r: float(r.get("coverage_fraction", 0.0))
        )[:5]

    # Top open gaps — limit 5 pushed into SQL to avoid full-table scan.
    top_gaps = []
    try:
        top_gaps = model.gaps.open_gaps(limit=5)
    except Exception:
        pass

    # Pending clarifications — limit 5
    clarifications = []
    try:
        clarifications = model.clarifications.get_pending(limit=5)
    except Exception:
        pass

    return {
        "matter_id": matter_id,
        "stats": stats,
        "so_metrics": so,
        "weakest_issues": weakest_issues,
        "top_gaps": top_gaps,
        "pending_clarifications": clarifications,
    }


@app.get(
    "/matter/{matter_id}/runs/{run_id}/events/stream",
    tags=["UI"],
    responses={404: {"model": ErrorResponse}},
)
async def stream_run_events(matter_id: str, run_id: str, after_seq: int = -1, request: Request = None):
    """Server-Sent Events stream of ledger events for a run.

    Clients connect and receive new ledger events as they are appended.
    Use after_seq=-1 (default) to receive all events from the beginning,
    or after_seq=N to receive only events with seq_no > N (resume/reconnect).

    The stream closes when the run reaches a terminal state (completed/failed/interrupted).
    """
    model = await _get_matter_model_or_404(matter_id)
    # Validate before opening the stream so HTTP 404 is sent as a real error response,
    # not buried inside a 200 SSE body (adv#031 MEDIUM).
    _run_check = model.db.execute(
        "SELECT id FROM run_session WHERE id=? AND matter_id=?",
        (run_id, model.matter_id),
    ).fetchone()
    if _run_check is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in this matter")

    async def _event_generator():
        last_seq = after_seq
        poll_interval = 0.5  # seconds between DB polls
        terminal_statuses = {"completed", "failed", "interrupted"}

        while True:
            # Check if client disconnected
            if request is not None and await request.is_disconnected():
                break

            # Fetch new events since last_seq
            try:
                rows = model.db.execute(
                    """SELECT id, run_id, seq_no, event_type, summary, why,
                              branch_issue_id, changed_object_type, changed_object_id,
                              snapshot_json, created_at
                       FROM ledger_event
                       WHERE run_id=? AND seq_no > ?
                       ORDER BY seq_no""",
                    (run_id, last_seq),
                ).fetchall()
                for row in rows:
                    event = dict(row)
                    last_seq = event["seq_no"]
                    data = _json.dumps(event, default=str)
                    yield f"data: {data}\n\n"
            except Exception as exc:
                yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
                break

            # Check run status — close stream when run is terminal
            try:
                run_row = model.db.execute(
                    "SELECT status FROM run_session WHERE id=?", (run_id,)
                ).fetchone()
                if run_row and run_row["status"] in terminal_statuses:
                    # Emit any remaining events before closing
                    yield f"data: {_json.dumps({'event': 'run_terminal', 'status': run_row['status']})}\n\n"
                    break
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/matter/{matter_id}/runs/{run_id}/stop",
    tags=["UI"],
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def stop_run(matter_id: str, run_id: str):
    """Stop a specific run by run_id.

    Preferred over the matter-level /matter/{matter_id}/stop when the UI
    has a specific run_id (e.g., from the live investigation panel).

    If run_id points to an interrupted run that has been resumed, the resolver
    auto-discovers the live resumed child via the resumed_from lineage column
    (adv#036 HIGH fix) so stop reliably targets the running investigation.
    """
    model = await _get_matter_model_or_404(matter_id)
    run_id = _resolve_resumed_run_id(model, run_id)
    run = model.ledger.get_run(run_id)
    if run is None or run.matter_id != model.matter_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in this matter")
    if run.status not in ("running", "RUNNING"):
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' is not running (status={run.status})")
    if run.objective in ("manual_flush", "background_flush"):
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' is a utility flush run and cannot be stopped via this endpoint")
    applied = model.ledger.request_stop(run_id)
    if not applied:
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' completed before stop could be applied")
    from irys.matter.enums import LedgerEventType
    model.ledger.append_event(
        run_id=run_id,
        event_type=LedgerEventType.USER_INTERRUPTED,
        summary="User requested stop via run-scoped API",
        why="User-initiated stop — investigation will halt after current iteration",
    )
    return {"status": "stop_requested", "run_id": run_id, "matter_id": matter_id}


@app.post(
    "/matter/{matter_id}/runs/{run_id}/resume",
    tags=["UI"],
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def resume_run(matter_id: str, run_id: str):
    """Resume an interrupted run from its checkpoint.

    Validates that:
    - The run exists and belongs to this matter
    - The run status is 'interrupted'
    - The run has a checkpoint path in next_action
    - The checkpoint file exists on disk
    - No other steerable run is currently running for this matter

    Returns the new run_id created for the resumed investigation.
    """
    config = get_config()  # HIGH r81: was missing — endpoint raised NameError before reaching resume
    from pathlib import Path as _Path
    from irys.api import Irys

    model = await _get_matter_model_or_404(matter_id)
    run = model.ledger.get_run(run_id)
    if run is None or run.matter_id != model.matter_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in this matter")
    if run.status != "interrupted":
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is not interrupted (status={run.status}) — only interrupted runs can be resumed",
        )
    if run.objective in ("manual_flush", "background_flush"):
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' is a utility flush run and cannot be resumed")
    checkpoint_path = run.next_action
    if not checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail=f"Run '{run_id}' has no checkpoint — was stopped before the first checkpoint interval",
        )
    if not _Path(checkpoint_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint file not found: {checkpoint_path}",
        )
    # Validate checkpoint can be loaded, belongs to this matter, and repository still exists
    try:
        from irys.rlm.state import InvestigationState
        ckpt_state = InvestigationState.load_checkpoint(checkpoint_path)
        # Cross-matter isolation: reject if checkpoint was written for a different matter
        ckpt_matter_id = getattr(ckpt_state, "_matter_id", None)
        if ckpt_matter_id and ckpt_matter_id != model.matter_id:
            raise HTTPException(
                status_code=400,
                detail=f"Checkpoint belongs to matter '{ckpt_matter_id}', not '{model.matter_id}'",
            )
        if not _Path(ckpt_state.repository_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Repository path no longer exists: {ckpt_state.repository_path}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot load checkpoint: {exc}") from exc

    # Reject if another steerable run is already running for this matter
    running = model.db.execute(
        "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
        " AND (objective IS NULL OR objective NOT IN ('manual_flush','background_flush'))",
        (matter_id,),
    ).fetchone()
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Matter '{matter_id}' already has a running investigation: {running['id']}",
        )

    irys = Irys(
        api_key=config.gemini_api_key,
        enable_matter_model=config.enable_matter_model,
        checkpoint_dir=config.checkpoint_dir,
    )
    irys._ensure_initialized()
    # Wire the already-open matter model so the resumed run uses the same DB
    irys._engine._matter_model = model

    # NOTE: resume_investigation() runs synchronously in-line here, blocking this
    # request handler for the full investigation duration. This is intentional for
    # the minimal SO-3 surface — matches the sync investigate endpoints. For
    # production use with long-running resumes, move this to the background-jobs
    # mechanism (same as the async /investigate endpoint).
    try:
        result = await irys.resume_investigation(checkpoint_path, original_run_id=run_id)
    except Exception as exc:
        # LOW r76: concurrent resume races are a 409, not a 500
        from ..rlm.engine import ConcurrentResumeError
        if isinstance(exc, ConcurrentResumeError):
            raise HTTPException(
                status_code=409,
                detail=f"Concurrent resume conflict: {exc}",
            ) from exc
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc

    new_run_id = getattr(result.state, "_run_id", None)
    return {
        "status": "resumed",
        "original_run_id": run_id,
        "new_run_id": new_run_id,
        "matter_id": matter_id,
    }
