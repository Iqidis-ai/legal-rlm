"""FastAPI REST API for Irys RLM service."""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx

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
                (now - job.completed_at).seconds > config.cleanup_after_seconds
            ]
            for job_id in expired:
                job = _jobs[job_id]
                if job.matter_id and job.matter_id in _active_matter_models:
                    del _active_matter_models[job.matter_id]
                del _jobs[job_id]
                logger.debug(f"Cleaned up job {job_id}")
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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
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


def _get_matter_model_or_404(matter_id: str):
    model = _active_matter_models.get(matter_id)
    if model is not None:
        return model
    # Try rehydrating from persistent storage (service restart recovery)
    config = get_config()
    if config.enable_matter_model and config.matter_db_dir:
        model = _try_rehydrate_matter_model(matter_id, config)
        if model is not None:
            _active_matter_models[matter_id] = model
            logger.info(f"Rehydrated matter model {matter_id} from persistent storage")
            return model
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
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model)
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
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at).total_seconds()

        # Record run_id from the matter model's most recent run
        if job.matter_id and job.matter_id in _active_matter_models:
            recent = _active_matter_models[job.matter_id].ledger.recent_runs(1)
            if recent:
                job.run_id = recent[0]["id"]

        logger.info(f"Job {job_id} completed in {job.duration_seconds:.1f}s")

        # Call webhook if provided
        if request.callback_url:
            await _send_callback(request.callback_url, job)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        # Cleanup temp files
        if s3_repo and temp_dir:
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


async def _save_uploaded_files(
    files: list[UploadFile],
    temp_dir: Path,
) -> int:
    """Save uploaded files to temp directory. Returns count saved."""
    saved = 0
    for file in files:
        if not file.filename:
            continue
        # Sanitize filename
        filename = Path(file.filename).name
        dest_path = temp_dir / filename
        async with aiofiles.open(dest_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        saved += 1
    return saved


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
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
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
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model)
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

        # Extract results
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at
        ).total_seconds()

        if job.matter_id and job.matter_id in _active_matter_models:
            recent = _active_matter_models[job.matter_id].ledger.recent_runs(1)
            if recent:
                job.run_id = recent[0]["id"]

        logger.info(f"Upload job {job_id} completed in {job.duration_seconds:.1f}s (mode={'local' if is_local else 's3'})")

        # Call webhook if provided
        if callback_url:
            await _send_callback(callback_url, job)

    except Exception as e:
        logger.error(f"Upload job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        if is_local:
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
    config = get_config()
    job_id = f"uploadsearch_{uuid.uuid4().hex[:8]}"
    s3_prefix = None
    s3_repo = None
    temp_dir = None

    try:
        # Read files into memory
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
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
    config = get_config()
    job_id = f"sync_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    s3_prefix = None
    s3_repo = None
    temp_dir = None

    try:
        # Check file count
        if len(files) > config.max_documents_per_job:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files ({len(files)}). Max: {config.max_documents_per_job}",
            )

        # Read files into memory
        file_data = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
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
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model)
        _wire_matter_model(irys, str(temp_dir), sync_corpus_key, config)

        result = await irys.investigate(
            query=query,
            repository=str(temp_dir),
        )

        # Cleanup temp files
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
        response = SyncInvestigateResponse(
            query=query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
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
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model)
        corpus_key = _compute_corpus_key(",".join(sorted(request.s3_urls)))
        matter_id = _wire_matter_model(irys, str(temp_dir), corpus_key, config)
        if matter_id:
            job.matter_id = matter_id
            job.corpus_key = corpus_key

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        # Extract results
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at
        ).total_seconds()

        if job.matter_id and job.matter_id in _active_matter_models:
            recent = _active_matter_models[job.matter_id].ledger.recent_runs(1)
            if recent:
                job.run_id = recent[0]["id"]

        logger.info(f"URLs job {job_id} completed in {job.duration_seconds:.1f}s")

        # Call webhook if provided
        if request.callback_url:
            await _send_callback(request.callback_url, job)

    except Exception as e:
        logger.error(f"URLs job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        # Cleanup temp files
        if s3_repo and temp_dir:
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
    config = get_config()
    job_id = f"urlsync_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    s3_repo = None
    temp_dir = None

    try:
        # Check URL count
        if len(request.s3_urls) > config.max_documents_per_job:
            raise HTTPException(
                status_code=400,
                detail=f"Too many URLs ({len(request.s3_urls)}). Max: {config.max_documents_per_job}",
            )

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
        irys = Irys(api_key=config.gemini_api_key, enable_matter_model=config.enable_matter_model)
        urls_corpus_key = _compute_corpus_key(",".join(sorted(request.s3_urls)))
        _wire_matter_model(irys, str(temp_dir), urls_corpus_key, config)

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        # Cleanup
        await s3_repo.cleanup(job_id)

        duration = time.time() - start_time
        logger.info(f"URL sync investigation {job_id} completed in {duration:.1f}s")

        citations, entities = _serialize_result(result)
        return SyncInvestigateResponse(
            query=request.query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL sync investigation failed: {e}")
        if s3_repo:
            await s3_repo.cleanup(job_id)
        raise HTTPException(status_code=500, detail=str(e))


# === MATTER MODEL ENDPOINTS ===


@app.get(
    "/matter/{matter_id}",
    response_model=MatterStatsResponse,
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_stats(matter_id: str):
    """Return counts and summary for an active matter model."""
    model = _get_matter_model_or_404(matter_id)
    return MatterStatsResponse(**model.stats())


@app.get(
    "/matter/{matter_id}/runs",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_runs(matter_id: str, limit: int = 10):
    """List recent investigation runs for a matter."""
    model = _get_matter_model_or_404(matter_id)
    return model.ledger.recent_runs(limit=limit)


@app.get(
    "/matter/{matter_id}/runs/{run_id}/events",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_run_events(matter_id: str, run_id: str):
    """Return the full reasoning ledger event sequence for a run."""
    model = _get_matter_model_or_404(matter_id)
    return model.ledger.get_events(run_id)


@app.get(
    "/matter/{matter_id}/clarifications",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_pending_clarifications(matter_id: str):
    """Return pending clarification questions for a matter."""
    model = _get_matter_model_or_404(matter_id)
    return model.clarifications.get_pending()


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
    model = _get_matter_model_or_404(matter_id)
    runs = model.ledger.recent_runs(1)
    if not runs or runs[0]["status"] != "running":
        raise HTTPException(status_code=409, detail="No running investigation to stop")
    run_id = runs[0]["id"]
    model.ledger.request_stop(run_id)
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
    """
    model = _get_matter_model_or_404(matter_id)
    run = model.ledger.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if run.status != "running":
        raise HTTPException(status_code=409, detail=f"Run is not active (status: {run.status})")
    issue = model.issues.get_issue(request.issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail=f"Issue '{request.issue_id}' not found")
    model.ledger.request_redirect(run_id, request.issue_id)
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
    model = _get_matter_model_or_404(matter_id)
    try:
        model.clarifications.answer_question(question_id, request.answer_text)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
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
    model = _get_matter_model_or_404(matter_id)
    try:
        override_id = model.trust_overrides.set(
            request.document_pattern, request.trust_level, request.note
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
    model = _get_matter_model_or_404(matter_id)
    return {"overrides": model.trust_overrides.list_all()}


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
    model = _get_matter_model_or_404(matter_id)
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
    model = _get_matter_model_or_404(matter_id)
    if document:
        return {"annotations": model.annotations.get_for_document(document)}
    return {"annotations": model.annotations.list_recent()}


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
    model = _get_matter_model_or_404(matter_id)
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
    model = _get_matter_model_or_404(matter_id)
    return {
        "total": model.assertions.count(),
        "limit": limit,
        "offset": offset,
        "assertions": model.assertions.list_recent(limit=limit, offset=offset),
    }


@app.get(
    "/matter/{matter_id}/issues",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_issues(matter_id: str, min_materiality: float = 0.0):
    """Return open issues with assertion coverage counts (SO-4).

    Each issue includes the count of supporting and attacking assertions
    so clients can see which claims are well-evidenced vs. proof-gap-exposed.
    """
    model = _get_matter_model_or_404(matter_id)
    issues = model.issues.get_open_issues(min_materiality=min_materiality)
    result = []
    for issue in issues:
        assertions = model.issues.get_assertions_for_issue(issue["id"])
        issue["supporting_assertions"] = sum(
            1 for a in assertions if a.get("relation_type") in ("supports", "establishes")
        )
        issue["attacking_assertions"] = sum(
            1 for a in assertions if a.get("relation_type") in ("attacks", "negates")
        )
        issue["total_assertions"] = len(assertions)
        result.append(issue)
    return result


@app.get(
    "/matter/{matter_id}/gaps",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}},
)
async def get_matter_gaps(matter_id: str, min_materiality: float = 0.0):
    """Return open gaps for a matter (SO-7 — missingness is modeled, not ignored).

    Each gap represents something the system knows is missing: a document,
    a predicate, an unresolved contradiction, or a needed clarification.
    Filtered by materiality threshold (0.0 = all gaps, 0.5 = significant only).
    """
    model = _get_matter_model_or_404(matter_id)
    return model.gaps.open_gaps(min_materiality=min_materiality)


@app.post(
    "/matter/{matter_id}/assertions/{assertion_id}/correct",
    tags=["Matter Model"],
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def correct_assertion(
    matter_id: str, assertion_id: str, request: CorrectAssertionRequest
):
    """Apply a user correction to an assertion's belief state (SO-2).

    Propagates the change through the assertion dependency graph, updating
    all downstream conclusions that depended on the corrected assertion.
    Returns the revision result including which dependent assertions changed.
    """
    from irys.matter.enums import BeliefState, RevisionCause

    model = _get_matter_model_or_404(matter_id)

    try:
        new_state = BeliefState(request.new_belief_state)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid belief_state '{request.new_belief_state}'. "
                   "Valid values: alleged/argued/admitted/operative/performed/"
                   "disputed/superseded/withdrawn/inferred/resolved/unknown",
        )

    try:
        result = model.correct_assertion(
            assertion_id=assertion_id,
            new_state=new_state,
            confidence=request.confidence,
            note=request.note,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # SO-3 active-run steering: inject correction as a synthetic clarification
    # so the currently-running investigation loop re-examines related evidence.
    try:
        assertion_record = model.assertions.get(assertion_id)
        if assertion_record is not None:
            active_run = model.db.execute(
                "SELECT id FROM run_session WHERE matter_id=? AND status='running'"
                " ORDER BY started_at DESC LIMIT 1",
                (matter_id,),
            ).fetchone()
            if active_run is not None:
                prop_text = assertion_record.proposition_text[:100]
                synth_note = request.note or ""
                synth_q_id = model.clarifications.add_question(
                    question_text=(
                        f"User correction: '{prop_text}' → {request.new_belief_state}"
                    ),
                    run_id=active_run["id"],
                    why_it_matters="User directly corrected an assertion during this run",
                )
                model.clarifications.answer_question(
                    synth_q_id,
                    f"Assertion '{prop_text}' corrected to {request.new_belief_state}. "
                    f"Re-examine evidence related to this claim. {synth_note}".strip(),
                )
    except Exception:
        pass  # steering injection is best-effort; never block the response

    return {
        "assertion_id": assertion_id,
        "old_belief_state": result.old_belief_state.value,
        "new_belief_state": result.new_belief_state.value,
        "propagated_to": result.propagated_to or [],
        "cause": result.cause.value,
    }
