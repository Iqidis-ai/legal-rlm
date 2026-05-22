"""FastAPI REST API for Irys RLM service."""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
import httpx

from ..core.utils import setup_logging, _log_message_id
from .config import ServiceConfig, get_config
from .models import (
    InvestigateRequest,
    InvestigateResponse,
    InvestigationContext,
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
)
from .s3_repository import S3Repository
from .session_store import SessionStore, SessionData

logger = logging.getLogger(__name__)

# In-memory job storage (use Redis in production for multi-worker)
_jobs: dict[str, JobResult] = {}
# _start_time: float = time.time()


async def _load_session(
    config: ServiceConfig, session_id: Optional[str],
) -> tuple[list[str], list[dict]]:
    """Load prior session facts/citations. Returns ([], []) if no session_id."""
    if not session_id:
        return [], []
    store = SessionStore(config)
    session = await store.load(session_id)
    if session is None:
        logger.info(f"Session {session_id}: new session")
        return [], []
    logger.info(
        f"Session {session_id}: loaded {len(session.facts)} facts, "
        f"{len(session.citations)} citations from investigation #{session.investigation_count}"
    )
    return session.facts, session.citations


async def _save_session(
    config: ServiceConfig, session_id: Optional[str], result,
) -> None:
    """Save accumulated facts/citations to session store."""
    if not session_id:
        return
    store = SessionStore(config)
    existing = await store.load(session_id) or SessionData()
    raw_facts = result.state.findings.get("accumulated_facts", [])
    facts = [
        entry[0] if isinstance(entry, (list, tuple)) else entry
        for entry in raw_facts
    ]
    citations_serialized, _ = _serialize_result(result)
    session = SessionData(
        facts=facts,
        citations=citations_serialized,
        investigation_count=existing.investigation_count + 1,
    )
    await store.save(session_id, session)
    logger.info(
        f"Session {session_id}: saved {len(facts)} facts, "
        f"{len(citations_serialized)} citations (investigation #{session.investigation_count})"
    )

def _parse_context_json(context_json: Optional[str]) -> Optional[InvestigationContext]:
    """Parse context JSON string into InvestigationContext object.

    Used for multipart form endpoints where nested objects can't be sent directly.
    """
    if not context_json:
        return None
    try:
        data = json.loads(context_json)
        return InvestigationContext(**data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse context JSON: {e}")
        return None


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

    # Ensure message-ID log filter is attached before any request arrives.
    setup_logging(level=config.log_level)

    # Create temp directory
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Irys RLM Service v{VERSION} starting...")
    logger.info(f"S3 Bucket: {config.s3_bucket}")
    logger.info(f"Temp Dir: {config.temp_dir}")

    # Start cleanup tasks
    cleanup_task = asyncio.create_task(_cleanup_loop(config))
    stale_dir_task = asyncio.create_task(_cleanup_stale_temp_dirs(config))

    yield

    # Cleanup on shutdown
    cleanup_task.cancel()
    stale_dir_task.cancel()
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
                del _jobs[job_id]
                logger.debug(f"Cleaned up job {job_id}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def _cleanup_stale_temp_dirs(config: ServiceConfig):
    """Periodically remove temp subdirectories older than cleanup_after_seconds."""
    import shutil
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        try:
            temp_root = Path(config.temp_dir)
            if not temp_root.exists():
                continue
            now = datetime.now().timestamp()
            for entry in temp_root.iterdir():
                if not entry.is_dir():
                    continue
                age_seconds = now - entry.stat().st_mtime
                if age_seconds > config.cleanup_after_seconds:
                    try:
                        await asyncio.to_thread(shutil.rmtree, entry)
                        logger.debug(f"Removed stale temp dir: {entry}")
                    except Exception as rm_err:
                        logger.warning(f"Failed to remove {entry}: {rm_err}")
        except Exception as e:
            logger.error(f"Stale temp dir cleanup error: {e}")


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

    # Mount Gradio Chat UI at /ui path (root path would override API routes)
    # Use the chat UI for consistent experience across local and S3 modes
    try:
        import gradio as gr
        from ..ui.chat_app import create_chat_app
        gradio_app = create_chat_app(api_key=config.gemini_api_key)
        app = gr.mount_gradio_app(app, gradio_app, path="/ui")
        logger.info("Gradio Chat UI mounted at /ui")
    except ImportError as e:
        logger.warning(f"Gradio not available, UI disabled: {e}")
    except Exception as e:
        logger.warning(f"Failed to mount Gradio UI: {e}")

    return app


# Create default app instance
app = create_app()


def _make_tracing_provider():
    """Create a tracing provider from environment variables, or NoOp if not configured.

    Tags every trace with:
      - "ar-service" (service identifier)
      - "env:<IRYS_ENV>" (e.g. "env:production", "env:staging", "env:local")
    """
    import os
    from irys.core.tracing import LangfuseProvider, NoOpProvider
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if public_key and secret_key:
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        irys_env = os.environ.get("IRYS_ENV", "unknown")
        default_tags = ["ar-service", f"env:{irys_env}"]
        try:
            return LangfuseProvider(
                public_key=public_key, secret_key=secret_key, host=host,
                default_tags=default_tags,
            )
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "langfuse package not installed — tracing disabled. "
                "Install with: pip install langfuse"
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to init Langfuse tracing: %s", exc)
    return NoOpProvider()


def _make_irys(config: ServiceConfig, s3_prefix: Optional[str] = None):
    """Create an Irys instance wired with service S3 config."""
    from irys import Irys
    return Irys(
        api_key=config.gemini_api_key,
        s3_bucket=config.s3_bucket or None,
        s3_region=config.s3_region,
        s3_prefix=s3_prefix,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        tracing_provider=_make_tracing_provider(),
    )


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


# === ENDPOINTS ===


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to Gradio UI."""
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")

    # --- Full health check (restore when S3/Gemini checks are needed) ---
    # config = get_config()
    #
    # # Check S3 connection
    # s3_connected = False
    # if config.s3_bucket:
    #     try:
    #         import boto3
    #         s3 = boto3.client("s3", region_name=config.s3_region)
    #         s3.head_bucket(Bucket=config.s3_bucket)
    #         s3_connected = True
    #     except Exception:
    #         pass
    #
    # # Check Gemini connection
    # gemini_connected = bool(config.gemini_api_key)
    #
    # # Calculate temp storage usage
    # temp_dir = Path(config.temp_dir)
    # temp_size_mb = 0.0
    # if temp_dir.exists():
    #     temp_size_mb = sum(
    #         f.stat().st_size for f in temp_dir.rglob("*") if f.is_file()
    #     ) / (1024 * 1024)
    #
    # # Count active jobs
    # active_jobs = sum(
    #     1 for job in _jobs.values()
    #     if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    # )
    #
    # return HealthResponse(
    #     status="healthy",
    #     version=VERSION,
    #     gemini_connected=gemini_connected,
    #     s3_connected=s3_connected,
    #     active_jobs=active_jobs,
    #     temp_storage_mb=round(temp_size_mb, 2),
    #     uptime_seconds=round(time.time() - _start_time, 2),
    # )


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
        _t0 = time.monotonic()
        temp_dir = await s3_repo.download_to_temp(job_id)
        _setup_ms = int((time.monotonic() - _t0) * 1000)

        # Run investigation
        irys = _make_irys(config, s3_prefix=request.s3_prefix)

        seed_facts, seed_citations = await _load_session(config, request.session_id)

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
            seed_facts=seed_facts,
            seed_citations=seed_citations,
            context=request.context,
            message_id=getattr(request, "message_id", None),
            user_id=getattr(request, "user_id", None),
            setup_duration_ms=_setup_ms,
        )

        await _save_session(config, request.session_id, result)

        # Extract results
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at).total_seconds()

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
    files: list[UploadFile] = File(...,
                                   description="Document files to analyze"),
    callback_url: Optional[str] = Form(
        None, description="Webhook URL for results"),
    keep_files: bool = Form(
        False, description="Keep files in S3 after processing"),
    session_id: Optional[str] = Form(
        None, description="Session ID for cross-investigation fact persistence"),
    context_json: Optional[str] = Form(
        None, description="Investigation context as JSON string (conversation_history, planning_instructions, output_instructions)"),
    background_tasks: BackgroundTasks = None,
):
    """Start investigation with uploaded files (async).

    Storage mode controlled by IRYS_STORAGE_MODE env var:
    - "local": Files stored on VM disk (for development)
    - "s3": Files streamed to S3 (for production, keeps VM light)

    Set keep_files=true to preserve files in S3 for later re-query (s3 mode only).

    context_json example: {"planning_instructions": "Focus on damages", "output_instructions": "Respond in Spanish"}
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
            raise HTTPException(
                status_code=400, detail="No valid files uploaded")

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
        )
        _jobs[job_id] = job

        # Parse context JSON
        context = _parse_context_json(context_json)

        # Start background investigation
        background_tasks.add_task(
            _run_upload_investigation,
            job_id,
            query,
            s3_prefix,
            callback_url,
            keep_files,
            config,
            session_id,
            context,
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
    session_id: Optional[str] = None,
    context: Optional[InvestigationContext] = None,
):
    """Background task to run investigation on uploaded files."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    s3_repo = None
    temp_dir = None
    is_local = s3_prefix.startswith("local:")

    try:
        _t0 = time.monotonic()
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
        _setup_ms = int((time.monotonic() - _t0) * 1000)

        irys = _make_irys(config, s3_prefix=s3_prefix)

        seed_facts, seed_citations = await _load_session(config, session_id)

        result = await irys.investigate(
            query=query,
            repository=str(temp_dir),
            seed_facts=seed_facts,
            seed_citations=seed_citations,
            context=context,
            setup_duration_ms=_setup_ms,
        )

        await _save_session(config, session_id, result)

        # Extract results
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at
        ).total_seconds()

        logger.info(
            f"Upload job {job_id} completed in {job.duration_seconds:.1f}s (mode={'local' if is_local else 's3'})")

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
                await asyncio.to_thread(shutil.rmtree, temp_dir)
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
                    logger.warning(
                        f"Failed to cleanup S3 prefix {s3_prefix}: {e}")


@app.post(
    "/upload/search",
    response_model=UploadSearchResponse,
    tags=["File Upload"],
)
async def upload_search(
    query: str = Form(..., description="Search query"),
    files: list[UploadFile] = File(...,
                                   description="Document files to search"),
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
            raise HTTPException(
                status_code=400, detail="No valid files uploaded")

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
                await asyncio.to_thread(shutil.rmtree, temp_dir)
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
                await asyncio.to_thread(shutil.rmtree, temp_dir)
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
    files: list[UploadFile] = File(...,
                                   description="Document files to analyze"),
    keep_files: bool = Form(
        False, description="Keep files in S3 after processing for re-query"),
    session_id: Optional[str] = Form(
        None, description="Session ID for cross-investigation fact persistence"),
    context_json: Optional[str] = Form(
        None, description="Investigation context as JSON string (conversation_history, planning_instructions, output_instructions)"),
):
    """Upload and investigate files synchronously.

    Storage mode controlled by IRYS_STORAGE_MODE env var:
    - "local": Files stored on VM disk (for development)
    - "s3": Files streamed to S3 (for production, keeps VM light)

    Set keep_files=true to preserve files in S3 for later re-query (only applies in s3 mode).

    context_json example: {"planning_instructions": "Focus on damages", "output_instructions": "Respond in Spanish"}

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
            raise HTTPException(
                status_code=400, detail="No valid files uploaded")

        # Branch based on storage mode
        if config.storage_mode == "local":
            # LOCAL MODE: Save directly to temp directory
            temp_dir = Path(config.temp_dir) / job_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in file_data:
                (temp_dir / filename).write_bytes(content)
            _setup_ms = 0
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
            _t0 = time.monotonic()
            temp_dir = await upload_repo.download_to_temp(job_id)
            _setup_ms = int((time.monotonic() - _t0) * 1000)

        # Parse context JSON
        context = _parse_context_json(context_json)

        # Run investigation
        irys = _make_irys(config, s3_prefix=s3_prefix)

        seed_facts, seed_citations = await _load_session(config, session_id)

        result = await irys.investigate(
            query=query,
            repository=str(temp_dir),
            seed_facts=seed_facts,
            seed_citations=seed_citations,
            context=context,
            setup_duration_ms=_setup_ms,
        )

        await _save_session(config, session_id, result)

        # Cleanup temp files
        if config.storage_mode == "local":
            import shutil
            if temp_dir and temp_dir.exists():
                await asyncio.to_thread(shutil.rmtree, temp_dir)
        else:
            if s3_repo:
                await s3_repo.cleanup(job_id)
            # Cleanup S3 files (unless keep_files=True)
            if not keep_files and s3_repo and s3_prefix:
                await s3_repo.delete_prefix(s3_prefix)

        duration = time.time() - start_time
        logger.info(
            f"Sync investigation {job_id} completed in {duration:.1f}s (mode={config.storage_mode})")

        citations, entities = _serialize_result(result)
        facts = result.state.findings.get("accumulated_facts", [])
        response = SyncInvestigateResponse(
            query=query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            facts=facts,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
            session_id=session_id,
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
                await asyncio.to_thread(shutil.rmtree, temp_dir)
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
        _t0 = time.monotonic()
        temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)
        _setup_ms = int((time.monotonic() - _t0) * 1000)

        # Run investigation
        irys = _make_irys(config)

        seed_facts, seed_citations = await _load_session(config, request.session_id)

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
            seed_facts=seed_facts,
            seed_citations=seed_citations,
            context=request.context,
            message_id=getattr(request, "message_id", None),
            user_id=getattr(request, "user_id", None),
            setup_duration_ms=_setup_ms,
        )

        await _save_session(config, request.session_id, result)

        # Extract results
        job.analysis = result.output
        job.citations, job.entities = _serialize_result(result)
        job.documents_processed = result.state.documents_read
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at
        ).total_seconds()

        logger.info(
            f"URLs job {job_id} completed in {job.duration_seconds:.1f}s")

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
        _t0 = time.monotonic()
        temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)
        _setup_ms = int((time.monotonic() - _t0) * 1000)

        # Run investigation
        irys = _make_irys(config)

        seed_facts, seed_citations = await _load_session(config, request.session_id)

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
            seed_facts=seed_facts,
            seed_citations=seed_citations,
            context=request.context,
            message_id=getattr(request, "message_id", None),
            user_id=getattr(request, "user_id", None),
            setup_duration_ms=_setup_ms,
        )

        await _save_session(config, request.session_id, result)

        # Cleanup
        await s3_repo.cleanup(job_id)

        duration = time.time() - start_time
        logger.info(
            f"URL sync investigation {job_id} completed in {duration:.1f}s")

        citations, entities = _serialize_result(result)
        facts = result.state.findings.get("accumulated_facts", [])
        return SyncInvestigateResponse(
            query=request.query,
            analysis=result.output,
            citations=citations,
            entities=entities,
            facts=facts,
            documents_processed=result.state.documents_read,
            duration_seconds=round(duration, 2),
            session_id=request.session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL sync investigation failed: {e}")
        if s3_repo:
            await s3_repo.cleanup(job_id)
        raise HTTPException(status_code=500, detail=str(e))


# === SSE STREAMING ENDPOINT ===


@app.post(
    "/investigate/urls/stream",
    tags=["S3 URLs"],
    responses={400: {"model": ErrorResponse}},
)
async def investigate_urls_stream(request: S3UrlsInvestigateRequest):
    """Investigate documents from URLs with real-time SSE streaming.

    Streams investigation progress as Server-Sent Events. Event types:

    **Hierarchical lead-centric events:**

    **investigation.started** - Investigation begins:
      `{query, document_count, repository}`

    **plan** - Investigation plan created:
      `{leads: [{id, type, description}], success_criteria, key_issues, strategy, iteration}`

    **lead.started** - Lead begins execution:
      `{lead_id, type, description, parent_lead_id}`

    **lead.update** - Real-time update within a lead:
      `{lead_id, kind, data}` — kind: matches|fact|ranking|reading|insight|spawned|external_results|analysis|triggers

    **lead.done** - Lead completes:
      `{lead_id, duration_ms}`

    **lead.error** - Lead fails:
      `{lead_id, error}`

    **checkpoint** - Sufficiency check after iteration:
      `{decision, total_facts, docs_read, reasoning}`

    **replan** - New leads added after checkpoint:
      `{new_leads: [{id, type, description}], iteration}`

    **synthesis.started** - Synthesis begins:
      `{fact_count, citation_count, case_law_count, web_count, model}`

    **synthesis.complete** - Synthesis done:
      `{output_length, duration_ms, docs_read, facts_used, citations}`

    **Legacy events (still emitted for backward compatibility):**

    **step** - Unmapped step types:
      `{id, step_type, content, details, depth, timestamp, duration_ms}`

    **citation** - Citation found:
      `{id, document, page, text, context, relevance, timestamp, url, mime}`

    **progress** - Progress update (emitted at boundaries only):
      `{status, elapsed_seconds, documents_read, searches_performed, citations,
        leads_investigated, leads_pending, facts_accumulated, entities_found}`

    **complete** - Final result:
      `{query, analysis, citations, entities, facts, documents_processed, duration_seconds}`

    **error** - Fatal error:
      `{error}`

    Example curl:
    ```
    curl -N -X POST http://localhost:8000/investigate/urls/stream \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What are the payment terms?", "s3_urls": ["https://..."]}'
    ```
    """
    config = get_config()

    # Validate URL count upfront
    if len(request.s3_urls) > config.max_documents_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"Too many URLs ({len(request.s3_urls)}). Max: {config.max_documents_per_job}",
        )

    job_id = f"stream_{uuid.uuid4().hex[:8]}"
    message_id = request.message_id
    user_id = request.user_id

    # Set context var — propagates automatically to all logs within this
    # request's async task and any child tasks spawned from it.
    _log_message_id.set(message_id or '')

    logger.info(
        f"stream request received | job={job_id} msg={message_id} user={user_id} session={request.session_id} "
        f"urls={len(request.s3_urls)} query={request.query[:80]!r}"
    )

    queue: asyncio.Queue = asyncio.Queue()

    async def run_investigation():
        """Download docs and run investigation, pushing events to queue."""
        s3_repo = None
        temp_dir = None
        start_time = time.time()

        try:
            # Download documents from URLs
            s3_repo = S3Repository(
                bucket=config.s3_bucket or "placeholder",
                prefix="",
                config=config,
            )
            _t0 = time.monotonic()
            temp_dir = await s3_repo.download_urls_to_temp(job_id, request.s3_urls)
            _setup_ms = int((time.monotonic() - _t0) * 1000)

            # Create Irys with callbacks wired to queue
            irys = _make_irys(config)

            # Map new hierarchical StepType values to named SSE events
            _STEP_TYPE_TO_SSE_EVENT = {
                "investigation_started": "investigation.started",
                "plan": "plan",
                "lead_started": "lead.started",
                "lead_update": "lead.update",
                "lead_done": "lead.done",
                "lead_error": "lead.error",
                "checkpoint": "checkpoint",
                "replan": "replan",
                "synthesis_started": "synthesis.started",
                "synthesis_complete": "synthesis.complete",
            }

            def on_step(step):
                step_type_str = step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type)
                sse_event = _STEP_TYPE_TO_SSE_EVENT.get(step_type_str)

                if sse_event:
                    # New hierarchical event — details IS the payload
                    queue.put_nowait({
                        "event": sse_event,
                        "data": step.details or {},
                    })
                else:
                    # Legacy/unmapped step type — keep old format
                    queue.put_nowait({
                        "event": "step",
                        "data": {
                            "id": step.id,
                            "step_type": step_type_str,
                            "content": step.content,
                            "details": step.details,
                            "depth": step.depth,
                            "timestamp": step.timestamp.isoformat(),
                            "duration_ms": step.duration_ms,
                        },
                    })

            def on_citation(citation):
                queue.put_nowait({
                    "event": "citation",
                    "data": {
                        "id": citation.id,
                        "document": citation.document,
                        "page": citation.page,
                        "text": citation.text,
                        "context": citation.context,
                        "relevance": citation.relevance,
                        "timestamp": citation.timestamp.isoformat(),
                        "url": citation.url,
                        "mime": citation.mime,
                    },
                })

            def on_fact(fact):
                queue.put_nowait({
                    "event": "fact",
                    "data": {"fact": fact},
                })

            def on_progress(progress):
                queue.put_nowait({
                    "event": "progress",
                    "data": progress,
                })

            irys.on_step(on_step)
            irys.on_citation(on_citation)
            irys.on_fact(on_fact)
            irys.on_progress(on_progress)

            # Load prior session data
            seed_facts, seed_citations = await _load_session(config, request.session_id)

            result = await irys.investigate(
                query=request.query,
                repository=str(temp_dir),
                seed_facts=seed_facts,
                seed_citations=seed_citations,
                context=request.context,
                message_id=getattr(request, "message_id", None),
                user_id=getattr(request, "user_id", None),
                setup_duration_ms=_setup_ms,
            )

            # Save session data
            await _save_session(config, request.session_id, result)

            duration = time.time() - start_time
            citations, entities = _serialize_result(result)
            facts = result.state.findings.get("accumulated_facts", [])
            queue.put_nowait({
                "event": "complete",
                "data": {
                    "query": request.query,
                    "analysis": result.output,
                    "citations": citations,
                    "entities": entities,
                    "facts": facts,
                    "documents_processed": result.state.documents_read,
                    "duration_seconds": round(duration, 2),
                    "session_id": request.session_id,
                },
            })
            logger.info(
                f"stream complete | job={job_id} msg={message_id} user={user_id} session={request.session_id} "
                f"duration={duration:.1f}s docs={result.state.documents_read}"
            )

        except Exception as e:
            logger.error(f"Stream investigation failed | job={job_id} msg={message_id} user={user_id} session={request.session_id}: {e}")
            queue.put_nowait({
                "event": "error",
                "data": {"error": str(e)},
            })

        finally:
            # Cleanup temp files
            if s3_repo and temp_dir:
                await s3_repo.cleanup(job_id)
            # Signal end of stream
            queue.put_nowait(None)

    async def event_generator():
        """Yield SSE-formatted events from the queue.

        Uses asyncio.wait_for with timeout to yield control back to
        the event loop frequently, allowing the investigation task
        to make progress and push events to the queue.
        """
        task = asyncio.create_task(run_investigation())
        try:
            while True:
                try:
                    # Short timeout forces event loop to context-switch
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check if task finished while we were waiting
                    if task.done():
                        # Drain remaining items
                        while not queue.empty():
                            item = queue.get_nowait()
                            if item is None:
                                return
                            event_type = item["event"]
                            data = json.dumps(item["data"])
                            yield f"event: {event_type}\ndata: {data}\n\n"
                        return
                    # Yield heartbeat to flush buffers and keep connection alive
                    yield ": heartbeat\n\n"
                    continue

                if item is None:
                    break
                event_type = item["event"]
                data = json.dumps(item["data"])
                yield f"event: {event_type}\ndata: {data}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
