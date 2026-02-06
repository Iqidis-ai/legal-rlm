"""Simple Gradio UI for testing RLM capabilities.

Features:
- Directory picker with native folder browser (local mode)
- File upload with S3 storage (cloud mode)
- Query input
- REAL-TIME thinking display (streams as investigation progresses)
- Citations panel
- Thinking trace panel
"""

import gradio as gr
import asyncio
import threading
import queue
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Generator
import os

from ..core.models import GeminiClient
from ..rlm.engine import RLMEngine, RLMConfig
from ..rlm.state import InvestigationState, ThinkingStep, Citation, StepType
from ..service.config import ServiceConfig

logger = logging.getLogger(__name__)


def get_storage_mode() -> str:
    """Get storage mode from environment."""
    return os.getenv("IRYS_STORAGE_MODE", "local")


def browse_folder() -> str:
    """Open native folder picker dialog and return selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front

        # Open folder picker
        folder_path = filedialog.askdirectory(
            title="Select Document Repository",
            initialdir=os.path.expanduser("~")
        )

        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"Folder picker error: {e}")
        return ""


class RLMApp:
    """Gradio application wrapper for RLM with real-time streaming."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[ServiceConfig] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.config = config or ServiceConfig.from_env()
        self.storage_mode = get_storage_mode()
        self.current_state: Optional[InvestigationState] = None
        self.thinking_log: list[str] = []
        self.citations_log: list[str] = []
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.stop_requested = False  # Flag to stop investigation midway
        self.final_output = ""
        self.error_msg = ""
        self._temp_dirs: dict[str, Path] = {}  # Track temp dirs for cleanup

    def _generate_session_id(self) -> str:
        """Generate a unique session/job ID."""
        return f"ui_{uuid.uuid4().hex[:12]}"

    async def _upload_files_to_s3(
        self,
        files: list[tuple[str, bytes, str]],
        session_id: str,
    ) -> str:
        """Upload files to S3 and return the S3 prefix.

        Files are uploaded with their DISPLAY names (original filenames) directly.

        Args:
            files: List of (display_name, content, actual_filename) tuples
            session_id: Unique session identifier

        Returns:
            S3 prefix where files were uploaded
        """
        from ..service.s3_repository import S3Repository

        s3_repo = S3Repository(
            bucket=self.config.s3_bucket,
            config=self.config,
        )

        # Upload files with their DISPLAY names (original filenames)
        upload_files = []
        for display_name, content, actual_filename in files:
            upload_files.append((display_name, content))

        prefix = await s3_repo.upload_files(session_id, upload_files)
        logger.info(f"Uploaded {len(files)} files to S3: {prefix}")
        return prefix

    async def _download_s3_to_temp(self, s3_prefix: str, session_id: str) -> Path:
        """Download files from S3 prefix to temp directory for processing.

        Args:
            s3_prefix: S3 prefix containing the files
            session_id: Session ID for temp dir naming

        Returns:
            Path to temp directory with downloaded files
        """
        from ..service.s3_repository import S3Repository

        s3_repo = S3Repository(
            bucket=self.config.s3_bucket,
            prefix=s3_prefix,
            config=self.config,
        )

        temp_dir = await s3_repo.download_to_temp(session_id)
        self._temp_dirs[session_id] = temp_dir
        logger.info(f"Downloaded S3 files to temp: {temp_dir}")
        return temp_dir

    def _save_files_to_temp(
        self,
        files: list[tuple[str, bytes, str]],
        session_id: str,
    ) -> Path:
        """Save uploaded files to local temp directory.

        Files are saved with their DISPLAY names (original filenames) directly.

        Args:
            files: List of (display_name, content, actual_filename) tuples
            session_id: Unique session identifier

        Returns:
            Path to temp directory with files
        """
        temp_dir = Path(self.config.temp_dir) / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        for display_name, content, actual_filename in files:
            # Save with DISPLAY name (original filename)
            file_path = temp_dir / display_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            logger.debug(f"Saved file: {file_path}")

        self._temp_dirs[session_id] = temp_dir
        logger.info(f"Saved {len(files)} files to temp: {temp_dir}")
        return temp_dir

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up temp directory for a session."""
        if session_id in self._temp_dirs:
            import shutil
            temp_dir = self._temp_dirs.pop(session_id)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")

    def stop_investigation(self) -> str:
        """Stop the current investigation."""
        if self.is_running:
            self.stop_requested = True
            self.update_queue.put(("stopped", "Investigation stopped by user"))
            return "Stopping..."
        return "No investigation running"

    def on_thinking_step(self, step: ThinkingStep):
        """Callback for thinking steps - pushes to queue for real-time updates."""
        self.thinking_log.append(step.display)
        self.update_queue.put(("thinking", step.display))

    def on_citation(self, citation: Citation):
        """Callback for citations."""
        page_str = f", p. {citation.page}" if citation.page else ""
        # Build full citation with text and context
        citation_parts = [f"[{len(self.citations_log) + 1}] {citation.document}{page_str}"]
        if citation.context:
            citation_parts.append(f"    {citation.context}")
        if citation.text:
            # Show full text (truncate only for display if extremely long)
            text_display = citation.text if len(citation.text) <= 500 else citation.text[:500] + "..."
            citation_parts.append(f"    Text: {text_display}")
        citation_text = "\n".join(citation_parts)
        self.citations_log.append(citation_text)
        self.update_queue.put(("citation", citation_text))

    def run_investigation_async(self, query: str, repo_path: str):
        """Run investigation in background thread."""
        async def _run():
            try:
                client = GeminiClient(api_key=self.api_key)
                engine = RLMEngine(
                    gemini_client=client,
                    config=RLMConfig(),
                    on_step=self.on_thinking_step,
                    on_citation=self.on_citation,
                )
                state = await engine.investigate(query, repo_path)
                self.current_state = state
                self.final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", state))
            except Exception as e:
                self.error_msg = str(e)
                self.update_queue.put(("error", str(e)))

        asyncio.run(_run())

    def run_investigation_with_upload_async(
        self,
        query: str,
        files: list[tuple[str, bytes]],
        session_id: str,
    ):
        """Run investigation with uploaded files in background thread."""
        async def _run():
            repo_path = None
            try:
                # Handle file storage based on mode
                if self.storage_mode != "local":
                    # S3 mode: upload files then download to temp
                    s3_prefix = await self._upload_files_to_s3(files, session_id)
                    repo_path = await self._download_s3_to_temp(s3_prefix, session_id)
                else:
                    # Local mode: save directly to temp
                    repo_path = self._save_files_to_temp(files, session_id)

                # Run investigation
                client = GeminiClient(api_key=self.api_key)
                engine = RLMEngine(
                    gemini_client=client,
                    config=RLMConfig(),
                    on_step=self.on_thinking_step,
                    on_citation=self.on_citation,
                )
                state = await engine.investigate(query, str(repo_path))
                self.current_state = state
                self.final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", state))
            except Exception as e:
                self.error_msg = str(e)
                self.update_queue.put(("error", str(e)))
            finally:
                # Cleanup temp directory
                self._cleanup_session(session_id)

        asyncio.run(_run())

    def _extract_files_from_upload(
        self,
        uploaded_files: list,
    ) -> tuple[list[tuple[str, bytes, str]], str | None]:
        """Extract files from Gradio upload, preserving folder structure.

        Args:
            uploaded_files: List of Gradio file objects (can be files or folder contents)

        Returns:
            Tuple of (list of (display_name, content, actual_filename) tuples, error message or None)
            - display_name: The original filename to show to users/LLM
            - content: File bytes
            - actual_filename: The filename to use on disk (may be hash-based)
        """
        if not uploaded_files:
            return [], None

        files: list[tuple[str, bytes, str]] = []

        def get_original_filename(file_obj) -> tuple[str | None, str]:
            """Extract original filename from Gradio file object.

            Returns:
                Tuple of (original_name or None, actual_disk_name)
            """
            actual_name = Path(file_obj.name).name

            # Method 1: Try orig_name (Gradio 4.x+)
            if hasattr(file_obj, 'orig_name') and file_obj.orig_name:
                orig = file_obj.orig_name
                orig_str = Path(orig).name if (os.path.sep in str(orig) or '/' in str(orig)) else str(orig)
                # Validate it's not a hash name (Gradio may return hash as orig_name at scale)
                if not _is_hash_filename(orig_str):
                    return orig_str, actual_name
                # Fall through to other methods if orig_name is a hash
                logger.debug(f"orig_name '{orig_str}' looks like a hash, trying other methods")

            # Method 2: Try path attribute (some Gradio versions)
            if hasattr(file_obj, 'path') and file_obj.path:
                path_name = Path(file_obj.path).name
                # Check if it looks like a real filename (has extension)
                if '.' in path_name and not _is_hash_filename(path_name):
                    return path_name, actual_name

            # Method 3: Check if actual_name looks like a real filename
            if '.' in actual_name and not _is_hash_filename(actual_name):
                return actual_name, actual_name

            # No original name found
            return None, actual_name

        def _is_hash_filename(name: str) -> bool:
            """Check if filename looks like a content hash."""
            base = Path(name).stem
            if len(base) >= 32 and all(c in '0123456789abcdef' for c in base.lower()):
                return True
            return False

        def _detect_extension(content: bytes) -> str:
            """Detect file extension from content magic bytes."""
            if content.startswith(b'%PDF'):
                return '.pdf'
            if content.startswith(b'PK\x03\x04'):
                return '.docx'
            if content.startswith(b'\xd0\xcf\x11\xe0'):
                return '.doc'
            if content.startswith(b'{\\rtf'):
                return '.rtf'
            try:
                content[:1000].decode('utf-8')
                return '.txt'
            except UnicodeDecodeError:
                pass
            return ''

        # Build mapping of temp paths to original names for folder structure detection
        file_info = []
        for idx, f in enumerate(uploaded_files):
            temp_path = Path(f.name)
            orig_name, actual_name = get_original_filename(f)
            file_info.append((f, temp_path, orig_name, actual_name, idx))

        # Check if this looks like a folder upload (paths have common parent structure)
        # NOTE: We need to be careful with Gradio's temp structure where each file
        # is in its own hash-named directory: /tmp/gradio/<hash>/original_filename.docx
        # We should NOT preserve these hash directories as folder structure.
        all_paths = [info[1] for info in file_info]
        common_prefix = None
        is_gradio_temp = False

        if len(all_paths) > 1:
            try:
                common_prefix = Path(os.path.commonpath([str(p) for p in all_paths]))
                # Check if this is Gradio's temp directory structure
                # Each file has its own unique parent dir (hash-named)
                unique_parents = set(p.parent for p in all_paths)
                if len(unique_parents) == len(all_paths):
                    # Each file has a unique parent - this is Gradio's structure, not user folders
                    is_gradio_temp = True
                    logger.debug("Detected Gradio temp structure - ignoring hash directories")
            except ValueError:
                common_prefix = None

        for file, file_path, orig_name, actual_name, idx in file_info:
            try:
                with open(file.name, "rb") as f:
                    content = f.read()

                # Determine the display name (what users/LLM see)
                if orig_name:
                    display_name = orig_name
                else:
                    # Generate a display name from hash + detected extension
                    ext = _detect_extension(content)
                    if ext:
                        display_name = f"document_{idx + 1}{ext}"
                    else:
                        display_name = f"document_{idx + 1}"
                    logger.warning(
                        f"Could not get original filename for {actual_name}, "
                        f"using generated name: {display_name}"
                    )

                # Determine relative path for folder structure
                # If it's Gradio's temp structure, don't preserve the hash directories
                if is_gradio_temp:
                    relative_display = display_name
                    relative_actual = actual_name
                elif common_prefix and common_prefix != file_path:
                    rel_dir = file_path.parent.relative_to(common_prefix)
                    relative_display = str(rel_dir / display_name)
                    relative_actual = str(rel_dir / actual_name)
                else:
                    relative_display = display_name
                    relative_actual = actual_name

                # Skip hidden files
                if any(part.startswith('.') for part in Path(relative_display).parts):
                    logger.debug(f"Skipping hidden file: {relative_display}")
                    continue

                files.append((relative_display, content, relative_actual))
                logger.debug(
                    f"Extracted file: display={relative_display}, "
                    f"actual={relative_actual}"
                )

            except Exception as e:
                logger.error(f"Error reading file {file.name}: {e}")
                return [], f"Error reading file {file.name}: {e}"

        return files, None

    def stream_investigation_with_upload(
        self,
        query: str,
        uploaded_files: list,
        uploaded_folder: list = None,
    ) -> Generator[tuple, None, None]:
        """
        Generator that streams investigation with uploaded files/folders.
        Yields: (output, thinking_trace, citations, status)

        Args:
            query: The investigation query
            uploaded_files: List of Gradio file objects from file upload
            uploaded_folder: List of Gradio file objects from folder upload
        """
        self.thinking_log = []
        self.citations_log = []
        self.final_output = ""
        self.error_msg = ""
        self.update_queue = queue.Queue()
        self.stop_requested = False  # Reset stop flag

        # Combine files from both upload sources
        all_uploaded = []
        if uploaded_files:
            all_uploaded.extend(uploaded_files)
        if uploaded_folder:
            all_uploaded.extend(uploaded_folder)

        if not all_uploaded:
            yield ("", "", "", "Error: Please upload files or a folder")
            return

        if not query.strip():
            yield ("", "", "", "Error: Please enter a query")
            return

        # Extract files preserving folder structure
        files, error = self._extract_files_from_upload(all_uploaded)
        if error:
            yield ("", "", "", f"Error: {error}")
            return

        if not files:
            yield ("", "", "", "Error: No valid files found in upload")
            return

        session_id = self._generate_session_id()
        logger.info(f"Starting investigation session {session_id} with {len(files)} files")

        # Start investigation in background thread
        self.is_running = True
        thread = threading.Thread(
            target=self.run_investigation_with_upload_async,
            args=(query, files, session_id)
        )
        thread.start()

        # Use the common streaming loop
        yield from self._stream_updates(thread)

    def stream_investigation(
        self,
        query: str,
        repo_path: str,
    ) -> Generator[tuple, None, None]:
        """
        Generator that streams investigation updates in real-time.
        Yields: (output, thinking_trace, citations, status)
        """
        self.thinking_log = []
        self.citations_log = []
        self.final_output = ""
        self.error_msg = ""
        self.update_queue = queue.Queue()
        self.stop_requested = False  # Reset stop flag

        if not repo_path or not Path(repo_path).exists():
            yield ("", "", "", "Error: Please select a valid directory")
            return

        if not query.strip():
            yield ("", "", "", "Error: Please enter a query")
            return

        # Start investigation in background thread
        self.is_running = True
        thread = threading.Thread(
            target=self.run_investigation_async,
            args=(query, repo_path)
        )
        thread.start()

        # Use the common streaming loop
        yield from self._stream_updates(thread)

    def _stream_updates(self, thread: threading.Thread) -> Generator[tuple, None, None]:
        """Common streaming loop for investigation updates."""

        start_time = time.time()

        # Stream updates as they come in
        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.5)

                if update_type == "thinking":
                    elapsed = time.time() - start_time
                    status = (
                        f"Status: INVESTIGATING...\n"
                        f"Elapsed: {elapsed:.1f}s\n"
                        f"Steps: {len(self.thinking_log)}\n"
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "Finding sources...",
                        status
                    )

                elif update_type == "citation":
                    # Just update citations
                    pass

                elif update_type == "complete":
                    self.is_running = False
                    state = data
                    elapsed = time.time() - start_time
                    status = (
                        f"Status: COMPLETE\n"
                        f"Documents read: {state.documents_read}\n"
                        f"Searches performed: {state.searches_performed}\n"
                        f"Citations found: {len(state.citations)}\n"
                        f"Duration: {elapsed:.1f}s"
                    )
                    yield (
                        self.final_output,
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "No citations found",
                        status
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        "",
                        "\n".join(self.thinking_log),
                        "",
                        f"Error: {data}"
                    )
                    return

                elif update_type == "stopped":
                    self.is_running = False
                    elapsed = time.time() - start_time
                    # Show partial findings from thinking log
                    partial_summary = "\n".join(self.thinking_log[-5:]) if self.thinking_log else "No findings yet"
                    stop_output = f"*Investigation stopped after {elapsed:.1f}s*\n\nPartial findings:\n{partial_summary}"
                    yield (
                        stop_output,
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "No citations found",
                        f"Status: STOPPED\nDuration: {elapsed:.1f}s\nSteps completed: {len(self.thinking_log)}"
                    )
                    return

            except queue.Empty:
                # No update, yield current state to keep UI responsive
                elapsed = time.time() - start_time
                if len(self.thinking_log) > 0:
                    status = (
                        f"Status: INVESTIGATING...\n"
                        f"Elapsed: {elapsed:.1f}s\n"
                        f"Steps: {len(self.thinking_log)}\n"
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "Finding sources...",
                        status
                    )

        # Final yield after thread completes
        thread.join()


def create_app(api_key: Optional[str] = None) -> gr.Blocks:
    """Create the Gradio application with real-time streaming.

    The UI adapts based on IRYS_STORAGE_MODE:
    - "local": Shows folder browser for local file system
    - "s3" or other: Shows file upload component for cloud storage
    """
    app = RLMApp(api_key=api_key)
    storage_mode = get_storage_mode()
    is_local_mode = storage_mode == "local"

    with gr.Blocks(
        title="Irys RLM - Legal Document Analysis",
    ) as demo:
        gr.Markdown("# Irys RLM - Recursive Legal Document Analysis")

        if is_local_mode:
            gr.Markdown(
                "Select a matter repository and ask a legal question. "
                "**Watch the thinking trace update in real-time as the system investigates!**"
            )
        else:
            gr.Markdown(
                "Upload your legal documents and ask a question. "
                "**Watch the thinking trace update in real-time as the system investigates!**"
            )

        with gr.Row():
            with gr.Column(scale=2):
                if is_local_mode:
                    # Local mode: folder browser
                    with gr.Row():
                        repo_path = gr.Textbox(
                            label="Repository Path",
                            placeholder="Enter path or click Browse...",
                            info="Full path to the folder containing legal documents",
                            value="",
                            scale=4,
                        )
                        browse_btn = gr.Button("Browse", scale=1)
                else:
                    # Cloud mode: file upload AND folder upload
                    with gr.Tabs():
                        with gr.TabItem("Upload Files"):
                            file_upload = gr.File(
                                label="Upload Individual Documents",
                                file_count="multiple",
                                file_types=[".pdf", ".docx", ".doc", ".txt", ".rtf", ".mht"],
                                type="filepath",
                            )
                            gr.Markdown(
                                "*Select one or more files. Supported formats: PDF, DOCX, DOC, TXT, RTF, MHT*",
                            )
                        with gr.TabItem("Upload Folder"):
                            folder_upload = gr.File(
                                label="Upload Document Folder",
                                file_count="directory",
                                type="filepath",
                            )
                            gr.Markdown(
                                "*Select a folder to upload all documents including subfolders. "
                                "Folder structure will be preserved.*",
                            )

                query = gr.Textbox(
                    label="Legal Query",
                    placeholder="What would you like to investigate?",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("Investigate", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", lines=8, interactive=False)

        with gr.Tabs():
            with gr.TabItem("Thinking Trace (LIVE)"):
                thinking = gr.Textbox(
                    label="Thinking Steps - Updates in real-time!",
                    lines=30,
                    interactive=False,
                    autoscroll=True,
                )

            with gr.TabItem("Analysis Output"):
                output = gr.Markdown(label="Analysis")

            with gr.TabItem("Citations"):
                citations = gr.Textbox(
                    label="Citations & Sources",
                    lines=25,
                    interactive=False,
                )

        # Wire up buttons based on mode
        if is_local_mode:
            # Local mode: folder browser
            browse_btn.click(
                fn=browse_folder,
                inputs=[],
                outputs=[repo_path],
            )
            submit_btn.click(
                fn=app.stream_investigation,
                inputs=[query, repo_path],
                outputs=[output, thinking, citations, status],
            )
        else:
            # Cloud mode: file upload and folder upload
            submit_btn.click(
                fn=app.stream_investigation_with_upload,
                inputs=[query, file_upload, folder_upload],
                outputs=[output, thinking, citations, status],
            )

        # Stop button (works for both modes)
        stop_btn.click(
            fn=app.stop_investigation,
            outputs=[status],
        )

        # Example queries (click to populate query field)
        gr.Markdown("### Example Queries (click to use)")
        gr.Examples(
            examples=[
                ["What are the key claims in this dispute?"],
                ["What was the initial cost estimate vs actual cost?"],
                ["What damages are being claimed and what is the basis?"],
                ["What is the timeline of events in this case?"],
                ["Who are the key parties and witnesses?"],
            ],
            inputs=[query],
        )

    return demo


def main():
    """Run the application."""
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Server hostname to bind to (default: 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Path to SSL certificate file for HTTPS",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Path to SSL key file for HTTPS",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="Root path for reverse proxy setups (e.g., /app)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No GEMINI_API_KEY provided")

    storage_mode = get_storage_mode()
    print(f"[INFO] Storage mode: {storage_mode}")

    app = create_app(api_key=api_key)

    # Build launch kwargs
    launch_kwargs = {
        "server_port": args.port,
        "server_name": args.server_name,
        "share": args.share,
    }

    # Add SSL if provided
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
        print(f"[INFO] SSL enabled with cert: {args.ssl_certfile}")

    # Add root path for reverse proxy
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path
        print(f"[INFO] Root path: {args.root_path}")

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
