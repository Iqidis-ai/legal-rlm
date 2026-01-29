"""Multi-turn Chat UI for Irys RLM.

Features:
- Chat interface with conversation history
- Multiple queries about the same repository
- Thinking logs stored separately (not sent to AI)
- Only user messages + AI responses used for context
- Supports both local folder browsing and S3 file upload modes
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
from dataclasses import dataclass, field
import os
import json
from datetime import datetime

from ..core.models import GeminiClient
from ..rlm.engine import RLMEngine, RLMConfig
from ..rlm.state import InvestigationState, ThinkingStep, Citation, StepType
from ..service.config import ServiceConfig

logger = logging.getLogger(__name__)


def get_storage_mode() -> str:
    """Get storage mode from environment."""
    return os.getenv("IRYS_STORAGE_MODE", "local")


@dataclass
class ChatSession:
    """Represents a chat session with a repository."""
    repo_path: str
    conversation: list[tuple[str, str]] = field(default_factory=list)  # (user_msg, ai_msg)
    thinking_logs: list[dict] = field(default_factory=list)  # Per-turn thinking logs
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_turn(self, user_msg: str, ai_msg: str, thinking: list[str], citations: list[str]):
        """Add a conversation turn."""
        self.conversation.append((user_msg, ai_msg))
        self.thinking_logs.append({
            "turn": len(self.conversation),
            "query": user_msg,
            "thinking": thinking,
            "citations": citations,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_for_ai(self, max_turns: int = 5) -> list[dict]:
        """Get conversation history formatted for AI context.

        Only includes user messages and AI responses, not thinking logs.
        """
        history = []
        # Get last N turns
        recent = self.conversation[-max_turns:] if len(self.conversation) > max_turns else self.conversation

        for user_msg, ai_msg in recent:
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "model", "content": ai_msg})

        return history

    def get_thinking_for_turn(self, turn: int) -> Optional[dict]:
        """Get thinking logs for a specific turn."""
        if 0 < turn <= len(self.thinking_logs):
            return self.thinking_logs[turn - 1]
        return None


class ChatApp:
    """Multi-turn chat application for Irys RLM."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[ServiceConfig] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.config = config or ServiceConfig.from_env()
        self.storage_mode = get_storage_mode()
        self.session: Optional[ChatSession] = None
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.stop_requested = False  # Flag to stop investigation midway
        self._temp_dirs: dict[str, Path] = {}  # Track temp dirs for cleanup

        # Current turn state
        self.current_thinking: list[str] = []
        self.current_citations: list[str] = []

    def on_thinking_step(self, step: ThinkingStep):
        """Callback for thinking steps."""
        self.current_thinking.append(step.display)
        self.update_queue.put(("thinking", step.display))

    def on_citation(self, citation: Citation):
        """Callback for citations."""
        page_str = f", p. {citation.page}" if citation.page else ""
        citation_text = f"[{len(self.current_citations) + 1}] {citation.document}{page_str}"
        if citation.context:
            citation_text += f"\n    {citation.context}"
        self.current_citations.append(citation_text)
        self.update_queue.put(("citation", citation_text))

    def _generate_session_id(self) -> str:
        """Generate a unique session/job ID."""
        return f"chat_{uuid.uuid4().hex[:12]}"

    async def _upload_files_to_s3(
        self,
        files: list[tuple[str, bytes]],
        session_id: str,
    ) -> str:
        """Upload files to S3 and return the S3 prefix.

        Args:
            files: List of (filename, content) tuples
            session_id: Unique session identifier

        Returns:
            S3 prefix where files were uploaded
        """
        from ..service.s3_repository import S3Repository

        s3_repo = S3Repository(
            bucket=self.config.s3_bucket,
            config=self.config,
        )

        prefix = await s3_repo.upload_files(session_id, files)
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
        files: list[tuple[str, bytes]],
        session_id: str,
    ) -> Path:
        """Save uploaded files to local temp directory.

        Args:
            files: List of (relative_path, content) tuples - path can include subdirectories
            session_id: Unique session identifier

        Returns:
            Path to temp directory with files
        """
        temp_dir = Path(self.config.temp_dir) / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, content in files:
            file_path = temp_dir / relative_path
            # Create parent directories if they don't exist (for folder uploads)
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

    def _extract_files_from_upload(
        self,
        uploaded_files: list,
    ) -> tuple[list[tuple[str, bytes]], str | None]:
        """Extract files from Gradio upload, preserving folder structure.

        Args:
            uploaded_files: List of Gradio file objects (can be files or folder contents)

        Returns:
            Tuple of (list of (relative_path, content) tuples, error message or None)
        """
        if not uploaded_files:
            return [], None

        files: list[tuple[str, bytes]] = []

        def get_original_filename(file_obj) -> str:
            """Extract original filename from Gradio file object.

            Gradio stores uploaded files in temp directories with hash-based names.
            The original filename is available via:
            - file.orig_name (Gradio 4.x+)
            - Falling back to extracting from the temp path basename
            """
            # Try orig_name first (available in Gradio 4.x+)
            if hasattr(file_obj, 'orig_name') and file_obj.orig_name:
                orig = file_obj.orig_name
                # orig_name might be full path in some versions, extract basename
                if os.path.sep in str(orig) or '/' in str(orig):
                    return Path(orig).name
                return str(orig)

            # Fallback: extract from the temp path
            # This may be a hash-based name if Gradio uses content hashing
            return Path(file_obj.name).name

        # Build mapping of temp paths to original names for folder structure detection
        file_info = []
        for f in uploaded_files:
            temp_path = Path(f.name)
            orig_name = get_original_filename(f)
            file_info.append((f, temp_path, orig_name))

        # Check if this looks like a folder upload (paths have common parent structure)
        # Folder uploads typically have paths like: /tmp/gradio/.../folder_name/subdir/file.pdf
        all_paths = [info[1] for info in file_info]
        common_prefix = None
        if len(all_paths) > 1:
            # Find the common ancestor directory
            try:
                common_prefix = Path(os.path.commonpath([str(p) for p in all_paths]))
            except ValueError:
                # No common path (different drives on Windows, etc.)
                common_prefix = None

        for file, file_path, orig_name in file_info:
            try:
                # Determine relative path for folder structure preservation
                if common_prefix and common_prefix != file_path:
                    # This is a folder upload - preserve structure relative to common prefix
                    # Use the directory structure from temp path but with original filename
                    rel_dir = file_path.parent.relative_to(common_prefix)
                    relative_path = rel_dir / orig_name
                else:
                    # Single file or no common structure - just use original filename
                    relative_path = Path(orig_name)

                # Skip hidden files and system files
                if any(part.startswith('.') for part in relative_path.parts):
                    logger.debug(f"Skipping hidden file: {relative_path}")
                    continue

                with open(file.name, "rb") as f:
                    content = f.read()

                files.append((str(relative_path), content))
                logger.debug(f"Extracted file: {relative_path} (from {file_path.name})")

            except Exception as e:
                logger.error(f"Error reading file {file.name}: {e}")
                return [], f"Error reading file {file.name}: {e}"

        return files, None

    def _build_context_prompt(self, query: str) -> str:
        """Build prompt with conversation context."""
        if not self.session or not self.session.conversation:
            return query

        # Get recent conversation history
        history = self.session.get_context_for_ai(max_turns=5)

        if not history:
            return query

        # Format history as context
        context_parts = ["Previous conversation about this matter:"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long responses to save tokens
            content = msg["content"]
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            context_parts.append(f"\n{role}: {content}")

        context_parts.append(f"\n\nCurrent question: {query}")
        context_parts.append("\nAnswer the current question, using context from the previous conversation if relevant.")

        return "\n".join(context_parts)

    def run_investigation_async(self, query: str, repo_path: str, use_context: bool = True):
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

                # Build query with context if we have conversation history
                if use_context and self.session and self.session.conversation:
                    enhanced_query = self._build_context_prompt(query)
                else:
                    enhanced_query = query

                state = await engine.investigate(enhanced_query, repo_path)
                final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", final_output))

            except Exception as e:
                self.update_queue.put(("error", str(e)))

        asyncio.run(_run())

    def run_investigation_with_upload_async(
        self,
        query: str,
        files: list[tuple[str, bytes]],
        session_id: str,
        use_context: bool = True,
    ):
        """Run investigation with uploaded files in background thread.

        Handles S3 vs local storage mode, maintains conversation context,
        and cleans up temp files after each turn.
        """
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

                # Build query with context if we have conversation history
                if use_context and self.session and self.session.conversation:
                    enhanced_query = self._build_context_prompt(query)
                else:
                    enhanced_query = query

                state = await engine.investigate(enhanced_query, str(repo_path))
                final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", final_output))

            except Exception as e:
                self.update_queue.put(("error", str(e)))
            finally:
                # Cleanup temp directory after each turn
                self._cleanup_session(session_id)

        asyncio.run(_run())

    def chat(
        self,
        message: str,
        history,
        repo_path: str,
    ) -> Generator[tuple[list, str, str, str], None, None]:
        """
        Process a chat message and yield updates.

        Args:
            message: User's message
            history: Gradio chat history (messages format)
            repo_path: Path to document repository

        Yields:
            (updated_history, thinking_log, citations_log, status)
        """
        # Ensure history is a mutable list of dicts
        if history is None:
            history = []
        else:
            # Convert to list of dicts if needed
            history = [
                msg if isinstance(msg, dict) else {"role": "user" if i % 2 == 0 else "assistant", "content": str(msg)}
                for i, msg in enumerate(history)
            ]

        # Validate inputs
        if not repo_path or not Path(repo_path).exists():
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Error: Please select a valid repository directory."}
            ]
            yield history, "", "", "Error: Invalid repository path"
            return

        if not message.strip():
            yield history, "", "", "Please enter a question"
            return

        # Initialize or update session
        if self.session is None or self.session.repo_path != repo_path:
            self.session = ChatSession(repo_path=repo_path)

        # Reset current turn state
        self.current_thinking = []
        self.current_citations = []
        self.update_queue = queue.Queue()
        self.is_running = True
        self.stop_requested = False  # Reset stop flag

        # Create new history with user message and placeholder
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "*Investigating...*"}
        ]

        # Start investigation in background
        thread = threading.Thread(
            target=self.run_investigation_async,
            args=(message, repo_path, True)
        )
        thread.start()

        start_time = time.time()

        # Stream updates
        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.3)

                if update_type == "thinking":
                    elapsed = time.time() - start_time
                    status = f"Investigating... ({elapsed:.1f}s) - {len(self.current_thinking)} steps"
                    # Show all steps - they're now more informative so we want to see them all
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "Finding sources..."

                    # Update assistant placeholder with progress
                    history = history[:-1] + [{"role": "assistant", "content": f"*Investigating... ({len(self.current_thinking)} steps)*"}]
                    yield history, thinking_display, citations_display, status

                elif update_type == "citation":
                    # Update citations display
                    citations_display = "\n".join(self.current_citations)
                    yield history, "\n".join(self.current_thinking), citations_display, f"Found {len(self.current_citations)} citations"

                elif update_type == "complete":
                    self.is_running = False
                    final_output = data
                    elapsed = time.time() - start_time

                    # Store in session
                    self.session.add_turn(
                        user_msg=message,
                        ai_msg=final_output,
                        thinking=self.current_thinking.copy(),
                        citations=self.current_citations.copy()
                    )

                    # Update history with final response
                    history = history[:-1] + [{"role": "assistant", "content": final_output}]

                    status = f"Complete ({elapsed:.1f}s) - Turn {len(self.session.conversation)}"
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "No citations found"

                    yield history, thinking_display, citations_display, status
                    return

                elif update_type == "stopped":
                    self.is_running = False
                    elapsed = time.time() - start_time
                    stop_msg = f"*Investigation stopped after {elapsed:.1f}s*\n\nPartial findings:\n" + "\n".join(self.current_thinking[-5:])
                    history = history[:-1] + [{"role": "assistant", "content": stop_msg}]
                    yield history, "\n".join(self.current_thinking), "\n".join(self.current_citations), "Stopped by user"
                    return

                elif update_type == "error":
                    self.is_running = False
                    error_msg = f"Error: {data}"
                    history = history[:-1] + [{"role": "assistant", "content": error_msg}]
                    yield history, "\n".join(self.current_thinking), "\n".join(self.current_citations), f"Error: {data}"
                    return

            except queue.Empty:
                # Keep UI responsive
                if self.current_thinking:
                    elapsed = time.time() - start_time
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "Finding sources..."
                    yield history, thinking_display, citations_display, f"Investigating... ({elapsed:.1f}s)"

        thread.join()

    def chat_with_upload(
        self,
        message: str,
        history,
        uploaded_files: list = None,
        uploaded_folder: list = None,
    ) -> Generator[tuple[list, str, str, str], None, None]:
        """
        Process a chat message with uploaded files and yield updates.

        This method is used for S3 mode where files are uploaded rather than
        selected from the local filesystem.

        Args:
            message: User's message
            history: Gradio chat history (messages format)
            uploaded_files: List of Gradio file objects from file upload
            uploaded_folder: List of Gradio file objects from folder upload

        Yields:
            (updated_history, thinking_log, citations_log, status)
        """
        # Ensure history is a mutable list of dicts
        if history is None:
            history = []
        else:
            # Convert to list of dicts if needed
            history = [
                msg if isinstance(msg, dict) else {"role": "user" if i % 2 == 0 else "assistant", "content": str(msg)}
                for i, msg in enumerate(history)
            ]

        # Combine files from both upload sources
        all_uploaded = []
        if uploaded_files:
            all_uploaded.extend(uploaded_files)
        if uploaded_folder:
            all_uploaded.extend(uploaded_folder)

        # Validate inputs
        if not all_uploaded:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Error: Please upload files or a folder."}
            ]
            yield history, "", "", "Error: No files uploaded"
            return

        if not message.strip():
            yield history, "", "", "Please enter a question"
            return

        # Extract files preserving folder structure
        files, error = self._extract_files_from_upload(all_uploaded)
        if error:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error: {error}"}
            ]
            yield history, "", "", f"Error: {error}"
            return

        if not files:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Error: No valid files found in upload."}
            ]
            yield history, "", "", "Error: No valid files found"
            return

        # Generate session ID for this upload
        session_id = self._generate_session_id()
        logger.info(f"Starting chat session {session_id} with {len(files)} files")

        # Initialize or reset session for uploaded files
        # Use session_id as repo_path identifier since files are temporary
        if self.session is None or self.session.repo_path != session_id:
            self.session = ChatSession(repo_path=session_id)

        # Reset current turn state
        self.current_thinking = []
        self.current_citations = []
        self.update_queue = queue.Queue()
        self.is_running = True
        self.stop_requested = False

        # Create new history with user message and placeholder
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "*Investigating...*"}
        ]

        # Start investigation in background
        thread = threading.Thread(
            target=self.run_investigation_with_upload_async,
            args=(message, files, session_id, True)
        )
        thread.start()

        start_time = time.time()

        # Stream updates (same loop as chat())
        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.3)

                if update_type == "thinking":
                    elapsed = time.time() - start_time
                    status = f"Investigating... ({elapsed:.1f}s) - {len(self.current_thinking)} steps"
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "Finding sources..."

                    history = history[:-1] + [{"role": "assistant", "content": f"*Investigating... ({len(self.current_thinking)} steps)*"}]
                    yield history, thinking_display, citations_display, status

                elif update_type == "citation":
                    citations_display = "\n".join(self.current_citations)
                    yield history, "\n".join(self.current_thinking), citations_display, f"Found {len(self.current_citations)} citations"

                elif update_type == "complete":
                    self.is_running = False
                    final_output = data
                    elapsed = time.time() - start_time

                    # Store in session
                    self.session.add_turn(
                        user_msg=message,
                        ai_msg=final_output,
                        thinking=self.current_thinking.copy(),
                        citations=self.current_citations.copy()
                    )

                    history = history[:-1] + [{"role": "assistant", "content": final_output}]

                    status = f"Complete ({elapsed:.1f}s) - Turn {len(self.session.conversation)}"
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "No citations found"

                    yield history, thinking_display, citations_display, status
                    return

                elif update_type == "stopped":
                    self.is_running = False
                    elapsed = time.time() - start_time
                    stop_msg = f"*Investigation stopped after {elapsed:.1f}s*\n\nPartial findings:\n" + "\n".join(self.current_thinking[-5:])
                    history = history[:-1] + [{"role": "assistant", "content": stop_msg}]
                    yield history, "\n".join(self.current_thinking), "\n".join(self.current_citations), "Stopped by user"
                    return

                elif update_type == "error":
                    self.is_running = False
                    error_msg = f"Error: {data}"
                    history = history[:-1] + [{"role": "assistant", "content": error_msg}]
                    yield history, "\n".join(self.current_thinking), "\n".join(self.current_citations), f"Error: {data}"
                    return

            except queue.Empty:
                if self.current_thinking:
                    elapsed = time.time() - start_time
                    thinking_display = "\n".join(self.current_thinking)
                    citations_display = "\n".join(self.current_citations) or "Finding sources..."
                    yield history, thinking_display, citations_display, f"Investigating... ({elapsed:.1f}s)"

        thread.join()

    def stop_investigation(self) -> str:
        """Stop the current investigation."""
        if self.is_running:
            self.stop_requested = True
            self.update_queue.put(("stopped", "Investigation stopped by user"))
            return "Stopping..."
        return "No investigation running"

    def clear_chat(self) -> tuple[list, str, str, str]:
        """Clear the current chat session."""
        self.session = None
        self.current_thinking = []
        self.current_citations = []
        return [], "", "", "Chat cleared. Start a new conversation."

    def get_turn_details(self, turn_number: int) -> str:
        """Get detailed thinking logs for a specific turn."""
        if not self.session:
            return "No active session"

        turn_data = self.session.get_thinking_for_turn(turn_number)
        if not turn_data:
            return f"No data for turn {turn_number}"

        output = [
            f"## Turn {turn_number} Details",
            f"**Query:** {turn_data['query']}",
            f"**Time:** {turn_data['timestamp']}",
            "",
            "### Thinking Steps",
        ]
        output.extend(turn_data['thinking'])

        if turn_data['citations']:
            output.append("\n### Citations")
            output.extend(turn_data['citations'])

        return "\n".join(output)


def browse_folder() -> str:
    """Open native folder picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        folder_path = filedialog.askdirectory(
            title="Select Document Repository",
            initialdir=os.path.expanduser("~")
        )

        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"Folder picker error: {e}")
        return ""


def create_chat_app(api_key: Optional[str] = None) -> gr.Blocks:
    """Create the multi-turn chat Gradio application.

    The UI adapts based on IRYS_STORAGE_MODE:
    - "local": Shows folder browser for local file system
    - "s3" or other: Shows file upload component for cloud storage
    """
    app = ChatApp(api_key=api_key)
    storage_mode = get_storage_mode()
    is_local_mode = storage_mode == "local"

    with gr.Blocks(
        title="Irys RLM - Legal Document Chat",
    ) as demo:
        gr.Markdown("# Irys RLM - Legal Document Chat")

        if is_local_mode:
            gr.Markdown(
                "Select a document repository and have a conversation about it. "
                "Ask follow-up questions - the system remembers your conversation."
            )
        else:
            gr.Markdown(
                "Upload your legal documents and have a conversation about them. "
                "Ask follow-up questions - the system remembers your conversation."
            )

        with gr.Row():
            # Left column: Chat
            with gr.Column(scale=2):
                if is_local_mode:
                    # Local mode: folder browser
                    with gr.Row():
                        repo_path = gr.Textbox(
                            label="Repository Path",
                            placeholder="Select a folder containing legal documents...",
                            scale=4,
                        )
                        browse_btn = gr.Button("Browse", scale=1)
                else:
                    # S3/Cloud mode: file upload tabs
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

                # Chat interface (Gradio 6.x uses messages format by default)
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about the documents...",
                        scale=4,
                        lines=2,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")

            # Right column: Status and thinking
            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

                with gr.Accordion("Thinking Trace (Current Turn)", open=True):
                    thinking = gr.Textbox(
                        label="Investigation Steps",
                        interactive=False,
                        lines=15,
                        autoscroll=True,
                    )

                with gr.Accordion("Citations (Current Turn)", open=True):
                    citations = gr.Textbox(
                        label="Sources Found",
                        interactive=False,
                        lines=10,
                        autoscroll=True,
                    )

        # Example queries
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["What are the main claims in this dispute?"],
                ["What is the timeline of key events?"],
                ["Who are the parties involved?"],
                ["What damages are being claimed?"],
                ["Can you explain more about the breach allegations?"],  # Follow-up style
            ],
            inputs=[msg],
        )

        # Event handlers - wired based on storage mode
        if is_local_mode:
            # Local mode: folder browser
            browse_btn.click(
                fn=browse_folder,
                outputs=[repo_path],
            )

            # Submit on button click
            submit_btn.click(
                fn=app.chat,
                inputs=[msg, chatbot, repo_path],
                outputs=[chatbot, thinking, citations, status],
            ).then(
                fn=lambda: "",  # Clear input after submit
                outputs=[msg],
            )

            # Submit on Enter
            msg.submit(
                fn=app.chat,
                inputs=[msg, chatbot, repo_path],
                outputs=[chatbot, thinking, citations, status],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            )
        else:
            # S3/Cloud mode: file upload
            submit_btn.click(
                fn=app.chat_with_upload,
                inputs=[msg, chatbot, file_upload, folder_upload],
                outputs=[chatbot, thinking, citations, status],
            ).then(
                fn=lambda: "",  # Clear input after submit
                outputs=[msg],
            )

            # Submit on Enter
            msg.submit(
                fn=app.chat_with_upload,
                inputs=[msg, chatbot, file_upload, folder_upload],
                outputs=[chatbot, thinking, citations, status],
            ).then(
                fn=lambda: "",
                outputs=[msg],
            )

        # Clear chat (same for both modes)
        clear_btn.click(
            fn=app.clear_chat,
            outputs=[chatbot, thinking, citations, status],
        )

        # Stop investigation (same for both modes)
        stop_btn.click(
            fn=app.stop_investigation,
            outputs=[status],
        )

    return demo


def main():
    """Run the chat application."""
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM Chat UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7863, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Server hostname to bind to (default: 0.0.0.0 for all interfaces)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No GEMINI_API_KEY provided")

    storage_mode = get_storage_mode()
    print(f"[INFO] Storage mode: {storage_mode}")

    app = create_chat_app(api_key=api_key)
    app.launch(
        server_port=args.port,
        server_name=args.server_name,
        share=args.share,
    )


if __name__ == "__main__":
    main()
