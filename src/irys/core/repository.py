"""Matter Repository - programmatic access to legal document collections.

No vectors. Direct file access, reading, and search.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator
import os
import json
import logging
import mimetypes

from .reader import DocumentReader, DocumentContent, OcrCallMetadata
from .search import DocumentSearch, SearchResults, SearchHit

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Metadata about a file in the repository."""
    path: Path
    filename: str
    file_type: str
    size_bytes: int
    relative_path: str

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class RepositoryStats:
    """Statistics about the repository."""
    total_files: int
    total_size_bytes: int
    files_by_type: dict[str, int]
    folders: list[str]
    skipped_legacy_files: int = 0  # Count of .doc/.rtf files that can't be processed

    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def has_legacy_files(self) -> bool:
        """True if there are legacy files that couldn't be processed."""
        return self.skipped_legacy_files > 0


@dataclass
class RepositoryMetadata:
    """Metadata about repository content size."""
    total_files: int
    total_chars: int
    learnings: dict[str, str] = field(default_factory=dict)  # query -> key facts


class MatterRepository:
    """
    Programmatic access to a legal matter document repository.

    Provides:
    - File listing and navigation
    - Document reading (PDF, DOCX, TXT, MHT)
    - Grep-style search across all documents
    - Parallel operations

    Note: Old .doc (binary) and .rtf formats are NOT supported.
    Convert to .docx or .pdf before adding to repository.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".mht", ".mhtml", ".png", ".jpg", ".jpeg"}
    # Extensions that require async read (OCR path) — sync read() will raise for these
    _ASYNC_ONLY_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise ValueError(f"Repository path does not exist: {base_path}")
        if not self.base_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {base_path}")

        self.reader = DocumentReader()
        self._doc_cache: dict[str, DocumentContent] = {}  # Global document cache
        self.search_engine = DocumentSearch(self.reader)
        self.search_engine._doc_cache = self._doc_cache  # Share cache
        self._file_cache: Optional[list[FileInfo]] = None
        self._metadata: Optional["RepositoryMetadata"] = None
        self._display_path_map: Optional[dict[str, Path]] = None  # display_name -> actual Path

        # Load filename mapping if it exists (for S3/upload mode)
        self._filename_mapping = self._load_filename_mapping()

        # Auto-detect and rename hash-named files if no mapping exists
        if not self._filename_mapping["actual_to_display"]:
            renamed = self._auto_detect_hash_files()
            if renamed > 0:
                logger.info(f"Auto-renamed {renamed} hash-named files with detected extensions")

        logger.info(f"Initialized repository: {base_path}")

    def _load_filename_mapping(self) -> dict:
        """Load filename mapping from _filename_mapping.json if it exists.

        Returns:
            Dict with four sub-dicts:
            - 'actual_to_display': actual_filename -> display_name
            - 'display_to_actual': display_name -> actual_filename
            - 'display_to_url': display_name -> url (if available)
            - 'display_to_mime': display_name -> mime type (if available)
        """
        mapping_path = self.base_path / "_filename_mapping.json"
        result = {
            "actual_to_display": {},
            "display_to_actual": {},
            "display_to_url": {},
            "display_to_mime": {},
        }

        if not mapping_path.exists():
            logger.debug(f"No filename mapping found at {mapping_path}")
            # List files in directory to help debug
            try:
                files_in_dir = list(self.base_path.glob("*"))[:10]
                file_names = [f.name for f in files_in_dir]
                logger.debug(f"Files in {self.base_path}: {file_names}")
            except Exception:
                pass
            return result

        try:
            with open(mapping_path) as f:
                raw_mapping = json.load(f)

            # Build bidirectional mappings
            for actual_name, info in raw_mapping.items():
                if isinstance(info, dict):
                    display_name = info.get("display_name", actual_name)
                    url = info.get("url")
                    mime = info.get("mime")
                else:
                    # Legacy format: just a string
                    display_name = str(info)
                    url = None
                    mime = None

                result["actual_to_display"][actual_name] = display_name
                result["display_to_actual"][display_name] = actual_name
                if url:
                    result["display_to_url"][display_name] = url
                if mime:
                    result["display_to_mime"][display_name] = mime

            logger.info(f"Loaded filename mapping with {len(raw_mapping)} entries")
            logger.debug(f"Sample mappings: {list(result['display_to_actual'].items())[:3]}")
        except Exception as e:
            logger.warning(f"Failed to load filename mapping: {e}")

        return result

    @staticmethod
    def _detect_extension_from_file(path: Path) -> str:
        """Detect file extension from magic bytes of a file on disk.

        Returns:
            Extension string (e.g., '.pdf') or empty string if undetectable.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except Exception:
            return ""

        if header.startswith(b'%PDF'):
            return '.pdf'
        if header.startswith(b'PK\x03\x04'):
            return '.docx'  # ZIP-based (could be docx, xlsx, etc.)
        if header.startswith(b'\xd0\xcf\x11\xe0'):
            return '.doc'
        if header.startswith(b'{\\rtf'):
            return '.rtf'
        # Try to detect text files
        try:
            with open(path, "rb") as f:
                sample = f.read(1000)
            sample.decode('utf-8')
            return '.txt'
        except (UnicodeDecodeError, Exception):
            pass
        return ''

    def _auto_detect_hash_files(self) -> int:
        """Detect hash-named files without extensions and rename with detected extension.

        Scans the repository for files that look like content hashes (32+ hex chars,
        no extension). For each, detects the file type from magic bytes and renames
        the file on disk to include the correct extension.

        Returns:
            Number of files renamed.
        """
        renamed = 0
        for path in self.base_path.glob("*"):
            if not path.is_file():
                continue
            name = path.name
            # Skip special files
            if name.startswith("_") or name.startswith("~$") or name.startswith("."):
                continue
            # Check if it's a hash filename (no extension, 32+ hex chars)
            if '.' in name or len(name) < 32:
                continue
            if not all(c in '0123456789abcdef' for c in name.lower()):
                continue
            # Detect extension from magic bytes
            ext = self._detect_extension_from_file(path)
            if ext:
                new_path = path.with_suffix(ext)
                if not new_path.exists():
                    path.rename(new_path)
                    renamed += 1
                    logger.debug(f"Renamed hash file: {name} -> {new_path.name}")
                else:
                    logger.warning(f"Cannot rename {name} to {new_path.name}: target exists")
        return renamed

    def _get_display_name(self, actual_filename: str) -> str:
        """Get display name for an actual filename."""
        return self._filename_mapping["actual_to_display"].get(
            actual_filename, actual_filename
        )

    def _get_actual_filename(self, display_name: str) -> str:
        """Get actual filename from a display name."""
        return self._filename_mapping["display_to_actual"].get(
            display_name, display_name
        )

    def get_document_url(self, document_name: str) -> Optional[str]:
        """Get URL for a document if available.

        Args:
            document_name: Display name or actual filename of the document

        Returns:
            URL string if available, None otherwise
        """
        url = self._filename_mapping["display_to_url"].get(document_name)
        if url:
            return url

        display_name = self._get_display_name(document_name)
        return self._filename_mapping["display_to_url"].get(display_name)

    def get_document_mime(self, document_name: str) -> Optional[str]:
        """Get MIME type for a document if available or infer it from its filename."""
        mime = self._filename_mapping["display_to_mime"].get(document_name)
        if mime:
            return mime

        display_name = self._get_display_name(document_name)
        mime = self._filename_mapping["display_to_mime"].get(display_name)
        if mime:
            return mime

        guessed_mime, _ = mimetypes.guess_type(display_name or document_name)
        return guessed_mime

    def _get_display_path_map(self) -> dict[str, Path]:
        """Lazily build mapping from display filenames to actual file Paths.

        This provides a robust fallback for _resolve_path when the
        _filename_mapping.json lookup fails (e.g., incomplete mapping,
        S3 download with hash filenames).
        """
        if self._display_path_map is None:
            self._display_path_map = {}
            for file_info in self.list_files():
                self._display_path_map[file_info.filename] = file_info.path
            logger.debug(f"Built display_path_map with {len(self._display_path_map)} entries")
        return self._display_path_map

    @property
    def is_small_repo(self) -> bool:
        """Check if repository is small enough for direct content loading."""
        SMALL_REPO_THRESHOLD = 100_000  # 100K chars
        if self._metadata is None:
            self._compute_metadata()
        return self._metadata.total_chars < SMALL_REPO_THRESHOLD

    @property
    def metadata(self) -> Optional["RepositoryMetadata"]:
        """Get repository metadata (lazy computed)."""
        if self._metadata is None:
            self._compute_metadata()
        return self._metadata

    def _compute_metadata(self) -> None:
        """Compute and cache repository metadata."""
        files = self.list_files()
        total_chars = 0
        for f in files[:50]:  # Sample first 50 files to estimate
            try:
                # Skip image files — they require async read (OCR)
                if Path(f.path).suffix.lower() in self._ASYNC_ONLY_EXTENSIONS:
                    continue
                doc = self.read(f.path)
                total_chars += len(doc.full_text)
            except Exception:
                pass
        # Extrapolate if we sampled
        if len(files) > 50:
            total_chars = int(total_chars * len(files) / 50)
        self._metadata = RepositoryMetadata(
            total_files=len(files),
            total_chars=total_chars,
        )

    def add_learning(self, query: str, facts: str) -> None:
        """Store learnings from a query for future reference."""
        if self._metadata is None:
            self._compute_metadata()
        self._metadata.learnings[query] = facts

    def get_learnings(self, limit: int = 5) -> list[dict[str, str]]:
        """Get recent learnings from this repository."""
        if self._metadata is None:
            return []
        learnings = []
        for query, finding in list(self._metadata.learnings.items())[:limit]:
            learnings.append({"query": query, "finding": finding})
        return learnings

    def get_all_content(self, max_chars: int = 500_000) -> str:
        """Get all document content concatenated (for small repos)."""
        files = self.list_files()
        content_parts = []
        total_chars = 0
        for f in files:
            if total_chars >= max_chars:
                break
            try:
                # Image files require async read (OCR) — skip in sync context
                if Path(f.path).suffix.lower() in self._ASYNC_ONLY_EXTENSIONS:
                    logger.debug(f"Skipping image file in get_all_content (use read_async): {f.filename}")
                    continue
                doc = self.read(f.path)
                text = doc.full_text
                content_parts.append(f"\n\n=== {f.filename} ===\n{text}")
                total_chars += len(text)
            except Exception as e:
                logger.warning(f"Failed to read {f.filename}: {e}")
        return "".join(content_parts)

    # === NAVIGATION ===

    def list_files(
        self,
        pattern: str = "**/*",
        file_types: Optional[list[str]] = None,
    ) -> list[FileInfo]:
        """
        List all documents matching pattern.

        Args:
            pattern: Glob pattern (default: all files recursively)
            file_types: Filter by extensions (e.g., [".pdf", ".docx"])

        Returns:
            List of FileInfo objects with display names (if mapping exists)
        """
        files = []
        file_types = file_types or list(self.SUPPORTED_EXTENSIONS)
        file_types = [ft.lower() if ft.startswith(".") else f".{ft.lower()}" for ft in file_types]

        for path in self.base_path.glob(pattern):
            # Skip mapping file and temp files
            if path.name == "_filename_mapping.json":
                continue
            if path.name.startswith("~$"):
                continue

            if not path.is_file():
                continue

            # Get display name from mapping FIRST (before extension filtering)
            actual_name = path.name
            display_name = self._get_display_name(actual_name)

            # Determine extension - prefer display name extension for hash files
            display_ext = Path(display_name).suffix.lower()
            actual_ext = path.suffix.lower()
            effective_ext = display_ext if display_ext else actual_ext

            # Last resort: detect extension from magic bytes for extensionless files
            if not effective_ext:
                effective_ext = self._detect_extension_from_file(path)

            # Filter by extension
            if effective_ext not in file_types:
                continue

            files.append(FileInfo(
                path=path,
                filename=display_name,  # Use display name for user/LLM
                file_type=effective_ext,
                size_bytes=path.stat().st_size,
                relative_path=str(path.relative_to(self.base_path)),
            ))

        return sorted(files, key=lambda f: f.relative_path)

    def get_structure(self) -> dict[str, int]:
        """Get folder tree with document counts."""
        structure: dict[str, int] = {}

        for file_info in self.list_files():
            folder = str(Path(file_info.relative_path).parent)
            if folder == ".":
                folder = "(root)"
            structure[folder] = structure.get(folder, 0) + 1

        return dict(sorted(structure.items()))

    def get_file_list(self) -> list[dict]:
        """Get list of files with names and metadata for LLM planning.

        Returns list of dicts with:
        - filename: The file name
        - path: Relative path for reading
        - type: File extension
        - size_kb: Size in KB (rounded)
        """
        files = []
        for file_info in self.list_files():
            files.append({
                "filename": file_info.filename,
                "path": file_info.relative_path,
                "type": file_info.file_type,
                "size_kb": round(file_info.size_bytes / 1024),
            })
        return files

    def get_stats(self) -> RepositoryStats:
        """Get repository statistics.

        Also detects and counts legacy files (.doc, .rtf) that exist
        but cannot be processed. Logs a single summary warning if any
        legacy files are found.
        """
        files_by_type: dict[str, int] = {}
        total_size = 0
        folders = set()
        total_files = 0

        # Legacy file tracking
        legacy_extensions = {".doc", ".rtf"}
        skipped_legacy = 0
        legacy_samples: list[str] = []  # Track a few for the warning

        # Single traversal: count supported and legacy files
        for path in self.base_path.glob("**/*"):
            if not path.is_file() or path.name.startswith("~$"):
                continue

            ext = path.suffix.lower()

            if ext in self.SUPPORTED_EXTENSIONS:
                total_files += 1
                files_by_type[ext] = files_by_type.get(ext, 0) + 1
                total_size += path.stat().st_size
                rel_path = str(path.relative_to(self.base_path))
                folder = str(Path(rel_path).parent)
                if folder == ".":
                    folder = "(root)"  # Normalize to match get_structure()
                folders.add(folder)
            elif ext in legacy_extensions:
                skipped_legacy += 1
                if len(legacy_samples) < 3:  # Collect up to 3 samples
                    legacy_samples.append(str(path.relative_to(self.base_path)))

        # Log a single summary warning if legacy files found
        if skipped_legacy > 0:
            samples_str = ", ".join(legacy_samples)
            if skipped_legacy > 3:
                samples_str += f", ... ({skipped_legacy - 3} more)"
            logger.warning(
                f"Repository contains {skipped_legacy} unsupported legacy file(s) "
                f"(.doc/.rtf): {samples_str}. Convert to .docx or .pdf for processing."
            )

        return RepositoryStats(
            total_files=total_files,
            total_size_bytes=total_size,
            files_by_type=files_by_type,
            folders=sorted(folders),
            skipped_legacy_files=skipped_legacy,
        )

    # === READING ===

    def read(self, path: str | Path) -> DocumentContent:
        """
        Read a document and extract text.

        Args:
            path: Absolute path or path relative to repository

        Returns:
            DocumentContent with pages and text
        """
        full_path = self._resolve_path(path)
        cache_key = str(full_path)

        if cache_key not in self._doc_cache:
            logger.debug(f"Reading document: {full_path.name}")
            self._doc_cache[cache_key] = self.reader.read(full_path)

        return self._doc_cache[cache_key]

    async def read_async(
        self,
        path: str | Path,
        ocr_timeout: float = 60.0,
    ) -> tuple[DocumentContent, Optional[OcrCallMetadata]]:
        """Async read — required for image files and OCR-fallback PDF/DOCX.

        Returns (DocumentContent, OcrCallMetadata | None).
        OcrCallMetadata is non-None only when a Mistral OCR call was made.
        Caches the DocumentContent result (same cache as read()).
        """
        full_path = self._resolve_path(path)
        cache_key = str(full_path)

        if cache_key in self._doc_cache:
            # Already cached from a previous read — no OCR metadata to report
            return self._doc_cache[cache_key], None

        logger.debug(f"Reading document (async): {full_path.name}")
        doc_content, ocr_meta = await self.reader.read_async(full_path, ocr_timeout=ocr_timeout)
        self._doc_cache[cache_key] = doc_content
        return doc_content, ocr_meta

    def read_pages(self, path: str | Path, start: int, end: int) -> str:
        """Read specific page range from a document."""
        doc = self.read(path)
        return doc.get_page_range(start, end)

    def read_excerpt(self, path: str | Path, max_chars: int = 5000) -> str:
        """Read excerpt of document."""
        doc = self.read(path)
        return doc.get_excerpt(max_chars)

    def batch_read(
        self,
        paths: list[str | Path],
        max_chars_per_doc: Optional[int] = None,
    ) -> dict[str, str]:
        """Read multiple documents."""
        results = {}
        for path in paths:
            try:
                doc = self.read(path)
                if max_chars_per_doc:
                    results[str(path)] = doc.get_excerpt(max_chars_per_doc)
                else:
                    results[str(path)] = doc.full_text
            except Exception as e:
                results[str(path)] = f"[ERROR: {e}]"
        return results

    def batch_read_parallel(
        self,
        paths: list[str | Path],
        max_chars_per_doc: Optional[int] = None,
        max_workers: int = 5,
    ) -> dict[str, str]:
        """Read multiple documents in parallel using threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        def read_one(path: str | Path) -> tuple[str, str]:
            try:
                doc = self.read(path)
                if max_chars_per_doc:
                    return str(path), doc.get_excerpt(max_chars_per_doc)
                else:
                    return str(path), doc.full_text
            except Exception as e:
                return str(path), f"[ERROR: {e}]"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(read_one, p): p for p in paths}
            for future in as_completed(futures):
                path_str, content = future.result()
                results[path_str] = content

        return results

    # === SEARCHING ===

    def _map_search_results_to_display_names(self, results: SearchResults) -> SearchResults:
        """Map filenames in search results to display names."""
        if not self._filename_mapping["actual_to_display"]:
            return results  # No mapping, return as-is

        # Update filenames in hits to use display names
        for hit in results.hits:
            actual_name = Path(hit.file_path).name
            display_name = self._get_display_name(actual_name)
            if display_name != actual_name:
                hit.filename = display_name

        return results

    def search(
        self,
        query: str,
        folder: Optional[str] = None,
        file_types: Optional[list[str]] = None,
        regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_workers: Optional[int] = None,
    ) -> SearchResults:
        """
        Search for query across all documents.

        Args:
            query: Search term or regex
            folder: Limit search to subfolder
            file_types: Limit to specific file types
            regex: Treat query as regex
            case_sensitive: Case sensitive matching
            context_lines: Lines of context around matches
            max_workers: Max parallel workers for search (default scales with file count)

        Returns:
            SearchResults with all matches (filenames mapped to display names)
        """
        # Get files to search
        if folder:
            pattern = f"{folder}/**/*"
        else:
            pattern = "**/*"

        files = [f.path for f in self.list_files(pattern, file_types)]

        # Scale workers based on file count if not specified
        if max_workers is None:
            max_workers = min(10, max(1, len(files)))

        results = self.search_engine.search(
            query=query,
            files=files,
            regex=regex,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_workers=max_workers,
        )

        # Map filenames to display names
        return self._map_search_results_to_display_names(results)

    def smart_search(
        self,
        query: str,
        folder: Optional[str] = None,
        file_types: Optional[list[str]] = None,
        regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 2,
    ) -> SearchResults:
        """
        Smart search with OR fallback for multi-word queries.

        If exact phrase returns no results, automatically splits into
        individual terms and searches for each, deduplicating results.
        """
        if folder:
            pattern = f"{folder}/**/*"
        else:
            pattern = "**/*"

        files = [f.path for f in self.list_files(pattern, file_types)]

        results = self.search_engine.smart_search(
            query=query,
            files=files,
            regex=regex,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
        )

        # Map filenames to display names
        return self._map_search_results_to_display_names(results)

    def search_multi(
        self,
        queries: list[str],
        folder: Optional[str] = None,
        require_all: bool = False,
    ) -> SearchResults:
        """Search for multiple terms."""
        if folder:
            pattern = f"{folder}/**/*"
        else:
            pattern = "**/*"

        files = [f.path for f in self.list_files(pattern)]
        return self.search_engine.search_multi(queries, files, require_all)

    # === UTILITIES ===

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to repository or absolute.

        Handles filename mapping: if a display name is provided and doesn't exist,
        tries to resolve it to the actual filename using the mapping.
        """
        path = Path(path)

        # If absolute path, use it directly
        if path.is_absolute():
            if path.exists():
                return path
            # Try resolving via filename mapping
            actual_name = self._get_actual_filename(path.name)
            if actual_name != path.name:
                resolved = path.parent / actual_name
                if resolved.exists():
                    return resolved
            return path  # Return original even if not found (error will be raised later)

        # Relative path - try direct resolution first
        resolved = self.base_path / path
        if resolved.exists():
            return resolved

        # Try using filename mapping (display_name -> actual_filename)
        path_str = str(path)
        actual_name = self._get_actual_filename(path_str)
        if actual_name != path_str:
            resolved = self.base_path / actual_name
            if resolved.exists():
                logger.debug(f"Resolved display name '{path_str}' to actual '{actual_name}'")
                return resolved

        # Also try just the filename portion (in case path has directory prefix)
        if '/' in path_str or os.sep in path_str:
            filename_only = Path(path_str).name
            actual_name = self._get_actual_filename(filename_only)
            if actual_name != filename_only:
                # Reconstruct path with actual filename
                parent = Path(path_str).parent
                resolved = self.base_path / parent / actual_name
                if resolved.exists():
                    logger.debug(f"Resolved display name '{filename_only}' to actual '{actual_name}'")
                    return resolved

        # Fallback: use display_path_map built from list_files()
        # This handles cases where _filename_mapping.json is missing/incomplete
        display_map = self._get_display_path_map()

        # Try exact display name match
        if path_str in display_map:
            logger.debug(f"Resolved via display_path_map: '{path_str}'")
            return display_map[path_str]

        # Try just the filename portion
        filename_only = Path(path_str).name
        if filename_only in display_map and filename_only != path_str:
            logger.debug(f"Resolved via display_path_map (filename only): '{filename_only}'")
            return display_map[filename_only]

        # Fall back to original path (may not exist - let caller handle error)
        return self.base_path / path

    def get_file_info(self, path: str | Path) -> FileInfo:
        """Get info about a specific file."""
        full_path = self._resolve_path(path)
        return FileInfo(
            path=full_path,
            filename=full_path.name,
            file_type=full_path.suffix.lower(),
            size_bytes=full_path.stat().st_size,
            relative_path=str(full_path.relative_to(self.base_path)),
        )

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"MatterRepository({self.base_path}, {stats.total_files} files, {stats.size_mb:.1f}MB)"
