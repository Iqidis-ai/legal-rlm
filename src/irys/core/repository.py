"""Matter Repository - programmatic access to legal document collections.

No vectors. Direct file access, reading, and search.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict
import logging

from .reader import DocumentReader, DocumentContent
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

    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)


class MatterRepository:
    """
    Programmatic access to a legal matter document repository.

    Provides:
    - File listing and navigation
    - Document reading (PDF, DOCX, TXT)
    - Grep-style search across all documents
    - Parallel operations
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

    def __init__(self, base_path: str | Path):
        # Always resolve to absolute path so that relative_to() and read_bytes() work
        # correctly regardless of the process CWD (e.g., FastAPI background tasks).
        self.base_path = Path(base_path).resolve()
        if not self.base_path.exists():
            raise ValueError(f"Repository path does not exist: {base_path}")
        if not self.base_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {base_path}")

        self.reader = DocumentReader()
        # LRU document cache capped at 200 entries (~200-400MB for typical legal docs).
        # Prevents unbounded RAM growth when searching broad corpora. OrderedDict maintains
        # insertion order so the oldest entry is always first (LRU eviction).
        self._doc_cache: OrderedDict[str, DocumentContent] = OrderedDict()
        self._doc_cache_maxsize = 200
        self.search_engine = DocumentSearch(self.reader)
        self.search_engine._doc_cache = self._doc_cache  # Share cache
        logger.info(f"Initialized repository: {base_path}")

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
            List of FileInfo objects
        """
        files = []
        file_types = file_types or list(self.SUPPORTED_EXTENSIONS)
        file_types = [ft.lower() if ft.startswith(".") else f".{ft.lower()}" for ft in file_types]

        # Directories to skip: .irys (metadata store), hidden dirs, common non-document dirs
        _SKIP_DIRS = {".irys", ".git", "__pycache__", ".venv", "node_modules"}

        for path in self.base_path.glob(pattern):
            if path.is_file() and path.suffix.lower() in file_types:
                # Skip temp files
                if path.name.startswith("~$"):
                    continue

                # Skip files inside excluded directories
                rel = path.relative_to(self.base_path)
                if any(part in _SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                    continue

                files.append(FileInfo(
                    path=path,
                    filename=path.name,
                    file_type=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                    relative_path=str(rel),
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

    def get_stats(self) -> RepositoryStats:
        """Get repository statistics."""
        files = self.list_files()

        files_by_type: dict[str, int] = {}
        total_size = 0
        folders = set()

        for f in files:
            files_by_type[f.file_type] = files_by_type.get(f.file_type, 0) + 1
            total_size += f.size_bytes
            folder = str(Path(f.relative_path).parent)
            folders.add(folder)

        return RepositoryStats(
            total_files=len(files),
            total_size_bytes=total_size,
            files_by_type=files_by_type,
            folders=sorted(folders),
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
            if len(self._doc_cache) >= self._doc_cache_maxsize:
                # Evict the oldest entry (LRU: first item in insertion order)
                self._doc_cache.popitem(last=False)
            self._doc_cache[cache_key] = self.reader.read(full_path)
        else:
            # Move to end to mark as recently used
            self._doc_cache.move_to_end(cache_key)

        return self._doc_cache[cache_key]

    # === SEARCHING ===

    def search(
        self,
        query: str,
        folder: Optional[str] = None,
        file_types: Optional[list[str]] = None,
        regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_workers: Optional[int] = None,
        file_paths: Optional[list] = None,
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
            file_paths: When provided, search only these files instead of walking
                the full repo. Paths should be relative to base_path or absolute.

        Returns:
            SearchResults with all matches
        """
        if file_paths is not None:
            # Candidate-first: search only the specified files
            files = [
                self._resolve_path(p) for p in file_paths
                if self._resolve_path(p).exists()
            ]
        else:
            # Full repo walk
            if folder:
                pattern = f"{folder}/**/*"
            else:
                pattern = "**/*"
            files = [f.path for f in self.list_files(pattern, file_types)]

        # Scale workers based on file count if not specified
        if max_workers is None:
            max_workers = min(10, max(1, len(files)))

        return self.search_engine.search(
            query=query,
            files=files,
            regex=regex,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_workers=max_workers,
        )

    def search_multi(
        self,
        queries: list[str],
        folder: Optional[str] = None,
        require_all: bool = False,
        file_paths: Optional[list] = None,
    ) -> SearchResults:
        """Search for multiple terms.

        When file_paths is provided, search only those files instead of
        walking the full repository.
        """
        if file_paths is not None:
            files = [
                self._resolve_path(p) for p in file_paths
                if self._resolve_path(p).exists()
            ]
        else:
            if folder:
                pattern = f"{folder}/**/*"
            else:
                pattern = "**/*"
            files = [f.path for f in self.list_files(pattern)]
        return self.search_engine.search_multi(queries, files, require_all)

    # === UTILITIES ===

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to repository or absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"MatterRepository({self.base_path}, {stats.total_files} files, {stats.size_mb:.1f}MB)"
