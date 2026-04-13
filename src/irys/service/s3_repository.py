"""S3-backed repository for document storage.

Downloads documents from S3 to temp storage for processing,
with automatic cleanup to keep disk usage low.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager
from urllib.parse import unquote_plus, urlparse, unquote as _url_unquote

import boto3
import httpx
from botocore.exceptions import ClientError

from .config import ServiceConfig, get_config

logger = logging.getLogger(__name__)

# Content-Type to file extension mapping
CONTENT_TYPE_TO_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}

# Magic bytes for file type detection (fallback when content-type is missing/generic)
MAGIC_BYTES = {
    b"%PDF": ".pdf",
    b"PK\x03\x04": ".docx",  # ZIP-based formats (docx, xlsx, pptx)
    b"\xd0\xcf\x11\xe0": ".doc",  # OLE compound document (old MS Office)
    b"{\\rtf": ".rtf",
    b"\x89PNG": ".png",       # PNG signature
    b"\xff\xd8\xff": ".jpg",  # JPEG signature
}

# Maximum number of concurrent file downloads
MAX_CONCURRENT_DOWNLOADS = 20


def _make_unique_filename(name: str, used: set[str]) -> str:
    """Return a unique filename, appending _1, _2 etc. to the stem if already taken."""
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def detect_extension_from_content_type(content_type: Optional[str]) -> Optional[str]:
    """Get file extension from Content-Type header."""
    if not content_type:
        return None
    # Handle content-type with charset, e.g., "text/plain; charset=utf-8"
    content_type = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(content_type)


def detect_extension_from_magic_bytes(data: bytes) -> Optional[str]:
    """Detect file extension from magic bytes (first few bytes of file)."""
    for magic, ext in MAGIC_BYTES.items():
        if data.startswith(magic):
            return ext
    # Check if it looks like plain text
    try:
        data[:1000].decode("utf-8")
        return ".txt"
    except UnicodeDecodeError:
        pass
    return None


class S3Repository:
    """S3-backed document repository with temp storage management."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        config: Optional[ServiceConfig] = None,
    ):
        """Initialize S3 repository.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            config: Service configuration
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.config = config or get_config()

        # Initialize S3 client
        self._s3 = boto3.client(
            "s3",
            region_name=self.config.s3_region,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Temp storage tracking
        self._temp_dirs: dict[str, tuple[Path, float]] = {}

        logger.info(f"Initialized S3Repository: s3://{bucket}/{prefix}")

    async def list_documents(
        self,
        extensions: Optional[list[str]] = None,
        include_hash_files: bool = True,
    ) -> list[str]:
        """List documents in S3 prefix.

        Args:
            extensions: Filter by file extensions (e.g., ['.pdf', '.txt'])
            include_hash_files: If True, also include files that look like content hashes
                               (no extension, 32+ hex chars). These are typically hash-named
                               files that need mapping to display names.

        Returns:
            List of S3 keys (relative to prefix)
        """
        extensions = extensions or [".txt", ".pdf", ".docx", ".md", ".png", ".jpg", ".jpeg"]

        def _is_hash_filename(name: str) -> bool:
            """Check if filename looks like a content hash (hex string, no extension)."""
            # Hash filenames are typically 32+ hex chars without extension
            if '.' in name:
                return False  # Has extension, not a hash
            if len(name) < 32:
                return False
            return all(c in '0123456789abcdef' for c in name.lower())

        def _list():
            documents = []
            paginator = self._s3.get_paginator("list_objects_v2")

            prefix = f"{self.prefix}/" if self.prefix else ""

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel_key = key[len(prefix):] if prefix else key

                    # Always include the filename mapping file
                    if rel_key == "_filename_mapping.json":
                        documents.append(rel_key)
                        continue

                    # Filter by extension for other files
                    if any(key.lower().endswith(ext) for ext in extensions):
                        documents.append(rel_key)
                        continue

                    # Include hash-named files (they may have display names in mapping)
                    if include_hash_files and _is_hash_filename(rel_key):
                        documents.append(rel_key)
                        logger.debug(f"Including hash-named file: {rel_key}")

            return documents

        return await asyncio.to_thread(_list)

    async def download_to_temp(self, job_id: str) -> Path:
        """Download all documents to temp directory.

        If a _filename_mapping.json exists in S3, it will be used to:
        1. Identify hash-named files that should be downloaded
        2. Save those files with their display names (not hash names)

        This ensures the repository can read files by their human-readable names.

        Args:
            job_id: Unique job identifier for tracking

        Returns:
            Path to temp directory containing downloaded files
        """
        # Create temp directory
        temp_dir = Path(self.config.temp_dir) / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Track for cleanup
        self._temp_dirs[job_id] = (temp_dir, time.time())

        # List all documents (including hash-named files)
        documents = await self.list_documents()
        logger.info(f"Downloading {len(documents)} documents for job {job_id}")

        # Check limits
        if len(documents) > self.config.max_documents_per_job:
            raise ValueError(
                f"Too many documents ({len(documents)}). "
                f"Max: {self.config.max_documents_per_job}"
            )

        # Download mapping file FIRST if it exists, to get hash -> display name mapping
        filename_mapping = {}  # hash_name -> display_name
        if "_filename_mapping.json" in documents:
            mapping_path = await self._download_file("_filename_mapping.json", temp_dir)
            if mapping_path and mapping_path.exists():
                try:
                    with open(mapping_path) as f:
                        raw_mapping = json.load(f)
                    # Build hash -> display_name mapping
                    for hash_name, info in raw_mapping.items():
                        if isinstance(info, dict):
                            display_name = info.get("display_name", hash_name)
                        else:
                            display_name = str(info)
                        filename_mapping[hash_name] = display_name
                    logger.info(f"Loaded filename mapping with {len(filename_mapping)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load filename mapping: {e}")

        # Build download list with unique filenames (handle same-name conflicts)
        used_names: set[str] = set()
        download_pairs: list[tuple[str, str]] = []
        for doc_key in documents:
            if doc_key == "_filename_mapping.json":
                continue
            save_name = filename_mapping.get(doc_key, doc_key)
            unique_name = _make_unique_filename(save_name, used_names)
            download_pairs.append((doc_key, unique_name))

        # Download concurrently with limit
        sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def _dl(doc_key: str, save_name: str) -> Optional[Path]:
            async with sem:
                logger.debug(f"[async-dl] start  {save_name}")
                result = await self._download_file(doc_key, temp_dir, save_as=save_name)
                logger.debug(f"[async-dl] done   {save_name}")
                return result

        logger.info(f"[async-dl] launching {len(download_pairs)} downloads (max {MAX_CONCURRENT_DOWNLOADS} concurrent)")
        dl_results = await asyncio.gather(*[_dl(k, n) for k, n in download_pairs], return_exceptions=True)
        downloaded_count = sum(1 for r in dl_results if isinstance(r, Path))
        skipped_count = sum(1 for r in dl_results if r is None or isinstance(r, Exception))

        # Verify files actually exist in temp directory
        actual_files = list(temp_dir.glob("**/*"))
        actual_file_count = sum(1 for f in actual_files if f.is_file())

        logger.info(
            f"Download complete for job {job_id}: "
            f"{downloaded_count}/{len(documents)} downloaded, {skipped_count} skipped, "
            f"{actual_file_count} files verified on disk at {temp_dir}"
        )

        # Donot raise error even if no files were downloaded
        # if actual_file_count == 0:
        #     raise ValueError(
        #         f"Failed to download any documents. Listed: {len(documents)}, "
        #         f"Downloaded: {downloaded_count}, Verified on disk: {actual_file_count}. "
        #         f"Temp dir: {temp_dir}"
        #     )

        return temp_dir

    async def _download_file(
        self, key: str, dest_dir: Path, save_as: Optional[str] = None
    ) -> Optional[Path]:
        """Download a single file from S3.

        Args:
            key: S3 key (relative to prefix) to download
            dest_dir: Destination directory
            save_as: Optional filename to save as (instead of the S3 key name).
                    Use this to save hash-named files with their display names.

        Returns:
            Path to downloaded file, or None if skipped
        """
        s3_key = f"{self.prefix}/{key}" if self.prefix else key
        # Use save_as name if provided, otherwise use the original key
        dest_filename = save_as if save_as else key
        dest_path = dest_dir / dest_filename

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        def _download():
            # Check file size first
            head = self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            size_mb = head["ContentLength"] / (1024 * 1024)

            if size_mb > self.config.max_document_size_mb:
                logger.warning(f"Skipping {key}: {size_mb:.1f}MB exceeds limit")
                return None

            self._s3.download_file(self.bucket, s3_key, str(dest_path))
            if save_as and save_as != key:
                logger.debug(f"Downloaded {key} -> {save_as}")
            return dest_path

        return await asyncio.to_thread(_download)

    async def cleanup(self, job_id: str) -> None:
        """Clean up temp directory for a job."""
        if job_id in self._temp_dirs:
            temp_dir, _ = self._temp_dirs.pop(job_id)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")

    async def cleanup_expired(self) -> int:
        """Clean up expired temp directories. Returns count cleaned."""
        now = time.time()
        expired = []

        for job_id, (temp_dir, created_at) in self._temp_dirs.items():
            if now - created_at > self.config.cleanup_after_seconds:
                expired.append(job_id)

        for job_id in expired:
            await self.cleanup(job_id)

        return len(expired)

    @staticmethod
    def is_s3_url(url: str) -> bool:
        """Check if URL is an S3 URL that should use the S3 SDK.

        Returns True for:
            - s3://bucket/key
            - https://bucket.s3.region.amazonaws.com/key (without presigned params)
            - https://s3.region.amazonaws.com/bucket/key (without presigned params)

        Returns False for:
            - Presigned S3 URLs (have X-Amz-* query params) - these use HTTP
            - Regular HTTP(S) URLs
        """
        # Presigned URLs should be fetched via HTTP, not S3 SDK
        # They contain query params like X-Amz-Signature, X-Amz-Credential, etc.
        if "X-Amz-" in url or "x-amz-" in url:
            return False

        if url.startswith("s3://"):
            return True

        # https://bucket.s3.region.amazonaws.com/key format
        if re.match(r"https://[a-zA-Z0-9.-]+\.s3\.[a-z0-9-]+\.amazonaws\.com/", url):
            return True

        # https://s3.region.amazonaws.com/bucket/key format
        if re.match(r"https://s3\.[a-z0-9-]+\.amazonaws\.com/", url):
            return True

        return False

    @staticmethod
    def parse_s3_url(url: str) -> tuple[str, str]:
        """Parse S3 URL into bucket and key.

        Supports:
            - s3://bucket/key
            - https://bucket.s3.region.amazonaws.com/key
            - https://s3.region.amazonaws.com/bucket/key

        Returns:
            Tuple of (bucket, key) - key is URL-decoded

        Raises:
            ValueError if URL format is not recognized
        """
        # s3:// format
        if url.startswith("s3://"):
            parts = url[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid S3 URL format: {url}")
            return parts[0], unquote_plus(parts[1])

        # https://bucket.s3.region.amazonaws.com/key format
        match = re.match(
            r"https://([a-zA-Z0-9.-]+)\.s3\.[a-z0-9-]+\.amazonaws\.com/(.+)",
            url,
        )
        if match:
            return match.group(1), unquote_plus(match.group(2))

        # https://s3.region.amazonaws.com/bucket/key format
        match = re.match(
            r"https://s3\.[a-z0-9-]+\.amazonaws\.com/([a-zA-Z0-9.-]+)/(.+)",
            url,
        )
        if match:
            return match.group(1), unquote_plus(match.group(2))

        raise ValueError(f"Unrecognized S3 URL format: {url}")

    async def download_urls_to_temp(
        self,
        job_id: str,
        url_inputs: list,
    ) -> Path:
        """Download documents from URLs to temp directory.

        Supports:
            - S3 URLs: s3://bucket/key, https://bucket.s3.region.amazonaws.com/key
            - Generic HTTP(S) URLs: https://example.com/document.pdf
            - Presigned S3 URLs (with X-Amz-* params)

        Args:
            job_id: Unique job identifier for tracking
            url_inputs: List of URLs (strings) or UrlWithMetadata objects

        Returns:
            Path to temp directory containing downloaded files
        """
        # Create temp directory
        temp_dir = Path(self.config.temp_dir) / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Track for cleanup
        self._temp_dirs[job_id] = (temp_dir, time.time())

        # Check limits
        if len(url_inputs) > self.config.max_documents_per_job:
            raise ValueError(
                f"Too many documents ({len(url_inputs)}). "
                f"Max: {self.config.max_documents_per_job}"
            )

        logger.info(f"Downloading {len(url_inputs)} documents for job {job_id}")

        # --- Step 1: parse every input and assign unique filenames up-front ---
        # Filenames are resolved before any I/O so the LLM always sees distinct,
        # human-readable names even when two inputs share the same base name.
        used_names: set[str] = set()

        def _resolve_input(url_input) -> tuple[str, Optional[str], Optional[str]]:
            """Return (url, filename, mime_type) from a str / object / dict input."""
            if isinstance(url_input, str):
                url, filename, mime_type = url_input, None, None
            elif hasattr(url_input, "url"):
                url, filename, mime_type = url_input.url, url_input.name, url_input.mime
            else:
                url = url_input.get("url", url_input)
                filename = url_input.get("name")
                mime_type = url_input.get("mime")

            # Fall back to the filename embedded in the URL path
            if not filename:
                filename = _url_unquote(Path(urlparse(url).path).name) or None

            # Ensure no two files land with the same name
            if filename:
                filename = _make_unique_filename(filename, used_names)

            return url, filename, mime_type

        parsed_inputs = [_resolve_input(u) for u in url_inputs]

        # --- Step 2: download all files concurrently, capped at MAX_CONCURRENT_DOWNLOADS ---
        sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def _download_one(url: str, filename: Optional[str], mime_type: Optional[str]) -> tuple[bool, Optional[str]]:
            async with sem:
                logger.debug(f"[async-dl] start  {filename or url}")
                try:
                    if self.is_s3_url(url):
                        bucket, key = self.parse_s3_url(url)
                        await self._download_url(bucket, key, temp_dir, filename, mime_type)
                    else:
                        await self._download_http_url(url, temp_dir, filename, mime_type)
                    logger.debug(f"[async-dl] done   {filename or url}")
                    return True, None
                except Exception as e:
                    error_msg = f"Failed to download {url}: {e}"
                    logger.warning(error_msg)
                    return False, error_msg

        logger.info(f"[async-dl] launching {len(parsed_inputs)} downloads (max {MAX_CONCURRENT_DOWNLOADS} concurrent)")
        dl_results = await asyncio.gather(*[_download_one(u, f, m) for u, f, m in parsed_inputs])
        downloaded = sum(1 for ok, _ in dl_results if ok)
        errors = [msg for _, msg in dl_results if msg]

        # Verify files actually exist in temp directory
        actual_files = list(temp_dir.glob("**/*"))
        actual_file_count = sum(1 for f in actual_files if f.is_file())

        logger.info(
            f"Download complete for job {job_id}: "
            f"{downloaded}/{len(url_inputs)} downloaded, {actual_file_count} files verified on disk at {temp_dir}"
        )

        # Create filename mapping with URLs for citation tracking
        mapping = {}
        for url, filename, mime_type in parsed_inputs:
            if filename:
                mapping[filename] = {
                    "display_name": filename,
                    "url": url,
                    "mime": mime_type,
                }

        if mapping:
            mapping_path = temp_dir / "_filename_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(mapping, f, indent=2)
            logger.info(f"Created filename mapping with {len(mapping)} entries including URLs")

        # Donot raise error even if no files were downloaded
        # if downloaded == 0:
        #     raise ValueError(
        #         f"Failed to download any documents. Errors: {'; '.join(errors)}"
        #     )

        # Raise error if files reported downloaded but not on disk (indicates bug)
        if downloaded > 0 and actual_file_count == 0:
            raise ValueError(
                f"Downloaded {downloaded} documents but 0 files found on disk at {temp_dir}. "
                f"This may indicate a path or storage issue."
            )

        return temp_dir

    async def _download_url(
        self,
        bucket: str,
        key: str,
        dest_dir: Path,
        provided_filename: Optional[str] = None,
        provided_mime: Optional[str] = None,
    ) -> Optional[Path]:
        """Download a single file from S3 by bucket and key.

        Args:
            bucket: S3 bucket name
            key: S3 object key
            dest_dir: Destination directory
            provided_filename: Optional filename with extension (from metadata)
            provided_mime: Optional MIME type (from metadata)
        """
        # Use provided filename or extract from key
        filename = provided_filename or Path(key).name
        current_ext = Path(filename).suffix.lower()

        # If we have provided mime type, use it for extension
        if provided_mime:
            ext_from_mime = detect_extension_from_content_type(provided_mime)
            if ext_from_mime:
                current_ext = ext_from_mime

        def _download():
            # Check file size first
            head = self._s3.head_object(Bucket=bucket, Key=key)
            size_mb = head["ContentLength"] / (1024 * 1024)

            if size_mb > self.config.max_document_size_mb:
                logger.warning(
                    f"Skipping {key}: {size_mb:.1f}MB exceeds limit"
                )
                return None

            # Determine file extension
            ext = current_ext
            if not ext or ext not in {".pdf", ".docx", ".doc", ".txt", ".rtf", ".png", ".jpg", ".jpeg"}:
                # Try to detect from Content-Type
                content_type = head.get("ContentType")
                ext = detect_extension_from_content_type(content_type)
                logger.debug(f"Content-Type '{content_type}' -> extension '{ext}'")

            # Download to temp file first
            temp_path = dest_dir / f"_temp_{filename}"
            self._s3.download_file(bucket, key, str(temp_path))

            # If still no extension, detect from magic bytes
            if not ext:
                with open(temp_path, "rb") as f:
                    header = f.read(16)
                ext = detect_extension_from_magic_bytes(header)
                logger.debug(f"Magic bytes detection -> extension '{ext}'")

            if not ext:
                logger.warning(f"Could not determine file type for {key}, skipping")
                temp_path.unlink()
                return None

            # Rename with proper extension
            final_filename = filename if filename.endswith(ext) else f"{filename}{ext}"
            dest_path = dest_dir / final_filename
            temp_path.rename(dest_path)

            logger.debug(f"Downloaded s3://{bucket}/{key} -> {dest_path}")
            return dest_path

        return await asyncio.to_thread(_download)

    async def _download_http_url(
        self,
        url: str,
        dest_dir: Path,
        provided_filename: Optional[str] = None,
        provided_mime: Optional[str] = None,
    ) -> Optional[Path]:
        """Download a file from a generic HTTP(S) URL.

        Args:
            url: HTTP(S) URL to download
            dest_dir: Destination directory
            provided_filename: Optional filename with extension (from metadata)
            provided_mime: Optional MIME type (from metadata)

        Returns:
            Path to downloaded file, or None if skipped/failed
        """
        # Use provided filename or extract from URL
        from urllib.parse import urlparse, unquote
        if provided_filename:
            filename = provided_filename
        else:
            parsed = urlparse(url)
            filename = unquote(Path(parsed.path).name)

        if not filename:
            # Generate a filename from URL hash
            import hashlib
            filename = hashlib.md5(url.encode()).hexdigest()[:12]

        current_ext = Path(filename).suffix.lower()

        # If we have provided mime type, use it for extension
        if provided_mime:
            ext_from_mime = detect_extension_from_content_type(provided_mime)
            if ext_from_mime:
                current_ext = ext_from_mime
                logger.debug(f"Using provided MIME '{provided_mime}' -> extension '{ext_from_mime}'")

        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            # First, do a HEAD request to check size and content type
            content_type = None
            try:
                head_response = await client.head(url)
                head_response.raise_for_status()

                content_length = head_response.headers.get("content-length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > self.config.max_document_size_mb:
                        logger.warning(
                            f"Skipping {url}: {size_mb:.1f}MB exceeds limit"
                        )
                        return None
                content_type = head_response.headers.get("content-type")
            except httpx.HTTPError as e:
                logger.warning(f"HEAD request failed for {url}: {e}")
                # Continue with GET anyway, some servers don't support HEAD

            # Download the file
            try:
                response = await client.get(url)
                response.raise_for_status()

                # Check size from actual content
                size_mb = len(response.content) / (1024 * 1024)
                if size_mb > self.config.max_document_size_mb:
                    logger.warning(
                        f"Skipping {url}: {size_mb:.1f}MB exceeds limit"
                    )
                    return None

                # Get content type from response if not from HEAD
                if not content_type:
                    content_type = response.headers.get("content-type")

                # Determine file extension (prefer provided metadata)
                ext = current_ext
                if not ext or ext not in {".pdf", ".docx", ".doc", ".txt", ".rtf", ".png", ".jpg", ".jpeg"}:
                    ext = detect_extension_from_content_type(content_type)
                    logger.debug(f"Content-Type '{content_type}' -> extension '{ext}'")

                # If still no extension, detect from magic bytes
                if not ext:
                    ext = detect_extension_from_magic_bytes(response.content[:16])
                    logger.debug(f"Magic bytes detection -> extension '{ext}'")

                if not ext:
                    logger.warning(f"Could not determine file type for {url}, skipping")
                    return None

                # Save with proper extension
                final_filename = filename if filename.endswith(ext) else f"{filename}{ext}"
                dest_path = dest_dir / final_filename
                dest_path.write_bytes(response.content)
                logger.debug(f"Downloaded {url} -> {dest_path}")
                return dest_path

            except httpx.HTTPError as e:
                logger.error(f"Failed to download {url}: {e}")
                raise

    async def upload_files(
        self,
        job_id: str,
        files: list[tuple[str, bytes]],
        prefix: str = "uploads",
    ) -> str:
        """Upload files to S3.

        Args:
            job_id: Unique job identifier
            files: List of (filename, content) tuples
            prefix: S3 prefix for uploads (default: "uploads")

        Returns:
            S3 prefix where files were uploaded
        """
        upload_prefix = f"{prefix}/{job_id}"

        def _upload():
            uploaded = 0
            for filename, content in files:
                key = f"{upload_prefix}/{filename}"
                self._s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content,
                )
                uploaded += 1
                logger.debug(f"Uploaded {filename} to s3://{self.bucket}/{key}")
            return uploaded

        count = await asyncio.to_thread(_upload)
        logger.info(f"Uploaded {count} files to s3://{self.bucket}/{upload_prefix}/")
        return upload_prefix

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix.

        Args:
            prefix: S3 prefix to delete

        Returns:
            Number of objects deleted
        """
        def _delete():
            deleted = 0
            paginator = self._s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                if objects:
                    delete_keys = [{"Key": obj["Key"]} for obj in objects]
                    self._s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": delete_keys},
                    )
                    deleted += len(delete_keys)

            return deleted

        count = await asyncio.to_thread(_delete)
        logger.info(f"Deleted {count} objects from s3://{self.bucket}/{prefix}")
        return count

