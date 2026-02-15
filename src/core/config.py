"""
Configuration settings for the Knowledge Graph system.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please create a .env file with GEMINI_API_KEY=your-api-key or set it in your environment."
    )
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Chunking Configuration
CHUNK_SIZE = 20000  # tokens (large context for Gemini)
CHUNK_OVERLAP = 1000  # tokens (5% overlap for large chunks)

# Iqidis: max document size for extraction (50 MB)
MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024

# --- Environment-based PostgreSQL URL ---
# APP_ENV controls which *_POSTGRES_URL is used.
# Supported values: "development" (default), "staging", "production"
APP_ENV = os.getenv("APP_ENV", "development").lower()

_POSTGRES_URLS = {
    "development": os.getenv("development_POSTGRES_URL"),
    "staging": os.getenv("staging_POSTGRES_URL"),
    "production": os.getenv("production_POSTGRES_URL"),
    "preview": os.getenv("preview_POSTGRES_URL"),
}

# Fallback chain: env-specific → generic POSTGRES_URL → DATABASE_URL
POSTGRES_URL = (
    _POSTGRES_URLS.get(APP_ENV)
    or os.getenv("POSTGRES_URL")
    or os.getenv("DATABASE_URL")
)


def get_postgres_url(env: Optional[str] = None) -> str:
    """Return the PostgreSQL URL for the given environment.

    Priority:
        1. Explicit *env* argument  → look up {env}_POSTGRES_URL
        2. APP_ENV env var          → look up {APP_ENV}_POSTGRES_URL
        3. Generic POSTGRES_URL / DATABASE_URL fallback

    Raises ValueError if no URL can be resolved.
    """
    if env:
        url = os.getenv(f"{env}_POSTGRES_URL") or _POSTGRES_URLS.get(env)
        if url:
            return url
    # Fall back to module-level default
    if POSTGRES_URL:
        return POSTGRES_URL
    raise ValueError(
        "No POSTGRES_URL configured. Set APP_ENV + {env}_POSTGRES_URL, "
        "or POSTGRES_URL / DATABASE_URL in .env"
    )


# Embedding Configuration
# Using 768 for gemini-embedding-001 with output_dimensionality
EMBEDDING_DIMENSION = 768
# Pre-computed chunk embeddings from Voyage AI (voyage-3)
VOYAGE_EMBEDDING_DIMENSION = 1024

# Paths (kept for backwards compatibility, but not used for KG storage anymore)
# Navigate from src/core/config.py up to project root
BASE_DIR = Path(__file__).parent.parent.parent
# DEPRECATED: KG now uses PostgreSQL, not local files
MATTERS_DIR = BASE_DIR / "matters"

# Entity Types
ENTITY_TYPES = [
    "Person",
    "Organization",
    "Document",
    "Clause",
    "Date",
    "Money",
    "Location",
    "Reference",
    "Fact"
]

# Relation Types (directional)
RELATION_TYPES = [
    "mentioned_in",
    "party_to",
    "represents",
    "signed",
    "defined_as",
    "references",
    "related_to",
    "attributed_to",
    "about",
    "binds",
    "testified",
    "employed_by",
    "affiliated_with"
]

# Confidence Levels
CONFIDENCE_CONFIRMED = "confirmed"  # User-asserted or structural anchors
CONFIDENCE_EXTRACTED = "extracted"  # Semantic pass, no user review
CONFIDENCE_INFERRED = "inferred"    # System-suggested, awaiting confirmation

# Resolution threshold
RESOLUTION_CONFIDENCE_THRESHOLD = 0.7
