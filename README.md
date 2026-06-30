# Irys RLM

**Recursive Language Model System for Legal Document Investigation**

Irys RLM is a sophisticated legal document analysis system that uses recursive language model techniques to investigate queries against document repositories. It combines tiered Gemini model usage with structured state tracking to extract facts, entities, and citations efficiently, with intelligent early termination to balance thoroughness and cost.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Model Tiering Strategy](#model-tiering-strategy)
- [Investigation Flow](#investigation-flow)
- [Output Formats](#output-formats)
- [Development](#development)
- [Limitations](#limitations)

## Overview

Irys RLM answers complex legal questions by recursively investigating document repositories. Unlike simple RAG systems, it:

1. **Plans investigations** - Analyzes query complexity and creates a strategic search plan
2. **Follows leads** - Iteratively pursues investigation avenues, discovering new leads along the way
3. **Tracks evidence** - Maintains citations with verification against source documents
4. **Terminates intelligently** - Stops when sufficient evidence is gathered or diminishing returns detected
5. **Synthesizes findings** - Produces a structured legal memorandum with citations

## Key Features

### Query-Aware Early Termination
Different query types require different investigation depths:

| Query Type | Confidence Target | Min Citations | Typical Depth |
|-----------|------------------|---------------|---------------|
| Factual | 60% | 5 | 2-3 iterations |
| Procedural | 65% | 6 | 3-4 iterations |
| Analytical | 75% | 8 | 4-5 iterations |
| Comparative | 80% | 10 | 4-5 iterations |
| Evaluative | 85% | 12 | 5-6 iterations |

### Diminishing Returns Detection
- Stops when recent iterations add fewer than 3 facts each
- Prevents wasteful continued searching on exhausted leads
- Balances thoroughness with efficiency

### Citation Tracking & Verification
- Extracts quotes with document and page references
- Verifies citations exist in source documents
- Tracks verification status for audit trails

### Entity Extraction
- Identifies people, organizations, dates, amounts, locations
- Tracks mention frequency across documents
- Links entities to their roles and relationships

### Parallel Processing
- Investigates multiple leads simultaneously
- Parallel document reading for performance
- Thread-safe search with concurrent workers

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Gradio Web App / API)                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           API LAYER                             │
│                    src/irys/api.py (Irys)                       │
│         investigate() | search() | summarize() | batch()        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RLM ENGINE                              │
│                   src/irys/rlm/engine.py                        │
│   orient() → investigate_loop() → verify() → synthesize()       │
└─────────────────────────────────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌───────────────────────────┐  ┌──────────────────────────────────┐
│      STATE MANAGEMENT     │  │         MODEL CLIENT             │
│   src/irys/rlm/state.py   │  │    src/irys/core/models.py       │
│  citations | leads |      │  │  LITE | FLASH | PRO tiering      │
│  entities | thinking      │  │  rate limiting | caching         │
└───────────────────────────┘  └──────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Repository    │  │     Search      │  │     Reader      │  │
│  │ (document list) │  │ (grep + ranking)│  │ (PDF/DOCX/TXT)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/irys/
├── __init__.py           # Public API exports
├── api.py                # High-level API (Irys class, IrysConfig)
├── core/
│   ├── models.py         # Gemini client, model tiering, rate limiting
│   ├── repository.py     # Document repository access
│   ├── reader.py         # PDF/DOCX/TXT/MHT text extraction
│   ├── search.py         # Full-text search with smart_search OR fallback
│   ├── cache.py          # LRU caching with TTL, disk cache, response cache
│   ├── clustering.py     # TF-IDF based document clustering
│   ├── external_search.py # CourtListener (case law) + Tavily (web search)
│   └── utils.py          # Retry logic, telemetry, validation, config
├── rlm/
│   ├── engine.py         # Core investigation engine
│   ├── decisions.py      # LLM decision functions organized by tier
│   ├── prompts.py        # Prompt templates organized by tier
│   ├── state.py          # Investigation state, citations, leads, entities
│   └── templates.py      # Investigation templates (contract, litigation, etc.)
├── service/
│   ├── api.py            # FastAPI REST API for cloud deployment
│   ├── config.py         # ServiceConfig (env-based configuration)
│   ├── models.py         # Pydantic request/response schemas
│   └── s3_repository.py  # S3Repository (download, cleanup)
├── ui/
│   └── app.py            # Gradio web interface with real-time streaming
└── output/
    └── formatters.py     # Output formatting (markdown, HTML, JSON, plain text)
```

## Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd RLMs

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai>=1.51.0` | Gemini API access |
| `pymupdf>=1.24.0` | PDF text extraction |
| `python-docx>=1.1.0` | DOCX text extraction |
| `gradio>=5.0.0` | Web interface |
| `aiofiles>=24.1.0` | Async file operations |
| `pydantic>=2.0.0` | Data validation |
| `rich>=13.0.0` | Terminal formatting |

## Quick Start

### Using the Web UI

```bash
# Set API key
export GEMINI_API_KEY=your_key_here

# Launch UI on port 7860
python run.py

# Or on port 7862
python run_ui.py
```

Navigate to `http://localhost:7860` in your browser.

### Using the Python API

```python
from irys import Irys

# Initialize
irys = Irys(api_key="your-gemini-api-key")

# Run investigation
result = await irys.investigate(
    query="What are the key obligations under the contract?",
    repository="./documents"
)

# Access results
print(result.output)           # Formatted legal memorandum
print(result.citations)        # List of citations with sources
print(result.entities)         # Extracted entities
print(result.confidence)       # Confidence score (0-100)
```

### Quick Functions

```python
from irys import quick_search, quick_summarize

# Search documents
results = await quick_search("damages", "./documents")

# Summarize files
summary = await quick_summarize(["contract.pdf", "amendment.pdf"])
```

### Synchronous Usage

```python
from irys import investigate_sync, search_sync

# For non-async contexts
result = investigate_sync("What happened?", "./documents")
```

## API Reference

### Main Classes

#### `Irys`
The primary interface for investigations.

```python
irys = Irys(
    api_key: str,                    # Gemini API key (or set GEMINI_API_KEY env var)
    config: IrysConfig = None        # Optional configuration
)
```

**Methods:**
- `investigate(query, repository, **kwargs)` - Run full investigation
- `search(query, repository)` - Search documents
- `summarize(files)` - Summarize documents
- `list_templates()` - Get available investigation templates
- `get_telemetry()` - Get usage statistics

#### `IrysConfig`
Configuration options for investigations.

```python
IrysConfig(
    max_depth: int = 5,              # Maximum recursion depth
    max_leads_per_level: int = 5,    # Leads to follow per iteration
    enable_cache: bool = True,       # Enable response caching
    cache_ttl: int = 3600,           # Cache TTL in seconds
    output_format: str = "markdown", # Output format
    log_level: str = "INFO"          # Logging level
)
```

#### `InvestigationResult`
Container for investigation results.

**Properties:**
- `query: str` - Original query
- `output: str` - Formatted output
- `status: str` - Investigation status
- `success: bool` - Whether investigation succeeded
- `citations: List[Citation]` - Extracted citations
- `entities: Dict` - Extracted entities
- `confidence: float` - Confidence score (0-100)
- `quality: str` - Quality level (Excellent/Good/Adequate/Poor)

### Core Components (Advanced Usage)

```python
from irys import (
    MatterRepository,    # Document repository access
    DocumentSearch,      # Full-text search engine
    DocumentReader,      # Text extraction
    RLMEngine,          # Investigation engine
    RLMConfig,          # Engine configuration
    InvestigationState   # State tracking
)
```

## Configuration

### RLMConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 3 | Maximum investigation depth |
| `max_leads_per_level` | 5 | Leads processed per iteration |
| `max_documents_per_search` | 10 | Top documents to deep-read |
| `min_lead_priority` | 0.3 | Skip leads below this priority |
| `excerpt_chars` | 8000 | Content size for LLM analysis |
| `parallel_reads` | 5 | Concurrent document reads |
| `adaptive_depth` | True | Adjust depth based on complexity |
| `min_depth` | 2 | Minimum investigation depth |
| `depth_citation_threshold` | 15 | Reduce depth if citations exceed this |
| `max_iterations` | 10 | Hard limit on iterations |

### Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key      # Required

# Optional: External search integration
COURTLISTENER_API_TOKEN=xxx             # For legal case law search
TAVILY_API_KEY=xxx                      # For web search enrichment

# Optional: S3/Cloud deployment
S3_BUCKET=your-bucket-name
S3_REGION=us-east-1
```

## External Search Integration

Irys can enrich investigations with external sources:

### CourtListener (Case Law)
- Searches legal case law database
- Returns relevant court opinions and citations
- Requires free API token from CourtListener

### Tavily (Web Search)
- General web search for current information
- Useful for regulatory updates, recent events
- Requires API key from Tavily

Enable by setting environment variables. The engine will automatically use available sources.

## REST API (Cloud Deployment)

For production deployment with S3 document storage:

```bash
# Start API server
python run_server.py --port 8000

# Or with Docker
docker-compose up -d
```

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/investigate` | Start investigation job |
| `GET` | `/investigate/{job_id}` | Get job status/results |
| `GET` | `/jobs` | List recent jobs |
| `POST` | `/search` | Quick keyword search |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | OpenAPI documentation |

See `DEPLOY.md` for full deployment instructions.

## Model Tiering Strategy

Irys uses a three-tier model strategy to optimize cost and performance:

| Tier | Model | Use Case | Cost |
|------|-------|----------|------|
| **LITE** | gemini-2.5-flash-lite | Bulk document reading, entity extraction | Lowest |
| **FLASH** | gemini-2.5-flash | Search analysis, routing, planning | Medium |
| **PRO** | gemini-2.5-pro | Final synthesis, complex analysis | Highest |

**Model Assignment by Task:**
- **Orientation** (Phase 1): FLASH with retry
- **Lead Analysis** (Phase 2): FLASH
- **Document Reading** (Phase 2): LITE (parallel)
- **Citation Verification** (Phase 2.5): LITE
- **Final Synthesis** (Phase 3): PRO with retry

## Investigation Flow

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 1: ORIENTATION                                      │
│ • Analyze repository structure                            │
│ • Classify query type (factual/analytical/evaluative)     │
│ • Generate initial hypothesis                             │
│ • Create strategic search plan with initial leads         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 2: ITERATIVE INVESTIGATION                          │
│ ┌────────────────────────────────────────────────────┐   │
│ │ For each iteration:                                 │   │
│ │ • Get pending leads (priority > 0.3)                │   │
│ │ • Process leads in parallel (up to 5)               │   │
│ │   ├─ Extract search terms                           │   │
│ │   ├─ Search repository                              │   │
│ │   ├─ Analyze results → extract facts                │   │
│ │   └─ Deep read top documents → citations            │   │
│ │ • Track facts added this iteration                  │   │
│ │ • Reprioritize leads (every 2 iterations)           │   │
│ │ • Check termination criteria                        │   │
│ └────────────────────────────────────────────────────┘   │
│                                                           │
│ Termination Criteria:                                     │
│ ✓ Confidence + citations meet query-type threshold        │
│ ✓ Diminishing returns (last 2 iterations < 3 facts each)  │
│ ✓ Extreme diminishing returns (last 3 iterations ≤1 fact) │
│ ✓ Max iterations reached                                  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 2.5: CITATION VERIFICATION                          │
│ • Verify quoted text exists in source documents           │
│ • Use fuzzy matching for truncated quotes                 │
│ • Update verification status for each citation            │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 3: SYNTHESIS                                        │
│ • Compile all findings                                    │
│ • Format citations and entities                           │
│ • Generate legal memorandum using PRO model               │
│ • Return InvestigationState with complete results         │
└──────────────────────────────────────────────────────────┘
```

## Output Formats

### Markdown (Default)
Structured legal memorandum with:
- Executive summary
- Key findings with citations
- Entity summary
- Timeline (if applicable)
- Recommendations

### JSON
Machine-readable format including:
- All citations with metadata
- Entity registry
- Thinking trace
- Confidence scores
- Investigation statistics

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest tests/

# Quick tests
python test_quick_rlm.py
python test_early_termination.py
```

### Test Files

| File | Purpose |
|------|---------|
| `tests/test_integration.py` | Full integration tests |
| `tests/test_engine.py` | RLM engine unit tests |
| `tests/test_search.py` | Search functionality tests |
| `test_quick_rlm.py` | Quick sanity check |
| `test_early_termination.py` | Early termination validation |
| `run_comparative_test.py` | Cross-model comparison |

### Code Quality

```bash
# Linting
ruff check src/

# Type checking (if configured)
mypy src/irys/
```

## Limitations

### Supported Document Formats
- **PDF** - Full support via PyMuPDF
- **DOCX** - Full support including tables
- **TXT** - Plain text files
- **MHT/MHTML** - Web archive files (emails, saved web pages)

**Not Supported:**
- Legacy `.doc` format (pre-2007)
- `.rtf` files
- Scanned PDFs without OCR

### Smart Search
The search engine includes a `smart_search` mode that:
- First attempts exact phrase matching
- Falls back to OR search on individual terms if no exact matches
- Automatically deduplicates results
- Particularly useful for multi-word legal queries

### Search Limitations
- Uses literal text matching (grep-style), not semantic search
- Legal synonym expansion helps but isn't comprehensive
- Very large repositories may require more iterations

### Cost Considerations
- PRO model calls (synthesis) are most expensive
- Large repositories increase token usage
- Consider setting `max_iterations` to limit costs

### Rate Limiting
- Default: 60 requests per minute with 10 burst capacity
- Adjust `RateLimiter` settings for different API quotas

## License

[Add license information]

## Contributing

[Add contribution guidelines]

---

**Version:** 0.1.0
**Python:** 3.11+
**Primary API:** Google Gemini (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro)
