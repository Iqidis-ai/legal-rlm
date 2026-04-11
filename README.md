# Irys RLM

**Recursive Language Model System for Legal Document Investigation**

Irys RLM is a legal intelligence system that builds and maintains a persistent matter model from legal document repositories. Every query reads from and writes to a durable matter model substrate — the system never rediscovers stable structure from scratch.

Current implementation snapshot for coding agents: `SYSTEM_STATE.md`.

## Architecture

The system is organized around seven Sacred Outcomes (SO-1 through SO-7) defined in `.claude/CLAUDE.md`. Key components:

- **Matter Model** (SO-1): Persistent SQLite-backed intelligence substrate per matter
- **Typed Assertion Graph** (SO-2): Facts stored with speech-act classification, source role, support/attack links, and belief revision
- **User Steering** (SO-3): Stop, redirect, resume investigations; correct assertions; annotate documents
- **Issue-Driven Architecture** (SO-4): Structured issue tree drives retrieval and synthesis
- **Source-Aware Intelligence** (SO-5): Distinguishes advocacy, operative, authoritative, and informal sources
- **Quantitative Intelligence** (SO-6): Structured numeric extraction, reconciliation, damages modeling
- **Missingness Modeling** (SO-7): Gaps, missing documents, and proof gaps are surfaced proactively

### Directory Structure

```
src/irys/
├── __init__.py          # Public API exports
├── api.py               # High-level API (Irys class)
├── core/
│   ├── models.py        # Gemini client, model tiering, rate limiting
│   ├── repository.py    # Document repository access
│   ├── reader.py        # PDF/DOCX/TXT text extraction
│   ├── search.py        # Full-text search with ranking
│   └── utils.py         # Logging, validation
├── rlm/
│   ├── engine.py        # Core investigation engine
│   └── state.py         # Investigation state tracking
├── matter/
│   ├── matter.py        # MatterModel: top-level orchestrator
│   ├── graph.py         # All canonical stores (assertions, issues, actors, gaps, quant, etc.)
│   ├── schema.py        # SQLite schema + migrations
│   ├── belief_revision.py  # Truth maintenance / BFS propagation
│   └── reasoning.py     # Reasoning ledger
├── service/
│   ├── api.py           # FastAPI service layer
│   ├── config.py        # Service configuration
│   └── s3_repository.py # S3-backed document repository
├── ui/
│   └── app.py           # Gradio web interface
└── output/
    └── formatters.py    # Output formatting (markdown, JSON, HTML)
```

## Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key

### Setup

```bash
pip install -e .
export GEMINI_API_KEY=your_key_here
```

## Usage

### Web UI

```bash
python -m irys.ui.app
```

### Service (FastAPI)

```bash
uvicorn irys.service.api:app --host 0.0.0.0 --port 8000
```

API documentation is auto-generated at `/docs`.

### Python API

```python
from irys import Irys

irys = Irys(api_key="your-gemini-api-key")
result = await irys.investigate(
    query="What are the key obligations under the contract?",
    repository="./documents"
)
print(result.output)
```

## Model Tiering

| Tier | Model | Use Case |
|------|-------|----------|
| **NANO** | gemini-2.5-flash-lite | Triage, classification (short output) |
| **LITE** | gemini-2.5-flash-lite | Bulk document reading, entity extraction |
| **FLASH** | gemini-3.1-flash-lite-preview | Search analysis, routing, planning |
| **PRO** | gemini-2.5-pro | Final synthesis, complex analysis |

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Supported Document Formats

- **PDF** — Full support via PyMuPDF
- **DOCX** — Full support including tables
- **TXT** — Plain text files

Scanned PDFs without embedded text are not supported (no OCR).

---

**Version:** 0.1.0 | **Python:** 3.11+ | **Primary API:** Google Gemini
