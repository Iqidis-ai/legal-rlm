# Irys RLM

Durable reasoning substrate for complex domain investigation.

Irys RLM builds and maintains persistent matter models from document
repositories across complex domains. Queries read from and write to that
durable substrate instead of rediscovering stable structure from scratch.

Currently legal-flavored; the neutral kernel (claims, objective nodes,
entities, artifacts, support edges, criteria, gaps) is domain-general and
targets finance, coding, academic research, and biomedical sciences.

Current coding-agent context: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)
API contract and target route roadmap: [API_CONTRACTS.md](API_CONTRACTS.md)
Portable ontology reference: [docs/IMPLEMENTED_REASONING_SYSTEM_SCHEMA.md](docs/IMPLEMENTED_REASONING_SYSTEM_SCHEMA.md)

## Architecture

The system is organized around seven Sacred Outcomes summarized in
`docs/PROJECT_CONTEXT.md`: durable matter model, typed assertion graph, user
steering, issue-driven retrieval, source-aware intelligence, quantitative
intelligence, and explicit missingness modeling.

Key components:

- `src/irys/api.py`: high-level `Irys` API.
- `src/irys/core`: Gemini client, document readers, repositories, and search.
- `src/irys/rlm`: investigation engine, state, checkpoints, and cascade governance.
- `src/irys/matter`: SQLite schema, matter model, stores, proof, trust, gaps,
  quantitative data, memory broker, and reasoning ledger.
- `src/irys/service`: FastAPI service layer and S3 repository support.
- `src/irys/ui`: Gradio dashboard and UI backends.

## Installation

Prerequisites:

- Python 3.11+
- Google Gemini API key

Setup:

```bash
pip install -e .
export GEMINI_API_KEY=your_key_here
```

## Usage

Web UI:

```bash
python -m irys.ui.app
```

FastAPI service:

```bash
irys-server
# or:
uvicorn irys.service.api:app --host 0.0.0.0 --port 8000
```

API documentation is auto-generated at `/docs`.

Python API:

```python
from irys import Irys

irys = Irys(api_key="your-gemini-api-key")
result = await irys.investigate(
    query="What are the key obligations under the contract?",
    repository="./documents",
)
print(result.output)
```

## Model Tiering

Configured model IDs live in `src/irys/core/models.py`.

| Tier | Use case |
| --- | --- |
| `NANO` | Short-output triage and classification |
| `LITE` | Bulk document reading and extraction |
| `FLASH` | Search analysis, routing, and planning |
| `PRO` | Final synthesis and heavier reasoning |

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Supported Document Formats

- PDF via PyMuPDF
- DOCX via python-docx
- TXT

Scanned PDFs without embedded text are not supported yet.
