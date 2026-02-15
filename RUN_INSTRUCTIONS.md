# How to Run Knowledge Graph

## Overview

- **Documents**: Fetched from Iqidis (PostgreSQL matter_documents → document → artifact, then S3)
- **Knowledge Graph storage**: Local SQLite at `matters/<matter_id>/graph.db` (same as CLI)

---

## Step 1: Configure

```bash
cd "d:\Office Work\KnowledgeGraphsIqidis"
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your-api-key
development_POSTGRES_URL=postgres://user:pass@host:5432/db?sslmode=require
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

---

## Step 2: Start Server (Iqidis mode)

```bash
python visualization_server.py --port 5000
```

Do **not** pass `--matter` – matter_id is sent per request from Iqidis.

---

## Step 3: Use from Iqidis

1. Add to Iqidis `.env`: `NEXT_PUBLIC_KG_API_URL=http://localhost:5000`
2. Run Iqidis, open a matter → Knowledge Graph → **Extract from Documents**
3. Documents flow: matter_documents → document → artifact → S3 → parse → store locally (graph.db)

---

## Local-only extraction (CLI)

```bash
python -m src.cli.extract --matter my_matter --dir ./path/to/documents/
```

---

## Supported document types

PDF, DOC, DOCX, TXT.
