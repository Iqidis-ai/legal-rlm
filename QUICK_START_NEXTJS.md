# Quick Start: Next.js → Knowledge Graph Backend

## 🎯 TL;DR - Get Started in 3 Steps

### Step 1: Start Backend
```bash
python3 start_server.py
```
Server runs on **http://localhost:8000**

### Step 2: Create API Client (Next.js)

```typescript
// lib/kgApi.ts
const API_URL = 'http://localhost:8000';

export async function extractKnowledgeGraph(matterId: string, documentIds: string[]) {
  const response = await fetch(`${API_URL}/api/extract-from-iqidis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ matter_id: matterId, document_ids: documentIds }),
  });
  return response.json();
}

export async function queryGraph(matterId: string, question: string) {
  const response = await fetch(`${API_URL}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ matter_id: matterId, query: question }),
  });
  return response.json();
}

export async function getGraphStats(matterId: string) {
  const response = await fetch(`${API_URL}/api/stats?matter_id=${matterId}`);
  return response.json();
}
```

### Step 3: Use in Your Components

```typescript
// app/matters/[id]/page.tsx
'use client';

import { extractKnowledgeGraph, queryGraph } from '@/lib/kgApi';

export default function MatterPage({ params }: { params: { id: string } }) {
  const handleExtract = async () => {
    const result = await extractKnowledgeGraph(params.id, ['doc1', 'doc2']);
    console.log(`Extracted ${result.total_entities} entities!`);
  };

  const handleQuery = async () => {
    const result = await queryGraph(params.id, 'Who are the main parties?');
    console.log(result.answer);
  };

  return (
    <div>
      <button onClick={handleExtract}>Extract Knowledge Graph</button>
      <button onClick={handleQuery}>Ask Question</button>
    </div>
  );
}
```

---

## 📡 Essential API Endpoints

### 1. Extract Knowledge Graph
```typescript
POST /api/extract-from-iqidis
Body: { matter_id: "123", document_ids: ["doc1", "doc2"] }
Response: { total_entities: 45, total_edges: 67, documents_processed: 2 }
```

### 2. Query with Natural Language
```typescript
POST /api/query
Body: { matter_id: "123", query: "Who are the parties?" }
Response: { answer: "The parties are...", entities: [...], subgraph: {...} }
```

### 3. Get Statistics
```typescript
GET /api/stats?matter_id=123
Response: { total_entities: 45, total_edges: 67, entity_types: {...} }
```

### 4. Search Entities
```typescript
GET /api/search?q=ACME&matter_id=123
Response: { entities: [{ id: "1", name: "ACME Corp", type: "Organization" }] }
```

### 5. Get Timeline
```typescript
GET /api/timeline?matter_id=123
Response: { events: [{ date: "2024-01-15", entities: [...] }] }
```

---

## 🔥 Complete Example: Extract & Query

```typescript
'use client';

import { useState } from 'react';

export default function KnowledgeGraphDemo({ matterId }: { matterId: string }) {
  const [status, setStatus] = useState('');
  const [answer, setAnswer] = useState('');

  const extractAndQuery = async () => {
    // Step 1: Extract
    setStatus('Extracting knowledge graph...');
    const extractResult = await fetch('http://localhost:8000/api/extract-from-iqidis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        matter_id: matterId,
        document_ids: ['doc1', 'doc2', 'doc3']
      }),
    }).then(r => r.json());

    setStatus(`✅ Extracted ${extractResult.total_entities} entities!`);

    // Step 2: Query
    setStatus('Querying graph...');
    const queryResult = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        matter_id: matterId,
        query: 'Who are the main parties in this case?'
      }),
    }).then(r => r.json());

    setAnswer(queryResult.answer);
    setStatus('✅ Complete!');
  };

  return (
    <div className="p-6">
      <button 
        onClick={extractAndQuery}
        className="px-4 py-2 bg-blue-500 text-white rounded"
      >
        Extract & Query
      </button>
      <div className="mt-4">
        <p><strong>Status:</strong> {status}</p>
        {answer && <p><strong>Answer:</strong> {answer}</p>}
      </div>
    </div>
  );
}
```

---

## ⚙️ Configuration

Add to `.env.local`:
```bash
NEXT_PUBLIC_KG_API_URL=http://localhost:8000
```

---

## 🧪 Test the Backend

```bash
# Test if server is running
curl http://localhost:8000/api/stats?matter_id=123

# Test extraction
curl -X POST http://localhost:8000/api/extract-from-iqidis \
  -H "Content-Type: application/json" \
  -d '{"matter_id":"123","document_ids":["doc1"]}'

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"matter_id":"123","query":"Who are the parties?"}'
```

---

## 📚 Full Documentation

See **NEXTJS_INTEGRATION.md** for:
- Complete API client class
- All available endpoints
- Advanced examples
- Error handling
- Production deployment

---

That's it! 🚀 Start the backend and start calling the APIs from your Next.js app!

