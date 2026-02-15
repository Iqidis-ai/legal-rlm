# Next.js Integration Guide - Knowledge Graph API

This guide shows how to integrate the Knowledge Graph backend with your Next.js frontend.

## 🚀 Backend Setup

### 1. Start the Backend Server

```bash
# In the KnowledgeGraphsIqidis directory
python3 start_server.py
```

The server will run on **http://localhost:8000** with CORS enabled.

---

## 📡 Frontend Integration (Next.js)

### API Base Configuration

Create an API client in your Next.js app:

```typescript
// lib/knowledgeGraphApi.ts

const API_BASE_URL = process.env.NEXT_PUBLIC_KG_API_URL || 'http://localhost:8000';

export class KnowledgeGraphAPI {
  private baseUrl: string;
  private matterId: string;

  constructor(matterId: string) {
    this.baseUrl = API_BASE_URL;
    this.matterId = matterId;
  }

  // Helper to add matter_id to requests
  private async request(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      'X-Matter-Id': this.matterId,
      ...options.headers,
    };

    const response = await fetch(url, { ...options, headers });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(error.error || `API Error: ${response.status}`);
    }
    
    return response.json();
  }

  // Extract knowledge graph from documents
  async extractFromDocuments(documentIds: string[]) {
    return this.request('/api/extract-from-iqidis', {
      method: 'POST',
      body: JSON.stringify({
        matter_id: this.matterId,
        document_ids: documentIds,
      }),
    });
  }

  // Get graph data for visualization
  async getGraph() {
    return this.request(`/api/graph?matter_id=${this.matterId}`);
  }

  // Get graph statistics
  async getStats() {
    return this.request(`/api/stats?matter_id=${this.matterId}`);
  }

  // Natural language query
  async query(question: string) {
    return this.request('/api/query', {
      method: 'POST',
      body: JSON.stringify({
        matter_id: this.matterId,
        query: question,
      }),
    });
  }

  // Search entities
  async searchEntities(searchTerm: string) {
    return this.request(`/api/search?q=${encodeURIComponent(searchTerm)}&matter_id=${this.matterId}`);
  }

  // Get timeline
  async getTimeline(limit = 200) {
    return this.request(`/api/timeline?limit=${limit}&matter_id=${this.matterId}`);
  }

  // Get graph schema
  async getSchema() {
    return this.request(`/api/schema?matter_id=${this.matterId}`);
  }

  // Get analytics (PageRank, centrality, etc.)
  async getAnalytics(limit = 50) {
    return this.request(`/api/analytics?limit=${limit}&matter_id=${this.matterId}`);
  }
}
```

---

## 🔥 Usage Examples

### Example 1: Extract Knowledge Graph from Documents

```typescript
// app/matters/[matterId]/knowledge-graph/page.tsx
'use client';

import { useState } from 'react';
import { KnowledgeGraphAPI } from '@/lib/knowledgeGraphApi';

export default function KnowledgeGraphPage({ params }: { params: { matterId: string } }) {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const handleExtract = async (documentIds: string[]) => {
    setLoading(true);
    setError('');
    setStatus('Starting extraction...');

    try {
      const api = new KnowledgeGraphAPI(params.matterId);
      
      const result = await api.extractFromDocuments(documentIds);
      
      setStatus(`✅ Extraction complete! 
        - ${result.total_entities} entities extracted
        - ${result.total_edges} relationships found
        - Processed ${result.documents_processed} documents`);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Knowledge Graph Extraction</h1>
      <button 
        onClick={() => handleExtract(['doc1', 'doc2', 'doc3'])}
        disabled={loading}
      >
        {loading ? 'Extracting...' : 'Extract Knowledge Graph'}
      </button>
      {status && <div className="status">{status}</div>}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
```

### Example 2: Query the Knowledge Graph

```typescript
// components/KnowledgeGraphQuery.tsx
'use client';

import { useState } from 'react';
import { KnowledgeGraphAPI } from '@/lib/knowledgeGraphApi';

export function KnowledgeGraphQuery({ matterId }: { matterId: string }) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const api = new KnowledgeGraphAPI(matterId);
      const result = await api.query(query);
      
      setAnswer(result.answer);
    } catch (err) {
      setAnswer(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your case..."
          className="w-full p-2 border rounded"
          onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
        />
        <button 
          onClick={handleQuery}
          disabled={loading}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded"
        >
          {loading ? 'Searching...' : 'Ask'}
        </button>
      </div>
      
      {answer && (
        <div className="p-4 bg-gray-50 rounded">
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
```


