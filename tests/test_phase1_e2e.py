"""End-to-end Phase 1 validation test (Task 1.10).

NOT A UNIT TEST — calls real Vertex AI embedding API.
Requires VERTEXAI_CREDENTIALS_B64 environment variable.
"""

import os
import uuid
import pytest
import sys
import tempfile
from pathlib import Path
import asyncio
import json
from unittest.mock import MagicMock

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.models import EmbeddingConfig
from irys.core.embeddings import EmbeddingClient
from irys.core.vector_store import LocalVectorStore
from irys.core.media_pipeline import ChunkRecord, MetadataStore, process_audio, process_image, process_video
from irys.core.retrieval import EvidenceRetriever, EvidenceCard
from irys.core.repository import MatterRepository
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import citation_from_evidence_card


# Check for Vertex AI credentials
SKIP_REASON = "Requires VERTEXAI_CREDENTIALS_B64 environment variable"
HAS_CREDENTIALS = bool(os.environ.get("VERTEXAI_CREDENTIALS_B64"))


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
class TestPhase1EndToEnd:
    """End-to-end validation for Phase 1 multimodal infrastructure.

    This test exercises the full stack from embedding to retrieval to engine integration,
    verifying that the isolation guarantee holds: multimodal_enabled flag produces
    zero changes when media index is populated but synthesis is unchanged.
    """

    @pytest.fixture
    def temp_matter(self, tmp_path):
        """Create a temporary matter with 3 short text documents."""
        matter_dir = tmp_path / "test_matter"
        matter_dir.mkdir()

        # Create 3 short text documents
        docs = [
            ("contract.txt", "This is a supply agreement between ACME Corp and Widget Inc, executed on January 15, 2024. The contract specifies delivery of 1000 units at $50 per unit."),
            ("email.txt", "From: john@acme.com\nTo: jane@widget.com\nDate: January 10, 2024\nSubject: Contract Discussion\n\nJane, I'm confirming our agreement for 1000 units. Let's finalize the contract next week."),
            ("invoice.txt", "INVOICE #INV-2024-001\nDate: February 1, 2024\nACME Corp → Widget Inc\n1000 units @ $50/unit = $50,000\nPayment terms: Net 30"),
        ]

        for filename, content in docs:
            (matter_dir / filename).write_text(content, encoding="utf-8")

        return matter_dir

    @pytest.fixture
    def embedding_config(self):
        """Create embedding config for testing."""
        return EmbeddingConfig(
            model_id="gemini-embedding-2-preview",
            fast_dimensionality=256,
            full_dimensionality=3072,
            index_dimensionality=768,
            top_k_candidates=50,
            top_k_results=10,
            similarity_threshold=0.3,  # Lower threshold for small test set
            chunk_strategy_version=1,
        )

    @pytest.fixture
    def vertex_project(self):
        """Extract Vertex AI project ID from credentials."""
        import base64
        creds_b64 = os.environ.get("VERTEXAI_CREDENTIALS_B64")
        if not creds_b64:
            pytest.skip("Missing VERTEXAI_CREDENTIALS_B64")

        creds_json = base64.b64decode(creds_b64).decode('utf-8')
        credentials_dict = json.loads(creds_json)
        return credentials_dict.get("project_id")

    @pytest.fixture
    def indexed_matter(self, temp_matter, tmp_path, embedding_config, vertex_project):
        """Index the matter through EmbeddingClient → VectorStore → MetadataStore.

        Returns:
            Tuple of (matter_dir, evidence_retriever)
        """
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        # Initialize stores
        vector_store = LocalVectorStore(
            index_dir=index_dir,
            fast_dim=embedding_config.fast_dimensionality,
            full_dim=embedding_config.full_dimensionality,
        )
        metadata_store = MetadataStore(index_dir / "metadata.db")

        # Initialize embedding client
        embedding_client = EmbeddingClient(
            config=embedding_config,
            project=vertex_project,
            region=embedding_config.region,
        )

        # Index each text document
        for text_file in temp_matter.glob("*.txt"):
            content = text_file.read_text(encoding="utf-8")

            # Create chunk (entire document as single chunk for simplicity)
            chunk = ChunkRecord.from_text(
                asset_path=text_file,
                chunk_index=0,
                text_content=content,
                start_char=0,
                end_char=len(content),
            )

            # Store metadata
            metadata_store.upsert(chunk)

            # Generate embeddings
            fast_vec = embedding_client.embed_text(
                content,
                output_dimensionality=embedding_config.fast_dimensionality,
            )
            full_vec = embedding_client.embed_text(
                content,
                output_dimensionality=embedding_config.full_dimensionality,
            )
            index_vec = embedding_client.embed_text(
                content,
                output_dimensionality=embedding_config.index_dimensionality,
            )

            # Add to vector store
            vector_store.add(chunk.chunk_id, fast_vec, index_vec)

        # Persist vector store
        vector_store.persist()

        # Create evidence retriever
        evidence_retriever = EvidenceRetriever(
            vector_store=vector_store,
            metadata_store=metadata_store,
            embedding_client=embedding_client,
            config=embedding_config,
        )

        return temp_matter, evidence_retriever

    @pytest.mark.asyncio
    async def test_multimodal_flag_isolation(self, indexed_matter, vertex_project):
        """Test that multimodal_enabled flag produces identical output when disabled.

        Steps 3-4 of Task 1.10:
        - Run investigation with multimodal_enabled = False
        - Run investigation with multimodal_enabled = True
        - Assert outputs are identical (media index populated but synthesis unchanged)
        """
        matter_dir, evidence_retriever = indexed_matter

        # Query that should match indexed documents
        query = "What is the contract value?"

        # Create engine with multimodal DISABLED
        config_disabled = RLMConfig(
            max_depth=1,
            max_iterations=1,
            multimodal_enabled=False,
        )
        engine_disabled = RLMEngine(config_disabled, api_key=os.environ["GEMINI_API_KEY"])

        # Run investigation with multimodal disabled
        state_disabled = await engine_disabled.investigate(query, matter_dir)

        # Create engine with multimodal ENABLED (but evidence_retriever not attached)
        config_enabled = RLMConfig(
            max_depth=1,
            max_iterations=1,
            multimodal_enabled=True,
        )
        engine_enabled = RLMEngine(config_enabled, api_key=os.environ["GEMINI_API_KEY"])

        # Run investigation with multimodal enabled but no evidence_retriever attached
        # This tests: "multimodal_enabled = True with an empty media index produces zero changes"
        state_enabled_no_media = await engine_enabled.investigate(query, matter_dir)

        # Assert outputs are identical (no media findings should be added)
        assert len(state_disabled.media_findings) == 0
        assert len(state_enabled_no_media.media_findings) == 0

        # Findings should be identical (multimodal flag had zero effect)
        assert state_disabled.findings.keys() == state_enabled_no_media.findings.keys()

    @pytest.mark.asyncio
    async def test_evidence_retriever_search(self, indexed_matter, embedding_config):
        """Test evidence_retriever.search() directly.

        Step 5 of Task 1.10:
        - Call evidence_retriever.search(query) with a query known to match
        - Assert correct chunk appears in results with score above threshold
        """
        matter_dir, evidence_retriever = indexed_matter

        # Query that should match the contract document
        query = "contract value 1000 units $50"

        # Search
        cards = evidence_retriever.search(query)

        # Assertions
        assert len(cards) > 0, "Should find at least one matching chunk"

        # Top result should have high similarity
        top_card = cards[0]
        assert top_card.similarity >= embedding_config.similarity_threshold

        # Should contain relevant text
        assert "1000" in top_card.text_content or "$50" in top_card.text_content

        # Check that it's a text chunk
        assert top_card.asset_type == "text"

    def test_citation_from_evidence_card(self, indexed_matter):
        """Test Citation built from EvidenceCard has correct fields.

        Step 6 of Task 1.10:
        - Build Citation from top result
        - Assert correct chunk_id, derived_text, and modality = "text"
        """
        matter_dir, evidence_retriever = indexed_matter

        # Get a search result
        query = "contract"
        cards = evidence_retriever.search(query)
        assert len(cards) > 0

        top_card = cards[0]

        # Build Citation
        citation = citation_from_evidence_card(top_card, relevance="Test relevance")

        # Assertions
        assert citation.chunk_id == top_card.chunk_id
        assert citation.text == top_card.text_content  # derived_text
        assert citation.asset_type == "text"  # modality
        assert citation.similarity == top_card.similarity
        assert citation.start_char == top_card.start_char
        assert citation.end_char == top_card.end_char

    @pytest.mark.asyncio
    async def test_multimodal_enabled_with_media(self, indexed_matter, vertex_project):
        """Test that multimodal_enabled=True with media calls evidence_retriever.search().

        Step 4 continuation of Task 1.10:
        - Attach evidence_retriever to engine
        - Run investigation with multimodal_enabled = True
        - Verify evidence_retriever.search() was called and results added to media_findings
        """
        matter_dir, evidence_retriever = indexed_matter

        # has_media() is already correctly implemented in EvidenceRetriever
        # (returns len(self._vs) > 0), no monkey-patching needed.

        query = "What is the contract value?"

        # Create engine with multimodal ENABLED
        config_enabled = RLMConfig(
            max_depth=1,
            max_iterations=1,
            multimodal_enabled=True,
        )
        engine_enabled = RLMEngine(config_enabled, api_key=os.environ["GEMINI_API_KEY"])

        # Attach evidence_retriever to engine
        engine_enabled.evidence_retriever = evidence_retriever

        # Run investigation
        state = await engine_enabled.investigate(query, matter_dir)

        # Verify media_findings were populated
        assert len(state.media_findings) > 0, "Should have media findings when multimodal enabled with media"

        # Verify media_findings contains EvidenceCards
        for chunk_id, card in state.media_findings.items():
            assert isinstance(card, EvidenceCard)
            assert card.chunk_id == chunk_id

        # Verify findings dict was NOT modified (media_findings is separate)
        assert "media" not in state.findings


class TestPhase1MockPipeline:
    """Mock-based Phase 1 pipeline tests — no Vertex AI credentials needed.

    Covers the same pipeline stages as TestPhase1EndToEnd but with
    deterministic numpy vectors substituted for real API embeddings.
    These tests MUST pass in CI without any external credentials.
    """

    def _norm_vec(self, dim: int, seed: int = 0) -> np.ndarray:
        """Return a reproducible normalized float32 vector."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    @pytest.fixture
    def cfg(self):
        return EmbeddingConfig(
            model_id="gemini-embedding-2-preview",
            fast_dimensionality=256,
            full_dimensionality=3072,
            index_dimensionality=768,
            top_k_candidates=50,
            top_k_results=10,
            similarity_threshold=0.0,  # accept all candidates in unit tests
            chunk_strategy_version=1,
        )

    @pytest.fixture
    def three_chunks(self, tmp_path):
        """Three ChunkRecords backed by real temp files (needed for IndexCache)."""
        docs = [
            ("contract.txt", "Supply agreement ACME Corp Widget Inc. 1000 units $50."),
            ("email.txt", "Confirming 1000 units contract. john@acme.com."),
            ("invoice.txt", "Invoice INV-2024-001. 1000 units @ $50 = $50,000."),
        ]
        chunks = []
        for fname, content in docs:
            p = tmp_path / fname
            p.write_text(content, encoding="utf-8")
            chunk = ChunkRecord.from_text(
                asset_path=str(p),
                chunk_index=0,
                text_content=content,
                start_char=0,
                end_char=len(content),
            )
            chunks.append(chunk)
        return chunks

    def test_chunk_record_from_text(self, three_chunks):
        """ChunkRecord.from_text produces correct fields."""
        c = three_chunks[0]
        assert c.asset_type == "text"
        assert c.chunk_index == 0
        assert c.start_char == 0
        assert c.end_char > 0
        assert len(c.chunk_id) > 0

    def test_metadata_store_crud(self, tmp_path, three_chunks):
        """MetadataStore upsert / get / get_many / list_by_asset / delete_by_asset."""
        store = MetadataStore(tmp_path / "meta.db")
        for c in three_chunks:
            store.upsert(c)

        # get
        r = store.get(three_chunks[0].chunk_id)
        assert r is not None and r.text_content == three_chunks[0].text_content

        # get_many
        ids = [c.chunk_id for c in three_chunks[:2]]
        assert len(store.get_many(ids)) == 2

        # list_by_asset
        assert len(store.list_by_asset(three_chunks[0].asset_path)) == 1

        # delete_by_asset
        assert store.delete_by_asset(three_chunks[0].asset_path) == 1
        assert store.get(three_chunks[0].chunk_id) is None

    def test_vector_store_add_search_persist_load(self, tmp_path, cfg):
        """LocalVectorStore: add, search_fast, persist, load round-trip."""
        idx_dir = tmp_path / "vs"
        store = LocalVectorStore(index_dir=idx_dir, fast_dim=256, full_dim=768)

        ids = [str(uuid.uuid4()) for _ in range(3)]
        fast_vecs = [self._norm_vec(256, seed=i) for i in range(3)]
        full_vecs = [self._norm_vec(768, seed=i + 10) for i in range(3)]
        for cid, fv, iv in zip(ids, fast_vecs, full_vecs):
            store.add(cid, fv, iv)
        assert len(store) == 3

        results = store.search_fast(self._norm_vec(256, seed=99), k=3)
        assert len(results) <= 3
        assert all(r in ids for r in results)

        store.persist()
        loaded = LocalVectorStore.load(idx_dir, fast_dim=256, full_dim=768)
        assert len(loaded) == 3

    def test_evidence_retriever_search_with_mock_embeddings(self, tmp_path, cfg, three_chunks):
        """EvidenceRetriever.search() returns EvidenceCards ranked by cosine similarity."""
        idx_dir = tmp_path / "vs"
        vector_store = LocalVectorStore(index_dir=idx_dir, fast_dim=256, full_dim=768)
        metadata_store = MetadataStore(tmp_path / "meta.db")

        fast_vecs = [self._norm_vec(256, seed=i) for i in range(3)]
        index_vecs = [self._norm_vec(768, seed=i + 20) for i in range(3)]

        for i, chunk in enumerate(three_chunks):
            metadata_store.upsert(chunk)
            vector_store.add(chunk.chunk_id, fast_vecs[i], index_vecs[i])

        # Mock client: query identical to chunk[0] vectors → chunk[0] scores 1.0
        mock_ec = MagicMock()
        def _embed(text, output_dimensionality=None):
            return fast_vecs[0].copy() if output_dimensionality == cfg.fast_dimensionality \
                else index_vecs[0].copy()
        mock_ec.embed_query.side_effect = _embed

        retriever = EvidenceRetriever(
            vector_store=vector_store,
            metadata_store=metadata_store,
            embedding_client=mock_ec,
            config=cfg,
        )
        cards = retriever.search("contract 1000 units")

        assert len(cards) >= 1
        assert cards[0].chunk_id == three_chunks[0].chunk_id
        assert cards[0].asset_type == "text"
        assert abs(cards[0].similarity - 1.0) < 0.001

    def test_has_media_reflects_store_size(self, tmp_path, cfg):
        """has_media() returns False when empty and True after indexing a chunk."""
        vs = LocalVectorStore(index_dir=tmp_path / "vs", fast_dim=256, full_dim=768)
        ms = MetadataStore(tmp_path / "meta.db")
        retriever = EvidenceRetriever(
            vector_store=vs, metadata_store=ms, embedding_client=MagicMock(), config=cfg
        )
        assert retriever.has_media() is False

        chunk = ChunkRecord.from_text("t.txt", 0, "text", 0, 4)
        ms.upsert(chunk)
        vs.add(chunk.chunk_id, self._norm_vec(256), self._norm_vec(768))
        assert retriever.has_media() is True

    def test_citation_from_evidence_card_fields(self):
        """citation_from_evidence_card maps all multimodal fields correctly."""
        card = EvidenceCard(
            chunk_id="cid-001",
            asset_path="/matter/contract.txt",
            asset_type="text",
            text_content="Agreement signed on 15 January.",
            similarity=0.91,
            query="agreement date",
            page_number=2,
            start_char=10,
            end_char=43,
        )
        cit = citation_from_evidence_card(card, relevance="relevant to date")
        assert cit.chunk_id == "cid-001"
        assert cit.asset_type == "text"
        assert cit.text == "Agreement signed on 15 January."
        assert abs(cit.similarity - 0.91) < 0.001
        assert cit.start_char == 10
        assert cit.end_char == 43
        assert cit.page == 2          # Citation uses .page, not .page_number
        assert cit.document == "contract.txt"

    def test_embed_media_stub_raises(self):
        """EmbeddingClient.embed_media raises NotImplementedError (Phase 2)."""
        from irys.core.embeddings import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._config = EmbeddingConfig()
        with pytest.raises(NotImplementedError, match="Phase 2"):
            client.embed_media(Path("/tmp/test.mp4"))

    def test_process_stubs_raise_not_implemented(self):
        """Phase 2 process_* stubs raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            process_audio(Path("/tmp/audio.mp3"))
        with pytest.raises(NotImplementedError):
            process_image(Path("/tmp/image.jpg"))
        with pytest.raises(NotImplementedError):
            process_video(Path("/tmp/video.mp4"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
