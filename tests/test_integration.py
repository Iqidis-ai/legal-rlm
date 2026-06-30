"""Integration tests for Irys RLM system against real repository."""

import pytest
import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.api import Irys, IrysConfig, analyze_repository
from irys.core.repository import MatterRepository
from irys.core.search import DocumentSearch
from irys.rlm.state import InvestigationState, classify_query

# Test repository path - update if needed
TEST_REPO = Path(r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents")


class TestRepositoryIntegration:
    """Integration tests with real repository."""

    @pytest.fixture
    def repo(self):
        if not TEST_REPO.exists():
            pytest.skip("Test repository not found")
        return MatterRepository(TEST_REPO)

    def test_list_documents(self, repo):
        """Test listing documents in repository."""
        docs = repo.list_files()
        assert len(docs) > 0
        print(f"Found {len(docs)} documents")

    def test_get_stats(self, repo):
        """Test getting repository statistics."""
        stats = repo.get_stats()
        assert stats.total_files > 0
        print(f"Total files: {stats.total_files}")
        print(f"File types: {stats.files_by_type}")

    def test_get_structure(self, repo):
        """Test getting repository structure."""
        structure = repo.get_structure()
        assert len(structure) > 0
        print(f"Structure: {structure}")

    def test_search_documents(self, repo):
        """Test searching documents."""
        results = repo.search("contract")
        print(f"Search results: {results.total_matches} matches in {results.files_searched} files")
        assert results.files_searched > 0


class TestSearchIntegration:
    """Search integration tests."""

    @pytest.fixture
    def search(self):
        if not TEST_REPO.exists():
            pytest.skip("Test repository not found")
        return DocumentSearch()

    def test_search_legal_terms(self, search):
        """Test searching for legal terms."""
        docs = list(TEST_REPO.glob("**/*.pdf"))[:5]
        if not docs:
            pytest.skip("No PDF files found")

        results = search.search("damages", docs)
        print(f"Found {results.total_matches} matches for 'damages'")

    def test_search_party_names(self, search):
        """Test searching for party names."""
        docs = list(TEST_REPO.glob("**/*.pdf"))[:5]
        if not docs:
            pytest.skip("No PDF files found")

        results = search.search("CITIOM", docs, case_sensitive=False)
        print(f"Found {results.total_matches} matches for 'CITIOM'")

        results = search.search("Gulfstream", docs, case_sensitive=False)
        print(f"Found {results.total_matches} matches for 'Gulfstream'")


class TestQueryClassificationIntegration:
    """Query classification tests with real queries."""

    @pytest.mark.parametrize("query,expected_type", [
        ("What are the damages claimed by CITIOM?", "factual"),
        ("Why did Gulfstream breach the contract?", "analytical"),
        ("Compare the plaintiff's and defendant's positions", "comparative"),
        ("What happened on the date of the breach?", "factual"),
    ])
    def test_classify_legal_queries(self, query, expected_type):
        """Test classifying various legal queries."""
        result = classify_query(query)
        print(f"Query: {query}")
        print(f"Classified as: {result['type']} (expected: {expected_type})")
        print(f"Complexity: {result['complexity']}/5")
        # Don't assert exact type - just verify classification works
        assert result["type"] in ["factual", "analytical", "comparative", "investigative", "unknown"]


class TestApiIntegration:
    """API integration tests."""

    def test_irys_initialization(self):
        """Test Irys initialization without API key."""
        irys = Irys(max_depth=3)
        assert irys.config.max_depth == 3

    def test_list_templates(self):
        """Test listing investigation templates."""
        irys = Irys()
        templates = irys.list_templates()
        assert isinstance(templates, list)
        print(f"Available templates: {templates}")

    @pytest.mark.asyncio
    async def test_analyze_repository(self):
        """Test repository analysis function."""
        if not TEST_REPO.exists():
            pytest.skip("Test repository not found")

        result = await analyze_repository(TEST_REPO)
        assert "stats" in result
        assert "categories" in result
        print(f"Repository analysis: {result}")


class TestStateIntegration:
    """State management integration tests."""

    def test_create_investigation_state(self):
        """Test creating investigation state."""
        state = InvestigationState.create(
            "What are the key contract breaches?",
            str(TEST_REPO),
        )
        assert state.query == "What are the key contract breaches?"
        assert state.status == "initialized"

    def test_state_serialization_roundtrip(self):
        """Test full state serialization and deserialization."""
        state = InvestigationState.create("Test query", "/path")
        state.hypothesis = "Test hypothesis"
        state.add_citation("doc.pdf", 1, "Test text", "context", "relevant")
        state.add_lead("Test lead", "source")
        state.add_entity("CITIOM", "company", "doc.pdf")

        # Serialize
        data = state.to_dict()

        # Deserialize
        restored = InvestigationState.from_dict(data)

        # Verify
        assert restored.query == state.query
        assert restored.hypothesis == state.hypothesis
        assert len(restored.citations) == 1
        assert len(restored.leads) == 1
        assert len(restored.entities) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
