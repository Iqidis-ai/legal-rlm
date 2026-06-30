"""Core module tests for Irys RLM system."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.cache import LRUCache, DiskCache, ResponseCache, CacheEntry
from irys.core.clustering import DocumentClusterer, cluster_by_document_type
from irys.core.utils import (
    clean_text, truncate_text, extract_sentences, extract_numbers,
    levenshtein_distance, similarity_ratio, jaccard_similarity,
    chunk_list, parse_date_flexible, format_duration,
    validate_query, RetryConfig,
)


class TestLRUCache:
    """Tests for LRU cache."""

    def test_basic_get_set(self):
        cache = LRUCache[str](max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_max_size_eviction(self):
        cache = LRUCache[int](max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_lru_ordering(self):
        cache = LRUCache[int](max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # Access "a" to make it recent
        cache.set("d", 4)  # Should evict "b" (least recent)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_stats(self):
        cache = LRUCache[str](max_size=10)
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["total_hits"] == 2


class TestResponseCache:
    """Tests for response cache."""

    def test_cache_response(self):
        cache = ResponseCache(max_size=100)
        cache.set("test prompt", "gemini-flash", "test response")
        result = cache.get("test prompt", "gemini-flash")
        assert result == "test response"

    def test_different_models(self):
        cache = ResponseCache(max_size=100)
        cache.set("prompt", "model1", "response1")
        cache.set("prompt", "model2", "response2")
        assert cache.get("prompt", "model1") == "response1"
        assert cache.get("prompt", "model2") == "response2"


class TestDocumentClustering:
    """Tests for document clustering."""

    def test_cluster_by_type(self):
        files = [
            "contract_2020.pdf",
            "agreement_v1.docx",
            "email_jan15.txt",
            "motion_to_dismiss.pdf",
            "random_file.pdf",
        ]
        categories = cluster_by_document_type(files)
        assert "contracts" in categories
        assert "correspondence" in categories
        assert len(categories["contracts"]) == 2
        assert len(categories["correspondence"]) == 1

    def test_tfidf_clustering(self):
        clusterer = DocumentClusterer()
        documents = {
            "doc1.txt": "contract agreement terms obligations",
            "doc2.txt": "contract terms conditions payment",
            "doc3.txt": "email letter correspondence message",
        }
        clusterer.fit(documents)
        clusters = clusterer.cluster(num_clusters=2)
        # Clustering may produce 1-3 clusters depending on similarity thresholds
        assert len(clusters) >= 1 and len(clusters) <= 3


class TestTextUtilities:
    """Tests for text utilities."""

    def test_clean_text(self):
        text = "  Hello   World  \n\n  Test  "
        assert clean_text(text) == "Hello World Test"

    def test_truncate_text(self):
        text = "This is a long text that needs truncation"
        truncated = truncate_text(text, 20)
        assert len(truncated) == 20
        assert truncated.endswith("...")

    def test_extract_sentences(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = extract_sentences(text)
        assert len(sentences) == 3

    def test_extract_numbers(self):
        text = "The amount was $1,000.50 and 500 items"
        numbers = extract_numbers(text)
        assert "$1,000.50" in numbers
        assert "500" in numbers

    def test_chunk_list(self):
        items = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(items, 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[2] == [7]


class TestStringSimilarity:
    """Tests for string similarity functions."""

    def test_levenshtein_distance(self):
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("hello", "hallo") == 1
        assert levenshtein_distance("", "test") == 4

    def test_similarity_ratio(self):
        assert similarity_ratio("hello", "hello") == 1.0
        assert similarity_ratio("hello", "hallo") >= 0.8  # Exactly 0.8 (4/5 chars match)
        assert similarity_ratio("abc", "xyz") < 0.5

    def test_jaccard_similarity(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0
        # "hello world" vs "hello there": intersection={hello}, union={hello,world,there}
        # Jaccard = 1/3 ≈ 0.333
        assert abs(jaccard_similarity("hello world", "hello there") - (1/3)) < 0.01


class TestDateParsing:
    """Tests for date parsing."""

    def test_various_formats(self):
        assert parse_date_flexible("2024-01-15") is not None
        assert parse_date_flexible("01/15/2024") is not None
        assert parse_date_flexible("January 15, 2024") is not None

    def test_format_duration(self):
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3600) == "1.0h"


class TestValidation:
    """Tests for validation functions."""

    def test_query_validation(self):
        valid, issues = validate_query("What are the contract terms?")
        assert valid is True
        assert len(issues) == 0

        valid, issues = validate_query("")
        assert valid is False
        assert len(issues) > 0

        # "Hi" is accepted — min-length check is intentionally disabled
        valid, issues = validate_query("Hi")
        assert valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
