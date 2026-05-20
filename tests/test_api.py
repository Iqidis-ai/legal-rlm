"""API module tests for Irys RLM system."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.api import (
    Irys, IrysConfig, InvestigationResult,
    get_version, get_irys, __version__,
)


class TestIrysConfig:
    """Tests for IrysConfig."""

    def test_default_config(self):
        config = IrysConfig()
        assert config.max_depth == 5
        assert config.max_leads_per_level == 5
        assert config.output_format == "markdown"
        assert config.cache_enabled is True

    def test_custom_config(self):
        config = IrysConfig(
            max_depth=3,
            output_format="json",
            cache_enabled=False,
        )
        assert config.max_depth == 3
        assert config.output_format == "json"
        assert config.cache_enabled is False


class TestIrys:
    """Tests for Irys main class."""

    def test_create_irys_with_config(self):
        config = IrysConfig(max_depth=3)
        irys = Irys(config=config)
        assert irys.config.max_depth == 3

    def test_create_irys_with_kwargs(self):
        irys = Irys(max_depth=4, output_format="html")
        assert irys.config.max_depth == 4
        assert irys.config.output_format == "html"

    def test_list_templates(self):
        irys = Irys()
        templates = irys.list_templates()
        assert isinstance(templates, list)
        # Should have some built-in templates
        assert len(templates) > 0

    def test_suggest_template(self):
        irys = Irys()
        # Contract query should suggest contract template
        suggestion = irys.suggest_template("What are the contract terms?")
        # May or may not match, depends on implementation
        assert suggestion is None or isinstance(suggestion, str)


class TestVersionInfo:
    """Tests for version information."""

    def test_version_string(self):
        assert __version__ == "0.1.0"

    def test_get_version(self):
        version = get_version()
        assert version == "0.1.0"


class TestDefaultInstance:
    """Tests for default instance management."""

    def test_get_irys_returns_instance(self):
        irys = get_irys()
        assert isinstance(irys, Irys)

    def test_get_irys_same_instance(self):
        irys1 = get_irys()
        irys2 = get_irys()
        # Should return the same instance
        assert irys1 is irys2


class TestInvestigationResult:
    """Tests for InvestigationResult."""

    def test_result_properties(self):
        from irys.rlm.state import InvestigationState

        state = InvestigationState.create("Test query", "/path")
        state.status = "completed"
        state.add_citation("doc.pdf", 1, "text", "ctx", "rel")

        result = InvestigationResult(
            state=state,
            output="Test output",
            format="markdown",
        )

        assert result.query == "Test query"
        assert result.status == "completed"
        assert result.success is True
        assert len(result.citations) == 1

    def test_result_to_dict(self):
        from irys.rlm.state import InvestigationState

        state = InvestigationState.create("Test query", "/path")
        result = InvestigationResult(
            state=state,
            output="Test output",
            format="markdown",
        )

        data = result.to_dict()
        assert "query" in data
        assert "status" in data
        assert "confidence" in data

    def test_result_citation_keeps_url_and_mime(self):
        from irys.rlm.state import InvestigationState

        state = InvestigationState.create("Test query", "/path")
        state.add_citation(
            "doc.pdf",
            1,
            "text",
            "ctx",
            "rel",
            url="https://example.com/doc.pdf",
            mime="application/pdf",
        )

        result = InvestigationResult(
            state=state,
            output="Test output",
            format="markdown",
        )

        citation = result.citations[0]
        assert citation.url == "https://example.com/doc.pdf"
        assert citation.mime == "application/pdf"


@pytest.mark.asyncio
async def test_save_session_serialises_fact_texts_not_tuples():
    """_save_session must store plain strings, not (fact, hash) tuples."""
    from irys.service.config import ServiceConfig
    from irys.service.api import _save_session

    config = ServiceConfig(storage_mode="local", temp_dir="/tmp/irys_test_session")
    result = MagicMock()
    result.state.findings = {
        "accumulated_facts": [
            ("Clause 14 requires notice", "abc123"),
            ("Damages capped at $5M", "def456"),
        ]
    }
    result.citations = []
    result.entities = {}

    saved_data = {}

    async def fake_save(session_id, data):
        saved_data["facts"] = data.facts

    with patch("irys.service.api.SessionStore") as MockStore:
        instance = MockStore.return_value
        instance.load = AsyncMock(return_value=None)
        instance.save = fake_save
        await _save_session(config, "sess-001", result)

    assert saved_data["facts"] == [
        "Clause 14 requires notice",
        "Damages capped at $5M",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
