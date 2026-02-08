"""Output formatters for investigation results.

Provides multiple output formats: Markdown, HTML, JSON, Plain text.
"""

from dataclasses import dataclass
from typing import Any, Protocol
from abc import abstractmethod
import json
import html


class OutputFormatter(Protocol):
    """Protocol for output formatters."""

    @abstractmethod
    def format(self, state: Any) -> str:
        """Format investigation state for output."""
        ...


class MarkdownFormatter:
    """Format investigation results as Markdown."""

    def format(self, state: Any) -> str:
        """Format as Markdown document."""
        lines = [
            f"# Investigation Report",
            "",
            f"**Query:** {state.query}",
            f"**Status:** {state.status}",
            f"**Duration:** {state.duration_seconds:.1f}s" if state.duration_seconds else "",
            "",
        ]

        # Confidence
        confidence = state.get_confidence_score()
        lines.extend([
            "## Confidence",
            f"- Score: {confidence['score']}/100 ({confidence['level']})",
            "",
        ])

        # Hypothesis
        if state.hypothesis:
            lines.extend([
                "## Hypothesis",
                state.hypothesis,
                "",
            ])

        # Key Findings
        facts = state.findings.get("accumulated_facts", [])
        if facts:
            lines.extend([
                "## Key Findings",
                "",
            ])
            for i, fact in enumerate(facts[:15], 1):
                lines.append(f"{i}. {fact}")
            lines.append("")

        # Citations
        if state.citations:
            lines.extend([
                "## Citations",
                "",
            ])
            for c in state.citations[:20]:
                page = f", p. {c.page}" if c.page else ""
                lines.append(f"- **{c.document}**{page}")
                lines.append(f"  > \"{c.text[:100]}...\"")
                lines.append(f"  - *{c.relevance}*")
                lines.append("")

        # Entities
        if state.entities:
            lines.extend([
                "## Key Entities",
                "",
            ])
            for entity in state.get_top_entities(10):
                lines.append(f"- **{entity.name}** ({entity.entity_type}): {entity.mentions} mentions")
            lines.append("")

        # Final Output
        final_output = state.findings.get("final_output")
        if final_output:
            lines.extend([
                "## Analysis",
                "",
                final_output,
                "",
            ])

        # Metrics
        lines.extend([
            "---",
            "## Investigation Metrics",
            f"- Documents read: {state.documents_read}",
            f"- Searches performed: {state.searches_performed}",
            f"- Citations collected: {len(state.citations)}",
            f"- Facts accumulated: {len(state.findings.get('accumulated_facts', []))}",
            f"- Entities found: {len(state.entities)}",
            f"- API calls: {state.api_calls}",
        ])

        return "\n".join(lines)


class HTMLFormatter:
    """Format investigation results as HTML."""

    def format(self, state: Any) -> str:
        """Format as HTML document."""
        confidence = state.get_confidence_score()
        facts = state.findings.get("accumulated_facts", [])
        final_output = state.findings.get("final_output", "")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Investigation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #555; }}
        .meta {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        .citation {{ border-left: 3px solid #007bff; padding-left: 10px; margin: 10px 0; }}
        .citation strong {{ color: #333; }}
        .entity {{ display: inline-block; background: #e0e0e0; padding: 2px 8px; margin: 2px; border-radius: 3px; }}
        blockquote {{ background: #f9f9f9; border-left: 3px solid #ccc; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Investigation Report</h1>

    <div class="meta">
        <p><strong>Query:</strong> {html.escape(state.query)}</p>
        <p><strong>Status:</strong> {state.status}</p>
        <p><strong>Confidence:</strong> {confidence['score']}/100 ({confidence['level']})</p>
    </div>

    {"<h2>Hypothesis</h2><p>" + html.escape(state.hypothesis) + "</p>" if state.hypothesis else ""}

    <h2>Key Findings</h2>
    <ol>
        {"".join(f"<li>{html.escape(f)}</li>" for f in facts[:15])}
    </ol>

    <h2>Citations</h2>
    {"".join(self._format_citation_html(c) for c in state.citations[:20])}

    <h2>Key Entities</h2>
    {"".join(f'<span class="entity">{html.escape(e.name)} ({e.mentions})</span>' for e in state.get_top_entities(10))}

    <h2>Analysis</h2>
    <div>{html.escape(final_output).replace(chr(10), '<br>')}</div>

    <hr>
    <h2>Metrics</h2>
    <ul>
        <li>Documents read: {state.documents_read}</li>
        <li>Searches performed: {state.searches_performed}</li>
        <li>Citations: {len(state.citations)}</li>
        <li>Entities: {len(state.entities)}</li>
    </ul>
</body>
</html>"""
        return html_content

    def _format_citation_html(self, citation: Any) -> str:
        page = f", p. {citation.page}" if citation.page else ""
        return f"""
        <div class="citation">
            <strong>{html.escape(citation.document)}</strong>{page}
            <blockquote>{html.escape(citation.text[:150])}...</blockquote>
            <em>{html.escape(citation.relevance)}</em>
        </div>"""


class JSONFormatter:
    """Format investigation results as JSON."""

    def format(self, state: Any) -> str:
        """Format as JSON."""
        return json.dumps(state.to_dict(), indent=2, default=str)


class PlainTextFormatter:
    """Format investigation results as plain text."""

    def format(self, state: Any) -> str:
        """Format as plain text."""
        lines = [
            "=" * 60,
            "INVESTIGATION REPORT",
            "=" * 60,
            "",
            f"Query: {state.query}",
            f"Status: {state.status}",
            "",
        ]

        confidence = state.get_confidence_score()
        lines.append(f"Confidence: {confidence['score']}/100 ({confidence['level']})")
        lines.append("")

        if state.hypothesis:
            lines.extend([
                "-" * 40,
                "HYPOTHESIS",
                "-" * 40,
                state.hypothesis,
                "",
            ])

        facts = state.findings.get("accumulated_facts", [])
        if facts:
            lines.extend([
                "-" * 40,
                "KEY FINDINGS",
                "-" * 40,
            ])
            for i, fact in enumerate(facts[:15], 1):
                lines.append(f"  {i}. {fact}")
            lines.append("")

        if state.citations:
            lines.extend([
                "-" * 40,
                "CITATIONS",
                "-" * 40,
            ])
            for c in state.citations[:15]:
                page = f", p. {c.page}" if c.page else ""
                lines.append(f"  {c.document}{page}")
                lines.append(f"      \"{c.text[:80]}...\"")
            lines.append("")

        final_output = state.findings.get("final_output")
        if final_output:
            lines.extend([
                "-" * 40,
                "ANALYSIS",
                "-" * 40,
                final_output,
            ])

        return "\n".join(lines)


def get_formatter(format_type: str) -> OutputFormatter:
    """Get formatter by type name."""
    formatters = {
        "markdown": MarkdownFormatter(),
        "md": MarkdownFormatter(),
        "html": HTMLFormatter(),
        "json": JSONFormatter(),
        "text": PlainTextFormatter(),
        "plain": PlainTextFormatter(),
    }
    return formatters.get(format_type.lower(), PlainTextFormatter())
