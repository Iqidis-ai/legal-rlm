"""External search integrations - CourtListener and Tavily APIs.

CourtListener: Free legal research API (case law, dockets, opinions)
Tavily: AI-native web search and URL extraction

Both integrate as tool functions for the RLM engine.
"""

import os
import logging
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LegalCase:
    """A legal case from CourtListener."""
    id: str
    case_name: str
    court: str
    date_filed: Optional[str] = None
    citation: Optional[str] = None
    docket_number: Optional[str] = None
    opinion_text: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None  # Search result snippet

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_name": self.case_name,
            "court": self.court,
            "date_filed": self.date_filed,
            "citation": self.citation,
            "docket_number": self.docket_number,
            "opinion_text": self.opinion_text[:2000] if self.opinion_text else None,
            "url": self.url,
            "snippet": self.snippet,
        }


@dataclass
class WebSearchResult:
    """A web search result from Tavily."""
    title: str
    url: str
    content: str  # Snippet or extracted content
    score: Optional[float] = None
    published_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
        }


@dataclass
class ExtractedContent:
    """Extracted content from a URL via Tavily."""
    url: str
    raw_content: str
    images: list[str] = field(default_factory=list)
    failed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "raw_content": self.raw_content[:5000] if self.raw_content else "",
            "images": self.images[:5],
            "failed": self.failed,
            "error": self.error,
        }


# =============================================================================
# CourtListener API
# =============================================================================

class CourtListenerClient:
    """Client for CourtListener REST API (free legal research).

    API Docs: https://www.courtlistener.com/help/api/rest/

    Provides access to:
    - Case law search (opinions)
    - Docket search
    - Court information
    - RECAP archive (PACER documents)

    Rate limits: Be respectful, no hard limits but they ask for reasonable use.
    Authentication: Optional API token for higher limits.
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search"

    def __init__(self, api_token: Optional[str] = None):
        """Initialize client.

        Args:
            api_token: Optional CourtListener API token for higher rate limits.
                      Get one free at: https://www.courtlistener.com/sign-in/
        """
        self.api_token = api_token or os.environ.get("COURTLISTENER_API_TOKEN")
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_headers(self) -> dict:
        headers = {
            "User-Agent": "Irys-RLM/1.0 (Legal Research Tool)",
        }
        if self.api_token:
            headers["Authorization"] = f"Token {self.api_token}"
        return headers

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._get_headers())
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_opinions(
        self,
        query: str,
        court: Optional[str] = None,
        filed_after: Optional[str] = None,
        filed_before: Optional[str] = None,
        max_results: int = 10,
    ) -> list[LegalCase]:
        """Search for legal opinions/case law.

        Args:
            query: Search query (supports boolean operators, phrases in quotes)
            court: Court ID filter (e.g., "scotus", "ca9", "nysd")
            filed_after: Filter cases filed after this date (YYYY-MM-DD)
            filed_before: Filter cases filed before this date (YYYY-MM-DD)
            max_results: Maximum results to return (default 10)

        Returns:
            List of LegalCase objects

        Example:
            cases = await client.search_opinions(
                query='"breach of contract" damages',
                court="scotus",
                filed_after="2020-01-01"
            )
        """
        session = await self._ensure_session()

        params = {
            "q": query,
            "type": "o",  # opinions
            "order_by": "score desc",
        }

        if court:
            params["court"] = court
        if filed_after:
            params["filed_after"] = filed_after
        if filed_before:
            params["filed_before"] = filed_before

        try:
            url = f"{self.SEARCH_URL}/?{urlencode(params)}"
            logger.info(f"CourtListener search: {query[:50]}...")

            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"CourtListener API error: {response.status}")
                    return []

                data = await response.json()
                results = []

                for item in data.get("results", [])[:max_results]:
                    case = LegalCase(
                        id=str(item.get("id", "")),
                        case_name=item.get("caseName", item.get("case_name", "Unknown")),
                        court=item.get("court", ""),
                        date_filed=item.get("dateFiled", item.get("date_filed")),
                        citation=item.get("citation", [None])[0] if item.get("citation") else None,
                        docket_number=item.get("docketNumber", item.get("docket_number")),
                        snippet=item.get("snippet", ""),
                        url=f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                    )
                    results.append(case)

                logger.info(f"CourtListener found {len(results)} cases")
                return results

        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")
            return []

    async def search_dockets(
        self,
        query: str,
        court: Optional[str] = None,
        max_results: int = 10,
    ) -> list[dict]:
        """Search for dockets (case filings, not opinions).

        Args:
            query: Search query
            court: Court ID filter
            max_results: Maximum results

        Returns:
            List of docket dictionaries
        """
        session = await self._ensure_session()

        params = {
            "q": query,
            "type": "r",  # RECAP dockets
            "order_by": "score desc",
        }

        if court:
            params["court"] = court

        try:
            url = f"{self.SEARCH_URL}/?{urlencode(params)}"

            async with session.get(url) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                return data.get("results", [])[:max_results]

        except Exception as e:
            logger.error(f"CourtListener docket search failed: {e}")
            return []

    async def get_opinion(self, opinion_id: str) -> Optional[LegalCase]:
        """Get full opinion text by ID.

        Args:
            opinion_id: The opinion ID from search results

        Returns:
            LegalCase with full opinion_text, or None
        """
        session = await self._ensure_session()

        try:
            url = f"{self.BASE_URL}/opinions/{opinion_id}/"

            async with session.get(url) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                # Get the cluster for case metadata
                cluster_url = data.get("cluster")
                case_name = "Unknown"
                court = ""
                date_filed = None

                if cluster_url:
                    async with session.get(cluster_url) as cluster_resp:
                        if cluster_resp.status == 200:
                            cluster = await cluster_resp.json()
                            case_name = cluster.get("case_name", "Unknown")
                            date_filed = cluster.get("date_filed")
                            # Get court from docket
                            docket_url = cluster.get("docket")
                            if docket_url:
                                async with session.get(docket_url) as docket_resp:
                                    if docket_resp.status == 200:
                                        docket = await docket_resp.json()
                                        court = docket.get("court", "")

                # Extract opinion text (prefer plain_text, fallback to html)
                opinion_text = data.get("plain_text") or data.get("html", "")

                return LegalCase(
                    id=str(opinion_id),
                    case_name=case_name,
                    court=court,
                    date_filed=date_filed,
                    opinion_text=opinion_text,
                    url=f"https://www.courtlistener.com/opinion/{opinion_id}/",
                )

        except Exception as e:
            logger.error(f"CourtListener get_opinion failed: {e}")
            return None

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[LegalCase]:
        """Semantic search using natural language (new 2025 feature).

        Unlike keyword search, this understands meaning and intent.
        Supports hybrid search: combine keywords (in quotes) with natural language.

        Args:
            query: Natural language query
            max_results: Maximum results

        Returns:
            List of LegalCase objects
        """
        # Semantic search uses same endpoint but different query handling
        # The API automatically detects semantic queries
        return await self.search_opinions(
            query=query,
            max_results=max_results,
        )


# =============================================================================
# Tavily API
# =============================================================================

class TavilyClient:
    """Client for Tavily AI-native search and extraction API.

    API Docs: https://docs.tavily.com/

    Provides:
    - Web search (optimized for LLM/RAG)
    - URL content extraction
    - Research endpoint (deep research)

    Free tier: 1,000 credits/month
    """

    SEARCH_URL = "https://api.tavily.com/search"
    EXTRACT_URL = "https://api.tavily.com/extract"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize client.

        Args:
            api_key: Tavily API key. Get one free at: https://tavily.com/
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set - Tavily searches will fail")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        """Search the web with AI-optimized results.

        Args:
            query: Search query
            search_depth: "basic" (fast) or "advanced" (thorough, 2x credits)
            max_results: Number of results (default 5)
            include_domains: Only include these domains
            exclude_domains: Exclude these domains
            include_answer: Include AI-generated answer summary
            include_raw_content: Include full page content (more tokens)

        Returns:
            Dict with 'answer', 'results', 'query', etc.

        Example:
            results = await client.search(
                query="latest Supreme Court ruling on patent law",
                include_domains=["supremecourt.gov", "law.cornell.edu"]
            )
        """
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return {"error": "API key not configured", "results": []}

        session = await self._ensure_session()

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_usage": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            logger.info(f"Tavily search: {query[:50]}...")

            async with session.post(self.SEARCH_URL, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Tavily API error: {response.status} - {error_text}")
                    return {"error": error_text, "results": []}

                data = await response.json()
                logger.info(f"Tavily found {len(data.get('results', []))} results")
                return data

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_legal(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search for legal information with domain filtering.

        Focuses on authoritative legal sources.

        Args:
            query: Legal search query
            max_results: Number of results

        Returns:
            List of WebSearchResult objects
        """
        legal_domains = [
            "law.cornell.edu",
            "supremecourt.gov",
            "uscourts.gov",
            "courtlistener.com",
            "justia.com",
            "findlaw.com",
            "law.com",
            "reuters.com/legal",
            "bloomberglaw.com",
        ]

        data = await self.search(
            query=query,
            max_results=max_results,
            include_domains=legal_domains,
            include_answer=True,
        )

        results = []
        for item in data.get("results", []):
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score"),
                published_date=item.get("published_date"),
            ))

        return results

    async def extract(
        self,
        urls: list[str],
        extract_depth: str = "basic",
        include_images: bool = False,
    ) -> list[ExtractedContent]:
        """Extract content from URLs.

        Args:
            urls: List of URLs to extract (max 20)
            extract_depth: "basic" or "advanced" (includes tables, embedded content)
            include_images: Include extracted images

        Returns:
            List of ExtractedContent objects

        Example:
            content = await client.extract([
                "https://www.supremecourt.gov/opinions/23pdf/22-451_7m58.pdf",
                "https://law.cornell.edu/supct/cert/22-451"
            ])
        """
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return []

        if len(urls) > 20:
            logger.warning("Tavily extract limited to 20 URLs, truncating")
            urls = urls[:20]

        session = await self._ensure_session()

        payload = {
            "api_key": self.api_key,
            "urls": urls,
            "extract_depth": extract_depth,
            "include_images": include_images,
        }

        try:
            logger.info(f"Tavily extracting {len(urls)} URLs...")

            async with session.post(self.EXTRACT_URL, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Tavily extract error: {response.status} - {error_text}")
                    return []

                data = await response.json()

                results = []
                for item in data.get("results", []):
                    results.append(ExtractedContent(
                        url=item.get("url", ""),
                        raw_content=item.get("raw_content", ""),
                        images=item.get("images", []),
                        failed=False,
                    ))

                # Handle failed extractions
                for failed in data.get("failed_results", []):
                    results.append(ExtractedContent(
                        url=failed.get("url", ""),
                        raw_content="",
                        failed=True,
                        error=failed.get("error", "Unknown error"),
                    ))

                logger.info(f"Tavily extracted {len(results)} URLs")
                return results

        except Exception as e:
            logger.error(f"Tavily extract failed: {e}")
            return []

    async def extract_single(self, url: str) -> Optional[str]:
        """Extract content from a single URL.

        Convenience method for single URL extraction.

        Args:
            url: URL to extract

        Returns:
            Extracted content string, or None if failed
        """
        results = await self.extract([url])
        if results and not results[0].failed:
            return results[0].raw_content
        return None


# =============================================================================
# Unified Search Interface
# =============================================================================

class ExternalSearchManager:
    """Unified interface for all external search APIs.

    Manages CourtListener and Tavily clients, provides high-level
    search functions for the RLM engine.
    """

    def __init__(
        self,
        courtlistener_token: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        self.courtlistener = CourtListenerClient(api_token=courtlistener_token)
        self.tavily = TavilyClient(api_key=tavily_api_key)

    async def close(self):
        """Close all client sessions."""
        await self.courtlistener.close()
        await self.tavily.close()

    async def search_case_law(
        self,
        query: str,
        court: Optional[str] = None,
        max_results: int = 5,
    ) -> list[dict]:
        """Search for case law via CourtListener.

        Returns results as dicts for easy LLM consumption.
        """
        cases = await self.courtlistener.search_opinions(
            query=query,
            court=court,
            max_results=max_results,
        )
        return [c.to_dict() for c in cases]

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        legal_only: bool = False,
    ) -> dict:
        """Search the web via Tavily.

        Args:
            query: Search query
            max_results: Number of results
            legal_only: Restrict to legal domains

        Returns:
            Dict with 'answer' and 'results'
        """
        if legal_only:
            results = await self.tavily.search_legal(query, max_results)
            return {
                "answer": None,
                "results": [r.to_dict() for r in results],
            }
        else:
            return await self.tavily.search(query, max_results=max_results)

    async def extract_url(self, url: str) -> Optional[str]:
        """Extract content from a URL via Tavily.

        Args:
            url: URL to extract

        Returns:
            Extracted content string, or None
        """
        return await self.tavily.extract_single(url)

    async def extract_urls(self, urls: list[str]) -> list[dict]:
        """Extract content from multiple URLs.

        Args:
            urls: List of URLs (max 20)

        Returns:
            List of extraction result dicts
        """
        results = await self.tavily.extract(urls)
        return [r.to_dict() for r in results]

    async def research_legal_question(
        self,
        question: str,
        include_case_law: bool = True,
        include_web: bool = True,
        max_results_each: int = 5,
    ) -> dict:
        """Comprehensive legal research combining all sources.

        Args:
            question: Legal research question
            include_case_law: Search CourtListener
            include_web: Search web via Tavily
            max_results_each: Max results per source

        Returns:
            Dict with 'case_law', 'web_results', 'answer'
        """
        results = {
            "question": question,
            "case_law": [],
            "web_results": [],
            "answer": None,
        }

        tasks = []

        if include_case_law:
            tasks.append(("case_law", self.search_case_law(question, max_results=max_results_each)))

        if include_web:
            tasks.append(("web", self.tavily.search(question, max_results=max_results_each)))

        if tasks:
            gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for i, (key, _) in enumerate(tasks):
                result = gathered[i]
                if isinstance(result, Exception):
                    logger.error(f"Research task {key} failed: {result}")
                    continue

                if key == "case_law":
                    results["case_law"] = result
                elif key == "web":
                    results["web_results"] = result.get("results", [])
                    results["answer"] = result.get("answer")

        return results


# =============================================================================
# Tool Functions for LLM Integration
# =============================================================================

async def tool_search_case_law(
    query: str,
    court: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """Tool function: Search for case law.

    For use with Gemini function calling.

    Args:
        query: Legal search query (e.g., "breach of contract damages")
        court: Optional court filter (e.g., "scotus", "ca9")
        max_results: Number of results (default 5)

    Returns:
        Formatted string of search results
    """
    client = CourtListenerClient()
    try:
        cases = await client.search_opinions(query, court=court, max_results=max_results)

        if not cases:
            return f"No case law found for query: {query}"

        lines = [f"Found {len(cases)} cases for '{query}':\n"]
        for i, case in enumerate(cases, 1):
            lines.append(f"{i}. {case.case_name}")
            if case.citation:
                lines.append(f"   Citation: {case.citation}")
            if case.court:
                lines.append(f"   Court: {case.court}")
            if case.date_filed:
                lines.append(f"   Filed: {case.date_filed}")
            if case.snippet:
                lines.append(f"   Snippet: {case.snippet[:200]}...")
            lines.append(f"   URL: {case.url}")
            lines.append("")

        return "\n".join(lines)
    finally:
        await client.close()


async def tool_web_search(
    query: str,
    legal_only: bool = False,
    max_results: int = 5,
) -> str:
    """Tool function: Search the web.

    For use with Gemini function calling.

    Args:
        query: Search query
        legal_only: Restrict to legal domains
        max_results: Number of results (default 5)

    Returns:
        Formatted string with answer and results
    """
    client = TavilyClient()
    try:
        if legal_only:
            results = await client.search_legal(query, max_results)
            lines = [f"Legal web search for '{query}':\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.title}")
                lines.append(f"   URL: {r.url}")
                lines.append(f"   {r.content[:300]}...")
                lines.append("")
            return "\n".join(lines)
        else:
            data = await client.search(query, max_results=max_results)

            lines = []
            if data.get("answer"):
                lines.append(f"Answer: {data['answer']}\n")

            lines.append(f"Search results for '{query}':\n")
            for i, r in enumerate(data.get("results", []), 1):
                lines.append(f"{i}. {r.get('title', 'Untitled')}")
                lines.append(f"   URL: {r.get('url', '')}")
                lines.append(f"   {r.get('content', '')[:300]}...")
                lines.append("")

            return "\n".join(lines)
    finally:
        await client.close()


async def tool_extract_url(url: str) -> str:
    """Tool function: Extract content from a URL.

    For use with Gemini function calling.

    Args:
        url: URL to extract content from

    Returns:
        Extracted content or error message
    """
    client = TavilyClient()
    try:
        content = await client.extract_single(url)
        if content:
            return f"Content from {url}:\n\n{content[:10000]}"
        else:
            return f"Failed to extract content from {url}"
    finally:
        await client.close()


# =============================================================================
# Gemini Tool Definitions
# =============================================================================

EXTERNAL_SEARCH_TOOLS = [
    {
        "name": "search_case_law",
        "description": "Search for legal case law, court opinions, and judicial decisions. Uses CourtListener database with millions of U.S. federal and state court cases.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Legal search query. Supports boolean operators (AND, OR), phrases in quotes, and legal terms."
                },
                "court": {
                    "type": "string",
                    "description": "Optional court filter. Examples: 'scotus' (Supreme Court), 'ca9' (9th Circuit), 'nysd' (Southern District of New York)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Can be restricted to legal domains only.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "legal_only": {
                    "type": "boolean",
                    "description": "If true, restrict to authoritative legal domains (law.cornell.edu, uscourts.gov, etc.)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "extract_url",
        "description": "Extract and read the content from a specific URL. Use this when you have a URL and need to read its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract content from"
                }
            },
            "required": ["url"]
        }
    },
]
