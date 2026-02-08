"""
Irys RLM Python Client

A simple client for integrating with the Irys RLM document investigation API.

Usage:
    from irys_client import IrysClient
    
    client = IrysClient("http://your-server:8000")
    result = await client.investigate("What are the payment terms?", "matters/case-001")
    print(result["analysis"])

Requirements:
    pip install httpx
"""

import asyncio
from typing import Optional, Any
import httpx


class IrysError(Exception):
    """Irys API error."""
    pass


class IrysClient:
    """Client for Irys RLM document investigation API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize the client.

        Args:
            base_url: Irys API server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health(self) -> dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status including version, connectivity, active jobs
        """
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def investigate(
        self,
        query: str,
        s3_prefix: str,
        callback_url: Optional[str] = None,
        wait: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """
        Run a document investigation.

        Args:
            query: The question to investigate
            s3_prefix: S3 folder path containing documents
            callback_url: Optional URL to POST results when complete
            wait: If True, poll until job completes
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Investigation results including analysis, citations, entities

        Raises:
            IrysError: If investigation fails
            TimeoutError: If wait times out
        """
        client = await self._get_client()

        # Start the investigation
        response = await client.post(
            "/investigate",
            json={
                "query": query,
                "s3_prefix": s3_prefix,
                "callback_url": callback_url,
            },
        )

        if response.status_code == 429:
            raise IrysError("Too many concurrent jobs. Try again later.")

        response.raise_for_status()
        job = response.json()

        if not wait:
            return job

        # Poll for completion
        job_id = job["job_id"]
        elapsed = 0.0

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            result = await self.get_job(job_id)

            if result["status"] == "completed":
                return result
            elif result["status"] == "failed":
                raise IrysError(f"Investigation failed: {result.get('error')}")

        raise TimeoutError(f"Investigation timed out after {timeout}s")

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """
        Get job status and results.

        Args:
            job_id: The job identifier

        Returns:
            Job status and results (if completed)
        """
        client = await self._get_client()
        response = await client.get(f"/investigate/{job_id}")
        response.raise_for_status()
        return response.json()

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List recent jobs.

        Args:
            status: Filter by status (pending, processing, completed, failed)
            limit: Maximum number of jobs to return

        Returns:
            List of job records
        """
        client = await self._get_client()
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = await client.get("/jobs", params=params)
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        query: str,
        s3_prefix: str,
        max_results: int = 20,
    ) -> dict[str, Any]:
        """
        Quick keyword search across documents.

        Args:
            query: Search terms
            s3_prefix: S3 folder path containing documents
            max_results: Maximum results to return

        Returns:
            Search results with file, page, and matched text
        """
        client = await self._get_client()
        response = await client.post(
            "/search",
            json={
                "query": query,
                "s3_prefix": s3_prefix,
                "max_results": max_results,
            },
        )
        response.raise_for_status()
        return response.json()

    async def investigate_urls_stream(
        self,
        query: str,
        s3_urls: list[str],
        on_step: Optional[Any] = None,
        on_citation: Optional[Any] = None,
        on_fact: Optional[Any] = None,
        on_progress: Optional[Any] = None,
        on_complete: Optional[Any] = None,
        on_error: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Stream an investigation via SSE, dispatching events to callbacks.

        Args:
            query: The question to investigate
            s3_urls: List of document URLs to analyze
            on_step: Callback for step events (receives dict)
            on_citation: Callback for citation events (receives dict)
            on_fact: Callback for fact events (receives dict)
            on_progress: Callback for progress events (receives dict)
            on_complete: Callback for complete event (receives dict)
            on_error: Callback for error events (receives dict)

        Returns:
            The complete event payload (final result), or error payload

        Raises:
            IrysError: If the request itself fails (not investigation errors)
        """
        import json

        callbacks = {
            "step": on_step,
            "citation": on_citation,
            "fact": on_fact,
            "progress": on_progress,
            "complete": on_complete,
            "error": on_error,
        }

        result = None
        url = f"{self.base_url}/investigate/urls/stream"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                json={"query": query, "s3_urls": s3_urls},
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise IrysError(
                        f"Stream request failed ({response.status_code}): {response.text}"
                    )

                event_type = None
                data_buf = []

                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        data_buf.append(line[6:])
                    elif line == "" and event_type and data_buf:
                        # End of event
                        try:
                            data = json.loads("".join(data_buf))
                        except json.JSONDecodeError:
                            data = {"raw": "".join(data_buf)}

                        cb = callbacks.get(event_type)
                        if cb:
                            cb(data)

                        if event_type == "complete":
                            result = data
                        elif event_type == "error":
                            result = data

                        event_type = None
                        data_buf = []

        return result or {}

    # Context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    async def main():
        """Example usage of IrysClient."""

        # Create client
        async with IrysClient("http://localhost:8000") as client:

            # Check health
            health = await client.health()
            print(f"Service status: {health['status']}")
            print(f"Active jobs: {health['active_jobs']}")

            # Run investigation
            print("\nStarting investigation...")
            result = await client.investigate(
                query="What are the payment terms and key obligations?",
                s3_prefix="matters/case-001",
                timeout=120.0,
            )

            # Print results
            print(f"\nStatus: {result['status']}")
            print(f"Documents processed: {result['documents_processed']}")
            print(f"Duration: {result['duration_seconds']:.1f}s")
            print(f"\nAnalysis:\n{result['analysis'][:500]}...")

            # Print citations
            if result.get("citations"):
                print("\nCitations:")
                for cite in result["citations"][:3]:
                    print(
                        f"  - {cite['file']}, page {cite.get('page', 'N/A')}")

            # Print entities
            if result.get("entities"):
                print("\nEntities found:")
                for entity_type, values in result["entities"].items():
                    if values:
                        print(f"  {entity_type}: {', '.join(values[:3])}")

    # Run the example
    asyncio.run(main())
