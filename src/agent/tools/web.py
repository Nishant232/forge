"""Web tools — fetch_url and search_web (both require trust=yolo)."""
from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool

from agent.safety.policy import is_network_allowed


@tool  # type: ignore[misc]
def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch the text content of a URL via HTTP GET.

    Blocked unless trust mode is 'yolo'. Use to read online documentation,
    API references, GitHub raw files, or any web resource needed for the task.

    Args:
        url: The URL to fetch (must start with http:// or https://)
        timeout: Request timeout in seconds (default 30)

    Returns:
        Response body as text. Responses over 50,000 characters are truncated.
    """
    trust = os.environ.get("AGENT_TRUST_MODE", "trusted")
    allowed, reason = is_network_allowed(trust)  # type: ignore[arg-type]
    if not allowed:
        raise PermissionError(
            f"fetch_url blocked by trust policy ({reason}). "
            "Set trust mode to 'yolo' to enable outbound HTTP."
        )

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https://, got: {url!r}")

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "agentX/0.2 (local-coding-agent)"},
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    text = response.text
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n[... response truncated at 50,000 chars ...]"
    return text


@tool  # type: ignore[misc]
def search_web(query: str, n: int = 5) -> str:
    """Search the web and return top results with titles, URLs, and snippets.

    Blocked unless trust mode is 'yolo'.
    Requires the search extra: pip install 'local-coding-agent[search]'

    Args:
        query: Search query string.
        n: Number of results to return (default 5, max 10).

    Returns:
        Formatted search results with title, URL, and snippet for each hit.
    """
    trust = os.environ.get("AGENT_TRUST_MODE", "trusted")
    allowed, reason = is_network_allowed(trust)  # type: ignore[arg-type]
    if not allowed:
        raise PermissionError(
            f"search_web blocked by trust policy ({reason}). "
            "Set trust mode to 'yolo' to enable web search."
        )

    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Web search requires the search extra: "
            "pip install 'local-coding-agent[search]'"
        )

    n = min(n, 10)
    results: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=n):
            title = r.get("title", "(no title)")
            href = r.get("href", "")
            body = r.get("body", "")[:300]
            results.append(f"**{title}**\n{href}\n{body}")

    if not results:
        return f"No results found for: {query!r}"
    return "\n\n---\n\n".join(results)
