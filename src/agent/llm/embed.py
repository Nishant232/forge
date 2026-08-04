from __future__ import annotations

import os

import httpx

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text produces 768-dim vectors; tiktoken over-counts Qwen tokens by ~10-15%,
# so apply a 15% safety margin on any token budget that uses these embeddings.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))


_MAX_EMBED_CHARS = 6_000  # ~1500 tokens — well within nomic-embed-text's 8192-token limit


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama's /api/embed endpoint.

    Raises httpx.HTTPStatusError on non-2xx responses and
    httpx.ConnectError / httpx.TimeoutException if Ollama is unreachable.
    """
    if not texts:
        return []
    truncated = [t[:_MAX_EMBED_CHARS] for t in texts]
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{_OLLAMA_BASE_URL}/api/embed",
            json={"model": _EMBED_MODEL, "input": truncated},
        )
        response.raise_for_status()
        data: dict[str, list[list[float]]] = response.json()
        return data["embeddings"]
