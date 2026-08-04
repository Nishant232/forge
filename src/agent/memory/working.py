"""Per-session in-memory ring buffer.

Holds longer-lived scratch data accumulated across multiple turns within one
agent session (e.g. aggregated retrieval results). Not persisted — resets when
the process exits. The LangGraph graph state carries the active turn's
transcript; this scratchpad is separate from that.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class WorkingMemory:
    """Fixed-capacity ring buffer of (key, value) entries.

    When the buffer is full, the oldest entry is evicted before the new one
    is appended. Thread-safety is not guaranteed — intended for single-threaded
    use within one agent session.
    """

    def __init__(self, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max = max_entries
        self._buf: deque[tuple[str, Any]] = deque()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, key: str, value: Any) -> None:
        """Append a (key, value) entry, evicting the oldest if at capacity."""
        if len(self._buf) >= self._max:
            self._buf.popleft()
        self._buf.append((key, value))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[tuple[str, Any]]:
        """Return all entries in insertion order (oldest first)."""
        return list(self._buf)

    def get_by_key(self, key: str) -> list[Any]:
        """Return all values stored under the exact key, in insertion order."""
        return [v for k, v in self._buf if k == key]

    def latest(self, key: str) -> Any | None:
        """Return the most recently stored value for a key, or None."""
        for k, v in reversed(self._buf):
            if k == key:
                return v
        return None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Empty the buffer."""
        self._buf.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._max

    @property
    def is_full(self) -> bool:
        return len(self._buf) >= self._max
