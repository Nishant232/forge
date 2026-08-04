from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

_LOG = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "FileWatcher") -> None:
        super().__init__()
        self._watcher = watcher

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._schedule_change(Path(str(event.src_path)))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._schedule_change(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._schedule_change(Path(str(event.dest_path)))  # type: ignore[attr-defined]
            self._watcher._schedule_delete(Path(str(event.src_path)))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher._schedule_delete(Path(str(event.src_path)))


class FileWatcher:
    """Watch a project directory for file changes and trigger incremental reindex.

    The debounce timer waits `debounce_s` seconds after the last event before
    invoking the `on_change` callback with (changed_paths, deleted_paths).
    """

    def __init__(
        self,
        project_root: Path,
        on_change: Callable[[set[Path], set[Path]], None],
        debounce_s: float = 1.0,
    ) -> None:
        self.project_root = project_root
        self._on_change = on_change
        self.debounce_s = debounce_s

        self._changed: set[Path] = set()
        self._deleted: set[Path] = set()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._observer: Observer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        handler = _Handler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.project_root), recursive=True)
        self._observer.start()
        _LOG.info("Watching %s (debounce=%.1fs)", self.project_root, self.debounce_s)

    def stop(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        _LOG.info("File watcher stopped")

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_change(self, path: Path) -> None:
        with self._lock:
            self._changed.add(path)
            self._reset_timer()

    def _schedule_delete(self, path: Path) -> None:
        with self._lock:
            self._deleted.add(path)
            self._changed.discard(path)
            self._reset_timer()

    def _reset_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_s, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            changed = set(self._changed)
            deleted = set(self._deleted)
            self._changed.clear()
            self._deleted.clear()
            self._timer = None

        if changed or deleted:
            _LOG.debug("Flushing watcher: %d changed, %d deleted", len(changed), len(deleted))
            try:
                self._on_change(changed, deleted)
            except Exception as exc:  # noqa: BLE001
                _LOG.error("on_change callback failed: %s", exc)
