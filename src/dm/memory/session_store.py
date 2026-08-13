"""
Session store — short-term sliding window of recent conversation messages.

Persists the last N {role, content} pairs to disk so sessions can resume
across process restarts.  The graph (Graphiti) provides long-term memory;
this file provides the recent-context window fed directly into the chat list.
"""

from __future__ import annotations

import json
from pathlib import Path


class SessionStore:
    """Persists and trims the recent message window for a single campaign."""

    def __init__(self, campaign_path: Path, window: int = 20) -> None:
        """
        Parameters
        ----------
        campaign_path:
            Directory under ``memory/<campaign_name>/``.  The session file is
            written as ``session.json`` inside this directory.
        window:
            Maximum number of messages to retain.  Older messages are trimmed
            from the front of the list after each ``append``.
        """
        self._path = campaign_path / "session.json"
        self._window = window
        self._messages: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[dict[str, str]]:
        """Load messages from disk.  Returns an empty list if the file is absent."""
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8")
            self._messages = json.loads(raw)
        return self._messages

    def append(self, role: str, content: str) -> None:
        """Add a message and persist to disk, trimming to the window size."""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._window:
            self._messages = self._messages[-self._window :]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def messages(self) -> list[dict[str, str]]:
        """Return a copy of the current message window."""
        return list(self._messages)

    def clear(self) -> None:
        """Empty the session window and overwrite the file with an empty list."""
        self._messages = []
        if self._path.exists():
            self._path.write_text(
                json.dumps([], ensure_ascii=False), encoding="utf-8"
            )
