"""Undo/compare history for photographer cues."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.photographer_cues import PhotographerCue, speakable_summary


@dataclass(frozen=True)
class CueSnapshot:
    sequence: int
    cue_text: tuple[str, ...]
    summary: str
    category: tuple[str, ...]
    cues: tuple[PhotographerCue, ...] = ()


class CueHistory:
    """Small bounded history with cursor-based undo/redo."""

    def __init__(self, capacity: int = 30):
        self.capacity = max(2, int(capacity))
        self._items: list[CueSnapshot] = []
        self._cursor = -1
        self._seq = 0

    def push(self, cues: Iterable[PhotographerCue]) -> CueSnapshot:
        cues = list(cues)
        if self._cursor < len(self._items) - 1:
            self._items = self._items[:self._cursor + 1]
        self._seq += 1
        snap = CueSnapshot(
            self._seq,
            tuple(c.cue for c in cues),
            speakable_summary(cues),
            tuple(c.category for c in cues),
            tuple(cues),
        )
        self._items.append(snap)
        if len(self._items) > self.capacity:
            self._items.pop(0)
        self._cursor = len(self._items) - 1
        return snap

    @property
    def current(self) -> CueSnapshot | None:
        return self._items[self._cursor] if 0 <= self._cursor < len(self._items) else None

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._cursor < len(self._items) - 1

    def undo(self) -> CueSnapshot | None:
        if not self.can_undo:
            return self.current
        self._cursor -= 1
        return self.current

    def redo(self) -> CueSnapshot | None:
        if not self.can_redo:
            return self.current
        self._cursor += 1
        return self.current

    def clear(self):
        self._items.clear()
        self._cursor = -1

    def snapshots(self) -> tuple[CueSnapshot, ...]:
        return tuple(self._items)
