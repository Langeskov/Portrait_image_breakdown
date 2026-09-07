"""Device-independent voice-ready cue formatting."""
from __future__ import annotations

import re


def voice_ready_text(summary: str) -> str:
    """Normalize field cues for TTS without binding to a speech engine."""
    text = re.sub(r"\s+", " ", str(summary)).strip()
    if not text:
        return "保持自然。"
    return text.replace("。 ", "。\n")


def ssml(text: str) -> str:
    """Minimal SSML-safe representation for engines that support it."""
    clean = voice_ready_text(text)
    escaped = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<speak><prosody rate=\"96%\">{escaped}</prosody></speak>"
