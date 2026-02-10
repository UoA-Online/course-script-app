\
from __future__ import annotations
from typing import Optional
from mutagen.mp3 import MP3

def mp3_duration_seconds_from_bytes(data: bytes) -> float:
    # mutagen can read from file-like object
    import io
    bio = io.BytesIO(data)
    audio = MP3(bio)
    return float(audio.info.length)

def duration_phrase_from_seconds(sec: float) -> str:
    mins = max(1, int(round(sec / 60.0)))
    return "in 1 minute" if mins == 1 else f"in {mins} minutes"
