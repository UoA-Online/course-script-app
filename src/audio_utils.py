from __future__ import annotations

import io
import wave

from mutagen.mp3 import MP3


def audio_duration_seconds_from_bytes(filename: str, data: bytes) -> float:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".wav") or (data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise ValueError("WAV file has invalid frame rate.")
            return float(wav_file.getnframes()) / float(frame_rate)

    audio = MP3(io.BytesIO(data))
    return float(audio.info.length)


def duration_phrase_from_seconds(sec: float) -> str:
    mins = max(1, int(round(sec / 60.0)))
    return "in 1 minute" if mins == 1 else f"in {mins} minutes"
