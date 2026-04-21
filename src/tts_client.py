from __future__ import annotations

import base64
import io
import json
import random
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
DEFAULT_SAMPLE_RATE_HERTZ = 24000
DEFAULT_MAX_CHUNK_BYTES = 900


@dataclass
class GeminiTTSClient:
    session: AuthorizedSession
    project_id: str


@dataclass
class Throttle:
    min_seconds_between_calls: float = 0.5
    last_call_time: float = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - elapsed)
        self.last_call_time = time.time()


def load_service_account_info(service_account_value: str) -> dict:
    raw_value = (service_account_value or "").strip()
    if not raw_value:
        raise ValueError("Service account JSON is required.")

    if raw_value.startswith("{"):
        service_account_info = json.loads(raw_value)
        if not isinstance(service_account_info, dict):
            raise ValueError("Service account JSON must decode to an object.")
        return service_account_info

    json_path = Path(raw_value).expanduser()
    try:
        if json_path.is_file():
            return json.loads(json_path.read_text(encoding="utf-8"))
    except OSError:
        pass

    service_account_info = json.loads(raw_value)
    if not isinstance(service_account_info, dict):
        raise ValueError("Service account JSON must decode to an object.")
    return service_account_info


def get_tts_client(*, service_account_json: Optional[str] = None) -> GeminiTTSClient:
    service_account_info = load_service_account_info(service_account_json or "")
    project_id = service_account_info.get("project_id")
    if not project_id:
        raise ValueError("Service account JSON is missing project_id.")

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES,
    )
    return GeminiTTSClient(
        session=AuthorizedSession(credentials),
        project_id=project_id,
    )


def split_text_into_chunks(text: str, max_bytes: int = DEFAULT_MAX_CHUNK_BYTES) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(sentence.encode("utf-8")) <= max_bytes:
                current = sentence
                continue

            words = sentence.split()
            oversized = ""
            for word in words:
                candidate = word if not oversized else f"{oversized} {word}"
                if len(candidate.encode("utf-8")) <= max_bytes:
                    oversized = candidate
                else:
                    chunks.append(oversized)
                    oversized = word
            if oversized:
                current = oversized

        if current:
            chunks.append(current)
            current = ""

    return chunks


def synthesize_chunk(
    *,
    client: GeminiTTSClient,
    text: str,
    language_code: str,
    voice_name: str,
    model_name: str,
    sample_rate_hertz: int,
    prompt: Optional[str] = None,
) -> bytes:
    payload = {
        "input": {
            "text": text,
        },
        "voice": {
            "languageCode": language_code,
            "name": voice_name,
            "modelName": model_name,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": sample_rate_hertz,
        },
    }
    if prompt:
        payload["input"]["prompt"] = prompt

    response = client.session.post(
        API_URL,
        headers={
            "x-goog-user-project": client.project_id,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"TTS request failed with HTTP {response.status_code}: {response.text}"
        )

    response_json = response.json()
    audio_content = response_json.get("audioContent")
    if not audio_content:
        raise RuntimeError(f"TTS response did not contain audioContent: {response_json}")

    return base64.b64decode(audio_content)


def build_wav_bytes(pcm_audio: bytes, sample_rate_hertz: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hertz)
        wav_file.writeframes(pcm_audio)
    return buffer.getvalue()


def synthesize_wav_with_retry(
    *,
    client: GeminiTTSClient,
    throttle: Throttle,
    text: str,
    language_code: str,
    voice_name: str,
    model_name: str,
    prompt: Optional[str] = None,
    sample_rate_hertz: int = DEFAULT_SAMPLE_RATE_HERTZ,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_retries: int = 6,
    base_backoff: float = 4.0,
) -> bytes:
    chunks = split_text_into_chunks(text, max_bytes=max_chunk_bytes)
    pcm_audio = bytearray()

    for index, chunk in enumerate(chunks, start=1):
        prompt_for_chunk = prompt if index == 1 else None

        for attempt in range(1, max_retries + 1):
            try:
                throttle.wait()
                pcm_audio.extend(
                    synthesize_chunk(
                        client=client,
                        text=chunk,
                        language_code=language_code,
                        voice_name=voice_name,
                        model_name=model_name,
                        sample_rate_hertz=sample_rate_hertz,
                        prompt=prompt_for_chunk,
                    )
                )
                break
            except Exception as exc:
                msg = str(exc)
                if "Prompt is only supported for Gemini TTS" in msg and prompt_for_chunk:
                    pcm_audio.extend(
                        synthesize_chunk(
                            client=client,
                            text=chunk,
                            language_code=language_code,
                            voice_name=voice_name,
                            model_name=model_name,
                            sample_rate_hertz=sample_rate_hertz,
                            prompt=None,
                        )
                    )
                    break
                if any(code in msg for code in ["429", "502", "503", "504", "RESOURCE_EXHAUSTED"]) and attempt < max_retries:
                    sleep_s = (base_backoff * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
                    time.sleep(sleep_s)
                    continue
                raise

    return build_wav_bytes(bytes(pcm_audio), sample_rate_hertz=sample_rate_hertz)
