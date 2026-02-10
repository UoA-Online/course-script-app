\
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import random

from google.cloud import texttospeech
from google.oauth2 import service_account, credentials as oauth_credentials
from google.auth.credentials import Credentials
import json
from google.api_core.client_options import ClientOptions

def get_tts_client(
    *,
    quota_project_id: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    service_account_json: Optional[str] = None,
    access_token: Optional[str] = None,
) -> texttospeech.TextToSpeechClient:
    opts = {}
    if quota_project_id:
        opts["quota_project_id"] = quota_project_id
    if api_endpoint:
        opts["api_endpoint"] = api_endpoint

    credentials: Optional[Credentials] = None
    if service_account_json:
        try:
            info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(info)
        except Exception:
            credentials = None
    if access_token:
        try:
            credentials = oauth_credentials.Credentials(token=access_token)
        except Exception:
            credentials = credentials

    if opts and credentials:
        return texttospeech.TextToSpeechClient(client_options=ClientOptions(**opts), credentials=credentials)
    if opts:
        return texttospeech.TextToSpeechClient(client_options=ClientOptions(**opts))
    if credentials:
        return texttospeech.TextToSpeechClient(credentials=credentials)
    return texttospeech.TextToSpeechClient()

def list_voices(language_code: str, quota_project_id: str = "") -> List[Dict[str, str]]:
    # Note: Streamlit cache decorator imported at runtime in app; this function is patched in app.
    raise RuntimeError("list_voices must be wrapped with st.cache_data in app context")

@dataclass
class Throttle:
    min_seconds_between_calls: float = 0.5
    last_call_time: float = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - elapsed)
        self.last_call_time = time.time()

def synthesize_mp3(
    *,
    client: texttospeech.TextToSpeechClient,
    text: str,
    language_code: str,
    voice_name: str,
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
) -> bytes:
    # SynthesisInput(prompt=...) + VoiceSelectionParams(model_name=...) are used in your notebook.
    # Some environments may not support prompt/model_name yet; we fall back gracefully.
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text, prompt=prompt) if prompt else texttospeech.SynthesisInput(text=text)
    except TypeError:
        synthesis_input = texttospeech.SynthesisInput(text=text)

    try:
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name, model_name=model_name) if model_name else texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
    except TypeError:
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return resp.audio_content

def synthesize_mp3_with_retry(
    *,
    client: texttospeech.TextToSpeechClient,
    throttle: Throttle,
    text: str,
    language_code: str,
    voice_name: str,
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    max_retries: int = 6,
    base_backoff: float = 4.0,
) -> bytes:
    for attempt in range(1, max_retries + 1):
        try:
            throttle.wait()
            return synthesize_mp3(
                client=client,
                text=text,
                language_code=language_code,
                voice_name=voice_name,
                model_name=model_name,
                prompt=prompt,
            )
        except Exception as e:
            msg = str(e)
            if "Prompt is only supported for Gemini TTS" in msg:
                # Retry once without prompt for non-Gemini voices.
                return synthesize_mp3(
                    client=client,
                    text=text,
                    language_code=language_code,
                    voice_name=voice_name,
                    model_name=model_name,
                    prompt=None,
                )
            if any(x in msg for x in ["502", "503", "RESOURCE_EXHAUSTED", "429"]) and attempt < max_retries:
                sleep_s = (base_backoff * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
                time.sleep(sleep_s)
                continue
            raise
