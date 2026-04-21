from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .audio_utils import audio_duration_seconds_from_bytes, duration_phrase_from_seconds
from .gemini_client import Throttle as GeminiThrottle
from .gemini_client import generate_long_script, generate_youtube_metadata
from .tts_client import Throttle as TTSThrottle
from .tts_client import synthesize_wav_with_retry
from .utils import safe_filename
from .web_fetch import fetch_page

ProgressFn = Callable[[int, int, str], None]


def run_script_generation(
    *,
    gemini_client,
    model: str,
    system_instruction: str,
    urls: List[str],
    min_seconds_between_calls: float = 3.0,
    max_retries: int = 6,
    base_backoff: float = 4.0,
    timeout: int = 25,
    progress: Optional[ProgressFn] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    throttle = GeminiThrottle(min_seconds_between_calls=min_seconds_between_calls)
    rows: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    for i, url in enumerate(urls, start=1):
        if progress:
            progress(i, len(urls), url)
        try:
            final_url, html = fetch_page(url, timeout=timeout)
            row = generate_long_script(
                client=gemini_client,
                model=model,
                system_instruction=system_instruction,
                url=final_url,
                html=html,
                throttle=throttle,
                max_retries=max_retries,
                base_backoff=base_backoff,
            )
            rows.append(row)
        except Exception as exc:
            failures.append({"link": url, "error": str(exc)[:500]})

    df = pd.DataFrame(rows, columns=["title", "link", "script_2_3min"])
    failed_df = pd.DataFrame(failures, columns=["link", "error"])
    return df, failed_df


def run_tts_generation(
    *,
    tts_client,
    df_scripts: pd.DataFrame,
    language_code: str,
    voice_name: str,
    model_name: str,
    prompt: Optional[str] = None,
    sample_rate_hertz: int = 24000,
    max_chunk_bytes: int = 900,
    min_seconds_between_calls: float = 0.5,
    max_retries: int = 6,
    base_backoff: float = 4.0,
    progress: Optional[ProgressFn] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, bytes]]]:
    throttle = TTSThrottle(min_seconds_between_calls=min_seconds_between_calls)
    audio_blobs: List[Tuple[str, bytes]] = []

    df = df_scripts.copy()
    if "audio_2_3min_file" not in df.columns:
        df["audio_2_3min_file"] = ""
    if "audio_model" not in df.columns:
        df["audio_model"] = ""
    if "audio_voice" not in df.columns:
        df["audio_voice"] = ""
    if "audio_lang" not in df.columns:
        df["audio_lang"] = ""
    if "audio_format" not in df.columns:
        df["audio_format"] = ""
    if "audio_sample_rate_hz" not in df.columns:
        df["audio_sample_rate_hz"] = pd.NA

    total = len(df)
    for position, (idx, row) in enumerate(df.iterrows(), start=1):
        title = str(row.get("title", "")).strip()
        script = str(row.get("script_2_3min", "")).strip()
        if progress:
            progress(position, total, title or "row")

        if not script:
            continue

        wav_bytes = synthesize_wav_with_retry(
            client=tts_client,
            throttle=throttle,
            text=script,
            language_code=language_code,
            voice_name=voice_name,
            model_name=model_name,
            prompt=prompt,
            sample_rate_hertz=sample_rate_hertz,
            max_chunk_bytes=max_chunk_bytes,
            max_retries=max_retries,
            base_backoff=base_backoff,
        )

        fname = f"{safe_filename(title)}.wav"
        audio_blobs.append((fname, wav_bytes))

        df.at[idx, "audio_2_3min_file"] = fname
        df.at[idx, "audio_model"] = model_name
        df.at[idx, "audio_voice"] = voice_name
        df.at[idx, "audio_lang"] = language_code
        df.at[idx, "audio_format"] = "wav"
        df.at[idx, "audio_sample_rate_hz"] = sample_rate_hertz

    return df, audio_blobs


def run_youtube_metadata(
    *,
    gemini_client,
    model: str,
    system_instruction: str,
    df: pd.DataFrame,
    audio_blobs_by_name: Dict[str, bytes],
    min_seconds_between_calls: float = 1.5,
    max_retries: int = 5,
    base_backoff: float = 2.0,
    progress: Optional[ProgressFn] = None,
) -> pd.DataFrame:
    throttle = GeminiThrottle(min_seconds_between_calls=min_seconds_between_calls)

    out_df = df.copy()
    for col in ["yt_title", "yt_description", "yt_tags", "audio_duration_sec", "yt_error", "yt_debug"]:
        if col not in out_df.columns:
            out_df[col] = pd.NA

    total = len(out_df)
    for position, (idx, row) in enumerate(out_df.iterrows(), start=1):
        title = str(row.get("title", "")).strip()
        link = str(row.get("link", "")).strip()
        script = str(row.get("script_2_3min", "")).strip()
        audio_file = str(row.get("audio_2_3min_file", "")).strip()

        if progress:
            progress(position, total, title or "row")

        if not title or not link or not script or not audio_file:
            out_df.at[idx, "yt_debug"] = "skipped_missing_fields"
            continue
        if audio_file not in audio_blobs_by_name:
            out_df.at[idx, "yt_debug"] = "skipped_missing_audio"
            continue

        audio_bytes = audio_blobs_by_name[audio_file]
        sec = audio_duration_seconds_from_bytes(audio_file, audio_bytes)
        out_df.at[idx, "audio_duration_sec"] = round(sec, 2)
        dur_phrase = duration_phrase_from_seconds(sec)

        existing = row.get("yt_description")
        if existing is not None and not pd.isna(existing) and str(existing).strip() != "" and str(existing).lower() != "nan" and str(existing).lower() != "<na>":
            out_df.at[idx, "yt_debug"] = "skipped_already_filled"
            continue

        try:
            meta = generate_youtube_metadata(
                client=gemini_client,
                model=model,
                system_instruction=system_instruction,
                title=title,
                link=link,
                script=script,
                duration_phrase=dur_phrase,
                throttle=throttle,
                max_retries=max_retries,
                base_backoff=base_backoff,
            )
            out_df.at[idx, "yt_title"] = meta.get("yt_title", "")
            out_df.at[idx, "yt_description"] = meta.get("yt_description", "")
            out_df.at[idx, "yt_tags"] = ", ".join(meta.get("yt_tags") or [])
            out_df.at[idx, "yt_error"] = ""
            out_df.at[idx, "yt_debug"] = str(meta)[:400]

            if not (out_df.at[idx, "yt_title"] or out_df.at[idx, "yt_description"] or out_df.at[idx, "yt_tags"]):
                out_df.at[idx, "yt_error"] = "empty_meta"
        except Exception as exc:
            out_df.at[idx, "yt_error"] = str(exc)[:400]
            out_df.at[idx, "yt_debug"] = ""

    return out_df
