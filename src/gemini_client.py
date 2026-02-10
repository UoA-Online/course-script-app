\
from __future__ import annotations
import os
import random
import time
from dataclasses import dataclass
from typing import Dict

from google import genai
from google.genai import types

from .schemas import CourseLongScript, YouTubeMetadata
from .utils import compact

def is_429(err: Exception) -> bool:
    msg = str(err)
    return ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg)

def is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(x in msg for x in ["429", "resource_exhausted", "503", "timeout", "bad gateway"])

@dataclass
class Throttle:
    min_seconds_between_calls: float = 3.0
    last_call_time: float = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - elapsed)
        self.last_call_time = time.time()

def get_client(api_key: str) -> genai.Client:
    os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client()

def generate_long_script(
    *,
    client: genai.Client,
    model: str,
    system_instruction: str,
    url: str,
    html: str,
    throttle: Throttle,
    max_retries: int = 6,
    base_backoff: float = 4.0,
) -> Dict[str, str]:
    for attempt in range(1, max_retries + 1):
        try:
            throttle.wait()
            resp = client.models.generate_content(
                model=model,
                contents=[system_instruction, f"URL: {url}", html],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CourseLongScript,
                ),
            )
            data: CourseLongScript = resp.parsed
            data.link = url
            out = data.model_dump()
            out["title"] = compact(out.get("title", ""))
            out["script_2_3min"] = compact(out.get("script_2_3min", ""))
            out["link"] = url
            return out
        except Exception as e:
            if is_429(e) and attempt < max_retries:
                sleep_s = (base_backoff * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
                time.sleep(sleep_s)
                continue
            raise

def generate_youtube_metadata(
    *,
    client: genai.Client,
    model: str,
    system_instruction: str,
    title: str,
    link: str,
    script: str,
    duration_phrase: str,
    throttle: Throttle,
    max_retries: int = 5,
    base_backoff: float = 2.0,
) -> Dict[str, object]:
    for attempt in range(1, max_retries + 1):
        try:
            throttle.wait()
            resp = client.models.generate_content(
                model=model,
                contents=[
                    system_instruction,
                    f"COURSE TITLE: {title}",
                    f"LINK: {link}",
                    f"DURATION PHRASE (must appear in yt_title exactly): {duration_phrase}",
                    "SCRIPT:",
                    script,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=YouTubeMetadata,
                ),
            )
            out = resp.parsed.model_dump()

            # enforce phrase in title
            if duration_phrase not in (out.get("yt_title") or ""):
                out["yt_title"] = f"{title} {duration_phrase}"
            out["yt_title"] = compact(out.get("yt_title", ""))

            # normalize description whitespace
            desc = (out.get("yt_description") or "").strip()
            desc = desc.replace("\r\n", "\n")
            import re
            desc = re.sub(r"\n{3,}", "\n\n", desc).strip()
            out["yt_description"] = desc

            # tags: dedupe and cap
            cleaned, seen = [], set()
            for t in out.get("yt_tags") or []:
                tt = compact(str(t))
                if not tt:
                    continue
                key = tt.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(tt)
            out["yt_tags"] = cleaned[:25]

            # Fallbacks if the model returned empty fields
            if not out.get("yt_title"):
                out["yt_title"] = f"{title} {duration_phrase}".strip()
            if not out.get("yt_description"):
                # Build a minimal, policy-safe description from title/link/script.
                script_snip = " ".join((script or "").split())
                if script_snip:
                    # naive sentence split
                    parts = [p.strip() for p in script_snip.split(".") if p.strip()]
                    snippet = ". ".join(parts[:2]).strip()
                    if snippet and not snippet.endswith("."):
                        snippet += "."
                else:
                    snippet = ""

                p1 = f"Find out about our online course or degree in {title}: {link}"
                p3 = snippet or f"This video introduces key ideas from the course: {title}."
                p4 = f"This course is a great fit for learners who want to explore {title}."
                p6 = f"Read all about the course or degree and how to register on the University of Aberdeen Online website: {link}"
                out["yt_description"] = "\n\n".join([p1, p3, p4, p6]).strip()
            if not out.get("yt_tags"):
                out["yt_tags"] = [
                    "University of Aberdeen",
                    "Aberdeen",
                    "Online Learning",
                    "University of Aberdeen Online",
                    title,
                ]
            return out
        except Exception as e:
            if is_retryable(e) and attempt < max_retries:
                sleep_s = (base_backoff * (2 ** (attempt - 1))) + random.uniform(0, 0.6)
                time.sleep(sleep_s)
                continue
            raise
