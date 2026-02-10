\
from __future__ import annotations
import re
from typing import Iterable, List, Optional, Tuple, Dict

def compact(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        it = (it or "").strip()
        if not it or it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\- ]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120] or "audio"

def safe_slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:80] if s else "course"
