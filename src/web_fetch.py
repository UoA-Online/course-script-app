\
from __future__ import annotations
from typing import Tuple
import requests

def fetch_page(url: str, timeout: int = 25) -> Tuple[str, str]:
    r = requests.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.url, r.text
