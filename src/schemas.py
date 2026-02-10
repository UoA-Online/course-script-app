\
from typing import List, Optional
from pydantic import BaseModel, Field

class CourseScripts3(BaseModel):
    title: str = Field(description="Clean course name from on-page H1 (no award prefixes).")
    link: str = Field(description="Canonical URL after redirects.")
    script_15s: str = Field(description="~15 seconds spoken script (about 35–50 words).")
    script_30s: str = Field(description="~30 seconds spoken script (about 65–90 words).")
    script_60s: str = Field(description="~60 seconds spoken script (about 140–180 words).")

class CourseLongScript(BaseModel):
    title: str = Field(description="Clean course name from on-page H1 (no award prefixes).")
    link: str = Field(description="Canonical URL after redirects.")
    script_2_3min: str = Field(
        description="Single 2–3 minute spoken script summarising the page (roughly 260–400 words)."
    )

class YouTubeMetadata(BaseModel):
    yt_title: str
    yt_description: str
    yt_tags: List[str]
