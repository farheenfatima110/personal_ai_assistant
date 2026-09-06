"""Wikipedia lookup via the public REST API (no API key required)."""

from __future__ import annotations

from typing import Type
from urllib.parse import quote

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_HEADERS = {"User-Agent": "personal-ai-assistant/0.1 (student project)"}


class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic or person to look up on Wikipedia")


class WikipediaTool(BaseTool):
    name: str = "Wikipedia"
    description: str = (
        "Returns a concise encyclopaedic summary of a topic, concept, place or "
        "person from Wikipedia. Best for established facts and definitions rather "
        "than current news."
    )
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(self, query: str) -> str:
        try:
            search = requests.get(
                _SEARCH_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=15,
            ).json()
            hits = search.get("query", {}).get("search", [])
            if not hits:
                return f"No Wikipedia article found for '{query}'."

            title = hits[0]["title"]
            summary = requests.get(
                _SUMMARY_URL + quote(title),
                headers=_HEADERS,
                timeout=15,
            ).json()

            extract = summary.get("extract")
            if not extract:
                return f"No summary available for '{title}'."
            url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
            return f"{title}\n\n{extract}\n\nSource: {url}".strip()
        except Exception as exc:  # noqa: BLE001
            return f"Wikipedia lookup error: {exc}"
