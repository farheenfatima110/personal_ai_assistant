"""Read/write access to the user's persistent personal profile.

The profile is a plain-text file (``knowledge/user_preference.txt``) so it stays
human-readable and easy to inspect. ``PersonalMemoryTool`` reads it; ``SaveMemoryTool``
appends a new fact, which is how the assistant "learns" about the user over time.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def _memory_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "knowledge" / "user_preference.txt"
        if candidate.parent.is_dir():
            return candidate
    return Path(__file__).resolve().parents[3] / "knowledge" / "user_preference.txt"


class PersonalMemoryTool(BaseTool):
    name: str = "Personal Memory"
    description: str = (
        "Reads the user's saved personal profile: their name, education, learning, "
        "skills, interests, projects, preferences, and anything previously remembered."
    )

    def _run(self, query: str = "") -> str:
        path = _memory_file()
        if not path.exists():
            return "Personal memory is empty."
        text = path.read_text(encoding="utf-8").strip()
        return text or "Personal memory is empty."


class SaveMemoryInput(BaseModel):
    fact: str = Field(
        ...,
        description=(
            "A single, self-contained fact about the user to remember for future "
            "sessions, e.g. 'User's ML exam is on 15 October' or "
            "'User prefers concise answers'."
        ),
    )


class SaveMemoryTool(BaseTool):
    name: str = "Save Memory"
    description: str = (
        "Saves one new fact about the user to persistent personal memory. Use this "
        "whenever the user shares a durable preference, deadline, goal, or detail "
        "they would want remembered later."
    )
    args_schema: Type[BaseModel] = SaveMemoryInput

    def _run(self, fact: str) -> str:
        fact = fact.strip().rstrip(".")
        if not fact:
            return "Nothing to save."

        path = _memory_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""

        if fact.lower() in existing.lower():
            return f"Already in memory: {fact}."

        line = f"[remembered {date.today().isoformat()}] {fact}."
        with path.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(line + "\n")
        return f"Saved to memory: {fact}."
