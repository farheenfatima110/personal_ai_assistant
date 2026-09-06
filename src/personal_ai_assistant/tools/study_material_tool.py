"""RAG tool: semantic search over the user's study material in knowledge/docs/."""

from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..rag.store import get_knowledge_base


class StudyMaterialInput(BaseModel):
    query: str = Field(
        ...,
        description="What to look up in the user's study notes / documents.",
    )
    k: int = Field(4, description="Number of passages to retrieve (1-8).")


class StudyMaterialTool(BaseTool):
    name: str = "Study Material Search"
    description: str = (
        "Semantically searches the user's own study documents (lecture notes, PDFs, "
        "textbook extracts) stored in the knowledge base and returns the most "
        "relevant passages with their source file. Use this before answering "
        "questions about the user's course content or uploaded material."
    )
    args_schema: Type[BaseModel] = StudyMaterialInput

    def _run(self, query: str, k: int = 4) -> str:
        kb = get_knowledge_base()
        kb.ingest()  # picks up any newly added files, no-op if unchanged

        if kb.is_empty():
            return (
                "The study knowledge base is empty. Add PDF/TXT/MD files to "
                "knowledge/docs/ to enable study-material search."
            )

        hits = kb.search(query, k=max(1, min(k, 8)))
        if not hits:
            return "No relevant passages found in the study material."

        blocks = [
            f"[{i}] (source: {h['source']}, relevance: {h['score']})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        ]
        return "\n\n".join(blocks)
