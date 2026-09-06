"""Read an ad-hoc document (PDF / CSV / TXT / MD) from a file path.

For one-off files the user points at, without adding them to the RAG index.
CSV files get a lightweight structural summary plus a preview.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

MAX_CHARS = 6000


class DocumentInput(BaseModel):
    path: str = Field(..., description="Path to a .pdf, .csv, .txt or .md file")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return f"PDF with {len(pages)} page(s).\n\n" + "\n\n".join(pages)


def _read_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "CSV is empty."
    header, *data = rows
    preview = io.StringIO()
    writer = csv.writer(preview)
    writer.writerow(header)
    writer.writerows(data[:10])
    return (
        f"CSV summary: {len(data)} data row(s), {len(header)} column(s).\n"
        f"Columns: {', '.join(header)}\n\n"
        f"First rows:\n{preview.getvalue()}"
    )


class DocumentTool(BaseTool):
    name: str = "Read Document"
    description: str = (
        "Reads ONE document from an EXACT file path the user has explicitly given "
        "(e.g. 'C:/files/report.pdf' or an attached-file path). Returns text (PDF, "
        "TXT, MD) or a structural summary + preview (CSV). Do NOT guess file names - "
        "for the user's course notes / study material use 'Study Material Search' "
        "instead."
    )
    args_schema: Type[BaseModel] = DocumentInput

    def _run(self, path: str) -> str:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return f"File not found: {file_path}"

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                content = _read_pdf(file_path)
            elif suffix == ".csv":
                content = _read_csv(file_path)
            elif suffix in {".txt", ".md"}:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                return f"Unsupported file type: {suffix}"
        except Exception as exc:  # noqa: BLE001
            return f"Could not read {file_path.name}: {exc}"

        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n[...truncated...]"
        return f"Content of {file_path.name}:\n\n{content}"
