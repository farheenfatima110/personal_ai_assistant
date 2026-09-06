"""Unit tests for the assistant's tools and RAG layer (no LLM calls)."""

from pathlib import Path

import pytest

from personal_ai_assistant.tools.calculator_tool import CalculatorTool
from personal_ai_assistant.tools.datetime_tool import DateTimeTool
from personal_ai_assistant.tools.document_tool import DocumentTool
from personal_ai_assistant.tools.personal_memory_tool import (
    PersonalMemoryTool,
    SaveMemoryTool,
    _memory_file,
)
from personal_ai_assistant.rag.store import KnowledgeBase


# --------------------------------------------------------------------------- calculator
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("25 * 4 + 10", "Result: 110"),
        ("2 ** 10", "Result: 1024"),
        ("sqrt(144)", "Result: 12"),
        ("18.5 / 100 * 2400", "Result: 444"),
        ("min(3, 9, 1)", "Result: 1"),
    ],
)
def test_calculator_valid(expr, expected):
    assert CalculatorTool()._run(expr) == expected


def test_calculator_blocks_code_execution():
    out = CalculatorTool()._run("__import__('os').system('echo hi')")
    assert "error" in out.lower()


def test_calculator_division_by_zero():
    assert "zero" in CalculatorTool()._run("1/0").lower()


# --------------------------------------------------------------------------- datetime
def test_datetime_offset():
    out = DateTimeTool()._run(offset_days=10)
    assert "Now:" in out
    assert "10 day(s) from now" in out


# --------------------------------------------------------------------------- memory
def test_personal_memory_reads_profile():
    out = PersonalMemoryTool()._run("name")
    assert "Farheen" in out


def test_save_memory_appends_and_dedupes(tmp_path, monkeypatch):
    fake = tmp_path / "knowledge" / "user_preference.txt"
    fake.parent.mkdir(parents=True)
    fake.write_text("User name is Test.\n", encoding="utf-8")
    monkeypatch.setattr(
        "personal_ai_assistant.tools.personal_memory_tool._memory_file", lambda: fake
    )

    assert "Saved" in SaveMemoryTool()._run("User exam is on 15 October")
    assert "15 October" in fake.read_text(encoding="utf-8")
    # second identical save is a no-op
    assert "Already" in SaveMemoryTool()._run("User exam is on 15 October")


# --------------------------------------------------------------------------- document
def test_document_tool_reads_csv(tmp_path):
    csv_file = tmp_path / "marks.csv"
    csv_file.write_text("name,score\nAsha,88\nRavi,72\n", encoding="utf-8")
    out = DocumentTool()._run(str(csv_file))
    assert "2 data row(s)" in out
    assert "Asha" in out


def test_document_tool_missing_file():
    assert "not found" in DocumentTool()._run("/no/such/file.pdf").lower()


# --------------------------------------------------------------------------- RAG
def test_rag_ingest_and_search():
    kb = KnowledgeBase()
    info = kb.ingest()
    assert info["total_chunks"] > 0
    hits = kb.search("when is the exam", k=3)
    assert hits
    assert any("exam" in h["text"].lower() for h in hits)
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
