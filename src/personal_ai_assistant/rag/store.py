"""Local RAG knowledge base over the user's study material.

Documents live in ``knowledge/docs/`` (PDF / TXT / MD). They are chunked,
embedded with a small local static-embedding model (model2vec - no API key,
runs offline) and cached in ``knowledge/.rag_index.npz`` so ingestion only
re-runs when a file is added or changed. Retrieval is plain cosine similarity
over the chunk vectors - lightweight and easy to reason about for a study-notes
sized corpus.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CHUNK_SIZE = 70           # words per chunk
CHUNK_OVERLAP = 20        # words shared between neighbouring chunks
EMBED_MODEL = "minishlab/potion-retrieval-32M"  # retrieval-tuned static embeddings


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "knowledge").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


DOCS_DIR = _project_root() / "knowledge" / "docs"
INDEX_PATH = _project_root() / "knowledge" / ".rag_index.npz"
META_PATH = _project_root() / "knowledge" / ".rag_index.json"


@lru_cache(maxsize=1)
def _model():
    from model2vec import StaticModel

    return StaticModel.from_pretrained(EMBED_MODEL)


def _embed(texts: list[str]) -> np.ndarray:
    vecs = np.asarray(_model().encode(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(words):
            break
    return chunks


def _fingerprint() -> dict:
    files = {}
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.suffix.lower() in {".pdf", ".txt", ".md"}:
            files[path.name] = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    return files


class KnowledgeBase:
    """Ingest and semantically search the user's study documents."""

    def __init__(self) -> None:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._fingerprint: dict = {}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if INDEX_PATH.exists() and META_PATH.exists():
            data = np.load(INDEX_PATH, allow_pickle=False)
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            self._vectors = data["vectors"]
            self._chunks = meta["chunks"]
            self._sources = meta["sources"]
            self._fingerprint = meta["fingerprint"]

    def _save(self) -> None:
        np.savez_compressed(INDEX_PATH, vectors=self._vectors)
        META_PATH.write_text(
            json.dumps(
                {
                    "chunks": self._chunks,
                    "sources": self._sources,
                    "fingerprint": self._fingerprint,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- public API --------------------------------------------------------
    def ingest(self, force: bool = False) -> dict:
        """(Re)build the index if any document changed."""
        current = _fingerprint()
        if current == self._fingerprint and not force and self._chunks:
            return {
                "status": "up-to-date",
                "documents_indexed": len(current),
                "total_chunks": len(self._chunks),
                "sources": sorted(current),
            }

        all_chunks: list[str] = []
        all_sources: list[str] = []
        for path in sorted(DOCS_DIR.glob("**/*")):
            if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            for chunk in _chunk(_read_document(path)):
                all_chunks.append(chunk)
                all_sources.append(path.name)

        self._vectors = (
            _embed(all_chunks) if all_chunks else np.zeros((0, 0), dtype=np.float32)
        )
        self._chunks = all_chunks
        self._sources = all_sources
        self._fingerprint = current
        self._save()

        return {
            "status": "rebuilt",
            "documents_indexed": len(current),
            "total_chunks": len(all_chunks),
            "sources": sorted(current),
        }

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self._chunks:
            return []
        q = _embed([query])[0]
        scores = self._vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [
            {
                "text": self._chunks[i],
                "source": self._sources[i],
                "score": round(float(scores[i]), 3),
            }
            for i in top
        ]

    def is_empty(self) -> bool:
        return not self._chunks

    def stats(self) -> dict:
        return {
            "chunks": len(self._chunks),
            "sources": sorted(set(self._sources)),
        }


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
