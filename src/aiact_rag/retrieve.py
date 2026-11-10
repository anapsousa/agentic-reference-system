"""Retrievers (BM25, dense, hybrid RRF) and the citation verifier."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .ingest import embed_text
from .index import VectorIndex


@dataclass
class Hit:
    chunk: dict
    score: float

    @property
    def id(self) -> str:
        return self.chunk["id"]

    @property
    def unit(self) -> str:
        return self.chunk["unit"]


def _top(scores, chunks, k) -> list[Hit]:
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i])) for i in order]


class Retriever:
    """mode: 'bm25' | 'dense' | 'hybrid' (reciprocal rank fusion, k=60)."""

    def __init__(self, chunks: list[dict], vindex: VectorIndex | None = None):
        self.chunks, self.vindex = chunks, vindex
        self.bm25 = BM25([embed_text(c) for c in chunks])

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode == "bm25":
            return _top(self.bm25.scores(query), self.chunks, k)
        if self.vindex is None:
            raise ValueError(f"mode {mode!r} needs a vector index")
        if mode == "dense":
            return _top(self.vindex.scores(query), self.chunks, k)
        if mode == "hybrid":
            n = len(self.chunks)
            fused = np.zeros(n)
            for sc in (self.bm25.scores(query), self.vindex.scores(query)):
                rank = np.empty(n, dtype=int)
                rank[np.argsort(-np.asarray(sc), kind="stable")] = np.arange(n)
                fused += 1.0 / (60 + rank + 1)
            return _top(fused, self.chunks, k)
        raise ValueError(f"unknown mode {mode!r}")


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def verify_citation(chunks_by_id: dict[str, dict], chunk_id: str, quote: str) -> tuple[bool, str]:
    """A citation is valid only if the chunk exists and the quote is verbatim in it
    (whitespace and case-insensitive). P2's agent must pass every claim through this."""
    c = chunks_by_id.get(chunk_id)
    if c is None:
        return False, "unknown chunk id"
    if len(_squash(quote)) < 12:
        return False, "quote too short to be evidence"
    if _squash(quote) not in _squash(c["text"]):
        return False, "quote not found verbatim in the cited chunk"
    return True, "ok"
