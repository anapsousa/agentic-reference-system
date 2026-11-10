"""File-based vector index: a float32 matrix (.npy) + chunk list + meta.json.

A flat, exact cosine index. At ~800 chunks a vector database adds nothing but
install weight; the interface (``search``) is what a later swap would keep.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .ingest import embed_text

EmbedFn = Callable[[Sequence[str]], np.ndarray]
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def fastembed_fns(model: str = DEFAULT_MODEL, cache_dir: str | None = None):
    """Return (embed_docs, embed_query) backed by fastembed (local ONNX, $0)."""
    from fastembed import TextEmbedding

    m = TextEmbedding(model_name=model, cache_dir=cache_dir)

    def docs(texts):
        return np.array(list(m.embed(list(texts))), dtype=np.float32)

    def query(texts):
        return np.array(list(m.query_embed(list(texts))), dtype=np.float32)

    return docs, query


def _norm(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def corpus_hash(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["id"].encode() + b"\0" + c["text"].encode() + b"\0")
    return h.hexdigest()[:16]


def build_index(chunks: list[dict], embed_docs: EmbedFn, out_dir: Path, model: str = DEFAULT_MODEL) -> None:
    mat = _norm(embed_docs([embed_text(c) for c in chunks]).astype(np.float32))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", mat)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": model, "dim": int(mat.shape[1]), "n": int(mat.shape[0]),
        "corpus_hash": corpus_hash(chunks), "metric": "cosine (flat, exact)",
    }, indent=2))


class VectorIndex:
    def __init__(self, chunks: list[dict], mat: np.ndarray, embed_query: EmbedFn):
        if len(chunks) != mat.shape[0]:
            raise ValueError("chunk count does not match index rows")
        self.chunks, self.mat, self.embed_query = chunks, mat, embed_query

    @classmethod
    def load(cls, chunks: list[dict], index_dir: Path, embed_query: EmbedFn) -> "VectorIndex":
        meta = json.loads((index_dir / "meta.json").read_text())
        if meta["corpus_hash"] != corpus_hash(chunks):
            raise ValueError("index is stale: corpus changed since it was built; rebuild")
        return cls(chunks, np.load(index_dir / "vectors.npy"), embed_query)

    def scores(self, query: str) -> np.ndarray:
        q = _norm(self.embed_query([query]).astype(np.float32))[0]
        return self.mat @ q
