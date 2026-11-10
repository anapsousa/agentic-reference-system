import numpy as np
import pytest

from aiact_rag.bm25 import BM25, tokenize
from aiact_rag.evaluate import score_question
from aiact_rag.index import VectorIndex, build_index, corpus_hash
from aiact_rag.retrieve import Hit, Retriever, verify_citation

CH = [
    {"id": "d/art-1#1", "unit": "d/art-1", "doc": "d", "kind": "art", "label": "Article 1 - Fines", "text": "Administrative fines apply to infringements."},
    {"id": "d/art-2#1", "unit": "d/art-2", "doc": "d", "kind": "art", "label": "Article 2 - Breach", "text": "A personal data breach must be notified within 72 hours."},
    {"id": "d/art-3#1", "unit": "d/art-3", "doc": "d", "kind": "art", "label": "Article 3 - Oversight", "text": "Humans must oversee the high-risk system."},
]
BY_ID = {c["id"]: c for c in CH}


def stub_embed(texts):
    # deterministic bag-of-words hashing embedder, enough to exercise the index
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            out[i, hash(w) % 64] += 1.0
    return out


def test_tokenize_drops_stopwords():
    assert tokenize("The breach of the data") == ["breach", "data"]


def test_bm25_ranks_matching_doc_first():
    s = BM25([c["text"] for c in CH]).scores("breach notified 72 hours")
    assert int(np.argmax(s)) == 1


def test_index_roundtrip_and_stale_detection(tmp_path):
    build_index(CH, stub_embed, tmp_path, model="stub")
    vi = VectorIndex.load(CH, tmp_path, stub_embed)
    assert int(np.argmax(vi.scores("humans oversee"))) == 2
    changed = [dict(CH[0], text="different")] + CH[1:]
    with pytest.raises(ValueError, match="stale"):
        VectorIndex.load(changed, tmp_path, stub_embed)
    assert corpus_hash(CH) != corpus_hash(changed)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_all_modes_return_k_hits(tmp_path, mode):
    build_index(CH, stub_embed, tmp_path, model="stub")
    r = Retriever(CH, VectorIndex.load(CH, tmp_path, stub_embed))
    hits = r.search("data breach 72 hours", k=2, mode=mode)
    assert len(hits) == 2 and hits[0].id == "d/art-2#1"


def test_dense_without_index_is_an_error():
    with pytest.raises(ValueError):
        Retriever(CH).search("x", mode="dense")


def test_verify_citation_accepts_verbatim_and_rejects_fabrication():
    ok, _ = verify_citation(BY_ID, "d/art-2#1", "must be notified  within 72 HOURS")
    assert ok
    assert not verify_citation(BY_ID, "d/art-2#1", "must be notified within 24 hours")[0]
    assert not verify_citation(BY_ID, "d/art-9#1", "must be notified within 72 hours")[0]
    assert not verify_citation(BY_ID, "d/art-2#1", "72")[0]  # too short to be evidence


def test_score_question_metrics():
    g = {"id": "x", "q": "q", "relevant": ["d/art-2"], "evidence": "notified within 72 hours"}
    hits = [Hit(CH[0], 1.0), Hit(CH[1], 0.9), Hit(CH[2], 0.1)]
    r = score_question(g, hits, BY_ID)
    assert r["hit@1"] == 0 and r["hit@3"] == 1 and r["rr"] == 0.5 and r["citation@1"] == 0
    r2 = score_question(g, hits[1:] + hits[:1], BY_ID)
    assert r2["hit@1"] == 1 and r2["citation@1"] == 1


def test_citation_requires_quote_to_be_in_chunk_not_just_right_unit():
    g = {"id": "x", "q": "q", "relevant": ["d/art-2"], "evidence": "not in this chunk at all"}
    r = score_question(g, [Hit(CH[1], 1.0)], BY_ID)
    assert r["hit@1"] == 1 and r["citation@1"] == 0


def test_kinds_filter_excludes_recitals():
    ch = CH + [{"id": "d/rct-1#1", "unit": "d/rct-1", "doc": "d", "kind": "rct", "label": "Recital 1", "text": "data breach 72 hours notified breach breach"}]
    r = Retriever(ch)
    assert r.search("data breach 72 hours", k=1, mode="bm25")[0].id == "d/rct-1#1"
    assert r.search("data breach 72 hours", k=1, mode="bm25", kinds=("art", "anx"))[0].id == "d/art-2#1"
