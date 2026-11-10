import json
from pathlib import Path

import pytest

from aiact_rag.ingest import load

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = load(ROOT / "data" / "processed" / "chunks.jsonl")
BY_ID = {c["id"]: c for c in CHUNKS}
UNITS = {}
for c in CHUNKS:
    UNITS.setdefault(c["unit"], []).append(c)


def test_counts_match_the_official_texts():
    n = lambda doc, kind: len([u for u in UNITS if u.startswith(f"{doc}/{kind}-")])
    assert n("ai-act", "art") == 113
    assert n("ai-act", "rct") == 180
    assert n("ai-act", "anx") == 13
    assert n("gdpr", "art") == 99
    assert n("gdpr", "rct") == 173


def test_ids_unique_and_parts_consistent():
    assert len(BY_ID) == len(CHUNKS)
    for u, cs in UNITS.items():
        assert [c["part"] for c in cs] == list(range(1, len(cs) + 1))
        assert all(c["parts"] == len(cs) for c in cs)


def test_no_empty_and_bounded_chunks():
    assert all(len(c["text"]) > 20 for c in CHUNKS)
    # one paragraph may exceed MAX_CHARS, but nothing is wildly large
    assert max(len(c["text"]) for c in CHUNKS) < 6000


def test_known_article_content():
    t = " ".join(c["text"] for c in UNITS["gdpr/art-33"])
    assert "72 hours" in t
    assert UNITS["ai-act/art-6"][0]["label"].startswith("Article 6 - Classification rules")
    assert "social behaviour" in " ".join(c["text"] for c in UNITS["ai-act/art-5"])


def test_no_oj_footnotes_or_signature():
    import re
    for c in CHUNKS:
        assert not c["text"].startswith("Done at ")
        assert not any(re.match(r"^\(\d+\) .*\(OJ L", ln) for ln in c["text"].split("\n"))
