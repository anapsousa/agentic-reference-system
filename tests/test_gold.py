import json
from pathlib import Path

from aiact_rag.ingest import load
from aiact_rag.retrieve import _squash

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = load(ROOT / "data" / "processed" / "chunks.jsonl")
UNITS = {}
for c in CHUNKS:
    UNITS.setdefault(c["unit"], []).append(c)
GOLD = [json.loads(l) for l in (ROOT / "eval" / "gold.jsonl").read_text().splitlines() if l.strip()]


def test_gold_is_nonempty_and_ids_unique():
    assert len(GOLD) >= 40
    assert len({g["id"] for g in GOLD}) == len(GOLD)


def test_every_relevant_unit_exists():
    for g in GOLD:
        for u in g["relevant"]:
            assert u in UNITS, (g["id"], u)


def test_evidence_quote_is_verbatim_in_a_relevant_unit():
    """The gold is only as good as its evidence: the quote must really be in the cited text."""
    bad = []
    for g in GOLD:
        text = _squash(" ".join(c["text"] for u in g["relevant"] for c in UNITS[u]))
        if _squash(g["evidence"]) not in text:
            bad.append(g["id"])
    assert not bad, bad
