"""Retrieval evaluation: recall@k, hit@k, MRR and citation correctness.

Relevance is judged at the *unit* (article / annex) level: a hit on any part of
a relevant article counts. Gold lives in eval/gold.jsonl with a verbatim
evidence quote per question (checked by tests).

Citation correctness (what P2's agent will be held to): the top-ranked chunk is
cited with the gold evidence quote; the citation is *correct* only if (a) the
chunk is in a relevant unit and (b) `verify_citation` accepts the quote for it.
Condition (b) is the anti-fabrication check; (a) is whether retrieval pointed
the agent at the right law. ``citation_at_1`` therefore equals precision@1
restricted to chunks that actually contain the evidence.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .retrieve import Retriever, verify_citation

KS = (1, 3, 5, 10)


def load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_question(g: dict, hits, by_id: dict, ks=KS) -> dict:
    rel = set(g["relevant"])
    units = [h.unit for h in hits]
    out: dict = {"id": g["id"]}
    for k in ks:
        got = set(units[:k]) & rel
        out[f"hit@{k}"] = 1.0 if got else 0.0
        out[f"recall@{k}"] = len(got) / len(rel)
    out["rr"] = next((1.0 / (i + 1) for i, u in enumerate(units) if u in rel), 0.0)
    top = hits[0] if hits else None
    ok = False
    if top is not None and top.unit in rel:
        ok, _ = verify_citation(by_id, top.id, g["evidence"])
    out["citation@1"] = 1.0 if ok else 0.0
    out["top1"] = top.id if top else None
    return out


def evaluate(retriever: Retriever, gold: list[dict], mode: str, ks=KS, kinds=None) -> dict:
    by_id = {c["id"]: c for c in retriever.chunks}
    rows = [score_question(g, retriever.search(g["q"], k=max(ks), mode=mode, kinds=kinds), by_id, ks) for g in gold]
    keys = [f"{m}@{k}" for m in ("hit", "recall") for k in ks] + ["rr", "citation@1"]
    summary = {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys}
    summary["mrr"] = summary.pop("rr")
    return {"mode": mode, "kinds": kinds, "n": len(rows), "summary": summary, "rows": rows}
