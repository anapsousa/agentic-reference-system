"""Run the retrieval eval for bm25 / dense / hybrid and write the report.

Usage:  python eval/run.py [--build]   (--build (re)builds data/index first)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aiact_rag.evaluate import KS, evaluate, load_gold  # noqa: E402
from aiact_rag.index import DEFAULT_MODEL, VectorIndex, build_index, fastembed_fns  # noqa: E402
from aiact_rag.ingest import load  # noqa: E402
from aiact_rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=None, help="fastembed model cache dir")
    a = ap.parse_args()
    chunks = load(ROOT / "data" / "processed" / "chunks.jsonl")
    gold = load_gold(ROOT / "eval" / "gold.jsonl")
    idx_dir = ROOT / "data" / "index"
    docs, query = fastembed_fns(a.model, a.cache)
    if a.build or not (idx_dir / "meta.json").exists():
        build_index(chunks, docs, idx_dir, a.model)
    r = Retriever(chunks, VectorIndex.load(chunks, idx_dir, query))
    OP = ("art", "anx")
    results = {m: evaluate(r, gold, m) for m in ("bm25", "dense", "hybrid")}
    results.update({m + "+operative": evaluate(r, gold, m, kinds=OP) for m in ("bm25", "dense", "hybrid")})
    meta = json.loads((idx_dir / "meta.json").read_text())
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "retrieval-eval.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    cols = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr", "citation@1"]
    lines = [
        "# P1 retrieval evaluation",
        "",
        f"Run: {datetime.date.today().isoformat()} | embedding model: `{meta['model']}` ({meta['dim']}d, {meta['metric']}) "
        f"| chunks: {meta['n']} | gold questions: {len(gold)}",
        "",
        "| retriever | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for m, res in results.items():
        lines.append(f"| {m} | " + " | ".join(f"{res['summary'][c]:.2f}" for c in cols) + " |")
    lines += ["", "`+operative` = recitals excluded (articles and annexes only). Recitals paraphrase the law in plain",
              "language and outrank the operative text on natural-language queries; an obligation must be cited from an article.",
              "", "## Misses at rank 1 (hybrid+operative)", ""]
    for row in results["hybrid+operative"]["rows"]:
        if row["hit@1"] == 0:
            q = next(g for g in gold if g["id"] == row["id"])
            lines.append(f"- {row['id']} \"{q['q']}\" -> top1 `{row['top1']}`, wanted {q['relevant']}, hit@5={row['hit@5']:.0f}")
    (out / "retrieval-eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
