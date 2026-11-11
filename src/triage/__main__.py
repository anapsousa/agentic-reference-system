"""python -m triage "<system description>" [--llm claude] [--model haiku] [--retriever bm25|hybrid]

Prints the answer and writes the node trace (JSON) with --trace. Live use runs `claude -p`
on the subscription; token usage is printed to stderr."""
import argparse
import json
import sys
from pathlib import Path

from aiact_rag.index import DEFAULT_MODEL, VectorIndex, fastembed_fns
from aiact_rag.ingest import load
from aiact_rag.retrieve import Retriever

from .graph import build_graph, run_case
from .llm import ClaudeCLI

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--retriever", choices=("bm25", "hybrid"), default="hybrid")
    ap.add_argument("--cache", default=None, help="fastembed model cache dir")
    ap.add_argument("--trace", default=None, help="write the full final state as JSON")
    a = ap.parse_args()
    chunks = load(ROOT / "data" / "processed" / "chunks.jsonl")
    vindex = None
    if a.retriever == "hybrid":
        _, query = fastembed_fns(DEFAULT_MODEL, a.cache)
        vindex = VectorIndex.load(chunks, ROOT / "data" / "index", query)  # build: python eval/run.py --build
    llm = ClaudeCLI(model=a.model)
    state = run_case(build_graph(chunks, Retriever(chunks, vindex), llm, mode=a.retriever), a.description, "cli")
    print(state["answer"])
    print(f"\nstatus={state['status']} usage={json.dumps(llm.usage)}", file=sys.stderr)
    if a.trace:
        Path(a.trace).write_text(json.dumps({**state, "usage": llm.usage}, indent=1))


main()
