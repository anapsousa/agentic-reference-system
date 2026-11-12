"""Inspect task: the P2 triage agent, end to end, under the agent-reliability-lab harness.

    inspect eval eval/inspect_task.py --model claude-code/claude-haiku-5-5 -T cache=<fastembed-cache>
    inspect eval eval/inspect_task.py --model claude-code/claude-haiku-5-5 -T retriever=bm25

`claude-code/<full model id>` is this repo's Inspect provider (src/triage/inspect_provider.py):
`claude -p` on the subscription, isolated and model-pinned with the lab's own code. The cases
are a dev set (eval/triage_cases.jsonl, 14, tuned on in P2) or the held-out test set
(-T cases=eval/triage_test.jsonl, 29, frozen in commit 364cfad before any P3 change). Both were
written by an LLM, not reviewed by a lawyer. P3 runs both with --epochs 3; triage.p3 turns the
logs into the 5-dimension report and the gate (docs/adr/0007).
"""
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from aiact_rag.index import DEFAULT_MODEL, VectorIndex, fastembed_fns
from aiact_rag.ingest import load
from aiact_rag.retrieve import Retriever
from triage.inspect_eval import citations_verified, load_cases, safety, status_appropriate, tier_correct, triage_agent

ROOT = Path(__file__).resolve().parents[1]


@task
def triage(retriever: str = "hybrid", cache: str | None = None, cases: str = "eval/triage_cases.jsonl",
           max_connections: int = 4) -> Task:
    """Static concurrency (Inspect's default is adaptive, starting at 20): gentler on the
    subscription, and latency figures stay comparable from run to run."""
    chunks = load(ROOT / "data" / "processed" / "chunks.jsonl")
    vindex = None
    if retriever == "hybrid":
        _, query = fastembed_fns(DEFAULT_MODEL, cache)
        vindex = VectorIndex.load(chunks, ROOT / "data" / "index", query)  # build: python eval/run.py --build
    return Task(
        dataset=load_cases(ROOT / cases),
        solver=triage_agent(chunks, Retriever(chunks, vindex), mode=retriever),
        scorer=[tier_correct(), citations_verified(chunks), status_appropriate(), safety(chunks)],
        config=GenerateConfig(max_connections=max_connections),
        metadata={"agent": "triage", "retriever": retriever, "cases": cases},
    )
