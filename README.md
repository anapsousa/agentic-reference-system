# agentic-reference-system

Reference system for the "agentic AI principal" build track: an **AI Act compliance triage agent**. Input: a plain-language description of an AI system. Output (planned): a proposed risk tier, obligations with verifiable citations, and escalation to a human when confidence is low.

**Status: P1 (retrieval), P2 (triage agent under the lab's Inspect harness) and P3 (five-dimension evaluation, pass^3, gate).** Private repo.

## What exists (P1)

- `src/aiact_rag/ingest.py` parses the EUR-Lex HTML of the AI Act (Reg. 2024/1689) and the GDPR (Reg. 2016/679) into 775 citable chunks (articles, annexes, recitals; long units split at paragraph boundaries). Ids look like `ai-act/art-6#1`. Raw HTML is in `data/raw/` (public text, fetched 10 Nov 2026), parsed output in `data/processed/chunks.jsonl`. Tests pin the unit counts to the official texts (113 + 13 annexes + 180 recitals; 99 + 173).
- `index.py`: file-based flat cosine index (`vectors.npy` + `meta.json` with model and corpus hash; refuses a stale index). Embeddings are local (fastembed, `BAAI/bge-small-en-v1.5`, ONNX, no API, no cost). A flat exact index is the honest choice at ~800 chunks; there is no vector database.
- `bm25.py`, `retrieve.py`: BM25 baseline, dense, and hybrid (reciprocal rank fusion). `search(..., kinds=("art","anx"))` restricts to operative text.
- `retrieve.verify_citation(chunks_by_id, chunk_id, quote)`: a citation is valid only if the chunk exists and the quote is verbatim in it. This is the anti-fabrication gate P2's agent must pass every claim through.
- `eval/gold.jsonl`: 44 questions (plain-language system descriptions and direct legal questions), each with the relevant article(s)/annex and a verbatim evidence quote. Tests check every quote really is in the cited text.
- `eval/run.py` writes `reports/retrieval-eval.{md,json}`.

## What exists (P2)

`src/triage/`: a LangGraph, `intake -> retrieve -> classify -> cite -> verify -> consistency -> decide -> answer`. Input: a plain-language system description. Output: a proposed risk tier (prohibited, high, limited, minimal, undetermined), obligations each carrying a chunk id and a verbatim quote, and a status: `answered`, `escalated` (human review needed) or `refused`. Every quote is checked with P1's `verify_citation` and must come from text the model was shown; unverifiable ones are dropped, and if nothing verifies the case is refused. The answer is rendered by code from verified items only. Design choices are in `docs/adr/` (0001 to 0005).

- `python -m pytest`: 91 tests, none calls a model (scripted fake LLM or Inspect's mockllm, real statute text).
- Live: `python -m triage "<description>" --cache <fastembed-cache> --trace out.json` runs `claude -p` on the subscription (tools disabled). One smoke case is recorded in `reports/smoke-2026-11-11.md` with token usage.
- Needs `langgraph` in addition to the P1 dependencies, and an index (`python eval/run.py --build --cache <dir>`) for the default hybrid retriever.

## Under the lab harness (P2 done-when)

`eval/inspect_task.py` runs the unchanged graph as an Inspect solver over 14 tier-labelled cases (`eval/triage_cases.jsonl`), with three scorers: `tier_correct`, `citations_verified` (independent re-check of every quote in the answer) and `status_appropriate`. The model is `claude-code/<full id>`, an Inspect provider in this repo that runs `claude -p` on the subscription with the lab's isolation and model-pinning code, imported from `agent-reliability-lab` at a pinned commit; shared packages are constrained to the lab's lockfile. Design: `docs/adr/0006`. Three runs on 12 Nov (haiku-5-5): tier 12, 13, 13 of 14; citations 14/14 every run; injection case never minimal. Details and failures: `reports/inspect-2026-11-12.md`.

```
uv sync --extra eval                  # venv off the exFAT volume: UV_PROJECT_ENVIRONMENT=<internal dir>
uv run inspect eval eval/inspect_task.py --model claude-code/claude-haiku-5-5 -T cache=<fastembed-cache>
uv run inspect view                   # browse the logs (logs/ is gitignored)
```

## P3: five dimensions, pass^3, gate (see `reports/p3-eval-2026-11-12.md`, `docs/adr/0007`)

Held-out set `eval/triage_test.jsonl` (29 cases, frozen before any P3 change), 3 epochs, haiku-5-5:

| Quality | Safety | Reliability | Latency | Cost |
|---|---|---|---|---|
| tier 99%, citations 100%, answer-vs-escalate 80% | 0 unsafe of 87; 5/5 injections held every epoch | pass^3 all layers 62% (tier alone 97%) | p50 13 s, p95 19 s per triage | $0.006 (EUR 0.0054) per triage, API-equivalent |

The weakness is needless escalation (borderline flags on clear cases), not wrong answers. 99% tier accuracy means the set is too easy: it has no contested edges yet.

```
uv run inspect eval eval/inspect_task.py --model claude-code/claude-haiku-5-5 --epochs 3 -T cache=<dir> [-T cases=eval/triage_test.jsonl]
uv run python -m triage.p3 report logs/<dev>.eval logs/<test>.eval --json reports/p3 --out tables.md --usd-eur <rate>
uv run python -m triage.p3 gate --base origin/main     # also in CI (.github/workflows/ci.yml)
```

## Result (see `reports/retrieval-eval.md`)

| retriever | hit@1 | hit@5 | recall@10 | MRR | citation@1 |
|---|---|---|---|---|---|
| bm25 | 0.32 | 0.70 | 0.77 | 0.47 | 0.27 |
| hybrid (all chunks) | 0.30 | 0.75 | 0.87 | 0.49 | 0.25 |
| dense, articles+annexes only | 0.73 | 0.98 | 0.96 | 0.84 | 0.55 |
| hybrid, articles+annexes only | 0.73 | 0.93 | 0.96 | 0.82 | 0.55 |

Main finding: recitals paraphrase the law in plain language and outrank the operative articles on natural-language queries (hit@1 0.30 with them, 0.73 without). Excluding recitals is the single design change made after seeing results; the gold set was written before the first run. `citation@1` is stricter than `hit@1` because the evidence quote must sit in the top-ranked chunk itself, not in another part of the same article.

## Run

```
python -m venv <somewhere-off-the-exFAT-volume> && pip install numpy fastembed pytest
python -m pytest                      # 91 tests, no model download needed
python src/aiact_rag/ingest.py        # rebuild chunks.jsonl from data/raw
python eval/run.py --build --cache <model-cache-dir>
```

## NOT done (read before trusting the numbers)

P2 gaps:
- **Held-out set: 29 LLM-written cases, no lawyer review, clear centre of each tier only.** The 14 P2 cases are a dev set. The P3 test set has now been seen: any escalation-policy change made from its results needs a fresh held-out set (test v2) with the contested edges (Art 6(3), Annex I, GPAI).
- **Needless escalation**: about one triage in five escalates a clear case on a borderline flag. Safe, costly, untuned on purpose (ADR 0007).
- One model, k = 3. The gate's floors are policy lines (about ±17 points of noise per run on 29 cases), not significance tests.
- The eval extra inherits the lab's lockfile, including the fsspec advisory held back by `inspect-ai==0.3.252`; bumping Inspect is a lab decision first.
- **Quotes are verified; claims only partly.** A deterministic gate (ADR 0005) checks numbers, durations, percentages, amounts, article/annex references and modal strength of each paraphrase against the quote or cited chunk. It does not check meaning: negation, scope and subject can still be wrong.
- No clarifying question, no second retrieval, no cross-reference following, no long-term memory, no human-approval UI (escalation is a status, not a workflow).
- The 0.6 confidence threshold is a guess, uncalibrated. Prompt injection: 5 direct-injection cases held in P3; indirect injection via retrieved text and tool abuse are P5. Tier "limited" and GPAI are coarsely handled.

P1 gaps:

- The gold set is 44 questions written by an LLM from its own knowledge of the Acts, with quotes machine-checked against the text, but **not reviewed by a human lawyer**. Relevance is judged at article level and some questions have several defensible answers (q39, q41, q44). Treat it as a smoke-test-grade set, not a benchmark.
- 44 questions: differences of a few points are noise. No confidence intervals.
- Only one embedding model, no chunk-size or reranker experiments, no cross-encoder. Dense vs hybrid is not distinguishable here.
- Cross-references are not followed (Art 6 points to Annex III; the retriever does not).
- Consolidated texts and later amendments (for example digital-omnibus changes) are not tracked; the corpus is the OJ version retrieved on the date above.
- CI runs tests and the eval gate on committed reports; it never calls a model. No Docker. No LLM call anywhere in P1.
