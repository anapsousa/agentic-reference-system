# ADR 0008: the same agent in CrewAI, measured by the same eval

Status: accepted (24 Nov 2026)

Context. P4 asks for one port to a second framework and the same eval, no framework tourism. ADR 0001 made the agent a fixed pipeline so the comparison is fair. CrewAI is chosen over AutoGen: its Flow gives explicit, ordered steps (a like-for-like of the LangGraph graph), AutoGen's conversational group chat does not.

Decision.
- **Shared code, not a rewrite.** `graph.make_nodes` returns the eight node functions (retriever, classifier, citer, verifier, consistency gate, escalation policy, renderer). LangGraph wires them into a StateGraph; CrewAI wires the same functions into a `crewai.flow.Flow` (`src/triage/crewai_flow.py`). Refactor was behaviour-neutral: 91 tests pass before and after.
- **CrewAI owns** orchestration (Flow with @start/@listen/@router) and the two model roles, each run as an Agent + Task + Crew.
- **Two environments.** CrewAI 1.15.45 cannot be installed next to the lab's pinned `inspect-ai==0.3.252` (regex, tenacity, tiktoken pins). The port runs as a worker in `.venv-crewai`; every model call it makes is sent as a JSON line to the Inspect process and answered by Inspect's model. Inspect therefore records usage, latency, retries and model pin exactly as for LangGraph, and the four scorers and the P3 report are unchanged. Run: `-T framework=crewai`.
- **Prompts.** CrewAI folds the system prompt into one user message and adds its own framing ("You are classifier.", "expected criteria", "Provide your complete response"). The worker sends the agent's own system/user split instead, so the ports differ in orchestration only. What CrewAI composed is stored in the trace (`messages_sent`). A run with CrewAI's native prompts (`TRIAGE_CREWAI_NATIVE=1`) was not done; it would measure prompt format, not the framework.
- **Test set.** v2 (32 cases, contested edges), committed before any P4 run (8ba6865). Model claude-haiku-5-5, 3 epochs, 4 connections, one run each, same day.

Result (reports/p4). Single runs of 32 cases: one run's 95% interval is roughly +-15 points, so differences below that are not evidence.

| | LangGraph | CrewAI |
|---|---|---|
| tier correct (mean of 3) | 90% | 86% |
| citations verified | 100% | 100% |
| status appropriate | 60% | 52% |
| pass^3, all layers | 14/32 = 44% | 13/32 = 41% |
| unsafe answers | 5 of 96 | 4 of 96 |
| latency p50 / p95, epochs 2-3 | 12.5 / 19.6 s | 12.9 / 21.6 s |
| cost per triage, epochs 2-3 (API-equivalent) | $0.0046 | $0.0045 |
| run wall time | 5 min 26 s | 7 min 51 s |

Reading. No measurable quality difference between the frameworks. The ordering of the flips differs (LangGraph 11 cases flip on some layer, CrewAI 7) but both are inside the noise. CrewAI costs: a second environment, a subprocess protocol, a slower start (the first epoch's p95 was 69 s against 20 s for LangGraph: four workers each loading CrewAI and the embedder; epoch 1 cost figures are also distorted by prompt-cache writes, so latency and cost above use epochs 2-3 only; the all-epoch figures are in the reports). What it bought: nothing the fixed pipeline needed.

Both ports fail the P3 safety floor on v2 (`unsafe_max` 0): v21, v24 and v27, the thin descriptions, are answered outright in some epochs. That is the agent's, not the framework's: the same nodes, the same prompts. It is the input to the P3b escalation-policy work and is not fixed here.

Consequence. Do not claim CrewAI is better or worse; claim the pipeline moved with its measured behaviour intact. Limits: one model, one run each, LLM-written cases, CrewAI native prompts untested, AutoGen not tried.
