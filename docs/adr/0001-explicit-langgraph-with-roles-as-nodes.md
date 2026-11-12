# ADR 0001: one explicit LangGraph; roles are nodes, not free-roaming agents

Status: accepted (11 Nov 2026)

Context. The plan asks for retriever, classifier and reviewer roles, tool use, memory and human oversight, in LangGraph. The risk in a compliance setting is not capability, it is an unauditable path.

Decision. A fixed graph `intake -> retrieve -> classify -> cite -> verify -> decide -> answer`. Retriever and verifier are code (P1 modules). Classifier and citer are the only model calls (two, separate prompts, so a wrong tier cannot silently shape the citations beyond the tier name). The reviewer is `decide`, a deterministic escalation policy, not a third model: its rules (confidence below 0.6, borderline flag, unverifiable tier basis, no verified obligation) are testable and explainable to an auditor. Intake can short-circuit to `answer` with a refusal before any model call.

Rejected. A tool-calling ReAct loop: more flexible, but the number and order of steps becomes model-dependent, which makes pass^k and cost per case noisy and the trace harder to audit. If P4 ports this to a second framework, the fixed pipeline also makes the comparison fair.

Consequence. The agent cannot ask a follow-up question or retrieve a second time. That is a known limit (see README, NOT done).
