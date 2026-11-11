# ADR 0004: one LLM seam; subscription CLI live, scripted fake in tests; checkpointed state as memory

Status: accepted

LLM seam. `LLM.complete(system, user) -> str`. Tests inject `FakeLLM` (scripted JSON keyed by role), so the policy in ADR 0003 is tested deterministically with no model, over the real statute text. Live runs use `ClaudeCLI`: `claude -p` with all tools disabled, no settings, no MCP, no saved session, empty working directory (the same isolation the lab runner uses), on the subscription. No API key or paid service anywhere.

Memory. LangGraph `MemorySaver`, one thread per case: the full state (candidates, proposals, rejections, trace) is kept and inspectable per case. That is the audit substrate P6 will build on. There is deliberately no long-term cross-case memory: nothing to learn from yet, and any cache of past answers would need its own verification story. In-process only; it does not survive a restart.

Rejected. Calling the Anthropic API directly (cost and key handling, against the subscription-first rule).
