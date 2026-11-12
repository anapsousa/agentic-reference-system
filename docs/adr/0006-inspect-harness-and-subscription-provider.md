# ADR 0006: measured under Inspect, through a subscription-backed model provider and a thread bridge

Status: accepted (12 Nov 2026)

Context. The plan's P2 done-when is "runs end to end under the lab harness". The lab (agent-reliability-lab) measures with Inspect and runs models on the Claude subscription via `claude -p`, never the paid API. The triage agent is a synchronous LangGraph whose two LLM nodes go through one seam (ADR 0004).

Decision.
1. **Provider.** `claude-code/<full model id>` is an Inspect `ModelAPI` (`src/triage/inspect_provider.py`, registered by entry point) that runs `claude -p` text-only. Isolation and pinning are the lab's own code, imported at a pinned commit (`agent-reliability-lab @ 5538ab8`), not copied: `JUDGE_DISALLOWED_TOOLS`, `is_full_model_id` (aliases refused), `pin_status` (per call, from the CLI's `modelUsage`), `redact`. Usage maps to Inspect's `ModelUsage`; `total_cost` is Claude Code's API-equivalent figure, not billed.
2. **Same packages as the lab.** `inspect-ai==0.3.252` and every shared package are constrained to the lab's `uv.lock` (`[tool.uv] constraint-dependencies`, list in `eval/lab-constraints.txt`). Found the hard way: unconstrained, `agent-client-protocol` resolved to 1.0.0 and Inspect failed to import.
3. **Bridge, not rewrite.** The solver runs the unchanged graph in a worker thread (`asyncio.to_thread`, which copies the sample's context variables); `InspectLLM.complete` schedules `get_model().generate(...)` back on the event loop. Each classifier/citer call is therefore an ordinary Inspect model event of its sample (usage, timing, retries, pin metadata in the log). A test asserts two model events per sample.
4. **Static concurrency.** Inspect 0.3.252 defaults to adaptive concurrency starting at 20, which overrides a provider's own default; the task sets `max_connections=4` so per-sample latency is comparable across runs.
5. **Scorers judge what the user sees.** `citations_verified` re-parses the rendered answer and re-runs `verify_citation` over the whole corpus, independently of the agent's own flags. `tier_correct` needs the expected tier and a verified basis in an expected provision. `status_appropriate` scores needless escalation as a miss (safe, but it costs a reviewer's time).

Rejected. Inspect's own tool loop driving a port of the prompts (the lab's research-agent task does this): it would measure a different agent. A `claude_code.py`-style runner outside Inspect: loses Inspect's log, viewer and scorer/metric machinery that P3 builds on. The paid Anthropic provider: against the subscription-first rule.

Consequences. The 14 cases in `eval/triage_cases.jsonl` drove two parsing fixes on 12 Nov (chunk-id resolution, see `graph._resolve`), so they are now a development set. P3 needs a held-out set written before its first run.
