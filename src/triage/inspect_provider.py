"""Inspect model provider `claude-code/<model-id>`: `claude -p` on the Claude subscription.

    inspect eval eval/inspect_task.py --model claude-code/claude-haiku-5-5

Text-only (no tool calls: the triage graph owns its tools). Isolation and model pinning are
the lab's, imported at the pinned commit rather than copied: every built-in tool disabled by
name (`JUDGE_DISALLOWED_TOOLS`), no MCP server, no settings, no saved session, an empty
working directory, a full model id required (`is_full_model_id`), and whether the run
measured the model it asked for recorded per call (`pin_status`).

`total_cost` in the Inspect log is Claude Code's API-equivalent figure. Nothing is billed:
the calls run on the subscription.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time

from evals.judges.run_llm_judge import JUDGE_DISALLOWED_TOOLS
from evals.runners.claude_code import is_full_model_id, pin_status, redact
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

DEFAULT_TIMEOUT = 300


def claude_argv(model: str, system: str) -> list[str]:
    return [
        "claude", "-p", "--model", model, "--system-prompt", system,
        "--tools", "", "--disallowedTools", ",".join(JUDGE_DISALLOWED_TOOLS),
        "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
        "--output-format", "json",
    ]


def render(messages: list[ChatMessage]) -> tuple[str, str]:
    """System messages become the system prompt. One user turn goes on stdin as is; a longer
    conversation is rendered as a labelled transcript (claude -p takes one prompt)."""
    system = "\n\n".join(m.text for m in messages if isinstance(m, ChatMessageSystem))
    turns = [m for m in messages if not isinstance(m, ChatMessageSystem)]
    if len(turns) == 1:
        return system, turns[0].text
    return system, "\n\n".join(f"{m.role.upper()}:\n{m.text}" for m in turns)


def to_output(model: str, envelope: dict, elapsed: float) -> ModelOutput:
    u = envelope.get("usage", {})
    inp, out = u.get("input_tokens", 0), u.get("output_tokens", 0)
    cw, cr = u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0)
    seen = sorted(envelope.get("modelUsage", {}))
    output = ModelOutput.from_content(model=model, content=envelope.get("result", ""))
    output.usage = ModelUsage(
        input_tokens=inp, output_tokens=out, total_tokens=inp + out + cw + cr,
        input_tokens_cache_write=cw, input_tokens_cache_read=cr,
        total_cost=envelope.get("total_cost_usd"),
    )
    output.time = elapsed
    output.metadata = {
        "models_seen": seen,
        "model_pin": pin_status(model, seen),
        "cost_basis": "api-equivalent, not billed (subscription)",
        "duration_api_ms": envelope.get("duration_api_ms"),
    }
    return output


@modelapi(name="claude-code")
class ClaudeCodeAPI(ModelAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig | None = None, allow_alias: bool = False,
                 timeout: int = DEFAULT_TIMEOUT, **model_args) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config or GenerateConfig())
        if not allow_alias and not is_full_model_id(model_name):
            raise ValueError(f"claude-code/{model_name}: pass a full model id (claude-haiku-5-5), "
                             "not an alias; aliases move without notice (-M allow_alias=true to override)")
        self.timeout = timeout

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo],
                       tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        if tools:
            raise ValueError("claude-code provider is text-only; tool calls are not supported")
        system, prompt = render(input)
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as empty:
            proc = await asyncio.create_subprocess_exec(
                *claude_argv(self.model_name, system), cwd=empty,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), self.timeout)
            except TimeoutError:
                proc.kill()
                raise
        if proc.returncode != 0:
            raise RuntimeError(f"claude exited {proc.returncode}: {redact(stderr.decode()[:500])}")
        envelope = json.loads(stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude reported an error: {redact(str(envelope.get('result'))[:500])}")
        envelope["result"] = redact(envelope.get("result", ""))
        return to_output(self.model_name, envelope, time.monotonic() - start)

    def should_retry(self, ex: BaseException) -> bool:
        return isinstance(ex, TimeoutError)

