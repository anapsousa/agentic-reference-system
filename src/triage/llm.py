"""The only seam to a language model. Tests inject FakeLLM; live runs use the `claude` CLI
on the subscription (no API key, no paid service)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass
class FakeLLM:
    """Scripted replies, matched by a marker the node puts in the system prompt
    (``ROLE: classifier`` / ``ROLE: citer``). Records every call."""
    replies: dict[str, str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        for role, reply in self.replies.items():
            if f"ROLE: {role}" in system:
                return reply
        raise AssertionError(f"no scripted reply for system prompt: {system[:80]!r}")


_DENY = "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent,Task,NotebookEdit,TodoWrite"


@dataclass
class ClaudeCLI:
    """`claude -p` with every tool disabled, no settings, no MCP, no session saved,
    run from an empty directory so no CLAUDE.md or memory reaches the model."""
    model: str = "haiku"
    timeout: int = 180
    usage: dict = field(default_factory=lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                                                "api_equivalent_usd": 0.0})

    def complete(self, system: str, user: str) -> str:
        argv = ["claude", "-p", "--model", self.model, "--system-prompt", system,
                "--tools", "", "--disallowedTools", _DENY, "--strict-mcp-config",
                "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
        with tempfile.TemporaryDirectory() as empty:
            p = subprocess.run(argv, input=user, capture_output=True, text=True, timeout=self.timeout, cwd=empty)
        if p.returncode != 0:
            raise RuntimeError(f"claude exited {p.returncode}: {p.stderr[:300]}")
        out = json.loads(p.stdout)
        u = out.get("usage", {})
        self.usage["calls"] += 1
        self.usage["input_tokens"] += u.get("input_tokens", 0)
        self.usage["output_tokens"] += u.get("output_tokens", 0)
        self.usage["cache_read_tokens"] += u.get("cache_read_input_tokens", 0)
        self.usage["cache_creation_tokens"] += u.get("cache_creation_input_tokens", 0)
        self.usage["api_equivalent_usd"] += out.get("total_cost_usd", 0.0)
        if out.get("is_error"):
            raise RuntimeError(f"claude reported an error: {str(out.get('result'))[:300]}")
        return out.get("result", "")


def parse_json(text: str) -> dict | None:
    """First JSON object in a reply (models wrap it in prose or fences). None if absent/invalid."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None
