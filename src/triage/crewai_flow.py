"""P4: the triage agent ported to CrewAI. Runs in `.venv-crewai` as a worker (docs/adr/0008).

Same pipeline as P2/P3, same eight node functions (`graph.make_nodes`: retriever, verifier,
consistency gate, escalation policy and answer renderer are code and are not re-written).
What CrewAI owns here:
  * orchestration: a `crewai.flow.Flow` with @start/@listen/@router steps, in place of the LangGraph StateGraph;
  * the two model roles: classifier and citer each run as a CrewAI Agent + Task + Crew, in place of a bare
    `llm.complete` call. Their system prompts are the same text (as the Agent backstory) and the user prompt
    is the Task description; the Agent's prompt/system templates are set to pass them through unwrapped.
What CrewAI does not change: the model. Every call leaves through `RPCLLM`, a CrewAI `BaseLLM`, to the parent
process, which answers it with Inspect's model (see inspect_crewai.py). The messages CrewAI actually sent are
recorded in the trace so any prompt difference the framework introduces is auditable, not assumed away.

    python -m triage.crewai_flow --mode hybrid [--cache DIR]      # speaks the JSON-lines protocol on stdin/stdout
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

# Keep the protocol channel clean: CrewAI and its dependencies print to stdout.
_PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import crewai  # noqa: E402
from crewai import Agent, BaseLLM, Crew, Task  # noqa: E402
from crewai.flow.flow import Flow, listen, or_, router, start  # noqa: E402

from aiact_rag.index import DEFAULT_MODEL, VectorIndex, fastembed_fns  # noqa: E402
from aiact_rag.ingest import load  # noqa: E402
from aiact_rag.retrieve import Retriever  # noqa: E402

from .graph import make_nodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_rid = itertools.count(1)
# CrewAI folds the system prompt into one user message and adds its own framing. By default the worker still
# sends the agent's own system/user split so the two ports differ in orchestration only; set TRIAGE_CREWAI_NATIVE=1
# to send what CrewAI composes instead (a separate, labelled run).
NATIVE_PROMPTS = os.environ.get("TRIAGE_CREWAI_NATIVE") == "1"
SEEN: list[dict] = []  # what CrewAI actually sent to the model, for the trace


def _send(obj: dict) -> None:
    _PROTO_OUT.write(json.dumps(obj) + "\n")
    _PROTO_OUT.flush()


def _rpc(system: str, user: str) -> str:
    rid = next(_rid)
    _send({"type": "llm", "rid": rid, "system": system, "user": user})
    reply = json.loads(sys.stdin.readline())
    if reply.get("rid") != rid:
        raise RuntimeError(f"protocol error: expected rid {rid}, got {reply}")
    if reply.get("error"):
        raise RuntimeError(f"model error: {reply['error']}")
    return reply["text"]


class RPCLLM(BaseLLM):
    """A CrewAI LLM whose only job is to hand the prompt to the parent process."""

    override: Any = None

    def __init__(self) -> None:
        super().__init__(model="inspect-bridge", temperature=None)

    def call(self, messages: str | list[dict], tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        system = "\n\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "system")
        user = "\n\n".join(str(m.get("content", "")) for m in msgs if m.get("role") != "system")
        SEEN.append({"system": system, "user": user})  # what CrewAI composed (audit trail)
        if NATIVE_PROMPTS or self.override is None:
            return _rpc(system, user)
        return _rpc(*self.override)  # the agent's own system/user split, byte for byte (ADR 0008)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 200_000


class CrewLLM:
    """The graph's `LLM` seam, answered by one CrewAI Agent + Task + Crew per call."""

    def __init__(self) -> None:
        self.rpc = RPCLLM()
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        role = "citer" if "ROLE: citer" in system else "classifier"
        agent = Agent(role=role, goal="Answer in the exact format the instructions give.", backstory=system,
                      llm=self.rpc, allow_delegation=False, verbose=False, max_iter=1,
                      system_template="{{ .System }}", prompt_template="{{ .Prompt }}", response_template="{{ .Response }}")
        task = Task(description=user, expected_output="One JSON object, as the instructions specify.", agent=agent)
        self.rpc.override = (system, user)
        out = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        self.calls += 1
        return out.raw


class TriageFlow(Flow):
    """intake -> (retrieve -> classify -> cite -> verify -> consistency -> decide | refuse) -> answer"""

    def __init__(self, nodes: dict, description: str):
        super().__init__()
        self.nodes = nodes
        self.s: dict[str, Any] = {"description": description}

    def _run(self, name: str) -> None:
        self.s.update(self.nodes[name](self.s))

    @start()
    def intake(self):
        self._run("intake")

    @router(intake)
    def gate(self):
        return "refused" if self.s.get("status") == "refused" else "accepted"

    @listen("accepted")
    def retrieve(self):
        self._run("retrieve")

    @listen(retrieve)
    def classify(self):
        self._run("classify")

    @listen(classify)
    def cite(self):
        self._run("cite")

    @listen(cite)
    def verify(self):
        self._run("verify")

    @listen(verify)
    def consistency(self):
        self._run("consistency")

    @listen(consistency)
    def decide(self):
        self._run("decide")

    @listen(or_(decide, "refused"))
    def answer(self):
        self._run("answer")


def run_case(nodes: dict, description: str) -> dict:
    flow = TriageFlow(nodes, description)
    flow.kickoff()
    return flow.s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    chunks = load(ROOT / "data" / "processed" / "chunks.jsonl")
    vindex = None
    if a.mode == "hybrid":
        _, query = fastembed_fns(DEFAULT_MODEL, a.cache)
        vindex = VectorIndex.load(chunks, ROOT / "data" / "index", query)
    llm = CrewLLM()
    nodes = make_nodes(chunks, Retriever(chunks, vindex), llm, mode=a.mode)
    _send({"type": "ready", "crewai": crewai.__version__})
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] != "case":
            continue
        SEEN.clear()
        state = run_case(nodes, msg["description"])
        state["trace"] = state.get("trace", []) + [{"node": "crewai", "messages_sent": SEEN[:]}]
        _send({"type": "done", "id": msg["id"], "state": state, "framework": f"crewai {crewai.__version__}"})


if __name__ == "__main__":
    main()
