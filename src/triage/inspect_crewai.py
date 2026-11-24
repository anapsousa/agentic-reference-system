"""P4: the CrewAI port of the triage agent, measured by the same Inspect task and scorers as P3.

CrewAI cannot share an environment with the lab's pinned Inspect (docs/adr/0008), so the port runs
in its own venv (`.venv-crewai`) as a worker process. The worker never talks to a model itself.
Every model call it wants is sent here as a JSON line, answered by Inspect's active model, and the
text goes back. So usage, latency, retries and model pinning are recorded by Inspect exactly as for
the LangGraph agent, and the scorers read the same store fields.

Wire format, one JSON object per line:
    parent -> worker   {"type": "case", "id": ..., "description": ...}
                       {"type": "llm_result", "rid": n, "text": ...} or {"type": "llm_result", "rid": n, "error": ...}
    worker -> parent   {"type": "llm", "rid": n, "system": ..., "user": ...}
                       {"type": "done", "state": {...}, "framework": "crewai x.y.z"}
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState, solver

from .inspect_eval import KEEP, STORE_KEY

ROOT = Path(__file__).resolve().parents[2]
WORKER_PYTHON = ROOT / ".venv-crewai" / "bin" / "python"


class Worker:
    def __init__(self, proc: asyncio.subprocess.Process):
        self.proc = proc

    @classmethod
    async def start(cls, mode: str, cache: str | None) -> "Worker":
        argv = [str(WORKER_PYTHON), "-m", "triage.crewai_flow", "--mode", mode] + (["--cache", cache] if cache else [])
        proc = await asyncio.create_subprocess_exec(*argv, cwd=ROOT, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=sys.stderr, limit=2**24)
        ready = json.loads(await proc.stdout.readline() or b"{}")
        if ready.get("type") != "ready":
            raise RuntimeError(f"crewai worker did not start: {ready}")
        return cls(proc)

    async def send(self, obj: dict) -> None:
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def run_case(self, case_id: str, description: str) -> dict:
        model = get_model()
        await self.send({"type": "case", "id": case_id, "description": description})
        llm_calls = 0
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("crewai worker exited mid-case")
            msg = json.loads(line)
            if msg["type"] == "llm":
                out = await model.generate([ChatMessageSystem(content=msg["system"]), ChatMessageUser(content=msg["user"])])
                llm_calls += 1
                await self.send({"type": "llm_result", "rid": msg["rid"],
                                 **({"error": str(out.error)} if out.error else {"text": out.completion})})
            elif msg["type"] == "done":
                return {**msg, "llm_calls": llm_calls}

    async def close(self) -> None:
        if self.proc.returncode is None:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except TimeoutError:
                self.proc.kill()


@solver
def triage_agent_crewai(mode: str = "hybrid", cache: str | None = None, workers: int = 4):
    """One worker process per concurrent sample (pool), each loaded once."""
    pool: asyncio.Queue[Worker] = asyncio.Queue()
    started = 0
    lock = asyncio.Lock()

    async def acquire() -> Worker:
        nonlocal started
        async with lock:
            if pool.empty() and started < workers:
                started += 1
                return await Worker.start(mode, cache)
        return await pool.get()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal started
        model = get_model()
        w = await acquire()
        try:
            res = await w.run_case(str(state.sample_id), state.input_text)
        except BaseException:
            await w.close()  # a worker that failed mid-case is discarded, not reused
            started -= 1
            raise
        pool.put_nowait(w)
        final = res["state"]
        state.store.set(STORE_KEY, {k: final.get(k) for k in KEEP} | {"llm_calls": res["llm_calls"]})
        transcript().info({"node_trace": final.get("trace", []), "status": final.get("status"),
                           "framework": res.get("framework")}, source="triage crewai flow")
        state.messages.append(ChatMessageAssistant(content=final["answer"], model=model.name))
        state.output = ModelOutput.from_content(model=model.name, content=final["answer"])
        return state

    return solve
