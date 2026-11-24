"""The CrewAI worker speaks the JSON-lines protocol and runs the whole pipeline (no live model)."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-crewai" / "bin" / "python"
pytestmark = pytest.mark.skipif(not PY.exists(), reason="no .venv-crewai")

CLS = json.dumps({"tier": "minimal", "confidence": 0.9, "borderline": False, "rationale": "x",
                  "basis": {"chunk_id": "nope", "quote": "nope"}})


def test_worker_runs_pipeline_and_keeps_prompt_split():
    p = subprocess.Popen([str(PY), "-m", "triage.crewai_flow", "--mode", "bm25"], cwd=ROOT, text=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert json.loads(p.stdout.readline())["type"] == "ready"

    def send(o):
        p.stdin.write(json.dumps(o) + "\n")
        p.stdin.flush()

    send({"type": "case", "id": "t", "description": "A start-up sells software that scores job applicants from their CVs."})
    roles = []
    while True:
        m = json.loads(p.stdout.readline())
        if m["type"] == "llm":
            assert m["system"].startswith("ROLE: ")  # the agent's own system prompt, not CrewAI's folded one
            roles.append(m["system"].split("\n")[0])
            send({"type": "llm_result", "rid": m["rid"], "text": CLS if "classifier" in roles[-1] else '{"obligations": []}'})
        else:
            break
    p.stdin.close()
    nodes = [t["node"] for t in m["state"]["trace"]]
    assert roles == ["ROLE: classifier", "ROLE: citer"]
    assert nodes[:8] == ["intake", "retrieve", "classify", "cite", "verify", "consistency", "decide", "answer"]
    assert m["state"]["trace"][-1]["messages_sent"], "CrewAI's composed messages must be kept for audit"
