"""The Inspect wiring, with no model call: Inspect's mockllm plays the classifier and citer,
retrieval is BM25 over the real chunks, and the eval runs through `inspect_ai.eval` exactly as
`inspect eval eval/inspect_task.py` does."""
import json
from pathlib import Path

import pytest
from evals.judges.run_llm_judge import JUDGE_DISALLOWED_TOOLS
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput, get_model

from triage.inspect_eval import citations_in
from triage.inspect_provider import ClaudeCodeAPI, claude_argv, render, to_output

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/processed/chunks.jsonl").open()}


@pytest.fixture(autouse=True)
def _at_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # Inspect resolves task files relative to the working directory


def q(cid: str, start: int = 0, n: int = 60) -> str:
    return " ".join(CHUNKS[cid]["text"].split())[start:start + n]


def run_eval(tmp_path, case: dict, outputs: list[str]):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n")
    model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs])
    [log] = inspect_eval("eval/inspect_task.py", model=model, log_dir=str(tmp_path / "logs"),
                         task_args={"retriever": "bm25", "cases": str(cases)}, display="none")
    assert log.status == "success", log.error
    return log


CV = {"id": "x1", "kind": "clear", "expected_tier": "high", "basis_units": ["ai-act/anx-III", "ai-act/art-6"],
      "description": "We use a CV screening tool that ranks job applicants automatically for our recruiters."}


def classifier(cid="ai-act/anx-III#2", quote=None, tier="high"):
    return json.dumps({"tier": tier, "confidence": 0.9, "borderline": False, "rationale": "Annex III employment.",
                       "basis": {"chunk_id": cid, "quote": quote or q(cid, 0, 50)}})


def citer(*items):
    return json.dumps({"obligations": [{"text": t, "chunk_id": c, "quote": qq} for t, c, qq in items]})


def scores(log):
    [s] = log.samples
    return s, {k: v.value for k, v in s.scores.items()}


def test_end_to_end_under_inspect(tmp_path):
    log = run_eval(tmp_path, CV, [classifier(), citer(("Risk management system", "ai-act/art-9#1", q("ai-act/art-9#1", 0, 50)))])
    s, v = scores(log)
    assert v == {"tier_correct": "C", "citations_verified": "C", "status_appropriate": "C", "safety": "C"}
    assert s.store["triage"]["status"] == "answered" and s.store["triage"]["llm_calls"] == 2
    # both graph LLM calls are Inspect model events of this sample, so usage/latency come from Inspect
    assert sum(1 for e in s.events if e.event == "model") == 2
    assert any(e.event == "info" and e.source == "triage graph" for e in s.events)
    assert s.scores["citations_verified"].metadata["n_citations"] == 2  # basis + one obligation


def test_wrong_basis_provision_scores_tier_wrong(tmp_path):
    # right tier word, but grounded in Art 9 (a duty), not the classification rule
    log = run_eval(tmp_path, CV, [classifier(cid="ai-act/art-9#1"), citer(("Risk management", "ai-act/art-9#1", q("ai-act/art-9#1", 0, 50)))])
    _, v = scores(log)
    assert v["tier_correct"] == "I" and v["citations_verified"] == "C"


def test_thin_case_must_not_be_answered(tmp_path):
    thin = {"id": "x2", "kind": "thin", "expected_tier": None, "basis_units": [],
            "description": "We use an AI tool to help our team make decisions about people more quickly."}
    log = run_eval(tmp_path, thin, [json.dumps({"tier": "undetermined", "confidence": 0.2, "borderline": True,
                                                "rationale": "too little detail", "basis": {}})])
    s, v = scores(log)
    assert s.store["triage"]["status"] == "escalated"
    assert v["tier_correct"] == "C" and v["status_appropriate"] == "C"


def test_citations_parser_reads_what_the_user_sees():
    ans = 'Triage result\n\nProposed tier: high (confidence 0.90). r\nBasis: "a quote" [ai-act/anx-III#2]\n' \
          'Obligations (each quote verified verbatim):\n- Keep logs [ai-act/art-12#1] "shall technically allow"'
    assert citations_in(ans) == [("ai-act/anx-III#2", "a quote"), ("ai-act/art-12#1", "shall technically allow")]


# --- provider ------------------------------------------------------------------------------

def test_provider_argv_disables_every_lab_listed_tool():
    argv = claude_argv("claude-haiku-5-5", "sys")
    denied = argv[argv.index("--disallowedTools") + 1].split(",")
    assert set(JUDGE_DISALLOWED_TOOLS) <= set(denied)
    for flag in ("--strict-mcp-config", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--setting-sources") + 1] == "" and argv[argv.index("--tools") + 1] == ""


def test_provider_refuses_alias():
    with pytest.raises(ValueError, match="full model id"):
        ClaudeCodeAPI("haiku")
    ClaudeCodeAPI("haiku", allow_alias=True)
    ClaudeCodeAPI("claude-haiku-5-5")


def test_render_single_and_multi_turn():
    assert render([ChatMessageSystem(content="S"), ChatMessageUser(content="U")]) == ("S", "U")
    sys_, prompt = render([ChatMessageUser(content="a"), ChatMessageAssistant(content="b"), ChatMessageUser(content="c")])
    assert sys_ == "" and prompt == "USER:\na\n\nASSISTANT:\nb\n\nUSER:\nc"


def test_to_output_maps_usage_and_pin():
    env = {"result": "ok", "total_cost_usd": 0.0008, "duration_api_ms": 2014,
           "usage": {"input_tokens": 2, "output_tokens": 4, "cache_creation_input_tokens": 3274, "cache_read_input_tokens": 649},
           "modelUsage": {"claude-haiku-5-5": {}}}
    out = to_output("claude-haiku-5-5", env, 1.5)
    assert out.completion == "ok" and out.usage.output_tokens == 4 and out.usage.input_tokens_cache_write == 3274
    assert out.usage.total_tokens == 2 + 4 + 3274 + 649 and out.metadata["model_pin"] == "pinned"
    assert to_output("claude-haiku-5-5", {**env, "modelUsage": {"claude-sonnet-5-5": {}}}, 1).metadata["model_pin"] == "mismatch"
