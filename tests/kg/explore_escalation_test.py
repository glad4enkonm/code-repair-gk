"""Escalation-on-rejection behavior of the KG exploration loop.

A rejected round is one whose tool calls fail extraction or validation.
Consecutive rejections must raise the request temperature (ladder capped
at ``escalation_temp_max``), rotate the feedback phrasing, and after
``rejection_restart_after`` consecutive rejections reset the
conversation to the initial prompt once, granting extra rounds.
"""

import json
import sys
from types import SimpleNamespace

from kg.config import KGConfig
from kg import explore as explore_mod

_BAD_CALL = json.dumps([{"function": {"name": "nonexistent_tool", "arguments": {}}}])
_GOOD_CALL = json.dumps(
    [
        {
            "function": {
                "name": "search_elements",
                "arguments": {"graph": "data", "query": "encoding"},
            }
        }
    ]
)


def _stub_graph_ops():
    """Hermetic replacement for the tool-package graph ops."""

    def _extract_function_calls(text, **kwargs):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = [data]
        return [obj for obj in data if isinstance(obj, dict)]

    def _search_elements(**kwargs):
        return {"success": True, "items": []}

    # explore() unpacks: GRAPH_READ_FUNCS, extract_function_calls
    return {"search_elements": _search_elements}, _extract_function_calls


def _install_fakes(monkeypatch, recorder):
    """Patch OpenAI (via sys.modules — hermetic without openai install)
    and the graph-ops import."""
    fake_openai = SimpleNamespace(OpenAI=_fake_openai(recorder))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(explore_mod, "_import_graph_ops", _stub_graph_ops)


class Recorder:
    """OpenAI-like client serving scripted responses, recording requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        # snapshot messages — explore() keeps mutating the same list
        self.requests.append(
            {**kwargs, "messages": [dict(m) for m in kwargs["messages"]]}
        )
        content = self.responses.pop(0) if self.responses else "stopping now"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _fake_openai(recorder):
    return lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=recorder.create))
    )


def _graph_dir(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "data.json").write_text(
        json.dumps({"nodes": [], "edges": [], "allValues": {}})
    )
    return str(graph_dir)


def _config(**overrides):
    params = dict(
        temperature=0.2,
        max_rounds=4,
        escalation_temp_step=0.15,
        escalation_temp_max=0.5,
        rejection_restart_after=3,
        restart_round_grant=6,
    )
    params.update(overrides)
    return KGConfig(**params)


def test_consecutive_rejections_escalate_temperature(monkeypatch, tmp_path):
    recorder = Recorder([_BAD_CALL, _BAD_CALL, _BAD_CALL])
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(
        _graph_dir(tmp_path), "bug", _config(max_rounds=3, rejection_restart_after=99)
    )
    temps = [r["temperature"] for r in recorder.requests]
    assert temps == [0.2, 0.35, 0.5]


def test_disable_thinking_passes_extra_body_on_every_round(monkeypatch, tmp_path):
    recorder = Recorder([_GOOD_CALL, "stopping now"])
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(_graph_dir(tmp_path), "bug", _config(disable_thinking=True))
    assert recorder.requests, "explore made no LLM calls"
    for request in recorder.requests:
        assert request.get("extra_body") == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


def test_thinking_untouched_by_default(monkeypatch, tmp_path):
    recorder = Recorder([_GOOD_CALL, "stopping now"])
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(_graph_dir(tmp_path), "bug", _config())
    for request in recorder.requests:
        assert "extra_body" not in request


def test_reset_truncates_conversation_and_grants_rounds(monkeypatch, tmp_path):
    recorder = Recorder(
        [_BAD_CALL, _BAD_CALL, _BAD_CALL, _GOOD_CALL, "stopping now", "stopping now"]
    )
    _install_fakes(monkeypatch, recorder)
    touched = explore_mod.explore(_graph_dir(tmp_path), "bug", _config())
    # 6 requests despite max_rounds=4: the reset granted extra rounds
    assert len(recorder.requests) == 6
    # request 4 runs on a fresh conversation (system + bug + restart note)
    assert len(recorder.requests[3]["messages"]) == 3
    # temperature recovers to base after a valid round
    assert recorder.requests[4]["temperature"] == 0.2
    assert touched == {}


def test_feedback_phrasing_rotates_across_rejections(monkeypatch, tmp_path):
    recorder = Recorder([_BAD_CALL, _BAD_CALL, _BAD_CALL])
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(
        _graph_dir(tmp_path), "bug", _config(max_rounds=3, rejection_restart_after=99)
    )
    feedback_1 = recorder.requests[1]["messages"][-1]["content"]
    feedback_2 = recorder.requests[2]["messages"][-1]["content"]
    assert feedback_1 != feedback_2


def test_reset_happens_at_most_once(monkeypatch, tmp_path):
    recorder = Recorder([_BAD_CALL] * 20)
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(_graph_dir(tmp_path), "bug", _config(max_rounds=10))
    # 10 budgeted rounds + a single grant of 6 (a reset per streak of 3
    # would give 10 + 6*3 = 28)
    assert len(recorder.requests) == 16
    restart_notes = [
        r
        for r in recorder.requests
        if "Restarting exploration" in r["messages"][-1]["content"]
    ]
    assert len(restart_notes) == 1


def test_happy_path_never_escalates(monkeypatch, tmp_path):
    recorder = Recorder([_GOOD_CALL, "stopping now", "stopping now"])
    _install_fakes(monkeypatch, recorder)
    explore_mod.explore(_graph_dir(tmp_path), "bug", _config())
    temps = [r["temperature"] for r in recorder.requests]
    assert temps == [0.2, 0.2, 0.2]
    for request in recorder.requests:
        for message in request["messages"]:
            assert "Restarting exploration" not in message["content"]


def test_escalation_defaults_on_kgconfig():
    config = KGConfig()
    assert config.escalation_temp_step > 0
    assert config.escalation_temp_max > config.temperature
    assert config.rejection_restart_after >= 2
    assert config.restart_round_grant >= 1
