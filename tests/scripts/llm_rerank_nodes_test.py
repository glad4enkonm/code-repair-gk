"""Tests for llm-rerank-nodes.py (node-level listwise rerank)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "llm-rerank-nodes.py"
_spec = importlib.util.spec_from_file_location("lrn", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_LRF = importlib.util.spec_from_file_location(
    "lrf_shared", Path(_SCRIPT).parent / "llm-rerank-files.py"
)
_lrf = importlib.util.module_from_spec(_LRF)
_LRF.loader.exec_module(_lrf)


class FakeClient:
    """OpenAI-like client returning canned completion contents in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0) if self.responses else "[]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _span(node_id, path, start, end, source="x" * 200):
    return _mod.ncr.NodeSpan(
        node_id=node_id,
        file=path,
        kind="Function",
        line_start=start,
        line_end=end,
        source_code=source,
    )


def _signal(node_id, freq, **kwargs):
    base = {
        "node_id": node_id,
        "path": "f.py",
        "kind": "Function",
        "name": node_id.rsplit("-", 1)[-1],
        "line_start": 1,
        "line_end": 9,
        "touch_freq": freq,
        "head": "def " + node_id,
    }
    base.update(kwargs)
    return base


# --- signal spec ---


def test_parse_node_signals_subset_and_order():
    assert _mod.parse_node_signals("sim,freq") == ["freq", "sim"]
    assert _mod.parse_node_signals("freq") == ["freq"]
    try:
        _mod.parse_node_signals("parts")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_parse_node_signals_empty_defaults_to_all():
    assert _mod.parse_node_signals("") == ["freq", "sim"]


# --- signal assembly ---


def test_assemble_node_signals_orders_and_fills_fields():
    nodes = [
        _span("Function-f.py-a", "f.py", 1, 10, source="def a():\n    pass"),
        _span("Function-f.py-b", "f.py", 20, 30),
        _span("Class-f.py-C", "f.py", 40, 50),
        _span("Function-h.py-z", "h.py", 1, 5),
    ]
    row = {
        "problem_statement": "fix the bug",
        "touched_nodes": [
            "Function-f.py-a",
            "Function-f.py-a",
            "Class-f.py-C",
            "Other-f.py-x",
        ],
    }
    signals = _mod.assemble_node_signals(
        row,
        graph_dir="unused",
        top_files=["f.py"],
        src_chars=8,
        sim_provider=lambda ps, gd: {"Function-f.py-b": 0.5},
        nodes=nodes,
    )
    assert [s["node_id"] for s in signals] == [
        "Function-f.py-a",
        "Class-f.py-C",
        "Function-f.py-b",
    ]
    assert signals[0]["touch_freq"] == 2
    assert signals[2]["touch_freq"] == 0
    assert signals[2]["sim"] == 0.5
    assert "sim" not in signals[0]
    assert signals[0]["head"] == "def a():"
    assert signals[0]["kind"] == "Function"
    assert signals[0]["path"] == "f.py"


def test_assemble_node_signals_without_sim_provider_omits_sim():
    nodes = [_span("Function-f.py-a", "f.py", 1, 10)]
    signals = _mod.assemble_node_signals(
        {"problem_statement": "ps", "touched_nodes": []},
        graph_dir="unused",
        top_files=["f.py"],
        nodes=nodes,
    )
    assert signals[0]["touch_freq"] == 0
    assert "sim" not in signals[0]


# --- prompt rendering ---


def test_render_node_block_and_prompt_content():
    signal = _signal("Function-f.py-a", 3, sim=0.7, line_start=45, line_end=78)
    block = _mod._render_node_block(signal, ["freq", "sim"])
    assert "### Function-f.py-a" in block
    assert "touch_freq: 3" in block
    assert "sim: 0.7000" in block
    assert "L45-L78" in block
    assert "path: f.py" in block
    prompt = _mod.build_node_list_prompt("the issue", [signal], ["freq", "sim"], 10)
    assert "top 10" in prompt
    assert "the issue" in prompt
    assert "Function-f.py-a" in prompt
    no_freq = _mod._render_node_block(signal, ["sim"])
    assert "touch_freq" not in no_freq


# --- chunking ---


def test_chunk_node_signals_splits_on_char_cap():
    signals = [_signal(f"Function-f.py-n{i}", i, head="h" * 300) for i in range(4)]
    chunks = _mod._chunk_node_signals(
        "issue", signals, ["freq"], 10, max_prompt_chars=800
    )
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    flat = [s["node_id"] for chunk in chunks for s in chunk]
    assert flat == [s["node_id"] for s in signals]


def test_chunk_node_signals_single_chunk_under_cap():
    signals = [_signal("Function-f.py-a", 1)]
    chunks = _mod._chunk_node_signals(
        "issue", signals, ["freq"], 10, max_prompt_chars=100_000
    )
    assert len(chunks) == 1 and len(chunks[0]) == 1


# --- list ranking ---


def test_rank_nodes_list_filters_and_pads_with_freq_tail():
    signals = [
        _signal("Function-f.py-a", 5),
        _signal("Function-f.py-b", 3),
        _signal("Function-f.py-c", 1),
    ]
    engine = _lrf.ListRanker(
        client=FakeClient(['["Function-f.py-b", "not-a-node"]']),
        model="m",
    )
    result = _mod.rank_nodes_list("issue", signals, 2, engine, ["freq"], 0)
    assert result["failed"] is False
    assert result["ranked"] == [
        "Function-f.py-b",
        "Function-f.py-a",
        "Function-f.py-c",
    ]
    assert result["n_calls"] == 1


def test_rank_nodes_list_failure_keeps_freq_order():
    signals = [_signal("Function-f.py-a", 5), _signal("Function-f.py-b", 3)]
    engine = _lrf.ListRanker(
        client=FakeClient(["garbage", "garbage", "garbage"]), model="m"
    )
    result = _mod.rank_nodes_list("issue", signals, 2, engine, ["freq"], 0)
    assert result["failed"] is True
    assert result["ranked"] == ["Function-f.py-a", "Function-f.py-b"]
    assert result["n_calls"] == 3


def test_rank_nodes_list_single_candidate_short_circuits():
    signals = [_signal("Function-f.py-a", 1)]
    engine = _lrf.ListRanker(client=FakeClient([]), model="m")
    result = _mod.rank_nodes_list("issue", signals, 2, engine, ["freq"], 0)
    assert result == {"ranked": ["Function-f.py-a"], "failed": False, "n_calls": 0}
    assert engine.client.calls == []


# --- progress checkpointing ---


def test_load_nodes_rerank_and_default_path():
    assert _mod.load_nodes_rerank("/nonexistent/file.jsonl") == {}
    assert _mod.node_signals_tag(["freq", "sim"]) == "freq-sim"
    path = _mod.default_node_progress_path("dev", "org/model", ["freq", "sim"], 3, 10)
    assert "node_rerank__list__freq-sim__f3__k10__dev__org-model" in str(path)


# --- GT nodes + evaluation ---


def _graph_fixture(tmp_path):
    graph_dir = tmp_path / "r__c1"
    graph_dir.mkdir()
    (graph_dir / "data.json").write_text(
        json.dumps(
            {
                "nodes": [{"id": "Function-f.py-fn"}],
                "edges": [],
                "allValues": {
                    "Function-f.py-fn": {
                        "meta_type": "Function",
                        "source_code": "def fn(): pass",
                        "line_start": 1,
                        "line_end": 5,
                    }
                },
            }
        )
    )
    return graph_dir


def test_gt_nodes_uses_overlap(tmp_path):
    graph_dir = _graph_fixture(tmp_path)
    row = {
        "instance_id": "i1",
        "patch": "--- a/f.py\n+++ b/f.py\n@@ -2,2 +2,2 @@\n-old\n+new\n",
    }
    gt = _mod.gt_nodes_for_row(
        row,
        str(graph_dir),
        node_paths={"Function-f.py-fn": "f.py"},
    )
    assert gt == {"Function-f.py-fn"}


def test_evaluate_nodes_recall_and_cover():
    rerank = {
        "i1": ["n1", "n2", "n3"],
        "i2": ["n9", "n7"],
        "i3": ["n1"],
    }
    gt = {"i1": {"n1"}, "i2": {"n7", "n8"}, "i3": set()}
    summary = _mod.evaluate_nodes(rerank, gt, ks=(1, 3))
    assert summary["n_instances"] == 3
    assert summary["n_with_gt"] == 2
    assert summary["n_empty_gt"] == 1
    assert summary["recall@1"] == 0.5  # i1 hit at 1, i2 only at 3
    assert summary["recall@3"] == 1.0
    assert summary["cover@1"] == 0.5  # i1 fully covered at 1
    assert summary["cover@3"] == 0.5  # i2 never (n8 absent from ranking)


# --- processing loop ---


def test_process_rows_checkpoint_and_skip(tmp_path):
    progress = tmp_path / "progress.jsonl"
    rows = [
        {
            "instance_id": "i1",
            "repo": "r",
            "base_commit": "c1",
            "problem_statement": "ps",
            "patch": "",
            "touched_nodes": [],
        }
    ]
    _graph_fixture(tmp_path)
    signals = [
        {
            "node_id": "Function-f.py-fn",
            "touch_freq": 1,
            "path": "f.py",
            "kind": "Function",
            "name": "fn",
            "line_start": 1,
            "line_end": 5,
            "head": "def fn",
        },
    ]
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _lrf.ListRanker(client=FakeClient(['["Function-f.py-fn"]']), model="m")

    stats = _mod.process_rows(
        rows,
        ranked_files_by_id={"i1": ["f.py"]},
        engine_factory=factory,
        progress_path=str(progress),
        graphs_dir=str(tmp_path),
        file_k=3,
        top_k=5,
        signals=["freq"],
        src_chars=120,
        max_prompt_chars=0,
        sim_provider=None,
        model="m",
        node_overrides={"i1": signals},
    )
    assert stats["processed"] == 1
    record = json.loads(progress.read_text().strip())
    assert record["ranked_nodes"] == ["Function-f.py-fn"]
    assert record["n_nodes"] == 1
    stats2 = _mod.process_rows(
        rows,
        ranked_files_by_id={"i1": ["f.py"]},
        engine_factory=factory,
        progress_path=str(progress),
        graphs_dir=str(tmp_path),
        file_k=3,
        top_k=5,
        signals=["freq"],
        src_chars=120,
        max_prompt_chars=0,
        sim_provider=None,
        model="m",
    )
    assert stats2["skipped"] == 1 and stats2["processed"] == 0
    assert calls["n"] == 1
