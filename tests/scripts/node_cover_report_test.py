"""Tests for node-cover-report.py (hunk→node ceiling + min-token floor)."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "node-cover-report.py"
_spec = importlib.util.spec_from_file_location("ncr", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _counter():
    tcr = importlib.util.spec_from_file_location(
        "tcr", Path(_SCRIPT).parent / "token-cost-report.py"
    )
    mod = importlib.util.module_from_spec(tcr)
    tcr.loader.exec_module(mod)
    return mod.make_counter(None)


def _node(nid, file, start, end, source="x" * 40):
    return _mod.NodeSpan(
        node_id=nid,
        file=file,
        kind="Function",
        line_start=start,
        line_end=end,
        source_code=source,
    )


# --- patch parsing ---


def test_parse_patch_hunks_ranges_and_counts():
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -12,7 +12,9 @@ def a():\n"
        " ctx\n"
        "-old\n"
        "+new\n"
        "@@ -5 +5,2 @@\n"
        "-x\n"
        "+y\n"
        "+z\n"
    )
    hunks = _mod.parse_patch_hunks(patch)
    assert [(h.file, h.start, h.end, h.count) for h in hunks] == [
        ("foo.py", 12, 18, 7),
        ("foo.py", 5, 5, 1),
    ]


def test_parse_patch_hunks_zero_count_insertion():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -10,0 +11,3 @@\n+a\n+b\n+c\n"
    hunks = _mod.parse_patch_hunks(patch)
    assert len(hunks) == 1
    assert (hunks[0].start, hunks[0].end, hunks[0].count) == (10, 10, 0)


def test_parse_patch_hunks_new_file_flagged():
    patch = "--- /dev/null\n+++ b/new_mod.py\n@@ -0,0 +1,5 @@\n+a\n+b\n"
    hunks = _mod.parse_patch_hunks(patch)
    assert len(hunks) == 1
    assert hunks[0].is_new_file is True


def test_parse_patch_hunks_deleted_file_uses_old_path():
    patch = "--- a/gone.py\n+++ /dev/null\n@@ -3,4 +1,2 @@\n-a\n-b\n"
    hunks = _mod.parse_patch_hunks(patch)
    assert hunks[0].file == "gone.py"
    assert hunks[0].is_new_file is False


def test_parse_patch_hunks_multiple_files():
    patch = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n"
        "--- a/b.py\n+++ b/b.py\n@@ -7,3 +7,3 @@\n-p\n+q\n"
    )
    hunks = _mod.parse_patch_hunks(patch)
    assert [h.file for h in hunks] == ["a.py", "b.py"]
    assert hunks[1].start == 7


def test_parse_patch_hunks_skips_binary():
    patch = "--- a/img.png\n+++ b/img.png\nGIT binary patch\nliteral 10\n"
    assert _mod.parse_patch_hunks(patch) == []


# --- containment / overlap semantics ---


def test_node_contains_regular_and_zero_width():
    node = _node("n", "f.py", 45, 78)
    inside = _mod.Hunk("f.py", 50, 60, 11, False)
    poking = _mod.Hunk("f.py", 40, 60, 21, False)
    assert _mod.node_contains(node, inside) is True
    assert _mod.node_contains(node, poking) is False
    at_last_line = _mod.Hunk("f.py", 78, 78, 1, False)
    assert _mod.node_contains(node, at_last_line) is True


def test_node_contains_zero_width_insertion_strictly_inside():
    node = _node("n", "f.py", 5, 20)
    mid = _mod.Hunk("f.py", 10, 10, 0, False)
    at_end_boundary = _mod.Hunk("f.py", 20, 20, 0, False)
    assert _mod.node_contains(node, mid) is True
    assert _mod.node_contains(node, at_end_boundary) is False


def test_node_overlaps_intersection():
    node = _node("n", "f.py", 10, 20)
    assert _mod.node_overlaps(node, _mod.Hunk("f.py", 20, 25, 6, False)) is True
    assert _mod.node_overlaps(node, _mod.Hunk("f.py", 21, 25, 5, False)) is False


# --- classification ---


def test_classify_hunks_contained_partial_outside_new():
    nodes = [
        _node("fn", "f.py", 45, 78),
        _node("cl", "f.py", 100, 140),
    ]
    hunks = [
        _mod.Hunk("f.py", 50, 60, 11, False),
        _mod.Hunk("f.py", 95, 110, 16, False),
        _mod.Hunk("f.py", 1, 3, 3, False),
        _mod.Hunk("other.py", 1, 1, 1, False),
    ]
    result = _mod.classify_hunks(hunks, nodes)
    assert [r.cls for r in result] == [1, 2, 3, 3]
    assert result[0].covering == ("fn",)
    assert result[1].overlapping == ("cl",)
    assert result[3].covering == ()
    assert result[3].cls == 3  # no nodes for that file → outside


def test_classify_hunks_new_file_gets_new_class():
    nodes = [_node("fn", "f.py", 1, 10)]
    hunks = [_mod.Hunk("new.py", 0, 4, 5, True)]
    result = _mod.classify_hunks(hunks, nodes)
    assert result[0].cls == "new"
    assert result[0].covering == ()


# --- greedy min-token cover ---


def _cls(hunk, cls, covering=()):
    return _mod.HunkClassification(hunk=hunk, cls=cls, covering=covering)


def test_min_token_cover_prefers_better_ratio():
    nodes = {
        "big": _node("big", "f.py", 1, 100, source="b" * 400),
        "a": _node("a", "f.py", 10, 20, source="a" * 160),
        "b": _node("b", "f.py", 30, 40, source="c" * 160),
    }
    h1 = _mod.Hunk("f.py", 10, 12, 3, False)
    h2 = _mod.Hunk("f.py", 30, 33, 4, False)
    classifications = [_cls(h1, 1, ("big", "a")), _cls(h2, 1, ("big", "b"))]
    result = _mod.min_token_cover(
        classifications,
        list(nodes.values()),
        lambda n: _counter().count(n.source_code),
    )
    assert sorted(result["chosen"]) == ["a", "b"]
    assert result["tokens"] == 80  # 2 × (160 chars / 4)
    assert result["covered_hunks"] == 2
    assert result["fallback_hunks"] == 0


def test_min_token_cover_counts_fallback_hunks():
    nodes = [_node("fn", "f.py", 45, 78)]
    hunks = [
        _mod.Hunk("f.py", 50, 60, 11, False),
        _mod.Hunk("f.py", 95, 110, 16, False),
        _mod.Hunk("new.py", 0, 4, 5, True),
    ]
    classifications = [
        _cls(hunks[0], 1, ("fn",)),
        _cls(hunks[1], 2),
        _cls(hunks[2], "new"),
    ]
    result = _mod.min_token_cover(classifications, nodes, lambda n: 10)
    assert result["chosen"] == ["fn"]
    assert result["covered_hunks"] == 1
    assert result["fallback_hunks"] == 2


# --- pool variants ---


def test_pool_nodes_filters_by_touched_and_top_files():
    nodes = [
        _node("Function-f.py-in", "f.py", 1, 10),
        _node("Function-f.py-out", "f.py", 20, 30),
        _node("Function-g.py-in", "g.py", 1, 10),
    ]
    pool = _mod.pool_nodes(
        nodes,
        touched_ids={"Function-f.py-in", "Function-g.py-in"},
        top_files=["f.py"],
    )
    assert [n.node_id for n in pool] == ["Function-f.py-in"]


def test_pool_variants_file_k_and_touched_and_baseline():
    nodes = [
        _node("Function-f.py-a", "f.py", 1, 10),
        _node("Function-f.py-b", "f.py", 20, 30),
        _node("Function-g.py-c", "g.py", 1, 10),
    ]
    variants = _mod.pool_variants(
        nodes,
        touched_ids={"Function-f.py-a", "Function-g.py-c"},
        ranked_files=["f.py", "g.py", "h.py"],
        file_ks=(1, 3),
    )
    assert [n.node_id for n in variants["file1"]] == [
        "Function-f.py-a",
        "Function-f.py-b",
    ]
    assert [n.node_id for n in variants["file3"]] == [
        "Function-f.py-a",
        "Function-f.py-b",
        "Function-g.py-c",
    ]
    assert [n.node_id for n in variants["touched"]] == [
        "Function-f.py-a",
        "Function-g.py-c",
    ]
    assert [n.node_id for n in variants["touched_file5"]] == [
        "Function-f.py-a",
        "Function-g.py-c",
    ]


# --- graph loader ---


def _write_graph(tmp_path, nodes, values):
    graph_dir = tmp_path / "r__c1"
    graph_dir.mkdir()
    (graph_dir / "data.json").write_text(
        json.dumps({"nodes": nodes, "edges": [], "allValues": values})
    )
    return graph_dir


def test_load_graph_nodes_with_injected_paths(tmp_path):
    nodes = [
        {"id": "Function-f.py-fn"},
        {"id": "File-f.py"},
        {"id": "Class-f.py-C"},
    ]
    values = {
        "Function-f.py-fn": {
            "meta_type": "Function",
            "source_code": "def fn(): pass",
            "line_start": 1,
            "line_end": 2,
        },
        "File-f.py": {"meta_type": "File", "relative_path": "f.py"},
        "Class-f.py-C": {
            "meta_type": "Class",
            "source_code": "class C: ...",
            "line_start": 5,
        },
    }
    graph_dir = _write_graph(tmp_path, nodes, values)
    paths = {
        "Function-f.py-fn": "f.py",
        "File-f.py": "f.py",
        "Class-f.py-C": "f.py",
    }
    loaded = _mod.load_graph_nodes(str(graph_dir), node_paths=paths)
    assert len(loaded) == 1
    assert loaded[0].node_id == "Function-f.py-fn"
    assert loaded[0].line_end == 2
    assert loaded[0].kind == "Function"


# --- report assembly ---


def _kg_row(iid, patch, touched, repo="r", commit="c1"):
    return {
        "instance_id": iid,
        "repo": repo,
        "base_commit": commit,
        "patch": patch,
        "touched_nodes": touched,
    }


def test_build_report_and_markdown(tmp_path):
    patch = "--- a/f.py\n+++ b/f.py\n@@ -2,2 +2,2 @@\n-old\n+new\n"
    nodes = [{"id": "Function-f.py-fn"}]
    values = {
        "Function-f.py-fn": {
            "meta_type": "Function",
            "source_code": "def fn(): pass",
            "line_start": 1,
            "line_end": 5,
        }
    }
    _write_graph(tmp_path, nodes, values)
    rows = [_kg_row("i1", patch, ["Function-f.py-fn"])]
    report = _mod.build_report(
        rows,
        rerank_ranked={"i1": ["f.py"]},
        graphs_dir=str(tmp_path),
        counter=_counter(),
        path_resolver=lambda gd: {"Function-f.py-fn": "f.py"},
    )
    inst = report["instances"][0]
    assert inst["instance_id"] == "i1"
    assert inst["n_hunks"] == 1
    assert inst["class_counts"][1] == 1
    assert inst["floor_oracle"]["covered_hunks"] == 1
    for name in ("file1", "file3", "file5", "touched", "touched_file5"):
        variant = inst["variants"][name]
        assert variant["floor"]["chosen"] == ["Function-f.py-fn"]
        assert variant["pool_recall"] == 1.0
    agg = report["aggregates"]
    assert agg["cls1_hunk_share"] == 1.0
    assert agg["n_instances"] == 1
    assert set(agg["variants"]) == {
        "file1",
        "file3",
        "file5",
        "touched",
        "touched_file5",
    }
    md = _mod.render_markdown(report)
    assert "Node-cover report" in md
    assert "i1" in md
    assert "file5" in md and "touched" in md
