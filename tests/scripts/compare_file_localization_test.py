"""Tests for compare-file-localization.py (metrics, GT extraction, loaders)."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "compare-file-localization.py"
)
_spec = importlib.util.spec_from_file_location("cfl", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


# --- recall_at_k ---
def test_recall_at_k_partial_hit():
    predicted = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    gt = {"a.py", "z.py"}
    assert _mod.recall_at_k(predicted, gt, 1) == 0.5
    assert _mod.recall_at_k(predicted, gt, 3) == 0.5
    assert _mod.recall_at_k(predicted, gt, 5) == 0.5


def test_recall_at_k_full_hit():
    predicted = ["a.py", "b.py", "c.py"]
    gt = {"a.py"}
    assert _mod.recall_at_k(predicted, gt, 1) == 1.0


def test_recall_at_k_no_hit():
    assert _mod.recall_at_k(["x.py"], {"a.py"}, 5) == 0.0


def test_recall_at_k_empty_gt():
    assert _mod.recall_at_k(["a.py"], set(), 5) == 0.0


def test_recall_at_k_multi_gt():
    predicted = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    gt = {"a.py", "b.py", "z.py"}
    # top-1: 1/3, top-3: 2/3, top-5: 2/3
    assert _mod.recall_at_k(predicted, gt, 1) == 1 / 3
    assert _mod.recall_at_k(predicted, gt, 3) == 2 / 3
    assert _mod.recall_at_k(predicted, gt, 5) == 2 / 3


# --- full_coverage_at_k ---
def test_coverage_all_in_top_k():
    predicted = ["a.py", "b.py", "c.py"]
    assert _mod.full_coverage_at_k(predicted, {"a.py"}, 1) is True
    assert _mod.full_coverage_at_k(predicted, {"a.py", "b.py"}, 1) is False
    assert _mod.full_coverage_at_k(predicted, {"a.py", "b.py"}, 3) is True


def test_coverage_empty_gt():
    assert _mod.full_coverage_at_k(["a.py"], set(), 5) is False


# --- unbounded_recall ---
def test_unbounded_recall():
    predicted = ["a.py", "b.py", "c.py"]
    gt = {"a.py", "z.py"}
    assert _mod.unbounded_recall(predicted, gt) == 0.5


def test_unbounded_recall_empty():
    assert _mod.unbounded_recall([], {"a.py"}) == 0.0


# --- gt_files_from_patch ---
def test_gt_files_from_single_file():
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert _mod.gt_files_from_patch(patch) == {"foo.py"}


def test_gt_files_from_multi_file():
    patch = (
        "+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        "+++ b/bar/baz.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert _mod.gt_files_from_patch(patch) == {"foo.py", "bar/baz.py"}


def test_gt_files_empty_patch():
    assert _mod.gt_files_from_patch("") == set()


# --- _is_readme ---
def test_is_readme_variants():
    assert _mod._is_readme("README.md") is True
    assert _mod._is_readme("readme.rst") is True
    assert _mod._is_readme("src/foo.py") is False
    assert _mod._is_readme("docs/README.txt") is True


# --- loaders (with temp files) ---
def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_bm25_ranks_by_score_and_filters_readmes(tmp_path):
    path = tmp_path / "bm25.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instance_id": "inst-1",
                "hits": [
                    {"docid": "README.md", "score": 100.0},
                    {"docid": "b.py", "score": 50.0},
                    {"docid": "a.py", "score": 80.0},
                ],
            },
        ],
    )
    result = _mod.load_bm25(str(path))
    # README filtered; remaining sorted by score desc: a.py(80) > b.py(50)
    assert result["inst-1"] == ["a.py", "b.py"]


def test_load_agentless(tmp_path):
    path = tmp_path / "agentless.jsonl"
    _write_jsonl(
        path,
        [
            {"instance_id": "inst-1", "found_files": ["a.py", "b.py", "README.md"]},
        ],
    )
    result = _mod.load_agentless(str(path))
    assert result["inst-1"] == ["a.py", "b.py"]


def test_load_kg_extracts_markers_and_filters_readmes(tmp_path):
    path = tmp_path / "kg.jsonl"
    text = (
        "[start of README.md]\nsome readme\n[end of README.md]\n"
        "[start of src/foo.py]\ncode\n[end of src/foo.py]\n"
        "[start of src/bar.py]\nmore code\n[end of src/bar.py]\n"
    )
    _write_jsonl(path, [{"instance_id": "inst-1", "text_inputs": text}])
    result = _mod.load_kg(str(path))
    assert result["inst-1"] == ["src/foo.py", "src/bar.py"]


def test_load_gt(tmp_path):
    path = tmp_path / "progress.jsonl"
    _write_jsonl(
        path,
        [
            {"instance_id": "inst-1", "patch": "+++ b/foo.py\n+++ b/bar.py\n"},
            {"instance_id": "inst-2", "patch": ""},
        ],
    )
    result = _mod.load_gt(str(path))
    assert result == {"inst-1": {"foo.py", "bar.py"}}


# --- aggregate ---
def test_aggregate_averages():
    rows = [
        {
            "recall@1": 1.0,
            "recall@3": 1.0,
            "recall@5": 1.0,
            "cov@5": 1,
            "unbounded_recall": 1.0,
        },
        {
            "recall@1": 0.0,
            "recall@3": 0.5,
            "recall@5": 0.5,
            "cov@5": 0,
            "unbounded_recall": 0.5,
        },
    ]
    agg = _mod.aggregate(rows)
    assert agg["recall@1"] == 0.5
    assert agg["recall@3"] == 0.75
    assert agg["cov@5"] == 0.5


def test_aggregate_empty():
    assert _mod.aggregate([]) == {}


# --- evaluate_method end-to-end with mock ---
def test_evaluate_method_missed_files():
    predicted = {"inst-1": ["a.py", "b.py", "c.py", "d.py", "e.py"]}
    gt = {"inst-1": {"a.py", "z.py"}}
    rows = _mod.evaluate_method(predicted, gt, ["inst-1"], "test_method")
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "test_method"
    assert r["recall@1"] == 0.5
    assert r["cov@5"] == 0
    # z.py missed (not in top-5)
    assert "z.py" in r["missed_gt_files"]


# --- variant-aware KG progress paths ---
def test_kg_progress_path_plain_file_level():
    path = _mod.kg_progress_path("file_level", "plain")
    assert path.endswith("localization" "/kg__k-auto.progress.jsonl")


def test_kg_progress_path_plain_modes():
    assert _mod.kg_progress_path("file_snippet", "plain").endswith(
        "kg__k-auto__file_snippet.progress.jsonl"
    )
    assert _mod.kg_progress_path("node_source", "plain").endswith(
        "kg__k-auto__node_source.progress.jsonl"
    )


def test_kg_progress_path_embed_variant():
    assert _mod.kg_progress_path("file_level", "embed").endswith(
        "kg__k-auto__embed.progress.jsonl"
    )
    assert _mod.kg_progress_path("file_snippet", "embed").endswith(
        "kg__k-auto__embed__file_snippet.progress.jsonl"
    )


def test_kg_progress_path_default_variant_is_plain():
    assert _mod.kg_progress_path("file_level") == _mod.kg_progress_path(
        "file_level", "plain"
    )


def test_kg_progress_paths_dict_matches_helper():
    for mode, path in _mod.KG_PROGRESS_PATHS.items():
        assert path == _mod.kg_progress_path(mode, "plain")
