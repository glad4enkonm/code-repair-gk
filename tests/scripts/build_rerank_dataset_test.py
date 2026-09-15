"""Tests for build-rerank-dataset.py (rerank-ordered top-k context assembly)."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "build-rerank-dataset.py"
_spec = importlib.util.spec_from_file_location("brd", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _sample_text():
    return (
        "You will be provided with code.\n<issue>\nbug\n</issue>\n\n<code>\n"
        "[start of src/foo.py]\nfoo line 1\nfoo line 2\n[end of src/foo.py]\n\n"
        "[start of src/bar.py]\nbar line\n[end of src/bar.py]\n\n"
        "[start of README.md]\nreadme\n[end of README.md]\n"
        "\n</code>\n\nRespond below:\n"
    )


# --- parse_blocks ---


def test_parse_blocks_splits_header_footer_separator():
    header, footer, separator, blocks, order = _mod.parse_blocks(_sample_text())
    assert header == (
        "You will be provided with code.\n<issue>\nbug\n</issue>\n\n<code>\n"
    )
    assert footer == "\n\n</code>\n\nRespond below:\n"
    assert separator == "\n\n"
    assert order == ["src/foo.py", "src/bar.py", "README.md"]
    assert blocks["src/foo.py"] == (
        "[start of src/foo.py]\nfoo line 1\nfoo line 2\n[end of src/foo.py]"
    )


def test_parse_blocks_single_block_separator_default():
    text = "head\n[start of a.py]\nx\n[end of a.py]\ntail"
    header, footer, separator, blocks, order = _mod.parse_blocks(text)
    assert order == ["a.py"]
    assert blocks["a.py"] == "[start of a.py]\nx\n[end of a.py]"
    assert header == "head\n"
    assert footer == "\ntail"
    assert separator == "\n"


def test_parse_blocks_no_blocks():
    header, footer, separator, blocks, order = _mod.parse_blocks("no markers here")
    assert header == "no markers here"
    assert footer == ""
    assert blocks == {} and order == []


# --- rebuild_context ---


def test_rebuild_reorders_cuts_and_preserves_structure():
    new_text, used, missing = _mod.rebuild_context(
        _sample_text(), ["src/bar.py", "src/foo.py"], 5
    )
    expected = (
        "You will be provided with code.\n<issue>\nbug\n</issue>\n\n<code>\n"
        "[start of src/bar.py]\nbar line\n[end of src/bar.py]\n\n"
        "[start of src/foo.py]\nfoo line 1\nfoo line 2\n[end of src/foo.py]\n"
        "\n</code>\n\nRespond below:\n"
    )
    assert new_text == expected
    assert used == ["src/bar.py", "src/foo.py"]
    assert missing == []


def test_rebuild_cuts_to_top_k():
    new_text, used, _ = _mod.rebuild_context(
        _sample_text(), ["src/bar.py", "src/foo.py", "README.md"], 2
    )
    assert used == ["src/bar.py", "src/foo.py"]
    assert "README.md" not in new_text


def test_rebuild_reports_missing_ranked_paths():
    _, used, missing = _mod.rebuild_context(
        _sample_text(), ["nope/x.py", "src/foo.py"], 5
    )
    assert used == ["src/foo.py"]
    assert missing == ["nope/x.py"]


def test_rebuild_fewer_than_k_takes_all():
    _, used, _ = _mod.rebuild_context(_sample_text(), ["src/bar.py"], 5)
    assert used == ["src/bar.py"]


def test_rebuild_zero_usable_falls_back_to_original():
    text = _sample_text()
    new_text, used, missing = _mod.rebuild_context(text, ["nope/a.py", "nope/b.py"], 5)
    assert new_text == text
    assert used == []
    assert missing == ["nope/a.py", "nope/b.py"]


# --- progress loading ---


def test_load_progress_dedupes_keep_last(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(
        json.dumps({"instance_id": "i1", "v": 1})
        + "\n"
        + json.dumps({"instance_id": "i2", "v": 2})
        + "\n"
        + json.dumps({"instance_id": "i1", "v": 3})
        + "\n",
        encoding="utf-8",
    )
    rows = _mod.load_progress(path)
    assert list(rows) == ["i1", "i2"]
    assert rows["i1"]["v"] == 3


def test_load_progress_skips_failed_rows(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"instance_id": "i1", "failed": True, "ranked_files": ["a"]})
        + "\n"
        + json.dumps({"instance_id": "i2", "failed": False, "ranked_files": ["b"]})
        + "\n",
        encoding="utf-8",
    )
    rows = _mod.load_progress(path, skip_failed=True)
    assert list(rows) == ["i2"]


# --- split + paths ---


def test_split_rows_dev_is_first_23_test_is_rest():
    rows = {f"i{n}": {"instance_id": f"i{n}"} for n in range(26)}
    dev = _mod.split_rows(rows, "dev")
    test = _mod.split_rows(rows, "test")
    assert len(dev) == 23 and dev[0]["instance_id"] == "i0"
    assert len(test) == 3 and test[0]["instance_id"] == "i23"


def test_output_names_embed_split_and_top_k():
    progress, dataset_dir = _mod.output_names("test", 5)
    assert progress.name == (
        "data__swe-bench_lite__style-3__fs-kg__k-auto__embed__rerank5" ".progress.jsonl"
    )
    assert (
        dataset_dir.name
        == "data__swe-bench_lite__style-3__fs-kg__k-auto__embed__rerank5"
    )


def test_rerank_progress_path_contains_split():
    path = _mod.rerank_progress_path("dev")
    assert "__dev__" in path.name
    assert path.name.endswith(".progress.jsonl")


def test_rerank_progress_path_default_model_is_qwen():
    path = _mod.rerank_progress_path("test")
    assert path.name.endswith("__test__qwen3-coder-next-Q8_0.progress.jsonl")


def test_rerank_progress_path_non_default_model():
    path = _mod.rerank_progress_path("test", model="gemma4-26b-A4B-Q8_0")
    assert path.name == (
        "llm_rerank__file_level__embed__list__freq-simmax-simmean-parts"
        "__test__gemma4-26b-A4B-Q8_0.progress.jsonl"
    )


def test_output_names_default_model_keeps_legacy_untagged_name():
    progress, dataset_dir = _mod.output_names("test", 5)
    assert "qwen" not in progress.name
    assert progress.parent == Path("outputs")
    assert dataset_dir.parent == Path("outputs")


def test_output_names_non_default_model_tagged_and_rooted():
    progress, dataset_dir = _mod.output_names(
        "test", 5, model="gemma4-26b-A4B-Q8_0", output_root="outputs/gemma26b"
    )
    assert progress == Path(
        "outputs/gemma26b/data__swe-bench_lite__style-3__fs-kg__k-auto"
        "__embed__rerank5__gemma4-26b-A4B-Q8_0.progress.jsonl"
    )
    assert dataset_dir == Path(
        "outputs/gemma26b/data__swe-bench_lite__style-3__fs-kg__k-auto"
        "__embed__rerank5__gemma4-26b-A4B-Q8_0"
    )


# --- build_entries ---


def _kg_row(iid, text):
    return {
        "instance_id": iid,
        "repo": "org/repo",
        "base_commit": "abc",
        "problem_statement": "ps",
        "patch": "+++ b/x.py",
        "text_inputs": text,
    }


def test_build_entries_rebuilds_and_marks_fallback():
    kg_rows = [_kg_row("i1", _sample_text()), _kg_row("i2", _sample_text())]
    rerank_rows = {
        "i1": {"instance_id": "i1", "ranked_files": ["src/bar.py", "src/foo.py"]},
    }
    entries, stats = _mod.build_entries(kg_rows, rerank_rows, 5)
    assert stats == {"rebuilt": 1, "fallback": 1, "missing_blocks": 0}
    by_id = {e["instance_id"]: e for e in entries}
    assert by_id["i1"]["rerank_used"] is True
    assert by_id["i2"]["rerank_used"] is False
    assert by_id["i2"]["text_inputs"] == _sample_text()
    assert by_id["i1"]["text_inputs"].index("[start of src/bar.py]") < by_id["i1"][
        "text_inputs"
    ].index("[start of src/foo.py]")
    assert "README.md" not in by_id["i1"]["text_inputs"]


def test_build_entries_counts_missing_ranked_paths():
    kg_rows = [_kg_row("i1", _sample_text())]
    rerank_rows = {"i1": {"instance_id": "i1", "ranked_files": ["nope.py"]}}
    _, stats = _mod.build_entries(kg_rows, rerank_rows, 5)
    assert stats == {"rebuilt": 0, "fallback": 1, "missing_blocks": 1}
