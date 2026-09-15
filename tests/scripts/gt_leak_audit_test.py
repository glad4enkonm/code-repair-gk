"""Tests for gt-leak-audit.py (gold-patch lines vs prompts vs base corpus)."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "gt-leak-audit.py"
_spec = importlib.util.spec_from_file_location("gla", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _write_jsonl(path: Path, rows: list) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _make_fixture(tmp_path: Path, prompt_lines: list, corpus_lines: list):
    """predictions jsonl + kg-progress jsonl + corpus dir for one instance."""
    pred_path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [{"instance_id": "org__repo-1", "text": "\n".join(prompt_lines)}],
    )
    patch = "--- a/lib/a.py\n+++ b/lib/a.py\n@@\n"
    patch += "".join(f"+{line}\n" for line in prompt_lines)
    patch += "+UNIQUE_GOLD_LINE_NEVER_IN_PROMPT_x = 1\n"
    progress_path = _write_jsonl(
        tmp_path / "progress.jsonl",
        [{"instance_id": "org__repo-1", "patch": patch}],
    )
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "org__repo-1").mkdir(parents=True)
    contents = "lib/a.py\n" + "\n".join(corpus_lines) + "\n"
    _write_jsonl(
        corpus_dir / "org__repo-1" / "documents.jsonl",
        [{"id": "lib/a.py", "contents": contents}],
    )
    return pred_path, progress_path, str(corpus_dir)


def test_no_leak_when_matched_lines_preexist_in_corpus(tmp_path):
    pred, prog, corpus = _make_fixture(
        tmp_path,
        prompt_lines=["    return compute(value=42)  # long enough"],
        corpus_lines=["    return compute(value=42)  # long enough"],
    )
    result = _mod.run_audit(pred, prog, corpus)
    assert result["totals"]["instances"] == 1
    assert result["totals"]["matched_lines"] == 1
    assert result["totals"]["leaked_lines"] == 0
    assert result["totals"]["corpus_missing_instances"] == 0


def test_flags_matched_line_absent_from_base_corpus(tmp_path):
    pred, prog, corpus = _make_fixture(
        tmp_path,
        prompt_lines=["    return compute(value=42)  # long enough"],
        corpus_lines=["def compute(value):", "    pass"],
    )
    result = _mod.run_audit(pred, prog, corpus)
    assert result["totals"]["matched_lines"] == 1
    assert result["totals"]["leaked_lines"] == 1
    # Leaked entries are plain strings (line text); location classification
    # is a manual step documented in the methodology audit md, not in-script.
    assert result["instances"]["org__repo-1"]["leaked"] == [
        "return compute(value=42)  # long enough"
    ]


def test_short_lines_are_ignored(tmp_path):
    pred, prog, corpus = _make_fixture(
        tmp_path,
        prompt_lines=["}"],
        corpus_lines=["def compute(value):"],
    )
    result = _mod.run_audit(pred, prog, corpus)
    assert result["totals"]["matched_lines"] == 0
    assert result["totals"]["leaked_lines"] == 0


def test_missing_corpus_dir_reported_not_leaked(tmp_path):
    pred, prog, _ = _make_fixture(
        tmp_path,
        prompt_lines=["    return compute(value=42)  # long enough"],
        corpus_lines=[],
    )
    result = _mod.run_audit(pred, prog, str(tmp_path / "nope"))
    assert result["totals"]["corpus_missing_instances"] == 1
    assert result["totals"]["matched_lines"] == 1
    assert result["totals"]["leaked_lines"] == 0
    assert result["instances"]["org__repo-1"]["corpus_missing"] is True


def test_instance_without_patch_is_reported(tmp_path):
    pred = _write_jsonl(
        tmp_path / "preds.jsonl",
        [{"instance_id": "org__repo-1", "text": "whatever"}],
    )
    prog = _write_jsonl(tmp_path / "progress.jsonl", [])
    result = _mod.run_audit(pred, prog, str(tmp_path / "corpus"))
    assert result["totals"]["instances"] == 1
    assert result["totals"]["patch_missing_instances"] == 1
    assert result["totals"]["leaked_lines"] == 0
