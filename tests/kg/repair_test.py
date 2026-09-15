"""Tests for repair.py: SEARCH/REPLACE and node-level code repair."""

import json
import shutil
import tempfile
from pathlib import Path

from kg.repair import (
    apply_search_replace,
    build_node_index,
    build_node_prompt,
    build_sr_prompt,
    generate_unified_diff,
    load_graph_data,
    parse_node_edits,
    parse_repair_output,
    parse_search_replace_blocks,
    strip_line_numbers,
    synthesize_node_patch,
    synthesize_sr_patch,
    truncate_file_sections,
    verify_and_replace_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_with_code():
    """Graph with Function/Method/Class nodes that have source_code + lines."""
    return {
        "nodes": [
            {"id": "File-src/app.py", "label": "app.py"},
            {"id": "Class-src/app.py-App", "label": "App"},
            {"id": "Method-src/app.py-App-run", "label": "run"},
            {"id": "Function-src/app.py-main", "label": "main"},
            {"id": "Import-File-src/app.py-0", "label": "os"},
            {"id": "File-src/utils.py", "label": "utils.py"},
            {"id": "Function-src/utils.py-helper", "label": "helper"},
        ],
        "edges": [
            {
                "id": "e1",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Class-src/app.py-App",
            },
            {
                "id": "e2",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Function-src/app.py-main",
            },
            {
                "id": "e3",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Import-File-src/app.py-0",
            },
            {
                "id": "e4",
                "label": "CONTAINS",
                "source": "Class-src/app.py-App",
                "target": "Method-src/app.py-App-run",
            },
            {
                "id": "e5",
                "label": "CONTAINS",
                "source": "File-src/utils.py",
                "target": "Function-src/utils.py-helper",
            },
        ],
        "allValues": {
            "File-src/app.py": {
                "meta_type": "File",
                "relative_path": "src/app.py",
            },
            "Class-src/app.py-App": {
                "meta_type": "Class",
                "name": "App",
                "source_code": (
                    "class App:\n" "    def run(self):\n" "        return 'run'\n"
                ),
                "line_start": 10,
                "line_end": 12,
            },
            "Method-src/app.py-App-run": {
                "meta_type": "Method",
                "name": "run",
                "source_code": ("    def run(self):\n" "        return 'run'\n"),
                "line_start": 11,
                "line_end": 12,
            },
            "Function-src/app.py-main": {
                "meta_type": "Function",
                "name": "main",
                "source_code": "def main():\n    print('hello')\n",
                "line_start": 20,
                "line_end": 21,
            },
            "Import-File-src/app.py-0": {"meta_type": "Import"},
            "File-src/utils.py": {
                "meta_type": "File",
                "relative_path": "src/utils.py",
            },
            "Function-src/utils.py-helper": {
                "meta_type": "Function",
                "name": "helper",
                "source_code": "def helper():\n    pass\n",
                "line_start": 5,
                "line_end": 6,
            },
        },
    }


def _write_repo(tmpdir: Path, files: dict):
    """Write files dict {rel_path: content} into tmpdir."""
    for rel_path, content in files.items():
        full = tmpdir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")


_APP_PY = (
    "# line 1\n"
    "# line 2\n"
    "# line 3\n"
    "# line 4\n"
    "# line 5\n"
    "# line 6\n"
    "# line 7\n"
    "# line 8\n"
    "# line 9\n"
    "class App:\n"  # 10
    "    def run(self):\n"  # 11
    "        return 'run'\n"  # 12
    "# line 13\n"
    "# line 14\n"
    "# line 15\n"
    "# line 16\n"
    "# line 17\n"
    "# line 18\n"
    "# line 19\n"
    "def main():\n"  # 20
    "    print('hello')\n"  # 21
    "# line 22\n"
)


# ---------------------------------------------------------------------------
# generate_unified_diff tests
# ---------------------------------------------------------------------------


def test_generate_unified_diff_basic():
    old = ["a", "b", "c"]
    new = ["a", "B", "c"]
    diff = generate_unified_diff("src/f.py", old, new)
    assert "--- a/src/f.py" in diff
    assert "+++ b/src/f.py" in diff
    assert "-b" in diff
    assert "+B" in diff


def test_generate_unified_diff_no_change():
    lines = ["a", "b", "c"]
    diff = generate_unified_diff("src/f.py", lines, lines)
    assert diff == ""


def test_generate_unified_diff_multi_hunk():
    old = ["a", "b", "c", "d", "e"]
    new = ["A", "b", "c", "D", "e"]
    diff = generate_unified_diff("src/f.py", old, new)
    assert diff.count("@@") == 2  # two hunks


# ---------------------------------------------------------------------------
# parse_search_replace_blocks tests
# ---------------------------------------------------------------------------


def test_parse_sr_single_block():
    raw = (
        "### src/app.py\n"
        "<<<<<<< SEARCH\n"
        "def foo():\n"
        "    pass\n"
        "=======\n"
        "def foo():\n"
        "    return 42\n"
        ">>>>>>> REPLACE"
    )
    edits = parse_search_replace_blocks(raw)
    assert len(edits) == 1
    assert edits[0][0] == "src/app.py"
    assert "def foo():\n    pass" in edits[0][1]
    assert "def foo():\n    return 42" in edits[0][2]


def test_parse_sr_multi_file():
    raw = (
        "### src/a.py\n"
        "<<<<<<< SEARCH\n"
        "old_a\n"
        "=======\n"
        "new_a\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "### src/b.py\n"
        "<<<<<<< SEARCH\n"
        "old_b\n"
        "=======\n"
        "new_b\n"
        ">>>>>>> REPLACE"
    )
    edits = parse_search_replace_blocks(raw)
    assert len(edits) == 2
    assert edits[0][0] == "src/a.py"
    assert edits[1][0] == "src/b.py"


def test_parse_sr_multi_block_same_file():
    raw = (
        "### src/app.py\n"
        "<<<<<<< SEARCH\n"
        "old1\n"
        "=======\n"
        "new1\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "old2\n"
        "=======\n"
        "new2\n"
        ">>>>>>> REPLACE"
    )
    edits = parse_search_replace_blocks(raw)
    assert len(edits) == 2
    assert edits[0][0] == "src/app.py"
    assert edits[1][0] == "src/app.py"


def test_parse_sr_no_blocks():
    raw = "I don't know how to fix this."
    assert parse_search_replace_blocks(raw) == []


def test_parse_sr_malformed_missing_markers():
    raw = "### src/app.py\n<<<<<<< SEARCH\nonly search\nno replace"
    edits = parse_search_replace_blocks(raw)
    assert edits == []


# ---------------------------------------------------------------------------
# apply_search_replace tests
# ---------------------------------------------------------------------------


def test_apply_sr_unique_match():
    lines = ["alpha", "beta", "gamma"]
    result = apply_search_replace(lines, "beta", "BETA")
    assert result == ["alpha", "BETA", "gamma"]


def test_apply_sr_not_found():
    lines = ["alpha", "beta"]
    assert apply_search_replace(lines, "missing", "x") is None


def test_apply_sr_non_unique():
    lines = ["dup", "mid", "dup"]
    assert apply_search_replace(lines, "dup", "unique") is None


def test_apply_sr_multi_line():
    lines = ["def foo():", "    pass", "def bar():", "    pass"]
    result = apply_search_replace(
        lines, "def foo():\n    pass", "def foo():\n    return 42"
    )
    assert result == ["def foo():", "    return 42", "def bar():", "    pass"]


def test_apply_sr_replacement_different_length():
    lines = ["a", "b", "c"]
    result = apply_search_replace(lines, "b", "x\ny\nz")
    assert result == ["a", "x", "y", "z", "c"]


# ---------------------------------------------------------------------------
# build_sr_prompt tests
# ---------------------------------------------------------------------------


def test_build_sr_prompt_contains_elements():
    prompt = build_sr_prompt("Fix the bug", "some code here")
    assert "Fix the bug" in prompt
    assert "some code here" in prompt
    assert "<issue>" in prompt
    assert "<code>" in prompt
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt
    assert "###" in prompt
    assert "src/" not in prompt


# ---------------------------------------------------------------------------
# parse_node_edits tests
# ---------------------------------------------------------------------------


def test_parse_node_single():
    raw = (
        "### calculate_gcd in src/math.py\n"
        "def calculate_gcd(a, b):\n"
        "    return math.gcd(a, b)"
    )
    edits = parse_node_edits(raw)
    assert len(edits) == 1
    assert edits[0][0] == "calculate_gcd"
    assert edits[0][1] == "src/math.py"
    assert "def calculate_gcd" in edits[0][2]


def test_parse_node_multi():
    raw = (
        "### foo in src/a.py\n"
        "def foo():\n"
        "    pass\n"
        "\n"
        "### bar in src/b.py\n"
        "def bar():\n"
        "    return 42"
    )
    edits = parse_node_edits(raw)
    assert len(edits) == 2
    assert edits[0][0] == "foo"
    assert edits[1][0] == "bar"


def test_parse_node_no_edits():
    raw = "I cannot fix this issue."
    assert parse_node_edits(raw) == []


def test_parse_node_skips_sr_sections():
    raw = (
        "### foo in src/a.py\n"
        "def foo():\n"
        "    pass\n"
        "\n"
        "### src/b.py\n"
        "<<<<<<< SEARCH\n"
        "old\n"
        "=======\n"
        "new\n"
        ">>>>>>> REPLACE"
    )
    edits = parse_node_edits(raw)
    assert len(edits) == 1
    assert edits[0][0] == "foo"


# ---------------------------------------------------------------------------
# parse_repair_output tests
# ---------------------------------------------------------------------------


def test_parse_repair_output_mixed():
    raw = (
        "### foo in src/a.py\n"
        "def foo():\n"
        "    return 42\n"
        "\n"
        "### src/b.py\n"
        "<<<<<<< SEARCH\n"
        "old_code\n"
        "=======\n"
        "new_code\n"
        ">>>>>>> REPLACE"
    )
    node_edits, sr_edits = parse_repair_output(raw)
    assert len(node_edits) == 1
    assert node_edits[0][0] == "foo"
    assert len(sr_edits) == 1
    assert sr_edits[0][0] == "src/b.py"


# ---------------------------------------------------------------------------
# build_node_index tests
# ---------------------------------------------------------------------------


def test_build_node_index_basic():
    index = build_node_index(_make_graph_with_code())
    assert "main|src/app.py" in index
    assert "helper|src/utils.py" in index
    info = index["main|src/app.py"]
    assert info["source_code"] == "def main():\n    print('hello')\n"
    assert info["line_start"] == 20
    assert info["line_end"] == 21
    assert info["file_path"] == "src/app.py"


def test_build_node_index_filters_non_code_nodes():
    index = build_node_index(_make_graph_with_code())
    # Import node should not be indexed
    assert not any("Import" in k for k in index)
    # Only Function/Method/Class
    for key, info in index.items():
        node_id = info["node_id"]
        graph = _make_graph_with_code()
        props = graph["allValues"][node_id]
        assert props["meta_type"] in ("Function", "Method", "Class")


def test_build_node_index_multi_hop_file_resolution():
    """Method → Class → File traversal."""
    index = build_node_index(_make_graph_with_code())
    assert "run|src/app.py" in index
    assert index["run|src/app.py"]["file_path"] == "src/app.py"


def test_build_node_index_empty_graph():
    index = build_node_index({"nodes": [], "edges": [], "allValues": {}})
    assert index == {}


# ---------------------------------------------------------------------------
# verify_and_replace_node tests
# ---------------------------------------------------------------------------


def test_verify_replace_match():
    file_lines = _APP_PY.splitlines()
    node_info = {
        "source_code": "class App:\n    def run(self):\n        return 'run'\n",
        "line_start": 10,
        "line_end": 12,
    }
    new_code = "class App:\n" "    def run(self):\n" "        return 'fixed'\n"
    result = verify_and_replace_node(file_lines, node_info, new_code)
    assert result is not None
    assert "return 'fixed'" in result[11]


def test_verify_replace_drift():
    file_lines = ["different", "content", "here"]
    node_info = {
        "source_code": "class App:\n    pass\n",
        "line_start": 1,
        "line_end": 2,
    }
    result = verify_and_replace_node(file_lines, node_info, "new code")
    assert result is None


def test_verify_replace_correct_line_range():
    file_lines = [
        "# 1",
        "# 2",
        "# 3",
        "# 4",
        "# 5",
        "def target():",
        "    pass",
        "# 8",
        "# 9",
    ]
    node_info = {
        "source_code": "def target():\n    pass",
        "line_start": 6,
        "line_end": 7,
    }
    new_code = "def target():\n    return 42"
    result = verify_and_replace_node(file_lines, node_info, new_code)
    assert result is not None
    assert result[5] == "def target():"
    assert result[6] == "    return 42"
    assert result[7] == "# 8"


# ---------------------------------------------------------------------------
# build_node_prompt tests
# ---------------------------------------------------------------------------


def test_build_node_prompt_contains_both_formats():
    prompt = build_node_prompt("Fix the bug", "some code")
    assert "Fix the bug" in prompt
    assert "some code" in prompt
    assert "<issue>" in prompt
    assert "<code>" in prompt
    # Node format example
    assert "calculate_gcd" in prompt
    assert " in " in prompt
    # S&R format example
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt
    assert "src/" not in prompt


# ---------------------------------------------------------------------------
# synthesize_sr_patch tests
# ---------------------------------------------------------------------------


def test_synthesize_sr_single_file():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_sr_"))
    try:
        _write_repo(tmpdir, {"src/app.py": "def foo():\n    pass\n"})
        edits = [
            ("src/app.py", "def foo():\n    pass", "def foo():\n    return 42"),
        ]
        patch = synthesize_sr_patch(str(tmpdir), edits)
        assert "--- a/src/app.py" in patch
        assert "+++ b/src/app.py" in patch
        assert "-    pass" in patch
        assert "+    return 42" in patch
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_sr_not_found_skipped():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_sr_"))
    try:
        _write_repo(tmpdir, {"src/app.py": "def foo():\n    pass\n"})
        edits = [("src/app.py", "nonexistent code", "replacement")]
        patch = synthesize_sr_patch(str(tmpdir), edits)
        assert patch == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_sr_multi_file():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_sr_"))
    try:
        _write_repo(
            tmpdir,
            {
                "src/a.py": "old_a\n",
                "src/b.py": "old_b\n",
            },
        )
        edits = [
            ("src/a.py", "old_a", "new_a"),
            ("src/b.py", "old_b", "new_b"),
        ]
        patch = synthesize_sr_patch(str(tmpdir), edits)
        assert "a/src/a.py" in patch
        assert "a/src/b.py" in patch
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_sr_file_not_found():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_sr_"))
    try:
        edits = [("src/missing.py", "old", "new")]
        patch = synthesize_sr_patch(str(tmpdir), edits)
        assert patch == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# synthesize_node_patch tests
# ---------------------------------------------------------------------------


def test_synthesize_node_single_edit():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_node_"))
    try:
        _write_repo(tmpdir, {"src/app.py": _APP_PY})
        graph = _make_graph_with_code()
        index = build_node_index(graph)

        node_edits = [
            (
                "main",
                "src/app.py",
                "def main():\n    print('fixed')\n",
            )
        ]
        patch = synthesize_node_patch(str(tmpdir), index, node_edits, [])
        assert "--- a/src/app.py" in patch
        assert "-    print('hello')" in patch
        assert "+    print('fixed')" in patch
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_node_source_mismatch():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_node_"))
    try:
        # File content doesn't match graph source_code
        _write_repo(tmpdir, {"src/app.py": "completely different\n"})
        graph = _make_graph_with_code()
        index = build_node_index(graph)

        node_edits = [("main", "src/app.py", "def main():\n    pass\n")]
        patch = synthesize_node_patch(str(tmpdir), index, node_edits, [])
        assert patch == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_node_not_found_in_index():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_node_"))
    try:
        _write_repo(tmpdir, {"src/app.py": _APP_PY})
        graph = _make_graph_with_code()
        index = build_node_index(graph)

        node_edits = [("nonexistent", "src/app.py", "def new():\n    pass\n")]
        patch = synthesize_node_patch(str(tmpdir), index, node_edits, [])
        assert patch == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthesize_node_with_sr_fallback():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_node_"))
    try:
        # main must be at lines 20-21 to match the graph index
        content = (
            "from typing import Optional\n"  # 1
            + "\n".join(f"# line {i}" for i in range(2, 20))
            + "\n"
            + "def main():\n"  # 20
            + "    print('hello')\n"  # 21
        )
        _write_repo(tmpdir, {"src/app.py": content})
        graph = _make_graph_with_code()
        index = build_node_index(graph)

        node_edits = [
            (
                "main",
                "src/app.py",
                "def main():\n    print('fixed')\n",
            )
        ]
        sr_edits = [
            (
                "src/app.py",
                "from typing import Optional",
                "from typing import Optional, List",
            )
        ]
        patch = synthesize_node_patch(str(tmpdir), index, node_edits, sr_edits)
        # Both changes should be in the diff
        assert "-    print('hello')" in patch
        assert "+    print('fixed')" in patch
        assert "-from typing import Optional" in patch
        assert "+from typing import Optional, List" in patch
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# load_graph_data tests
# ---------------------------------------------------------------------------


def test_load_graph_data():
    tmpdir = Path(tempfile.mkdtemp(prefix="repair_graph_"))
    try:
        graph = _make_graph_with_code()
        (tmpdir / "data.json").write_text(json.dumps(graph), encoding="utf-8")
        loaded = load_graph_data(str(tmpdir))
        assert len(loaded["nodes"]) == len(graph["nodes"])
        assert "allValues" in loaded
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# strip_line_numbers tests
# ---------------------------------------------------------------------------


def test_strip_line_numbers_basic():
    code = "1 def foo():\n2     pass\n3 \n"
    result = strip_line_numbers(code)
    assert result == "def foo():\n    pass\n"


def test_strip_line_numbers_preserves_markers():
    code = "[start of app.py]\n1 def foo():\n2     pass\n[end of app.py]"
    result = strip_line_numbers(code)
    assert "[start of app.py]" in result
    assert "[end of app.py]" in result
    assert "def foo():" in result
    assert "    pass" in result


def test_strip_line_numbers_strips_metadata():
    code = "## Function: foo | complexity=3\n10 def foo():\n11     pass"
    result = strip_line_numbers(code)
    assert "## Function" not in result
    assert "def foo():" in result
    assert "    pass" in result


def test_strip_line_numbers_strips_edge_metadata():
    code = "##   --CALLS--> Function: bar\n10 def foo():"
    result = strip_line_numbers(code)
    assert "##" not in result
    assert "def foo():" in result


def test_strip_line_numbers_preserves_gap_markers():
    code = "10 a\n... (5 lines omitted) ...\n16 b"
    result = strip_line_numbers(code)
    assert "... (5 lines omitted) ..." in result
    assert result == "a\n... (5 lines omitted) ...\nb"


def test_strip_line_numbers_no_prefixes():
    code = "def foo():\n    pass\n"
    assert strip_line_numbers(code) == "def foo():\n    pass"


# ---------------------------------------------------------------------------
# truncate_file_sections
# ---------------------------------------------------------------------------


def _make_section(path: str, body: str) -> str:
    return f"[start of {path}]\n{body}\n[end of {path}]\n"


def test_truncate_small_context_unchanged():
    code = "intro text\n" + _make_section("a.py", "x = 1")
    assert truncate_file_sections(code) == code


def test_truncate_caps_oversized_file_head_and_tail():
    body = "".join(f"line {i}\n" for i in range(20_000))  # ~240k chars
    code = _make_section("big.py", body)
    result = truncate_file_sections(code, max_file_chars=10_000)
    assert result.startswith("[start of big.py]\nline 0\n")
    assert result.endswith("[end of big.py]\n")
    assert "characters truncated" in result
    assert len(result) < 20_000  # head + tail + note only


def test_truncate_total_budget_stub_later_files():
    files = "".join(
        _make_section(f"f{i}.py", f"marker{i}\n" + "x\n" * 4_000)  # ~8k each
        for i in range(10)
    )
    result = truncate_file_sections(
        files, max_file_chars=60_000, max_total_chars=12_000
    )
    # early files keep real content
    assert "marker0" in result
    assert "marker1" in result
    # later files become stubs with markers intact
    assert "[start of f9.py]" in result
    assert "[end of f9.py]" in result
    assert "content omitted" in result
    # overall size bounded (stubs may add a little beyond the budget)
    assert len(result) < 20_000


def test_truncate_max_files_drops_extra_sections():
    files = "".join(_make_section(f"f{i}.py", f"v = {i}") for i in range(8))
    result = truncate_file_sections(files, max_files=5)
    assert "[start of f4.py]" in result and "v = 4" in result
    assert "[start of f5.py]" not in result
    assert "[start of f7.py]" not in result
    assert result.count("[start of ") == 5


def test_truncate_preserves_text_outside_sections():
    code = "before\n" + _make_section("a.py", "x = 1") + "after"
    result = truncate_file_sections(code, max_file_chars=10)
    assert result.startswith("before\n")
    assert result.endswith("after")
