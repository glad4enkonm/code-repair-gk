"""Tests for context.py file resolution and snippet assembly."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from kg.config import KGConfig
from kg.context import (
    _collect_graph_context,
    _format_node_snippets,
    _format_snippets,
    _merge_ranges,
    _rank_files,
    assemble_file_snippet,
    assemble_node_source,
    read_all_files,
    read_files_within_budget,
)


def _write_graph(dir_path: Path, graph: dict):
    """Write a minimal graph dataset to dir_path."""
    for name, payload in [
        ("data.json", graph),
        ("origin.json", {"nodes": [], "edges": [], "allValues": {}}),
        ("meta.json", {"nodes": [], "edges": [], "allValues": {}}),
    ]:
        (dir_path / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_code_graph():
    """Build a code-like graph with multi-level containment.

    File-src/app.py --CONTAINS--> Class-src/app.py-App
    File-src/app.py --CONTAINS--> Function-src/app.py-main
    File-src/app.py --CONTAINS--> Import-File-src/app.py-0
    Class-src/app.py-App --CONTAINS--> Method-src/app.py-App-run
    Class-src/app.py-App --CONTAINS--> Variable-Class-src/app.py-App-name
    """
    return {
        "nodes": [
            {"id": "Directory-src", "label": "src"},
            {"id": "File-src/app.py", "label": "app.py"},
            {"id": "Class-src/app.py-App", "label": "App"},
            {"id": "Method-src/app.py-App-run", "label": "run"},
            {"id": "Function-src/app.py-main", "label": "main"},
            {"id": "Import-File-src/app.py-0", "label": "os"},
            {"id": "Variable-Class-src/app.py-App-name", "label": "name"},
        ],
        "edges": [
            {
                "id": "e1",
                "label": "CONTAINS",
                "source": "Directory-src",
                "target": "File-src/app.py",
            },
            {
                "id": "e2",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Class-src/app.py-App",
            },
            {
                "id": "e3",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Function-src/app.py-main",
            },
            {
                "id": "e4",
                "label": "CONTAINS",
                "source": "File-src/app.py",
                "target": "Import-File-src/app.py-0",
            },
            {
                "id": "e5",
                "label": "CONTAINS",
                "source": "Class-src/app.py-App",
                "target": "Method-src/app.py-App-run",
            },
            {
                "id": "e6",
                "label": "CONTAINS",
                "source": "Class-src/app.py-App",
                "target": "Variable-Class-src/app.py-App-name",
            },
        ],
        "allValues": {
            "Directory-src": {"meta_type": "Directory"},
            "File-src/app.py": {
                "meta_type": "File",
                "relative_path": "src/app.py",
            },
            "Class-src/app.py-App": {"meta_type": "Class"},
            "Method-src/app.py-App-run": {"meta_type": "Method"},
            "Function-src/app.py-main": {"meta_type": "Function"},
            "Import-File-src/app.py-0": {"meta_type": "Import"},
            "Variable-Class-src/app.py-App-name": {"meta_type": "Variable"},
        },
    }


def test_rank_files_resolves_method_to_file():
    """Method → Class → File traversal via CONTAINS edges."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_rank_"))
    try:
        _write_graph(tmpdir, _make_code_graph())
        touched = {
            "Method-src/app.py-App-run": {"label": "run"},
            "Function-src/app.py-main": {"label": "main"},
        }
        ranked = _rank_files(touched, str(tmpdir))
        assert ranked == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rank_files_resolves_variable_and_import():
    """Variable-Class-... and Import-File-... nodes resolve to their File."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_rank_"))
    try:
        _write_graph(tmpdir, _make_code_graph())
        touched = {
            "Variable-Class-src/app.py-App-name": {"label": "name"},
            "Import-File-src/app.py-0": {"label": "os"},
        }
        ranked = _rank_files(touched, str(tmpdir))
        assert ranked == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rank_files_by_touch_frequency():
    """Multiple touched nodes in same file should rank it higher."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_rank_"))
    try:
        graph = _make_code_graph()
        # Add a second file
        graph["nodes"].append({"id": "File-src/utils.py", "label": "utils.py"})
        graph["nodes"].append({"id": "Function-src/utils.py-helper", "label": "helper"})
        graph["edges"].append(
            {
                "id": "e7",
                "label": "CONTAINS",
                "source": "File-src/utils.py",
                "target": "Function-src/utils.py-helper",
            }
        )
        graph["edges"].append(
            {
                "id": "e8",
                "label": "CONTAINS",
                "source": "Directory-src",
                "target": "File-src/utils.py",
            }
        )
        graph["allValues"]["File-src/utils.py"] = {
            "meta_type": "File",
            "relative_path": "src/utils.py",
        }
        graph["allValues"]["Function-src/utils.py-helper"] = {
            "meta_type": "Function",
        }
        _write_graph(tmpdir, graph)

        # 3 nodes in app.py, 1 in utils.py
        touched = {
            "Method-src/app.py-App-run": {},
            "Function-src/app.py-main": {},
            "Import-File-src/app.py-0": {},
            "Function-src/utils.py-helper": {},
        }
        ranked = _rank_files(touched, str(tmpdir))
        assert ranked[0] == "src/app.py"
        assert ranked[1] == "src/utils.py"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rank_files_empty_touched():
    """No touched nodes → empty ranking."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_rank_"))
    try:
        _write_graph(tmpdir, _make_code_graph())
        ranked = _rank_files({}, str(tmpdir))
        assert ranked == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rank_files_file_node_itself():
    """A File node should resolve to itself."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_rank_"))
    try:
        _write_graph(tmpdir, _make_code_graph())
        touched = {"File-src/app.py": {"label": "app.py"}}
        ranked = _rank_files(touched, str(tmpdir))
        assert ranked == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _merge_ranges tests
# ---------------------------------------------------------------------------


def test_merge_ranges_empty():
    assert _merge_ranges([]) == []


def test_merge_ranges_single_no_padding():
    assert _merge_ranges([(10, 20)]) == [(10, 20)]


def test_merge_ranges_single_with_padding():
    assert _merge_ranges([(10, 20)], padding=5) == [(5, 25)]


def test_merge_ranges_padding_clamps_to_one():
    assert _merge_ranges([(3, 10)], padding=5) == [(1, 15)]


def test_merge_ranges_overlapping_merge():
    result = _merge_ranges([(10, 20), (15, 25)])
    assert result == [(10, 25)]


def test_merge_ranges_non_overlapping():
    result = _merge_ranges([(10, 20), (30, 40)])
    assert result == [(10, 20), (30, 40)]


def test_merge_ranges_padding_causes_merge():
    """Adjacent ranges with padding overlap → merged."""
    result = _merge_ranges([(10, 20), (25, 35)], padding=3)
    assert result == [(7, 38)]


def test_merge_ranges_contained_dropped():
    """Range fully inside another is absorbed."""
    result = _merge_ranges([(10, 30), (15, 20)])
    assert result == [(10, 30)]


def test_merge_ranges_unsorted_input():
    result = _merge_ranges([(30, 40), (10, 20)])
    assert result == [(10, 20), (30, 40)]


# ---------------------------------------------------------------------------
# _format_snippets tests
# ---------------------------------------------------------------------------


def test_format_snippets_single_file_single_snippet():
    snippets = {
        "src/app.py": [(10, 12, "line ten\nline eleven\nline twelve")],
    }
    text = _format_snippets(snippets)
    assert "[start of src/app.py]" in text
    assert "[end of src/app.py]" in text
    assert "10 line ten" in text
    assert "11 line eleven" in text
    assert "12 line twelve" in text


def test_format_snippets_gap_marker():
    snippets = {
        "src/app.py": [
            (10, 12, "alpha\nbeta\ngamma"),
            (20, 21, "delta"),
        ],
    }
    text = _format_snippets(snippets)
    assert "10 alpha" in text
    assert "12 gamma" in text
    assert "... (7 lines omitted) ..." in text
    assert "20 delta" in text


def test_format_snippets_no_gap_when_adjacent():
    snippets = {
        "src/app.py": [
            (10, 12, "alpha\nbeta\ngamma"),
            (13, 14, "delta\nepsilon"),
        ],
    }
    text = _format_snippets(snippets)
    assert "... (" not in text
    assert "13 delta" in text
    assert "14 epsilon" in text


def test_format_snippets_multiple_files_sorted():
    snippets = {
        "src/utils.py": [(1, 1, "util")],
        "src/app.py": [(1, 1, "app")],
    }
    text = _format_snippets(snippets)
    app_pos = text.index("src/app.py")
    util_pos = text.index("src/utils.py")
    assert app_pos < util_pos


def test_format_snippets_empty():
    assert _format_snippets({}) == ""


# ---------------------------------------------------------------------------
# Graph-with-source helper for snippet tests
# ---------------------------------------------------------------------------


def _make_code_graph_with_source():
    """Extend _make_code_graph with source_code / line_start / line_end."""
    graph = _make_code_graph()
    graph["allValues"]["Class-src/app.py-App"] = {
        "meta_type": "Class",
        "name": "App",
        "source_code": ("class App:\n" "    def run(self):\n" "        return 'run'\n"),
        "line_start": 10,
        "line_end": 12,
    }
    graph["allValues"]["Method-src/app.py-App-run"] = {
        "meta_type": "Method",
        "name": "run",
        "source_code": "    def run(self):\n        return 'run'\n",
        "line_start": 11,
        "line_end": 12,
    }
    graph["allValues"]["Function-src/app.py-main"] = {
        "meta_type": "Function",
        "name": "main",
        "source_code": "def main():\n    print('hello')\n",
        "line_start": 20,
        "line_end": 21,
    }
    return graph


_APP_PY_CONTENT = (
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
    "# line 23\n"
)


def _setup_repo_with_source(tmpdir: Path):
    """Create graph + repo files for snippet assembly tests."""
    graph_dir = tmpdir / "graph"
    graph_dir.mkdir()
    _write_graph(graph_dir, _make_code_graph_with_source())

    repo_dir = tmpdir / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "app.py").write_text(_APP_PY_CONTENT, encoding="utf-8")
    (repo_dir / "README.md").write_text("# Test Repo\n", encoding="utf-8")

    return graph_dir, repo_dir


def _noop_build_prompt(problem_statement, readmes, code_text):
    """Mock _build_prompt that returns code_text only."""
    return code_text


# ---------------------------------------------------------------------------
# assemble_file_snippet tests
# ---------------------------------------------------------------------------


def test_assemble_file_snippet_extracts_touched_ranges():
    """File snippet mode reads only the relevant line ranges from the file."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_snip_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()
        config.node_context_padding = 0

        touched = {
            "Class-src/app.py-App": {"label": "App", "line_start": 10, "line_end": 12},
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_file_snippet(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        # Should include lines 10-12 from the file, with original numbers
        assert "10 class App:" in text
        assert "11     def run(self):" in text
        assert "12         return 'run'" in text
        # Should NOT include lines outside the range
        assert "# line 1" not in text
        assert "# line 22" not in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_file_snippet_with_padding():
    """Padding expands the extracted range by ±N lines."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_snip_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()
        config.node_context_padding = 2

        touched = {
            "Function-src/app.py-main": {
                "label": "main",
                "line_start": 20,
                "line_end": 21,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_file_snippet(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        # Padding=2 → range 18..23
        assert "18 # line 18" in text
        assert "21     print('hello')" in text
        assert "23 # line 23" in text
        # Line 10 should not be in the output
        assert "10 class App:" not in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_file_snippet_multiple_ranges_gap():
    """Two non-adjacent ranges produce a gap marker."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_snip_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()
        config.node_context_padding = 0

        touched = {
            "Class-src/app.py-App": {"label": "App", "line_start": 10, "line_end": 12},
            "Function-src/app.py-main": {
                "label": "main",
                "line_start": 20,
                "line_end": 21,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_file_snippet(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        assert "10 class App:" in text
        assert "20 def main():" in text
        assert "... (" in text  # gap marker between 12 and 20
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_file_snippet_empty_touched():
    """No touched nodes with source → empty string."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_snip_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()

        touched = {"Import-File-src/app.py-0": {"label": "os"}}

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                text = assemble_file_snippet(
                    touched,
                    "test-repo",
                    "abc",
                    "bug",
                    str(graph_dir),
                    config,
                )

        assert text == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# assemble_node_source tests
# ---------------------------------------------------------------------------


def test_assemble_node_source_uses_graph_source_code():
    """Node source mode uses source_code from graph properties."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_node_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()
        config.node_context_padding = 0

        touched = {
            "Class-src/app.py-App": {
                "label": "App",
                "meta_type": "Class",
                "source_code": (
                    "class App:\n" "    def run(self):\n" "        return 'run'\n"
                ),
                "line_start": 10,
                "line_end": 12,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_node_source(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        assert "10 class App:" in text
        assert "11     def run(self):" in text
        assert "12         return 'run'" in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_node_source_no_file_padding():
    """Node source mode uses graph source_code only — no file padding."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_node_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()

        touched = {
            "Function-src/app.py-main": {
                "label": "main",
                "meta_type": "Function",
                "source_code": "def main():\n    print('hello')\n",
                "line_start": 20,
                "line_end": 21,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_node_source(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        # Graph source_code shown with original line numbers
        assert "20 def main():" in text
        assert "21     print('hello')" in text
        # NO file padding lines
        assert "18 # line 18" not in text
        assert "22 # line 22" not in text
        # Metadata header present
        assert "## Function: main" in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_node_source_deduplicates_class_method():
    """When both Class and its Method are touched, Class covers Method."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_node_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()
        config.node_context_padding = 0

        touched = {
            "Class-src/app.py-App": {
                "label": "App",
                "meta_type": "Class",
                "source_code": (
                    "class App:\n" "    def run(self):\n" "        return 'run'\n"
                ),
                "line_start": 10,
                "line_end": 12,
            },
            "Method-src/app.py-App-run": {
                "label": "run",
                "meta_type": "Method",
                "source_code": "    def run(self):\n        return 'run'\n",
                "line_start": 11,
                "line_end": 12,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_node_source(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        # Lines 10-12 should appear only once (no duplicate)
        assert text.count("10 class App:") == 1
        assert "... (" not in text  # no gap within the class
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_node_source_empty_touched():
    """No touched nodes with source_code → empty string."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_node_"))
    try:
        graph_dir, repo_dir = _setup_repo_with_source(tmpdir)
        config = KGConfig()

        touched = {"Import-File-src/app.py-0": {"label": "os"}}

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                text = assemble_node_source(
                    touched,
                    "test-repo",
                    "abc",
                    "bug",
                    str(graph_dir),
                    config,
                )

        assert text == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_node_source_overlapping_nodes_shown_separately():
    """Two partially overlapping nodes are shown as separate blocks.

    Function foo (6-16) and bar (9-18) overlap but neither contains
    the other. In pure-graph mode each node's source_code is shown
    independently with its own metadata header.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_overlap_"))
    try:
        # Graph with two overlapping functions in the same file
        graph = {
            "nodes": [
                {"id": "File-src/app.py", "label": "app.py"},
                {"id": "Function-src/app.py-foo", "label": "foo"},
                {"id": "Function-src/app.py-bar", "label": "bar"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "label": "CONTAINS",
                    "source": "File-src/app.py",
                    "target": "Function-src/app.py-foo",
                },
                {
                    "id": "e2",
                    "label": "CONTAINS",
                    "source": "File-src/app.py",
                    "target": "Function-src/app.py-bar",
                },
            ],
            "allValues": {
                "File-src/app.py": {
                    "meta_type": "File",
                    "relative_path": "src/app.py",
                },
                "Function-src/app.py-foo": {
                    "meta_type": "Function",
                    "source_code": "def foo():\n    pass\n",
                    "line_start": 6,
                    "line_end": 16,
                },
                "Function-src/app.py-bar": {
                    "meta_type": "Function",
                    "source_code": "def bar():\n    pass\n",
                    "line_start": 9,
                    "line_end": 18,
                },
            },
        }
        graph_dir = tmpdir / "graph"
        graph_dir.mkdir()
        _write_graph(graph_dir, graph)

        repo_dir = tmpdir / "repo"
        repo_dir.mkdir()

        config = KGConfig()

        touched = {
            "Function-src/app.py-foo": {
                "label": "foo",
                "meta_type": "Function",
                "source_code": "def foo():\n    pass\n",
                "line_start": 6,
                "line_end": 16,
            },
            "Function-src/app.py-bar": {
                "label": "bar",
                "meta_type": "Function",
                "source_code": "def bar():\n    pass\n",
                "line_start": 9,
                "line_end": 18,
            },
        }

        with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
            with patch("kg.build_graphs._checkout"):
                with patch("kg.context._build_prompt", side_effect=_noop_build_prompt):
                    text = assemble_node_source(
                        touched,
                        "test-repo",
                        "abc",
                        "bug",
                        str(graph_dir),
                        config,
                    )

        # Both functions appear with their metadata headers
        assert "## Function: foo" in text
        assert "## Function: bar" in text
        # Source lines with original numbers
        assert "6 def foo():" in text
        assert "9 def bar():" in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _collect_graph_context tests
# ---------------------------------------------------------------------------


def test_collect_graph_context_filters_to_code_nodes():
    """Only Function/Method/Class nodes pass through enrichment."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_cgc_"))
    try:
        graph_dir = tmpdir / "graph"
        graph_dir.mkdir()
        _write_graph(graph_dir, _make_code_graph_with_source())

        touched = {
            "Class-src/app.py-App": {
                "meta_type": "Class",
                "source_code": "class App:\n    pass\n",
                "line_start": 10,
                "line_end": 11,
            },
            "Import-File-src/app.py-0": {"label": "os"},
        }

        enriched, edges, file_map = _collect_graph_context(touched, str(graph_dir))

        assert "Class-src/app.py-App" in enriched
        assert "Import-File-src/app.py-0" not in enriched
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_collect_graph_context_excludes_source_from_edges():
    """Connected node props should not include source_code."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_cgc_"))
    try:
        graph_dir = tmpdir / "graph"
        graph_dir.mkdir()
        _write_graph(graph_dir, _make_code_graph_with_source())

        touched = {
            "Method-src/app.py-App-run": {
                "meta_type": "Method",
                "source_code": "    def run(self):\n        pass\n",
                "line_start": 11,
                "line_end": 12,
            },
        }

        enriched, edges, file_map = _collect_graph_context(touched, str(graph_dir))

        assert len(enriched) == 1
        # File mapping should be found via CONTAINS edge
        assert file_map.get("Method-src/app.py-App-run") == "src/app.py"
        # Edges collected for the method
        method_edges = edges.get("Method-src/app.py-App-run", [])
        assert len(method_edges) > 0
        # No connected_props should have source_code
        for e in method_edges:
            assert "source_code" not in e.get("connected_props", {})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _format_node_snippets tests
# ---------------------------------------------------------------------------


def test_format_node_snippets_metadata_header():
    """Metadata header shows type, name, complexity, docstring."""
    snippets = {
        "src/app.py": [
            {
                "line_start": 10,
                "line_end": 12,
                "source_code": "class App:\n    pass\n",
                "node_props": {
                    "meta_type": "Class",
                    "name": "App",
                    "complexity": 3,
                    "docstring": "Main class",
                },
                "edges": [],
            },
        ],
    }
    text = _format_node_snippets(snippets)
    assert "## Class: App | complexity=3" in text
    assert 'docstring="Main class"' in text
    assert "10 class App:" in text


def test_format_node_snippets_edge_lines():
    """Edge metadata lines show direction and connected node info."""
    snippets = {
        "src/app.py": [
            {
                "line_start": 20,
                "line_end": 21,
                "source_code": "def main():\n    pass\n",
                "node_props": {"meta_type": "Function", "name": "main"},
                "edges": [
                    {
                        "edge_label": "CALLS",
                        "direction": "outgoing",
                        "connected_props": {
                            "meta_type": "Function",
                            "name": "helper",
                            "line_start": 30,
                            "line_end": 35,
                        },
                    },
                    {
                        "edge_label": "CONTAINS",
                        "direction": "incoming",
                        "connected_props": {
                            "meta_type": "File",
                            "name": "app.py",
                        },
                    },
                ],
            },
        ],
    }
    text = _format_node_snippets(snippets)
    assert "##   --CALLS--> Function: helper (L30-35)" in text
    assert "##   <--CONTAINS-- File: app.py" in text


def test_format_node_snippets_gap_between_nodes():
    """Gap markers appear between non-adjacent nodes."""
    snippets = {
        "src/app.py": [
            {
                "line_start": 10,
                "line_end": 12,
                "source_code": "a\nb\nc\n",
                "node_props": {},
                "edges": [],
            },
            {
                "line_start": 20,
                "line_end": 21,
                "source_code": "d\ne\n",
                "node_props": {},
                "edges": [],
            },
        ],
    }
    text = _format_node_snippets(snippets)
    assert "... (7 lines omitted) ..." in text


def test_format_node_snippets_empty():
    assert _format_node_snippets({}) == ""


def test_format_node_snippets_file_uses_relative_path():
    """File nodes have ``relative_path`` not ``name`` — should show path."""
    snippets = {
        "src/app.py": [
            {
                "line_start": 10,
                "line_end": 10,
                "source_code": "x = 1\n",
                "node_props": {"meta_type": "Function", "name": "foo"},
                "edges": [
                    {
                        "edge_label": "CONTAINS",
                        "direction": "incoming",
                        "connected_props": {
                            "meta_type": "File",
                            "relative_path": "src/app.py",
                        },
                    },
                ],
            },
        ],
    }
    text = _format_node_snippets(snippets)
    assert "##   <--CONTAINS-- File: src/app.py" in text
    assert "File: ?" not in text


# ---------------------------------------------------------------------------
# read_files_within_budget tests
# ---------------------------------------------------------------------------


def _setup_budget_repo(tmp: Path) -> Path:
    """Repo with a small file, a big file, and a missing path."""
    repo_dir = tmp / "repo"
    repo_dir.mkdir()
    (repo_dir / "small.py").write_text("s = 1\n", encoding="utf-8")
    (repo_dir / "big.py").write_text("b" * 3000, encoding="utf-8")
    return repo_dir


def test_read_files_within_budget_breaks_on_overflow():
    """Files after the budget break are not read."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_budget_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_files_within_budget(str(repo_dir), ["small.py", "big.py"], 20)
        assert list(contents) == ["small.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_files_within_budget_first_file_always_included():
    """The first file is included even if it alone exceeds the budget."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_budget_first_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_files_within_budget(str(repo_dir), ["big.py"], 10)
        assert list(contents) == ["big.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_files_within_budget_skips_missing_and_continues():
    """A missing file is skipped; later files are still read."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_budget_missing_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_files_within_budget(
            str(repo_dir), ["gone.py", "small.py"], 128_000
        )
        assert list(contents) == ["small.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# read_all_files tests
def test_read_all_files_includes_every_file_without_budget():
    """All touched files must reach the reranker; the token budget
    dropped single-touch files (e.g. gt files) before the rerank
    stage could ever see them."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_all_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_all_files(str(repo_dir), ["small.py", "big.py"])
        assert list(contents) == ["small.py", "big.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_all_files_preserves_ranked_order():
    """Files stay in frequency-ranked order so the prompt shows the
    most-touched files first."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_all_order_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_all_files(str(repo_dir), ["big.py", "small.py"])
        assert list(contents) == ["big.py", "small.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_all_files_skips_missing():
    """A missing file is skipped; later files are still read."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ctx_all_missing_"))
    try:
        repo_dir = _setup_budget_repo(tmpdir)
        contents = read_all_files(str(repo_dir), ["gone.py", "small.py"])
        assert list(contents) == ["small.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
