"""Tests for AstStructureExecutor — parses .py files, creates AST nodes."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult
from code_kg_builder.executors.python.ast_structure import AstStructureExecutor

SAMPLE_PY = """\
import os
from typing import Optional

VERSION = "1.0"

class Base:
    '''Base class.'''
    class_attr = 42

    def method(self):
        return self.class_attr

    @staticmethod
    def static_method():
        return 1

    @classmethod
    def cls_method(cls):
        return cls


def standalone(x: int) -> str:
    return str(x)


async def async_func():
    pass
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text(SAMPLE_PY)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "main.py"]
    return c


def collect_batches(executor, ctx):
    """Run executor and merge all batches into one."""
    merged = BatchResult()
    for batch in executor.run(ctx):
        merged = merged.merge(batch)
    return merged


def find_node(batch, node_id):
    for n in batch.nodes_data:
        if n["id"] == node_id:
            return n
    return None


def find_props(batch, node_id):
    for av in batch.all_values_data:
        if av.get("id") == node_id:
            return av
    return None


def find_edge(batch, source, label, target):
    for e in batch.edges_data:
        if e["source"] == source and e["label"] == label and e["target"] == target:
            return e
    return None


class TestClassNodes:
    def test_creates_class_node(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        assert find_node(batch, "Class-main.py-Base") is not None

    def test_class_properties(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Class-main.py-Base")
        assert props is not None
        assert props["meta_type"] == "Class"
        assert props["name"] == "Base"
        assert props["source_code"] is not None
        assert "class Base" in props["source_code"]
        assert props["docstring"] == "Base class."

    def test_class_line_numbers(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Class-main.py-Base")
        assert props["line_start"] == 6
        assert props["line_end"] is not None

    def test_class_complexity(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Class-main.py-Base")
        assert props["complexity"] is not None


class TestFunctionNodes:
    def test_creates_function_node(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        assert find_node(batch, "Function-main.py-standalone") is not None

    def test_function_properties(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Function-main.py-standalone")
        assert props["meta_type"] == "Function"
        assert props["name"] == "standalone"
        assert props["return_type"] is not None

    def test_async_function(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Function-main.py-async_func")
        assert props["is_async"] is True


class TestMethodNodes:
    def test_creates_method_nodes(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        assert find_node(batch, "Method-main.py-Base-method") is not None
        assert find_node(batch, "Method-main.py-Base-static_method") is not None
        assert find_node(batch, "Method-main.py-Base-cls_method") is not None

    def test_static_method_flag(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Method-main.py-Base-static_method")
        assert props["is_staticmethod"] is True

    def test_classmethod_flag(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        props = find_props(batch, "Method-main.py-Base-cls_method")
        assert props["is_classmethod"] is True


class TestVariableNodes:
    def test_module_level_variable(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        # Variable-Function-...-VERSION or Variable-{parent}-VERSION
        var_nodes = [n for n in batch.nodes_data if n["id"].startswith("Variable-")]
        version_vars = [v for v in var_nodes if "VERSION" in v["id"]]
        assert len(version_vars) == 1

    def test_class_level_variable(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        var_nodes = [n for n in batch.nodes_data if "class_attr" in n["id"]]
        assert len(var_nodes) == 1


class TestImportNodes:
    def test_creates_import_nodes(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        import_nodes = [n for n in batch.nodes_data if n["id"].startswith("Import-")]
        assert len(import_nodes) >= 2  # import os + from typing import Optional

    def test_import_properties(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        for av in batch.all_values_data:
            if av.get("id", "").startswith("Import-"):
                assert av["meta_type"] == "Import"
                assert "module_path" in av or "imported_name" in av


class TestContainsEdges:
    def test_file_contains_class(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        e = find_edge(
            batch,
            "File-main.py",
            "CONTAINS",
            "Class-main.py-Base",
        )
        assert e is not None

    def test_file_contains_function(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        e = find_edge(
            batch,
            "File-main.py",
            "CONTAINS",
            "Function-main.py-standalone",
        )
        assert e is not None

    def test_class_contains_method(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        e = find_edge(
            batch,
            "Class-main.py-Base",
            "CONTAINS",
            "Method-main.py-Base-method",
        )
        assert e is not None


class TestDecoratorNodes:
    def test_creates_decorator_for_static_method(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        dec_nodes = [n for n in batch.nodes_data if "staticmethod" in n["id"]]
        assert len(dec_nodes) == 1

    def test_decorates_edge(self, ctx):
        batch = collect_batches(AstStructureExecutor(), ctx)
        target = "Method-main.py-Base-static_method"
        dec_edges = [
            e
            for e in batch.edges_data
            if e["label"] == "DECORATES" and e["target"] == target
        ]
        assert len(dec_edges) == 1


class TestErrorHandling:
    def test_syntax_error_file_skipped(self, ctx, repo):
        bad_file = repo / "bad.py"
        bad_file.write_text("def broken(:\n")
        ctx.python_files.append(bad_file)

        executor = AstStructureExecutor()
        # Should not raise, just log error
        batches = list(executor.run(ctx))
        assert len(batches) > 0  # good file still processed

    def test_empty_file(self, ctx, repo):
        empty = repo / "empty.py"
        empty.write_text("")
        ctx.python_files = [empty]

        executor = AstStructureExecutor()
        batches = list(executor.run(ctx))
        # No error, just no nodes
        assert all(b.is_empty for b in batches) or len(batches) == 0
