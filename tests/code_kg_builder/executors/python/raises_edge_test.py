"""Tests for RaisesEdgeExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.raises_edge import RaisesEdgeExecutor

SAMPLE = """\
def risky():
    raise ValueError("bad")

def safe():
    pass

class Handler:
    def process(self):
        raise RuntimeError("oops")
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "errors.py").write_text(SAMPLE)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "errors.py"]
    c.node_index["Function-errors.py-risky"] = NodeInfo(
        id="Function-errors.py-risky",
        label="risky",
        meta_type="Function",
        properties={"name": "risky"},
    )
    c.node_index["Function-errors.py-safe"] = NodeInfo(
        id="Function-errors.py-safe",
        label="safe",
        meta_type="Function",
        properties={"name": "safe"},
    )
    c.node_index["Method-errors.py-Handler-process"] = NodeInfo(
        id="Method-errors.py-Handler-process",
        label="process",
        meta_type="Method",
        properties={"name": "process"},
    )
    return c


class TestRaisesEdgeExecutor:
    def test_creates_exception_nodes(self, ctx):
        batches = list(RaisesEdgeExecutor().run(ctx))
        nodes = []
        for b in batches:
            nodes.extend(b.nodes_data)
        exc_ids = {n["id"] for n in nodes}
        assert "Exception-ValueError" in exc_ids
        assert "Exception-RuntimeError" in exc_ids

    def test_creates_raises_edges(self, ctx):
        batches = list(RaisesEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        raises = [e for e in edges if e["label"] == "RAISES"]
        assert len(raises) == 2

    def test_raises_from_correct_function(self, ctx):
        batches = list(RaisesEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        sources = {e["source"] for e in edges if e["label"] == "RAISES"}
        assert "Function-errors.py-risky" in sources
        assert "Method-errors.py-Handler-process" in sources


SAMPLE_VAR = """\
def compose():
    result = UnitsError("Cannot represent unit")
    raise result

def handler():
    try:
        pass
    except ValueError as exc:
        raise exc
"""


@pytest.fixture
def var_repo(tmp_path: Path) -> Path:
    (tmp_path / "var_raises.py").write_text(SAMPLE_VAR)
    return tmp_path


@pytest.fixture
def var_ctx(var_repo: Path) -> BuildContext:
    c = BuildContext(repo_path=var_repo)
    c.python_files = [var_repo / "var_raises.py"]
    c.node_index["Function-var_raises.py-compose"] = NodeInfo(
        id="Function-var_raises.py-compose",
        label="compose",
        meta_type="Function",
        properties={"name": "compose"},
    )
    c.node_index["Function-var_raises.py-handler"] = NodeInfo(
        id="Function-var_raises.py-handler",
        label="handler",
        meta_type="Function",
        properties={"name": "handler"},
    )
    return c


class TestRaisesVariableResolution:
    """When ``raise variable_name`` is used, the executor must resolve the
    variable to the actual exception class assigned to it, not create a
    false Exception node named after the variable.
    """

    def test_resolves_variable_to_exception_class(self, var_ctx):
        """``raise result`` where ``result = UnitsError(...)`` should create
        Exception-UnitsError, not Exception-result."""
        batches = list(RaisesEdgeExecutor().run(var_ctx))
        nodes = []
        for b in batches:
            nodes.extend(b.nodes_data)
        exc_ids = {n["id"] for n in nodes}

        assert "Exception-UnitsError" in exc_ids
        assert "Exception-result" not in exc_ids

    def test_skips_unresolvable_variable(self, var_ctx):
        """``raise exc`` (from except clause) cannot be resolved — skip it
        rather than creating a false Exception-exc node."""
        batches = list(RaisesEdgeExecutor().run(var_ctx))
        nodes = []
        for b in batches:
            nodes.extend(b.nodes_data)
        exc_ids = {n["id"] for n in nodes}

        assert "Exception-exc" not in exc_ids
