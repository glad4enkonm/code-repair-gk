"""Tests for CatchesEdgeExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.catches_edge import CatchesEdgeExecutor

SAMPLE = """\
def handler():
    try:
        risky()
    except ValueError:
        pass
    except KeyError:
        pass


class Worker:
    def run(self):
        try:
            self.do()
        except Exception:
            raise
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "handlers.py").write_text(SAMPLE)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "handlers.py"]
    c.node_index["Function-handlers.py-handler"] = NodeInfo(
        id="Function-handlers.py-handler",
        label="handler",
        meta_type="Function",
        properties={"name": "handler"},
    )
    c.node_index["Method-handlers.py-Worker-run"] = NodeInfo(
        id="Method-handlers.py-Worker-run",
        label="run",
        meta_type="Method",
        properties={"name": "run"},
    )
    return c


class TestCatchesEdgeExecutor:
    def test_creates_catches_edges(self, ctx):
        batches = list(CatchesEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        catches = [e for e in edges if e["label"] == "CATCHES"]
        assert len(catches) >= 3

    def test_catches_correct_exceptions(self, ctx):
        batches = list(CatchesEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        targets = {e["target"] for e in edges if e["label"] == "CATCHES"}
        assert "Exception-ValueError" in targets
        assert "Exception-KeyError" in targets
