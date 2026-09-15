"""Tests for InheritsEdgeExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.inherits_edge import InheritsEdgeExecutor

SAMPLE = """\
class Animal:
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "animals.py").write_text(SAMPLE)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "animals.py"]
    for name in ("Animal", "Dog", "Cat"):
        nid = f"Class-animals.py-{name}"
        c.node_index[nid] = NodeInfo(
            id=nid,
            label=name,
            meta_type="Class",
            properties={"name": name},
        )
    return c


class TestInheritsEdgeExecutor:
    def test_creates_inherits_edges(self, ctx):
        batches = list(InheritsEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)

        inherits = [e for e in edges if e["label"] == "INHERITS"]
        assert len(inherits) == 2

        targets = {e["target"] for e in inherits}
        assert "Class-animals.py-Animal" in targets

    def test_no_bogus_edges(self, ctx):
        batches = list(InheritsEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        sources = {e["source"] for e in edges}
        assert "Class-animals.py-Animal" not in sources
