"""Tests for HasTypeEdgeExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.has_type_edge import HasTypeEdgeExecutor

SAMPLE = """\
class Config:
    timeout: int = 30
    name: str = "default"

    def get(self) -> str:
        return self.name
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "config.py").write_text(SAMPLE)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "config.py"]
    c.node_index["Class-config.py-Config"] = NodeInfo(
        id="Class-config.py-Config",
        label="Config",
        meta_type="Class",
        properties={"name": "Config"},
    )
    c.node_index["Variable-Class-config.py-Config-timeout"] = NodeInfo(
        id="Variable-Class-config.py-Config-timeout",
        label="timeout",
        meta_type="Variable",
        properties={"name": "timeout"},
    )
    c.node_index["Variable-Class-config.py-Config-name"] = NodeInfo(
        id="Variable-Class-config.py-Config-name",
        label="name",
        meta_type="Variable",
        properties={"name": "name"},
    )
    return c


class TestHasTypeEdgeExecutor:
    def test_creates_type_annotation_nodes(self, ctx):
        batches = list(HasTypeEdgeExecutor().run(ctx))
        nodes = []
        for b in batches:
            nodes.extend(b.nodes_data)
        ta_ids = {n["id"] for n in nodes}
        assert "TypeAnnotation-int" in ta_ids
        assert "TypeAnnotation-str" in ta_ids

    def test_creates_has_type_edges(self, ctx):
        batches = list(HasTypeEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        has_type = [e for e in edges if e["label"] == "HAS_TYPE"]
        assert len(has_type) == 2

    def test_edge_connects_variable_to_type(self, ctx):
        batches = list(HasTypeEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        for e in edges:
            if e["label"] == "HAS_TYPE" and e["target"] == "TypeAnnotation-int":
                assert e["source"] == "Variable-Class-config.py-Config-timeout"
                return
        assert False, "int edge not found"
