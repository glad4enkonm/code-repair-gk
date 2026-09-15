"""Tests for DependencyExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.dependency import DependencyExecutor


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "requirements.txt").write_text(
        "django==4.2.1\nrequests>=2.0\nflask\n# comment\n-r other.txt\n"
    )
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.node_index["File-main.py"] = NodeInfo(
        id="File-main.py", label="main.py", meta_type="File"
    )
    return c


class TestDependencyExecutor:
    def test_creates_dependency_nodes(self, ctx):
        batches = list(DependencyExecutor().run(ctx))
        dep_ids = set()
        for b in batches:
            for n in b.nodes_data:
                dep_ids.add(n["id"])
        assert "Dependency-django" in dep_ids
        assert "Dependency-requests" in dep_ids
        assert "Dependency-flask" in dep_ids

    def test_version_parsed(self, ctx):
        batches = list(DependencyExecutor().run(ctx))
        for b in batches:
            for av in b.all_values_data:
                if av["id"] == "Dependency-django":
                    assert av["version"] == "==4.2.1"
                if av["id"] == "Dependency-flask":
                    assert "version" not in av

    def test_depends_on_edges(self, ctx):
        batches = list(DependencyExecutor().run(ctx))
        edge_labels = set()
        for b in batches:
            for e in b.edges_data:
                edge_labels.add(e["label"])
        assert "DEPENDS_ON" in edge_labels

    def test_no_requirements(self, tmp_path):
        c = BuildContext(repo_path=tmp_path)
        batches = list(DependencyExecutor().run(c))
        assert len(batches) == 0
