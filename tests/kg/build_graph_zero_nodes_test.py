"""Debug test: reproduce 0-nodes bug on pylint/sphinx commits.

Run locally to investigate why the orchestrator produces 0 nodes for
certain repos (pylint, sphinx) while working fine for others (seaborn).

Usage:
    pytest tests/kg/build_graph_zero_nodes_test.py -s -v
"""

import json
from pathlib import Path

import pytest
from git import Repo

from kg.build_graphs import build_graph
from kg.config import KGConfig

pytestmark = pytest.mark.slow


PYLINT_COMMIT = "3c5eca2ded3dd2b59ebaf23eb289453b5d2930f0"
SPHINX_COMMIT = "2c2335bbb8af99fa132e1573bbf45dc91584d5a2"


def _make_config(tmpdir: str) -> KGConfig:
    config = KGConfig()
    config.graph_cache_dir = str(Path(tmpdir) / "kg_graphs")
    config.repo_cache_dir = str(Path(tmpdir) / "kg_repos")
    config.meta_schema_dir = "swe_bench_kg"
    return config


def _clone_repo(repo: str, commit: str, dest: Path):
    """Clone a repo and checkout a specific commit."""
    repo_dir = dest / repo.replace("/", "__")
    if not repo_dir.exists():
        url = f"https://github.com/swe-bench-repos/{repo.replace('/', '__')}.git"
        print(f"Cloning {repo} from {url} ...")
        Repo.clone_from(url, repo_dir)
    repo_obj = Repo(str(repo_dir))
    repo_obj.git.checkout(commit)
    return repo_dir


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with graph cache + repo cache."""
    config = _make_config(str(tmp_path))
    Path(config.graph_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.repo_cache_dir).mkdir(parents=True, exist_ok=True)
    yield config, tmp_path


class TestPylintZeroNodes:
    """Reproduce: pylint produces 0 nodes."""

    def test_pylint_produces_nodes(self, tmp_workspace):
        config, tmp_path = tmp_workspace

        repo_dir = _clone_repo(
            "pylint-dev/pylint", PYLINT_COMMIT, Path(config.repo_cache_dir)
        )

        py_count = sum(1 for _ in repo_dir.rglob("*.py"))
        print(f"pylint .py files in repo: {py_count}")
        assert py_count > 0, "Repo has no .py files"

        graph_dir = build_graph("pylint-dev/pylint", PYLINT_COMMIT, config)

        data_file = graph_dir / "data.json"
        assert data_file.exists(), "data.json not created"

        with open(data_file) as f:
            data = json.load(f)

        node_count = len(data.get("nodes", []))
        edge_count = len(data.get("edges", []))
        print(f"pylint graph: {node_count} nodes, {edge_count} edges")

        assert node_count > 0, (
            f"Expected nodes > 0 but got {node_count}. "
            f"Repo has {py_count} .py files."
        )


class TestSphinxZeroNodes:
    """Reproduce: sphinx produces 0 nodes."""

    def test_sphinx_produces_nodes(self, tmp_workspace):
        config, tmp_path = tmp_workspace

        repo_dir = _clone_repo(
            "sphinx-doc/sphinx", SPHINX_COMMIT, Path(config.repo_cache_dir)
        )

        py_count = sum(1 for _ in repo_dir.rglob("*.py"))
        print(f"sphinx .py files in repo: {py_count}")
        assert py_count > 0, "Repo has no .py files"

        graph_dir = build_graph("sphinx-doc/sphinx", SPHINX_COMMIT, config)

        data_file = graph_dir / "data.json"
        assert data_file.exists(), "data.json not created"

        with open(data_file) as f:
            data = json.load(f)

        node_count = len(data.get("nodes", []))
        edge_count = len(data.get("edges", []))
        print(f"sphinx graph: {node_count} nodes, {edge_count} edges")

        assert node_count > 0, (
            f"Expected nodes > 0 but got {node_count}. "
            f"Repo has {py_count} .py files."
        )


class TestFilesystemExecutorDebug:
    """Standalone test: run FilesystemExecutor directly on pylint repo."""

    def test_filesystem_executor_finds_files(self, tmp_path):
        """Run FilesystemExecutor on a cloned pylint repo and check output."""
        from code_kg_builder.context import BuildContext
        from code_kg_builder.executors.python.filesystem import (
            FilesystemExecutor,
        )

        repo_dir = _clone_repo("pylint-dev/pylint", PYLINT_COMMIT, tmp_path / "repos")

        ctx = BuildContext(
            repo_path=repo_dir,
            meta_schema=None,
            graph_dir=str(tmp_path / "graph"),
        )

        executor = FilesystemExecutor()
        batches = list(executor.run(ctx))

        total_nodes = sum(len(b.nodes) for b in batches if hasattr(b, "nodes"))
        total_files = len(ctx.python_files)

        print("FilesystemExecutor on pylint:")
        print(f"  batches: {len(batches)}")
        print(f"  nodes:   {total_nodes}")
        print(f"  python_files in ctx: {total_files}")

        assert total_files > 0, "FilesystemExecutor found 0 .py files in pylint repo"
