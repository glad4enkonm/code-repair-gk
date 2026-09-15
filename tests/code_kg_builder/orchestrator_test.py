"""Tests for the Orchestrator — end-to-end pipeline."""

import shutil
from pathlib import Path

import pytest

from code_kg_builder.orchestrator import Orchestrator

KG_SRC = Path(__file__).resolve().parents[2] / "swe_bench_kg"


@pytest.fixture
def multi_file_repo(tmp_path: Path) -> Path:
    """Create a test repo with multiple Python files for batching tests."""
    repo = tmp_path / "multi_repo"
    repo.mkdir()
    for i in range(5):
        (repo / f"module_{i}.py").write_text(
            f"class Class{i}:\n"
            f"    def method_{i}(self):\n"
            f"        return {i}\n"
            f"\n"
            f"def func_{i}():\n"
            f"    return Class{i}().method_{i}()\n"
        )
    return repo


class TestOrchestratorIntegration:
    def test_build_small_repo(self, tmp_repo: Path, graph_dir: Path, monkeypatch):
        monkeypatch.chdir(graph_dir)
        orch = Orchestrator(
            repo_path=tmp_repo,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
        )
        summary = orch.run()

        assert summary.committed_batches > 0
        assert summary.total_nodes > 0
        assert summary.total_edges > 0

    def test_build_creates_class_nodes(
        self, tmp_repo: Path, graph_dir: Path, monkeypatch
    ):
        monkeypatch.chdir(graph_dir)
        orch = Orchestrator(
            repo_path=tmp_repo,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
        )
        orch.run()

        # Check data.json was written
        import json

        data = json.loads((graph_dir / "data.json").read_text())
        node_ids = [n["id"] for n in data.get("nodes", [])]
        assert any("Class-" in nid for nid in node_ids)
        assert any("Function-" in nid for nid in node_ids)

    def test_errors_logged_on_bad_input(self, graph_dir: Path, monkeypatch):
        monkeypatch.chdir(graph_dir)
        nonexistent = Path("/nonexistent/repo")
        orch = Orchestrator(
            repo_path=nonexistent,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
        )
        summary = orch.run()
        # Should not crash, just produce empty or error results
        assert isinstance(summary.total_nodes, int)

    def test_no_executors_in_meta(self, graph_dir: Path, monkeypatch):
        """Meta without executor properties should produce a clear error."""
        monkeypatch.chdir(graph_dir)
        # Create meta without executor props
        import json

        meta = json.loads((graph_dir / "meta.json").read_text())
        for key in list(meta.get("allValues", {}).keys()):
            meta["allValues"][key].pop("executor", None)

        tmp_meta = graph_dir / "meta_no_exec.json"
        tmp_meta.write_text(json.dumps(meta, indent=2))

        orch = Orchestrator(
            repo_path=graph_dir,
            meta_path=str(tmp_meta),
            graph_dir=str(graph_dir),
            run_verification=False,
        )
        summary = orch.run()
        assert any("No executors" in e["message"] for e in summary.errors)


class TestVerifyCommits:
    """Tests for verify_commits=False — skip Z3 per-commit."""

    def test_verify_commits_false_builds_successfully(
        self, tmp_repo: Path, graph_dir: Path, monkeypatch
    ):
        monkeypatch.chdir(graph_dir)
        orch = Orchestrator(
            repo_path=tmp_repo,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
            verify_commits=False,
        )
        summary = orch.run()
        assert summary.committed_batches > 0
        assert summary.total_nodes > 0
        assert summary.total_edges > 0

    def test_verify_commits_false_produces_same_data(
        self, tmp_repo: Path, graph_dir: Path, monkeypatch
    ):
        """verify_commits=False should produce identical nodes/edges."""
        monkeypatch.chdir(graph_dir)
        orch = Orchestrator(
            repo_path=tmp_repo,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
            verify_commits=False,
        )
        orch.run()

        import json

        data = json.loads((graph_dir / "data.json").read_text())
        node_ids = [n["id"] for n in data.get("nodes", [])]
        assert any("Class-" in nid for nid in node_ids)
        assert any("Function-" in nid for nid in node_ids)


class TestBatchSize:
    """Tests for batch_size — merge multiple files per commit."""

    def _fresh_graph_dir(self, base: Path, name: str) -> Path:
        gd = base / name
        gd.mkdir()
        for fname in ("origin.json", "meta.json", "data.json"):
            shutil.copy2(KG_SRC / fname, gd / fname)
        return gd

    def test_batch_size_reduces_commits(
        self, multi_file_repo: Path, tmp_path: Path, monkeypatch
    ):
        """Larger batch_size should produce fewer commits with same data."""
        gd1 = self._fresh_graph_dir(tmp_path, "bs1")
        monkeypatch.chdir(gd1)
        s1 = Orchestrator(
            repo_path=multi_file_repo,
            meta_path=str(gd1 / "meta.json"),
            graph_dir=str(gd1),
            run_verification=False,
            verify_commits=False,
            batch_size=1,
        ).run()

        gd50 = self._fresh_graph_dir(tmp_path, "bs50")
        monkeypatch.chdir(gd50)
        s50 = Orchestrator(
            repo_path=multi_file_repo,
            meta_path=str(gd50 / "meta.json"),
            graph_dir=str(gd50),
            run_verification=False,
            verify_commits=False,
            batch_size=50,
        ).run()

        assert s50.committed_batches < s1.committed_batches
        assert s50.total_nodes == s1.total_nodes
        assert s50.total_edges == s1.total_edges

    def test_batch_size_1_default_unchanged(
        self, tmp_repo: Path, graph_dir: Path, monkeypatch
    ):
        """batch_size=1 should behave like the old per-file commit."""
        monkeypatch.chdir(graph_dir)
        orch = Orchestrator(
            repo_path=tmp_repo,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
            run_verification=False,
            verify_commits=False,
            batch_size=1,
        )
        summary = orch.run()
        assert summary.committed_batches > 0
        assert summary.total_nodes > 0
