"""Tests for GraphWriter — wraps update_graph() for batched commits."""

import json
import shutil
from pathlib import Path

import pytest

from code_kg_builder.executors.base import BatchResult
from code_kg_builder.graph_writer import GraphWriter

KG_SRC = Path(__file__).resolve().parents[2] / "swe_bench_kg"


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """Copy swe_bench_kg JSON files into a temp dir."""
    for name in ("origin.json", "meta.json", "data.json"):
        shutil.copy2(KG_SRC / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def writer(graph_dir: Path, monkeypatch) -> GraphWriter:
    monkeypatch.chdir(graph_dir)
    return GraphWriter()


class TestCommit:
    def test_empty_batch_returns_sat(self, writer):
        result = writer.commit(BatchResult())
        assert result["status"] == "sat"

    def test_single_node(self, writer):
        batch = BatchResult(
            nodes_data=[{"id": "Directory-src", "label": "src"}],
            all_values_data=[
                {
                    "id": "Directory-src",
                    "meta_type": "Directory",
                    "relative_path": "src",
                },
            ],
        )
        result = writer.commit(batch)
        assert result["status"] == "sat"
        assert result["committed"] is True

    def test_node_with_edge(self, writer):
        batch = BatchResult(
            nodes_data=[
                {"id": "Directory-src", "label": "src"},
                {"id": "File-src/main.py", "label": "main.py"},
            ],
            edges_data=[
                {
                    "id": "Dir-CONTAINS-File",
                    "source": "Directory-src",
                    "target": "File-src/main.py",
                    "label": "CONTAINS",
                },
            ],
            all_values_data=[
                {
                    "id": "Directory-src",
                    "meta_type": "Directory",
                    "relative_path": "src",
                },
                {
                    "id": "File-src/main.py",
                    "meta_type": "File",
                    "relative_path": "src/main.py",
                },
            ],
        )
        result = writer.commit(batch)
        assert result["status"] == "sat"

    def test_data_written_to_file(self, writer, graph_dir):
        batch = BatchResult(
            nodes_data=[{"id": "Directory-src", "label": "src"}],
            all_values_data=[
                {
                    "id": "Directory-src",
                    "meta_type": "Directory",
                    "relative_path": "src",
                },
            ],
        )
        writer.commit(batch)
        data = json.loads((graph_dir / "data.json").read_text())
        nodes = data.get("nodes_data") or data.get("nodes") or []
        node_ids = [n["id"] for n in nodes]
        assert "Directory-src" in node_ids

    def test_unsat_returns_unsat(self, writer):
        """Node with invalid meta_type should fail validation."""
        batch = BatchResult(
            nodes_data=[{"id": "BadNode-1", "label": "bad"}],
            all_values_data=[
                {"id": "BadNode-1", "meta_type": "NonExistentType"},
            ],
        )
        result = writer.commit(batch)
        assert result["status"] == "unsat"
        assert result["committed"] is False


class TestCommitVerifyFalse:
    """Tests for verify=False — skip Z3 per-commit, commit without validation."""

    def test_verify_false_commits_invalid_data(self, writer, graph_dir):
        """Data that would fail Z3 is committed when verify=False."""
        batch = BatchResult(
            nodes_data=[{"id": "BadNode-1", "label": "bad"}],
            all_values_data=[
                {"id": "BadNode-1", "meta_type": "NonExistentType"},
            ],
        )
        result = writer.commit(batch, verify=False)
        assert result["status"] == "sat"
        assert result["committed"] is True

        data = json.loads((graph_dir / "data.json").read_text())
        node_ids = [n["id"] for n in data.get("nodes", [])]
        assert "BadNode-1" in node_ids

    def test_verify_false_writes_valid_data(self, writer, graph_dir):
        """Valid data is also committed with verify=False."""
        batch = BatchResult(
            nodes_data=[{"id": "Directory-src", "label": "src"}],
            all_values_data=[
                {
                    "id": "Directory-src",
                    "meta_type": "Directory",
                    "relative_path": "src",
                },
            ],
        )
        result = writer.commit(batch, verify=False)
        assert result["status"] == "sat"

        data = json.loads((graph_dir / "data.json").read_text())
        assert any(n["id"] == "Directory-src" for n in data["nodes"])

    def test_verify_true_default_still_rejects_invalid(self, writer):
        """Default verify=True still rejects invalid data (backward compat)."""
        batch = BatchResult(
            nodes_data=[{"id": "BadNode-2", "label": "bad"}],
            all_values_data=[
                {"id": "BadNode-2", "meta_type": "NonExistentType"},
            ],
        )
        result = writer.commit(batch)
        assert result["status"] == "unsat"
