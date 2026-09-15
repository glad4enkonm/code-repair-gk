"""Tests for FilesystemExecutor — walks repo, creates Directory + File nodes."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.python.filesystem import FilesystemExecutor
from code_kg_builder.meta_schema import MetaSchema


@pytest.fixture
def schema() -> MetaSchema:
    return MetaSchema.from_data(
        {
            "nodes": [
                {"id": "Directory", "label": "Directory"},
                {"id": "File", "label": "File"},
            ],
            "edges": [],
            "allValues": {
                "Directory": {"meta_type": "NodeType"},
                "File": {"meta_type": "NodeType"},
            },
        }
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a small test repo."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "src" / "utils" / "helper.py").write_text("def f():\n    pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("assert True\n")
    (tmp_path / "README.md").write_text("# project")
    return tmp_path


@pytest.fixture
def ctx(repo: Path, schema: MetaSchema) -> BuildContext:
    return BuildContext(repo_path=repo, meta_schema=schema)


class TestFilesystemExecutor:
    def test_creates_directory_nodes(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        dir_ids = set()
        for batch in batches:
            for n in batch.nodes_data:
                if n["id"].startswith("Directory-"):
                    dir_ids.add(n["id"])

        # root, src, src/utils, tests
        assert "Directory-src" in dir_ids
        assert "Directory-src/utils" in dir_ids
        assert "Directory-tests" in dir_ids

    def test_creates_file_nodes(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        file_ids = set()
        for batch in batches:
            for n in batch.nodes_data:
                if n["id"].startswith("File-"):
                    file_ids.add(n["id"])

        assert "File-src/main.py" in file_ids
        assert "File-src/utils/helper.py" in file_ids
        assert "File-tests/test_main.py" in file_ids

    def test_ignores_non_py_files(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        file_ids = set()
        for batch in batches:
            for n in batch.nodes_data:
                file_ids.add(n["id"])

        assert not any("README" in fid for fid in file_ids)

    def test_contains_edges_directory_to_file(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        edge_ids = set()
        for batch in batches:
            for e in batch.edges_data:
                edge_ids.add(e["id"])

        # Directory→File CONTAINS edges
        assert any("CONTAINS" in eid and "File-src/main.py" in eid for eid in edge_ids)

    def test_contains_edges_directory_to_directory(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        edge_ids = set()
        for batch in batches:
            for e in batch.edges_data:
                edge_ids.add(e["id"])

        # src → src/utils CONTAINS
        assert any("CONTAINS" in eid and "src/utils" in eid for eid in edge_ids)

    def test_file_has_relative_path_property(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        for batch in batches:
            for av in batch.all_values_data:
                if av.get("id") == "File-src/main.py":
                    assert av["relative_path"] == "src/main.py"
                    assert av["meta_type"] == "File"

    def test_directory_has_relative_path_property(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        for batch in batches:
            for av in batch.all_values_data:
                if av.get("id") == "Directory-src":
                    assert av["relative_path"] == "src"
                    assert av["meta_type"] == "Directory"

    def test_file_has_line_count(self, ctx, repo):
        exec_ = FilesystemExecutor()
        batches = list(exec_.run(ctx))

        for batch in batches:
            for av in batch.all_values_data:
                if av.get("id") == "File-src/main.py":
                    assert av["line_count"] == 1

    def test_populates_python_files_in_context(self, ctx, repo):
        exec_ = FilesystemExecutor()
        list(exec_.run(ctx))

        py_files = [str(f.relative_to(repo)) for f in ctx.python_files]
        assert "src/main.py" in py_files
        assert "src/utils/helper.py" in py_files
        assert "tests/test_main.py" in py_files

    def test_phase_zero(self):
        assert FilesystemExecutor.phase == 0
