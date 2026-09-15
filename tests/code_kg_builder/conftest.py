"""Shared fixtures for code_kg_builder tests."""

import shutil
from pathlib import Path

import pytest

KG_SRC = Path(__file__).resolve().parents[2] / "swe_bench_kg"


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """Copy swe_bench_kg JSON files into a temp dir."""
    for name in ("origin.json", "meta.json", "data.json"):
        shutil.copy2(KG_SRC / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small test Python repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "import os\n\n\nVERSION = '1.0'\n\n\n"
        "class App:\n    def run(self):\n        return self.helper()\n\n"
        "    def helper(self):\n        return 1\n\n\n"
        "def main():\n    app = App()\n    return app.run()\n"
    )
    (repo / "requirements.txt").write_text("click>=8.0\n")
    return repo
