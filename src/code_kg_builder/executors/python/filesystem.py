"""FilesystemExecutor — walks the repo, creates Directory + File nodes.

Phase 0.  Creates:
  - Directory nodes for every directory in a subtree containing .py files
  - File nodes for every .py file
  - CONTAINS edges: Directory→File, Directory→sub-Directory

Also populates ``ctx.python_files`` for use by later executors.
"""

import logging
import os
from pathlib import Path
from typing import Iterator

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_directory_id,
    make_edge_id,
    make_file_id,
    relative_path_str,
)

# Directories to always skip
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    ".venv_alpine",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "htmlcov",
    "build",
    "dist",
    ".eggs",
    ".tox",
}


def _has_py_in_subtree(path: Path) -> bool:
    """Quick check: does *path* or any descendant contain a .py file?"""
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if any(f.endswith(".py") for f in files):
            return True
    return False


class FilesystemExecutor(Executor):
    """Walk the repository filesystem and emit Directory/File nodes."""

    phase = 0

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        repo = ctx.repo_path
        all_nodes: list[dict] = []
        all_edges: list[dict] = []
        all_values: list[dict] = []

        for dirpath, dirnames, filenames in os.walk(repo):
            # Prune unwanted directories in-place
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)

            py_files = sorted(f for f in filenames if f.endswith(".py"))

            dir_path = Path(dirpath)
            try:
                rel_dir = dir_path.relative_to(repo).as_posix()
                if rel_dir == ".":
                    rel_dir = ""
            except ValueError as exc:
                logging.getLogger(__name__).warning(
                    "directory %s not under repo root %s — skipped: %s",
                    dir_path,
                    repo,
                    exc,
                )
                continue

            # Skip directories with no .py files in subtree
            if not py_files:
                if not any(_has_py_in_subtree(dir_path / d) for d in dirnames):
                    continue

            dir_id = make_directory_id(rel_dir)
            all_nodes.append({"id": dir_id, "label": rel_dir or "."})
            all_values.append(
                {
                    "id": dir_id,
                    "meta_type": "Directory",
                    "relative_path": rel_dir,
                }
            )

            # File nodes + CONTAINS Directory→File
            for py_file in py_files:
                file_path = dir_path / py_file
                rel_file = relative_path_str(repo, file_path)
                file_id = make_file_id(rel_file)

                all_nodes.append({"id": file_id, "label": py_file})
                line_count = sum(
                    1
                    for _ in file_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                )
                all_values.append(
                    {
                        "id": file_id,
                        "meta_type": "File",
                        "relative_path": rel_file,
                        "line_count": line_count,
                    }
                )

                all_edges.append(
                    {
                        "id": make_edge_id(dir_id, "CONTAINS", file_id),
                        "source": dir_id,
                        "target": file_id,
                        "label": "CONTAINS",
                    }
                )

                ctx.python_files.append(file_path)

            # CONTAINS Directory→sub-Directory
            for d in dirnames:
                sub_path = dir_path / d
                if not _has_py_in_subtree(sub_path):
                    continue
                sub_rel = relative_path_str(repo, sub_path)
                sub_id = make_directory_id(sub_rel)

                all_edges.append(
                    {
                        "id": make_edge_id(dir_id, "CONTAINS", sub_id),
                        "source": dir_id,
                        "target": sub_id,
                        "label": "CONTAINS",
                    }
                )

        if all_nodes or all_edges:
            yield BatchResult(
                nodes_data=all_nodes,
                edges_data=all_edges,
                all_values_data=all_values,
            )
