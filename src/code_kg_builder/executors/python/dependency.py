"""DependencyExecutor — parse requirements.txt, create Dependency nodes.

Phase 1.  Parses ``requirements.txt`` at the repository root.  Each dependency
creates a Dependency node with name and optional version.  Creates DEPENDS_ON
edges from all File nodes to each Dependency.
"""

import re
from typing import Iterator, List

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_dependency_id,
    make_edge_id,
)

_REQ_RE = re.compile(r"^([a-zA-Z0-9_.-]+)\s*([<>=~!]+[\d.\*]+)?")
_FILES = ("requirements.txt", "requirements.in")


def _parse_requirements(content: str) -> List[tuple]:
    """Return [(name, version_or_none), ...] from requirements file content."""
    result = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _REQ_RE.match(line)
        if m:
            name = m.group(1)
            version = m.group(2) if m.group(2) else None
            result.append((name, version))
    return result


class DependencyExecutor(Executor):
    """Parse dependency files and emit Dependency nodes + DEPENDS_ON edges."""

    phase = 1

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        deps = []
        for fname in _FILES:
            req_path = ctx.repo_path / fname
            if req_path.exists():
                deps = _parse_requirements(
                    req_path.read_text(encoding="utf-8", errors="replace")
                )
                break

        if not deps:
            return

        batch = BatchResult()
        file_ids = [
            nid for nid, info in ctx.node_index.items() if info.meta_type == "File"
        ]

        for name, version in deps:
            dep_id = make_dependency_id(name)
            batch.nodes_data.append({"id": dep_id, "label": name})

            props: dict = {"id": dep_id, "meta_type": "Dependency", "name": name}
            if version:
                props["version"] = version
            batch.all_values_data.append(props)

            for fid in file_ids:
                batch.edges_data.append(
                    {
                        "id": make_edge_id(fid, "DEPENDS_ON", dep_id),
                        "source": fid,
                        "target": dep_id,
                        "label": "DEPENDS_ON",
                    }
                )

        if not batch.is_empty:
            yield batch
