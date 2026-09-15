"""InheritsEdgeExecutor — resolve class inheritance, create INHERITS edges.

Phase 2.  For each Python file, finds ``ast.ClassDef`` nodes and their bases.
For each base that is a simple ``ast.Name``, resolves it to a Class node in
the index by name.  Creates INHERITS edges.
"""

import ast
from typing import Iterator

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_class_id,
    make_edge_id,
    relative_path_str,
)


class InheritsEdgeExecutor(Executor):
    """Resolve class inheritance and emit INHERITS edges."""

    phase = 2

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        repo = ctx.repo_path
        for file_path in ctx.python_files:
            try:
                tree = ctx.get_ast(file_path)
            except Exception as exc:
                ctx.log_error(str(file_path), str(exc))
                continue

            rel = relative_path_str(repo, file_path)
            batch = BatchResult()

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                child_id = make_class_id(rel, node.name)
                if child_id not in ctx.node_index:
                    continue

                for base in node.bases:
                    base_name = self._base_name(base)
                    if base_name is None:
                        continue

                    parent_info = ctx.find_node("Class", name=base_name)
                    if parent_info is None:
                        continue

                    batch.edges_data.append(
                        {
                            "id": make_edge_id(child_id, "INHERITS", parent_info.id),
                            "source": child_id,
                            "target": parent_info.id,
                            "label": "INHERITS",
                        }
                    )

            if not batch.is_empty:
                yield batch

    @staticmethod
    def _base_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None
