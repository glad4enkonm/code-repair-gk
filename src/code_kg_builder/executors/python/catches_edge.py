"""CatchesEdgeExecutor — resolve except handlers, create CATCHES edges.

Phase 2.  For each Python file, walks Function/Method bodies to find
``ast.Try`` handlers.  Extracts exception names and emits CATCHES edges.
Reuses Exception nodes created by RaisesEdgeExecutor or creates new ones.
"""

import ast
from typing import Any, Iterator, List, Tuple

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_edge_id,
    make_exception_id,
    make_function_id,
    make_method_id,
    relative_path_str,
)
from code_kg_builder.executors.python.raises_edge import _exception_name


class CatchesEdgeExecutor(Executor):
    """Resolve try/except handlers and emit CATCHES edges."""

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

            for fn_id, fn_node in self._iter_functions(tree, rel):
                if fn_id not in ctx.node_index:
                    continue

                for child in ast.walk(fn_node):
                    if not isinstance(child, ast.Try):
                        continue
                    for handler in child.handlers:
                        names = self._handler_names(handler)
                        for exc_name in names:
                            exc_id = make_exception_id(exc_name)
                            self._ensure_exception(exc_id, exc_name, batch, ctx)

                            batch.edges_data.append(
                                {
                                    "id": make_edge_id(fn_id, "CATCHES", exc_id),
                                    "source": fn_id,
                                    "target": exc_id,
                                    "label": "CATCHES",
                                }
                            )

            if not batch.is_empty:
                yield batch

    def _iter_functions(self, tree: ast.Module, rel: str) -> List[Tuple[str, Any]]:
        result: List[Tuple[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.append((make_function_id(rel, node.name), node))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mid = make_method_id(rel, node.name, child.name)
                        result.append((mid, child))
        return result

    @staticmethod
    def _handler_names(handler: ast.ExceptHandler) -> List[str]:
        if handler.type is None:
            return ["Exception"]
        name = _exception_name(handler.type)
        return [name] if name else []

    def _ensure_exception(
        self,
        exc_id: str,
        exc_name: str,
        batch: BatchResult,
        ctx: BuildContext,
    ) -> None:
        if exc_id in ctx.node_index:
            return
        if any(n["id"] == exc_id for n in batch.nodes_data):
            return
        batch.nodes_data.append({"id": exc_id, "label": exc_name})
        batch.all_values_data.append(
            {
                "id": exc_id,
                "meta_type": "Exception",
                "name": exc_name,
            }
        )
