"""RaisesEdgeExecutor — resolve raise statements, create RAISES edges + Exception nodes.

Phase 2.  For each Python file, walks the AST to find ``ast.Raise`` nodes
inside Function/Method bodies.  Extracts the exception name, creates an
Exception node if needed, and emits a RAISES edge.
"""

import ast
from typing import Dict, Iterator, Optional, Tuple

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_edge_id,
    make_exception_id,
    make_function_id,
    make_method_id,
    relative_path_str,
)


def _exception_name(node: Optional[ast.expr]) -> Optional[str]:
    """Extract exception class name from a Raise's exc or handler type."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _exception_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _build_var_to_exception(fn_node: ast.AST) -> Dict[str, str]:
    """Map variable names to exception class names from assignments.

    Scans the function body for patterns like::

        result = UnitsError("message")
        err = MyException

    Returns a dict mapping variable name → exception class name.
    Only class-like names (first char uppercase) on the RHS are mapped,
    to avoid false positives from non-exception assignments.
    """
    var_map: Dict[str, str] = {}
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Assign):
            continue
        exc_name = _exception_name(node.value)
        if exc_name is None or not exc_name[0].isupper():
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_map[target.id] = exc_name
    return var_map


class RaisesEdgeExecutor(Executor):
    """Resolve raise statements and emit RAISES edges + Exception nodes."""

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

                var_to_exc = _build_var_to_exception(fn_node)

                for child in ast.walk(fn_node):
                    if not isinstance(child, ast.Raise):
                        continue
                    exc_name = _exception_name(child.exc)
                    if exc_name is None:
                        continue

                    if exc_name in var_to_exc:
                        exc_name = var_to_exc[exc_name]
                    elif not exc_name[0].isupper():
                        continue

                    exc_id = make_exception_id(exc_name)
                    self._ensure_exception(exc_id, exc_name, batch, ctx)

                    batch.edges_data.append(
                        {
                            "id": make_edge_id(fn_id, "RAISES", exc_id),
                            "source": fn_id,
                            "target": exc_id,
                            "label": "RAISES",
                        }
                    )

            if not batch.is_empty:
                yield batch

    def _iter_functions(
        self, tree: ast.Module, rel: str
    ) -> Iterator[Tuple[str, ast.stmt]]:
        """Yield (function_or_method_id, ast_node) for all functions in tree."""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = make_function_id(rel, node.name)
                yield fid, node
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mid = make_method_id(rel, node.name, child.name)
                        yield mid, child

    def _ensure_exception(
        self,
        exc_id: str,
        exc_name: str,
        batch: BatchResult,
        ctx: BuildContext,
    ) -> None:
        if exc_id in ctx.node_index:
            return
        existing = [n for n in batch.nodes_data if n["id"] == exc_id]
        if existing:
            return
        batch.nodes_data.append({"id": exc_id, "label": exc_name})
        batch.all_values_data.append(
            {
                "id": exc_id,
                "meta_type": "Exception",
                "name": exc_name,
            }
        )
