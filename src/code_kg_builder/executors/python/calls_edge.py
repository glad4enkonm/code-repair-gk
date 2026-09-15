"""CallsEdgeExecutor — resolve function/method calls, create CALLS edges.

Phase 2.  For each Python file, walks Function/Method bodies to find
``ast.Call`` nodes.  Resolves the called function/method name using the
scope resolver.  Emits CALLS edges between resolvable pairs.
"""

import ast
from typing import Iterator, List, Optional, Tuple

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_edge_id,
    make_function_id,
    make_method_id,
    relative_path_str,
)
from code_kg_builder.executors.python.utils.scope_resolver import (
    build_name_index,
    resolve_call,
)


class CallsEdgeExecutor(Executor):
    """Resolve calls and emit CALLS edges."""

    phase = 2

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        repo = ctx.repo_path
        name_index = build_name_index(ctx)

        for file_path in ctx.python_files:
            try:
                tree = ctx.get_ast(file_path)
            except Exception as exc:
                ctx.log_error(str(file_path), str(exc))
                continue

            rel = relative_path_str(repo, file_path)
            batch = BatchResult()

            for fn_id, fn_node, class_name in self._iter_functions(tree, rel):
                if fn_id not in ctx.node_index:
                    continue

                for child in ast.walk(fn_node):
                    if not isinstance(child, ast.Call):
                        continue
                    target_id = resolve_call(child.func, class_name, name_index, ctx)
                    if target_id is None or target_id == fn_id:
                        continue

                    batch.edges_data.append(
                        {
                            "id": make_edge_id(fn_id, "CALLS", target_id),
                            "source": fn_id,
                            "target": target_id,
                            "label": "CALLS",
                        }
                    )

            if not batch.is_empty:
                yield batch

    def _iter_functions(
        self, tree: ast.Module, rel: str
    ) -> List[Tuple[str, ast.stmt, Optional[str]]]:
        """Return (function_id, ast_node, enclosing_class_name_or_None)."""
        result: List[Tuple[str, ast.stmt, Optional[str]]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.append((make_function_id(rel, node.name), node, None))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mid = make_method_id(rel, node.name, child.name)
                        result.append((mid, child, node.name))
        return result
