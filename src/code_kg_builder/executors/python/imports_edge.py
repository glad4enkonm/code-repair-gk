"""ImportsEdgeExecutor — resolve import statements, create IMPORTS edges.

Phase 2.  For each Import node in the index, attempts to resolve the
imported module to a Function or Class node within the project.  Emits
IMPORTS edges for resolved imports.
"""

import ast
from typing import Dict, Iterator, List

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_edge_id,
    relative_path_str,
)
from code_kg_builder.executors.python.utils.scope_resolver import build_name_index


class ImportsEdgeExecutor(Executor):
    """Resolve imports and emit IMPORTS edges."""

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
            file_id = f"File-{rel}"
            batch = BatchResult()

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.names and node.names[0].name == "*":
                        continue
                    for alias in node.names:
                        imported = alias.asname or alias.name
                        target = self._resolve_name(imported, name_index, ctx)
                        if target and target != file_id:
                            source_fn = self._find_importing_function(
                                tree, rel, node.lineno
                            )
                            source_id = source_fn or file_id
                            batch.edges_data.append(
                                {
                                    "id": make_edge_id(source_id, "IMPORTS", target),
                                    "source": source_id,
                                    "target": target,
                                    "label": "IMPORTS",
                                }
                            )

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        target = self._resolve_name(name, name_index, ctx)
                        if target and target != file_id:
                            batch.edges_data.append(
                                {
                                    "id": make_edge_id(file_id, "IMPORTS", target),
                                    "source": file_id,
                                    "target": target,
                                    "label": "IMPORTS",
                                }
                            )

            if not batch.is_empty:
                yield batch

    @staticmethod
    def _resolve_name(
        name: str,
        name_index: Dict[str, List[str]],
        ctx: BuildContext,
    ) -> str | None:
        candidates = name_index.get(name, [])
        for nid in candidates:
            mt = ctx.node_index[nid].meta_type
            if mt in ("Function", "Class", "Method"):
                return nid
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _find_importing_function(tree: ast.Module, rel: str, lineno: int) -> str | None:
        """Return the function ID that contains the given line, if any."""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                if start <= lineno <= end:
                    return f"Function-{rel}-{node.name}"
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        start = child.lineno
                        end = getattr(child, "end_lineno", start)
                        if start <= lineno <= end:
                            return f"Method-{rel}-{node.name}-{child.name}"
        return None
