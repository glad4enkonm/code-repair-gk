"""HasTypeEdgeExecutor — resolve type annotations, create HAS_TYPE edges.

Phase 2.  For each Python file, finds ``ast.AnnAssign`` nodes (annotated
assignments like ``x: int = 5``).  Extracts the variable name and annotation
type name.  Creates TypeAnnotation nodes and HAS_TYPE edges from Variable
nodes.
"""

import ast
from typing import Any, Iterator, List, Optional, Tuple

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    make_edge_id,
    make_type_annotation_id,
    make_variable_id,
    relative_path_str,
)


def _annotation_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _target_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    return None


class HasTypeEdgeExecutor(Executor):
    """Resolve type annotations and emit HAS_TYPE edges + TypeAnnotation nodes."""

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

            for parent_id, parent_node in self._iter_scopes(tree, rel):
                for child in ast.walk(parent_node):
                    if not isinstance(child, ast.AnnAssign):
                        continue
                    var_name = _target_name(child.target)
                    if var_name is None:
                        continue
                    ann_name = _annotation_name(child.annotation)
                    if ann_name is None:
                        continue

                    var_id = make_variable_id(parent_id, var_name)
                    if var_id not in ctx.node_index:
                        continue

                    ta_id = make_type_annotation_id(ann_name)
                    self._ensure_type_annotation(ta_id, ann_name, batch, ctx)

                    batch.edges_data.append(
                        {
                            "id": make_edge_id(var_id, "HAS_TYPE", ta_id),
                            "source": var_id,
                            "target": ta_id,
                            "label": "HAS_TYPE",
                        }
                    )

            if not batch.is_empty:
                yield batch

    def _iter_scopes(self, tree: ast.Module, rel: str) -> List[Tuple[str, ast.stmt]]:
        """Return (scope_id, ast_node) for file, classes, and methods."""
        scopes: List[Tuple[str, Any]] = [(f"File-{rel}", tree)]
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_id = f"Class-{rel}-{node.name}"
                scopes.append((class_id, node))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mid = f"Method-{rel}-{node.name}-{child.name}"
                        scopes.append((mid, child))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"Function-{rel}-{node.name}"
                scopes.append((fid, node))
        return scopes

    def _ensure_type_annotation(
        self,
        ta_id: str,
        ta_name: str,
        batch: BatchResult,
        ctx: BuildContext,
    ) -> None:
        if ta_id in ctx.node_index:
            return
        if any(n["id"] == ta_id for n in batch.nodes_data):
            return
        batch.nodes_data.append({"id": ta_id, "label": ta_name})
        batch.all_values_data.append(
            {
                "id": ta_id,
                "meta_type": "TypeAnnotation",
                "name": ta_name,
            }
        )
