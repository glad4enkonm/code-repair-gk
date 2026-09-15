"""AstStructureExecutor — parse Python files, create code structure nodes.

Phase 1.  Processes each ``.py`` file in ``ctx.python_files`` and yields one
BatchResult per file.  Creates:

NodeTypes: Class, Function, Method, Variable, Decorator, Import
EdgeTypes: CONTAINS (File→Class, File→Function, Class→Method, Class→Variable),
           DECORATES (Decorator→Method, Decorator→Class)
"""

import ast
from typing import Iterator, Optional, Union

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.complexity import cyclomatic_complexity
from code_kg_builder.executors.python.utils.id_convention import (
    make_class_id,
    make_decorator_id,
    make_edge_id,
    make_function_id,
    make_import_id,
    make_method_id,
    make_variable_id,
    relative_path_str,
)
from code_kg_builder.executors.python.utils.source_extract import (
    extract_source_for_node,
)


def _ann_to_str(node: Optional[ast.expr]) -> Optional[str]:
    """Convert an annotation AST node to a string representation."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _decorator_name(node: ast.expr) -> str:
    """Extract decorator name from AST."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return "unknown"


def _is_decorator(node: ast.expr, name: str) -> bool:
    """Check if a decorator node matches *name* (e.g. 'staticmethod')."""
    return _decorator_name(node) == name


def _var_name(node: ast.expr) -> Optional[str]:
    """Extract variable name from an assignment target."""
    if isinstance(node, ast.Name):
        return node.id
    return None


class AstStructureExecutor(Executor):
    """Parse Python files and emit structural code nodes."""

    phase = 1

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        repo = ctx.repo_path
        for file_path in ctx.python_files:
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(file_path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                ctx.log_error(str(file_path), f"Parse error: {exc}")
                continue
            except Exception as exc:
                ctx.log_error(str(file_path), f"Unexpected error: {exc}")
                continue

            rel = relative_path_str(repo, file_path)
            file_id = f"File-{rel}"
            yield self._process_file(tree, source, rel, file_id, ctx)

    def _process_file(
        self,
        tree: ast.Module,
        source: str,
        rel_path: str,
        file_id: str,
        ctx: BuildContext,
    ) -> BatchResult:
        batch = BatchResult()
        import_idx = 0

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._handle_import(node, file_id, import_idx, batch)
                import_idx += 1
            elif isinstance(node, ast.ClassDef):
                self._handle_class(node, source, rel_path, file_id, batch)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_function(node, source, rel_path, file_id, batch)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._handle_module_variable(node, file_id, batch)

        # Strip None values — graph API rejects null for typed properties
        batch.all_values_data = [
            {k: v for k, v in av.items() if v is not None}
            for av in batch.all_values_data
        ]

        return batch

    # ------------------------------------------------------------------ #
    #  Imports
    # ------------------------------------------------------------------ #

    def _handle_import(
        self,
        node: ast.stmt,
        file_id: str,
        idx: int,
        batch: BatchResult,
    ) -> None:
        import_id = make_import_id(file_id, idx)

        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
            module_path = ", ".join(names)
            imported_name = ""
            label = ", ".join(n.split(".")[0] for n in names)
        elif isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            module_path = node.module or ""
            imported_name = ", ".join(names)
            label = ", ".join(names)
        else:
            return

        batch.nodes_data.append({"id": import_id, "label": label})

        props: dict = {"id": import_id, "meta_type": "Import"}
        props["module_path"] = module_path
        props["imported_name"] = imported_name
        props["line_number"] = node.lineno
        batch.all_values_data.append(props)

        # CONTAINS File→Import
        batch.edges_data.append(
            {
                "id": make_edge_id(file_id, "CONTAINS", import_id),
                "source": file_id,
                "target": import_id,
                "label": "CONTAINS",
            }
        )

    # ------------------------------------------------------------------ #
    #  Classes
    # ------------------------------------------------------------------ #

    def _handle_class(
        self,
        node: ast.ClassDef,
        source: str,
        rel_path: str,
        file_id: str,
        batch: BatchResult,
    ) -> None:
        class_id = make_class_id(rel_path, node.name)
        batch.nodes_data.append({"id": class_id, "label": node.name})

        # CONTAINS File→Class
        batch.edges_data.append(
            {
                "id": make_edge_id(file_id, "CONTAINS", class_id),
                "source": file_id,
                "target": class_id,
                "label": "CONTAINS",
            }
        )

        end_line = getattr(node, "end_lineno", node.lineno)
        props: dict = {
            "id": class_id,
            "meta_type": "Class",
            "name": node.name,
            "source_code": extract_source_for_node(source, node),
            "line_start": node.lineno,
            "line_end": end_line,
            "docstring": ast.get_docstring(node),
            "complexity": cyclomatic_complexity(node),
        }
        batch.all_values_data.append(props)

        # Decorators
        for dec in node.decorator_list:
            self._handle_decorator(dec, class_id, batch)

        # Methods and class-level variables
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_method(child, source, rel_path, node.name, class_id, batch)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                self._handle_class_variable(child, class_id, batch)

    # ------------------------------------------------------------------ #
    #  Functions (module-level)
    # ------------------------------------------------------------------ #

    def _handle_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
        rel_path: str,
        file_id: str,
        batch: BatchResult,
    ) -> None:
        func_id = make_function_id(rel_path, node.name)
        batch.nodes_data.append({"id": func_id, "label": node.name})

        # CONTAINS File→Function
        batch.edges_data.append(
            {
                "id": make_edge_id(file_id, "CONTAINS", func_id),
                "source": file_id,
                "target": func_id,
                "label": "CONTAINS",
            }
        )

        end_line = getattr(node, "end_lineno", node.lineno)
        props: dict = {
            "id": func_id,
            "meta_type": "Function",
            "name": node.name,
            "source_code": extract_source_for_node(source, node),
            "line_start": node.lineno,
            "line_end": end_line,
            "return_type": _ann_to_str(node.returns),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "docstring": ast.get_docstring(node),
            "complexity": cyclomatic_complexity(node),
        }
        batch.all_values_data.append(props)

        # Decorators
        for dec in node.decorator_list:
            self._handle_decorator(dec, func_id, batch)

    # ------------------------------------------------------------------ #
    #  Methods (inside classes)
    # ------------------------------------------------------------------ #

    def _handle_method(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
        rel_path: str,
        class_name: str,
        class_id: str,
        batch: BatchResult,
    ) -> None:
        method_id = make_method_id(rel_path, class_name, node.name)
        batch.nodes_data.append({"id": method_id, "label": node.name})

        # CONTAINS Class→Method
        batch.edges_data.append(
            {
                "id": make_edge_id(class_id, "CONTAINS", method_id),
                "source": class_id,
                "target": method_id,
                "label": "CONTAINS",
            }
        )

        end_line = getattr(node, "end_lineno", node.lineno)
        props: dict = {
            "id": method_id,
            "meta_type": "Method",
            "name": node.name,
            "source_code": extract_source_for_node(source, node),
            "line_start": node.lineno,
            "line_end": end_line,
            "return_type": _ann_to_str(node.returns),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "is_staticmethod": any(
                _is_decorator(d, "staticmethod") for d in node.decorator_list
            ),
            "is_classmethod": any(
                _is_decorator(d, "classmethod") for d in node.decorator_list
            ),
            "docstring": ast.get_docstring(node),
            "complexity": cyclomatic_complexity(node),
        }
        batch.all_values_data.append(props)

        # Decorators
        for dec in node.decorator_list:
            self._handle_decorator(dec, method_id, batch)

    # ------------------------------------------------------------------ #
    #  Variables
    # ------------------------------------------------------------------ #

    def _handle_module_variable(
        self,
        node: ast.stmt,
        parent_id: str,
        batch: BatchResult,
    ) -> None:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _var_name(target)
                if name is None:
                    continue
                self._make_variable(name, node.lineno, node.value, parent_id, batch)
        elif isinstance(node, ast.AnnAssign):
            name = _var_name(node.target)
            if name is None:
                return
            self._make_variable(name, node.lineno, node.value, parent_id, batch)

    def _handle_class_variable(
        self,
        node: ast.stmt,
        class_id: str,
        batch: BatchResult,
    ) -> None:
        self._handle_module_variable(node, class_id, batch)

    def _make_variable(
        self,
        name: str,
        line_start: int,
        value_node: Optional[ast.expr],
        parent_id: str,
        batch: BatchResult,
    ) -> None:
        var_id = make_variable_id(parent_id, name)
        batch.nodes_data.append({"id": var_id, "label": name})

        # CONTAINS parent→Variable
        batch.edges_data.append(
            {
                "id": make_edge_id(parent_id, "CONTAINS", var_id),
                "source": parent_id,
                "target": var_id,
                "label": "CONTAINS",
            }
        )

        value_str = None
        if value_node is not None:
            try:
                value_str = ast.unparse(value_node)
            except Exception:
                value_str = type(value_node).__name__

        props: dict = {
            "id": var_id,
            "meta_type": "Variable",
            "name": name,
            "line_start": line_start,
            "value": value_str,
        }
        batch.all_values_data.append(props)

    # ------------------------------------------------------------------ #
    #  Decorators
    # ------------------------------------------------------------------ #

    def _handle_decorator(
        self,
        dec_node: ast.expr,
        target_id: str,
        batch: BatchResult,
    ) -> None:
        dec_name = _decorator_name(dec_node)
        dec_id = make_decorator_id(target_id, dec_name)
        batch.nodes_data.append({"id": dec_id, "label": dec_name})

        # DECORATES Decorator→target
        batch.edges_data.append(
            {
                "id": make_edge_id(dec_id, "DECORATES", target_id),
                "source": dec_id,
                "target": target_id,
                "label": "DECORATES",
            }
        )

        props: dict = {
            "id": dec_id,
            "meta_type": "Decorator",
            "name": dec_name,
            "line_start": getattr(dec_node, "lineno", 0),
        }
        batch.all_values_data.append(props)
