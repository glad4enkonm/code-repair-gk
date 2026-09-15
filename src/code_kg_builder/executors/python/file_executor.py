"""Template base for executors that visit each Python file's cached AST.

Phase-2 executors (:mod:`inherits_edge`, :mod:`raises_edge`,
:mod:`catches_edge`, :mod:`has_type_edge`) share identical file-iteration
and error-handling logic; only the per-file AST processing differs.  This
class captures the common ``run`` loop (template method) and delegates to
:meth:`process_file`.
"""

import ast
from abc import abstractmethod
from typing import Iterator

from code_kg_builder.context import BuildContext
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.executors.python.utils.id_convention import (
    relative_path_str,
)


class PythonFileExecutor(Executor):
    """Visit each file in ``ctx.python_files`` and delegate AST handling."""

    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        repo = ctx.repo_path
        for file_path in ctx.python_files:
            try:
                tree = ctx.get_ast(file_path)
            except Exception as exc:
                ctx.log_error(str(file_path), f"Parse error: {exc}")
                continue
            rel_path = relative_path_str(repo, file_path)
            batch = self.process_file(tree, rel_path, ctx)
            if not batch.is_empty:
                yield batch

    @abstractmethod
    def process_file(
        self,
        tree: ast.Module,
        rel_path: str,
        ctx: BuildContext,
    ) -> BatchResult:
        """Process a single file's AST and return a batch (may be empty)."""
        raise NotImplementedError
