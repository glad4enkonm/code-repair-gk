"""Extract source code snippets by line range from file content."""

import ast
from typing import Union


def extract_lines(source: str, line_start: int, line_end: int) -> str:
    """Return lines ``[line_start, line_end]`` (1-indexed, inclusive).

    If *line_end* exceeds the actual number of lines, returns what exists.
    """
    lines = source.splitlines()
    start_idx = max(line_start - 1, 0)
    end_idx = min(line_end, len(lines))
    return "\n".join(lines[start_idx:end_idx])


def extract_source_for_node(
    source: str,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> str:
    """Extract source text for an AST node using its line span."""
    end_line = getattr(node, "end_lineno", None) or node.lineno
    return extract_lines(source, node.lineno, end_line)
