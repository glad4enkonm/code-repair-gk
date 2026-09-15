"""McCabe cyclomatic complexity calculation for AST function/method nodes.

Starts at 1, then +1 for each decision point:
  if, elif (via If.test), for, while, except handler, with, assert,
  boolean and/or, and comprehension (ListComp, SetComp, DictComp, GeneratorExp).
"""

import ast
from typing import Union

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)

_COMP_TYPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# Node types that each add exactly one decision point
_DECISION_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
) + _COMP_TYPES

# Boolean operators: each operand boundary adds a decision point
_BOOL_OPS = (ast.And, ast.Or)


def cyclomatic_complexity(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> int:
    """Compute McCabe complexity for an AST function or class node.

    For class nodes, sums the complexity of all methods.
    """
    if isinstance(node, ast.ClassDef):
        total = 0
        for child in ast.walk(node):
            if isinstance(child, _FUNC_TYPES):
                total += _walk_for_complexity(child)
        return total if total > 0 else 1

    return _walk_for_complexity(node)


def _walk_for_complexity(node: ast.AST) -> int:
    """Count decision points in *node*'s subtree, starting from 1."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, _DECISION_TYPES):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # 'a and b and c' has 2 operators → +2
            complexity += len(child.values) - 1
    return complexity
