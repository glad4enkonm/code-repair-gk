"""Extract exception class names from ``ast.Raise`` / ``ast.ExceptHandler``.

Shared by :mod:`raises_edge` and :mod:`catches_edge`.

Supported shapes:

    raise ValueError            -> ["ValueError"]
    raise ValueError("msg")     -> ["ValueError"]
    raise mod.CustomError       -> ["CustomError"]
    except ValueError           -> ["ValueError"]
    except (A, B)               -> ["A", "B"]
    bare ``raise`` / ``except`` -> []
"""

import ast
from typing import List, Optional


def extract_exception_name(node: Optional[ast.AST]) -> Optional[str]:
    """Return the exception name for a single exception reference.

    Handles ``Name`` (``ValueError``), ``Call`` (``ValueError("x")``) and
    ``Attribute`` (``mod.CustomError``).  Returns ``None`` otherwise.
    """
    if node is None:
        return None
    if isinstance(node, ast.Call):
        return extract_exception_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def extract_exception_names(node: Optional[ast.AST]) -> List[str]:
    """Return all exception names referenced by *node*.

    A ``Tuple`` (used in ``except (A, B)``) yields one name per element;
    any other node yields zero or one name.
    """
    if node is None:
        return []
    if isinstance(node, ast.Tuple):
        names: List[str] = []
        for elt in node.elts:
            name = extract_exception_name(elt)
            if name:
                names.append(name)
        return names
    name = extract_exception_name(node)
    return [name] if name else []
