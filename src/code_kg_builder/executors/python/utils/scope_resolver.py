"""ScopeResolver — build name→node_id maps for call/import resolution.

Provides simple name resolution for Python source code:
  - ``self.method()`` → Method in current class
  - ``cls.method()`` → Method in current class
  - ``foo()`` → Function in same file
  - ``ClassName()`` → Class in project
"""

import ast
from typing import Dict, List, Optional

from code_kg_builder.context import BuildContext


def build_name_index(ctx: BuildContext) -> Dict[str, List[str]]:
    """Build {name: [node_ids]} from the node index."""
    index: Dict[str, List[str]] = {}
    for nid, info in ctx.node_index.items():
        name = info.properties.get("name")
        if name:
            index.setdefault(name, []).append(nid)
    return index


def resolve_call(
    func_node: ast.expr,
    enclosing_class_name: Optional[str],
    name_index: Dict[str, List[str]],
    ctx: BuildContext,
) -> Optional[str]:
    """Resolve a call target to a node ID in the index.

    Returns None if unresolvable.
    """
    if isinstance(func_node, ast.Attribute):
        if isinstance(func_node.value, ast.Name):
            if func_node.value.id in ("self", "cls"):
                method_name = func_node.attr
                candidates = name_index.get(method_name, [])
                methods = [
                    c for c in candidates if ctx.node_index[c].meta_type == "Method"
                ]
                if enclosing_class_name:
                    prefixed = [c for c in methods if enclosing_class_name in c]
                    if len(prefixed) == 1:
                        return prefixed[0]
                    if len(prefixed) > 1:
                        return None
                if len(methods) == 1:
                    return methods[0]
                return None

    if isinstance(func_node, ast.Name):
        bare_name = func_node.id
        candidates = name_index.get(bare_name, [])
        funcs = [c for c in candidates if ctx.node_index[c].meta_type == "Function"]
        if len(funcs) == 1:
            return funcs[0]
        classes = [c for c in candidates if ctx.node_index[c].meta_type == "Class"]
        if len(classes) == 1:
            return classes[0]
        if len(candidates) == 1:
            return candidates[0]

    return None
