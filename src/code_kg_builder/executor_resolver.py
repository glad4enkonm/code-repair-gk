"""ExecutorResolver — resolve ``"module:Class"`` paths via importlib.

Fail-fast: if any path is broken, raise immediately with a clear error
identifying the problematic type.
"""

import importlib
from typing import Any, Dict, List

_DEFAULT_SEP = ":"


class ExecutorResolverError(Exception):
    """Raised when an executor path cannot be resolved."""


def resolve_executor(path: str) -> Any:
    """Resolve a ``"module:Class"`` string and return an instance.

    Args:
        path: Dotted module path + ``:`` + class name, e.g.
            ``"code_kg_builder.executors.base:BatchResult"``.

    Returns:
        An instantiated object of the resolved class.

    Raises:
        ExecutorResolverError: If the path is malformed, the module
            cannot be imported, or the class does not exist.
    """
    if not path or not path.strip():
        raise ExecutorResolverError(f"Empty executor path: {path!r}")

    if _DEFAULT_SEP not in path:
        raise ExecutorResolverError(
            f"Executor path must contain '{_DEFAULT_SEP}' separating module "
            f"and class: {path!r}"
        )

    module_path, _, class_name = path.rpartition(_DEFAULT_SEP)
    if not module_path or not class_name:
        raise ExecutorResolverError(
            f"Malformed executor path (missing module or class): {path!r}"
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ExecutorResolverError(
            f"Cannot import module '{module_path}' for executor {path!r}: {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ExecutorResolverError(
            f"Class '{class_name}' not found in module '{module_path}' "
            f"for executor {path!r}"
        )

    return cls()


def resolve_executors(
    paths_to_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Resolve multiple executor paths.

    Args:
        paths_to_types: ``{executor_path: [type_names handled by this executor]}``.

    Returns:
        ``{executor_path: instance}`` for each resolved executor.

    Raises:
        ExecutorResolverError: If any path cannot be resolved. The error
            message includes the type names affected.
    """
    instances: Dict[str, Any] = {}
    for path, type_names in paths_to_types.items():
        try:
            instances[path] = resolve_executor(path)
        except ExecutorResolverError:
            raise ExecutorResolverError(
                f"Failed to resolve executor for type(s) {type_names}: "
                f"path {path!r}"
            )
    return instances
