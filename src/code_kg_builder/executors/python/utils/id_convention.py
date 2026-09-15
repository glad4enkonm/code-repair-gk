"""ID generation utilities following the SWE-bench KG ID convention.

Patterns (see DESIGN.md section 8):

    Directory    Directory-{rel_path}
    File         File-{rel_path}
    Class        Class-{rel_path}-{name}
    Function     Function-{rel_path}-{name}
    Method       Method-{rel_path}-{ClassName}-{name}
    Variable     Variable-{parent_id}-{name}
    Decorator    Decorator-{target_id}-{name}
    Import       Import-{file_id}-{idx}
    Dependency   Dependency-{name}
    Exception    Exception-{name}
    TypeAnnot    TypeAnnotation-{name}
    Edge         {source}-{RELATION}-{target}
"""

from pathlib import Path


def relative_path_str(repo_root: Path, file_path: Path) -> str:
    """Return forward-slash relative path from *repo_root* to *file_path*."""
    return file_path.relative_to(repo_root).as_posix()


# --------------------------------------------------------------------------- #
#  Node IDs
# --------------------------------------------------------------------------- #


def make_directory_id(rel_path: str) -> str:
    return f"Directory-{rel_path}"


def make_file_id(rel_path: str) -> str:
    return f"File-{rel_path}"


def make_class_id(rel_path: str, name: str) -> str:
    return f"Class-{rel_path}-{name}"


def make_function_id(rel_path: str, name: str) -> str:
    return f"Function-{rel_path}-{name}"


def make_method_id(rel_path: str, class_name: str, name: str) -> str:
    return f"Method-{rel_path}-{class_name}-{name}"


def make_variable_id(parent_id: str, name: str) -> str:
    return f"Variable-{parent_id}-{name}"


def make_decorator_id(target_id: str, name: str) -> str:
    return f"Decorator-{target_id}-{name}"


def make_import_id(file_id: str, idx: int) -> str:
    return f"Import-{file_id}-{idx}"


def make_dependency_id(name: str) -> str:
    return f"Dependency-{name}"


def make_exception_id(name: str) -> str:
    return f"Exception-{name}"


def make_type_annotation_id(name: str) -> str:
    return f"TypeAnnotation-{name}"


# --------------------------------------------------------------------------- #
#  Edge IDs
# --------------------------------------------------------------------------- #


def make_edge_id(source_id: str, relation: str, target_id: str) -> str:
    return f"{source_id}-{relation}-{target_id}"
