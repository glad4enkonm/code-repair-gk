from typing import Any, Dict, List

from ._common import clear_cache
from .read import (
    get_element,
    search_elements,
    get_neighbors,
    traverse,
    find_connected,
    get_get_element_function_definition,
    get_search_elements_function_definition,
    get_get_neighbors_function_definition,
    get_traverse_function_definition,
    get_find_connected_function_definition,
)
from .write import (
    update_graph,
    get_update_graph_function_definition,
)


def get_function_definitions() -> List[Dict[str, Any]]:
    """
    Return all graph file operation function definitions.
    """
    return [
        get_get_element_function_definition(),
        get_search_elements_function_definition(),
        get_get_neighbors_function_definition(),
        get_traverse_function_definition(),
        get_find_connected_function_definition(),
        get_update_graph_function_definition(),
    ]


__all__ = [
    # functions
    "get_element",
    "search_elements",
    "get_neighbors",
    "traverse",
    "find_connected",
    "update_graph",
    "clear_cache",
    # defs
    "get_get_element_function_definition",
    "get_search_elements_function_definition",
    "get_get_neighbors_function_definition",
    "get_traverse_function_definition",
    "get_find_connected_function_definition",
    "get_update_graph_function_definition",
    "get_function_definitions",
]
