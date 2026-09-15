"""
Shared utility functions for graph file operations.

Holds the common helpers used by read.py, write.py, and layout.py:
- ``_graph_path`` — resolve a graph name to its JSON file path
- ``_normalize`` — normalize various storage shapes to a common internal format
- ``clear_cache`` / ``_graph_cache`` — module-level cache for loaded graphs
"""

from pathlib import Path
from typing import Any, Dict, Optional

# ---- Internal cache ----

_graph_cache: Dict[str, Dict[str, Any]] = {}


def clear_cache(graph: Optional[str] = None) -> None:
    """Clear the internal cache for one or all graphs.

    If *graph* is a name (``"data"``, ``"meta"``, ``"origin"``),
    resolves it to the current file path and removes that entry only.
    If *graph* is ``None``, clears the entire cache.
    """
    global _graph_cache  # noqa: F824
    if graph:
        cache_key = str(_graph_path(graph.lower()).resolve())
        _graph_cache.pop(cache_key, None)
    else:
        _graph_cache.clear()


# ---- Helpers ----

GRAPH_NAMES = {"origin", "meta", "data"}


def _graph_path(graph: str) -> Path:
    if graph not in GRAPH_NAMES:
        raise ValueError(f"Invalid graph: {graph}. Must be one of {GRAPH_NAMES}")
    cwd_path = Path(f"{graph}.json")
    if cwd_path.is_file():
        return cwd_path
    return Path("graph_store") / f"{graph}.json"


def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize various storage shapes to common internal keys.

    Returns dict with keys: ``nodes`` (list), ``edges`` (list), ``all_values`` (dict).
    Accepts either ``nodes_data``/``edges_data``/``all_values_data`` or
    ``nodes``/``edges``/``allValues`` key conventions.
    """
    nodes = raw.get("nodes_data") or raw.get("nodes") or []
    edges = raw.get("edges_data") or raw.get("edges") or []
    all_values = raw.get("all_values_data") or raw.get("allValues") or {}

    # If all_values provided as a list of {id, ...}, convert to dict keyed by id
    if isinstance(all_values, list):
        converted = {}
        for item in all_values:
            if isinstance(item, dict) and "id" in item:
                elem_id = item["id"]
                props = {k: v for k, v in item.items() if k != "id"}
                converted[elem_id] = props
        all_values = converted

    return {"nodes": nodes, "edges": edges, "all_values": all_values}
