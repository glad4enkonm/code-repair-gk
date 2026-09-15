from typing import Any, Dict, Generator, List, Optional, Tuple
import json
import fnmatch

"""
Graph Read Operations

Implements read-only endpoints for origin, meta, and data graphs as per graph_file_ops_spec.md:
- get_element
- search_elements
- get_neighbors
- traverse

Notes:
- Expects the current working directory (dataset folder) to contain origin.json, meta.json, data.json,
  or a graph_store/ subdirectory with the graph JSON files.
- Supports datasets that use either keys: nodes_data/edges_data/all_values_data or nodes/edges/allValues.
- Deterministic ordering where required.
"""
from ._common import (
    _graph_cache,
    _graph_path,
    _normalize,
)


def _load_graph(
    graph: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    graph = graph.lower()
    path = _graph_path(graph)
    cache_key = str(path.resolve())
    if cache_key in _graph_cache:
        return _graph_cache[cache_key], None
    if not path.is_file():
        return None, {
            "success": False,
            "code": "GRAPH_FILE_NOT_FOUND",
            "error": f"Graph file not found: {path.resolve()}",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, {
            "success": False,
            "code": "GRAPH_FILE_INVALID_JSON",
            "error": f"Invalid JSON in {path}: {e}",
        }
    except Exception as e:
        return None, {
            "success": False,
            "code": "GRAPH_FILE_READ_ERROR",
            "error": f"Failed to read {path}: {e}",
        }

    norm = _normalize(data)
    _graph_cache[cache_key] = norm
    return norm, None


def _element_kind(element: Dict[str, Any]) -> str:
    return "edge" if ("source" in element and "target" in element) else "node"


def _get_properties(
    all_values: Dict[str, Dict[str, Any]], elem_id: str
) -> Dict[str, Any]:
    return dict(all_values.get(elem_id, {}))


# ---- API: get_element ----


def get_element(
    graph: str, id: str, include: Optional[List[str]] = None
) -> Dict[str, Any]:
    include = include or []
    data, error = _load_graph(graph)
    if error:
        return error
    nodes = data["nodes"]  # type: ignore[index]
    edges = data["edges"]  # type: ignore[index]
    all_values = data["all_values"]  # type: ignore[index]

    found: Optional[Dict[str, Any]] = next(
        (n for n in nodes if n.get("id") == id), None
    )
    if not found:
        found = next((e for e in edges if e.get("id") == id), None)
    if not found:
        return {
            "success": False,
            "code": "ELEMENT_NOT_FOUND",
            "error": f"Element '{id}' not found in graph '{graph}'",
        }

    kind = _element_kind(found)
    result: Dict[str, Any] = {
        "success": True,
        "kind": kind,
        "id": found.get("id"),
        "label": found.get("label", found.get("id")),
        "properties": (
            _get_properties(all_values, id)
            if ("properties" in include or "neighbors" in include or "edges" in include)
            else {}
        ),
    }

    # Include source/target if edge
    if kind == "edge":
        result.update(
            {
                "source": found.get("source"),
                "target": found.get("target"),
            }
        )

    # Optionally include edges and neighbors for node
    if kind == "node" and ("edges" in include or "neighbors" in include):
        # gather incident edges
        inc_edges = [e for e in edges if e.get("source") == id or e.get("target") == id]
        if "edges" in include:
            result["edges"] = [
                {
                    "id": e.get("id"),
                    "label": e.get("label", e.get("id")),
                    "source": e.get("source"),
                    "target": e.get("target"),
                    **(
                        {"properties": _get_properties(all_values, e.get("id"))}
                        if "properties" in include
                        else {}
                    ),
                }
                for e in inc_edges
            ]
        if "neighbors" in include:
            neighbor_ids = [
                (e.get("target") if e.get("source") == id else e.get("source"))
                for e in inc_edges
            ]
            # dedupe and order deterministically
            neighbor_ids = sorted(set([nid for nid in neighbor_ids if nid]))
            id_map = {n.get("id"): n for n in nodes}
            result["neighbors"] = [
                {
                    "id": nid,
                    "label": (id_map.get(nid) or {}).get("label", nid),
                    **(
                        {"properties": _get_properties(all_values, nid)}
                        if "properties" in include
                        else {}
                    ),
                }
                for nid in neighbor_ids
            ]

    return result


def get_get_element_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_element",
            "description": "Fetch a single node or edge from the specified graph; optionally include properties, edges, and neighbors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "string", "enum": ["origin", "meta", "data"]},
                    "id": {"type": "string"},
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["properties", "edges", "neighbors"],
                        },
                    },
                },
                "required": ["graph", "id"],
                "additionalProperties": False,
            },
        },
    }


# ---- API: search_elements ----


def _match_properties(props: Dict[str, Any], prop_filter: Dict[str, Any]) -> bool:
    for k, expected in prop_filter.items():
        v = props.get(k)
        if v is None:
            return False
        sv = str(v)
        # LLM callers may send native JSON values (bool/int/None) instead
        # of strings; match via their string forms instead of crashing.
        expected = str(expected)
        if "*" in expected:
            if not fnmatch.fnmatchcase(sv, expected):
                return False
        else:
            if sv != expected:
                return False
    return True


def search_elements(
    graph: str,
    kind: str = "any",
    meta_type: Optional[str] = None,
    label_contains: Optional[str] = None,
    properties_filter: Optional[Dict[str, str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_by: str = "id",
    sort_order: str = "asc",
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    include = include or []
    properties_filter = properties_filter or {}
    data, error = _load_graph(graph)
    if error:
        return error

    nodes = data["nodes"]  # type: ignore[index]
    edges = data["edges"]  # type: ignore[index]
    all_values = data["all_values"]  # type: ignore[index]

    def base_iter() -> Generator[Dict[str, Any], None, None]:
        if kind == "node":
            for n in nodes:
                yield n
        elif kind == "edge":
            for e in edges:
                yield e
        else:
            for n in nodes:
                yield n
            for e in edges:
                yield e

    items: List[Dict[str, Any]] = []
    for el in base_iter():
        props = all_values.get(el.get("id"), {})
        if meta_type and props.get("meta_type") != meta_type:
            continue
        if label_contains:
            label = el.get("label", el.get("id", "")) or ""
            if label_contains.lower() not in str(label).lower():
                continue
        if properties_filter and not _match_properties(props, properties_filter):
            continue
        # Shape like get_element (without neighbors unless requested)
        item = {
            "kind": _element_kind(el),
            "id": el.get("id"),
            "label": el.get("label", el.get("id")),
            "properties": props if "properties" in include else {},
        }
        if item["kind"] == "edge":
            item.update({"source": el.get("source"), "target": el.get("target")})
        items.append(item)

    reverse = sort_order == "desc"

    def sort_key(it: Dict[str, Any]) -> Any:
        if sort_by == "meta_type":
            return (it.get("properties", {}).get("meta_type"), it.get("id"))
        return (it.get(sort_by),)

    items.sort(key=sort_key, reverse=reverse)

    total = len(items)
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[: max(0, int(limit))]

    return {"success": True, "total": total, "items": items}


def get_search_elements_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_elements",
            "description": "Search nodes/edges in a graph by kind, type, label substring, or properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "string", "enum": ["origin", "meta", "data"]},
                    "kind": {
                        "type": "string",
                        "enum": ["node", "edge", "any"],
                        "default": "any",
                    },
                    "meta_type": {"type": ["string", "null"]},
                    "label_contains": {"type": ["string", "null"]},
                    "properties_filter": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "limit": {"type": ["integer", "null"]},
                    "offset": {"type": "integer", "default": 0},
                    "sort_by": {"type": "string", "default": "id"},
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "default": "asc",
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["properties"]},
                    },
                },
                "required": ["graph"],
                "additionalProperties": False,
            },
        },
    }


# ---- API: get_neighbors ----


def get_neighbors(
    graph: str,
    id: str,
    direction: str = "both",
    edge_label: Optional[str] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    include = include or []
    data, error = _load_graph(graph)
    if error:
        return error

    nodes = data["nodes"]  # type: ignore[index]
    edges = data["edges"]  # type: ignore[index]
    all_values = data["all_values"]  # type: ignore[index]

    def dir_ok(e: Dict[str, Any]) -> bool:
        if direction == "outgoing":
            return e.get("source") == id
        if direction == "incoming":
            return e.get("target") == id
        return e.get("source") == id or e.get("target") == id

    inc_edges = [
        e
        for e in edges
        if dir_ok(e) and (edge_label is None or e.get("label") == edge_label)
    ]
    edges_out = [
        {
            "id": e.get("id"),
            "label": e.get("label", e.get("id")),
            "source": e.get("source"),
            "target": e.get("target"),
            **(
                {"properties": _get_properties(all_values, e.get("id"))}
                if "properties" in include
                else {}
            ),
        }
        for e in inc_edges
    ]

    neighbor_ids: List[str] = []
    for e in inc_edges:
        if e.get("source") == id:
            neighbor_ids.append(e.get("target"))
        elif e.get("target") == id:
            neighbor_ids.append(e.get("source"))
    neighbor_ids = sorted(set([nid for nid in neighbor_ids if nid]))

    id_map = {n.get("id"): n for n in nodes}
    nodes_out = [
        {
            "id": nid,
            "label": (id_map.get(nid) or {}).get("label", nid),
            **(
                {"properties": _get_properties(all_values, nid)}
                if "properties" in include
                else {}
            ),
        }
        for nid in neighbor_ids
    ]

    return {"success": True, "center_id": id, "edges": edges_out, "nodes": nodes_out}


def get_get_neighbors_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "List direct neighbors and connecting edges for a center element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "string", "enum": ["origin", "meta", "data"]},
                    "id": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "default": "both",
                    },
                    "edge_label": {"type": ["string", "null"]},
                    "include": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["properties"]},
                    },
                },
                "required": ["graph", "id"],
                "additionalProperties": False,
            },
        },
    }


# ---- API: traverse ----


def _available_links_per_hop(
    start_id: str,
    max_depth: int,
    out_adj: Dict[str, List[Dict[str, Any]]],
    in_adj: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Unconstrained BFS from *start_id*, tallying REAL (label, direction)
    pairs available at each hop.  Diagnostic companion for empty traversals:
    shows the caller what the graph actually offers per hop."""
    hops: List[Dict[str, Any]] = []
    frontier = {start_id}
    visited = {start_id}
    for depth in range(max(0, max_depth)):
        link_counts: Dict[Tuple[Any, str], int] = {}
        next_frontier: set = set()
        for nid in frontier:
            for e in out_adj.get(nid, []):
                key = (e.get("label"), "out")
                link_counts[key] = link_counts.get(key, 0) + 1
                target = e.get("target")
                if target is not None and target not in visited:
                    next_frontier.add(target)
            for e in in_adj.get(nid, []):
                key = (e.get("label"), "in")
                link_counts[key] = link_counts.get(key, 0) + 1
                source = e.get("source")
                if source is not None and source not in visited:
                    next_frontier.add(source)
        if not link_counts:
            break
        links = [
            {"edge_label": label, "direction": direction, "count": count}
            for (label, direction), count in sorted(
                link_counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
            )
        ]
        hops.append({"hop": depth + 1, "links": links})
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return hops


def _format_links(links: List[Dict[str, Any]]) -> str:
    return ", ".join(
        "{label} ({direction}, {count})".format(
            label=link.get("edge_label"),
            direction=link.get("direction"),
            count=link.get("count"),
        )
        for link in links
    )


def _empty_traverse_hint(
    graph: str,
    start_id: str,
    max_depth: int,
    pattern: List[Dict[str, Any]],
    out_adj: Dict[str, List[Dict[str, Any]]],
    in_adj: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a self-correction hint for a traversal that matched nothing.

    Lists the real per-hop links reachable from *start_id*, diagnoses which
    pattern step failed (unknown label, direction mismatch, or exhausted
    reachability), and suggests a valid ``traverse`` call built from the
    graph's actual edges."""
    hops = _available_links_per_hop(start_id, max_depth, out_adj, in_adj)

    if not hops:
        return {
            "message": (
                f"No paths found: start node '{start_id}' has no incident "
                "edges, so there is nothing to traverse."
            ),
            "available_links_per_hop": [],
            "suggested_call": None,
        }

    # Diagnose the first pattern step that cannot match the real links.
    message = None
    for step_index, step in enumerate(pattern[:max_depth]):
        if step_index >= len(hops):
            message = (
                f"No path matched: nothing is reachable beyond hop "
                f"{len(hops)} from '{start_id}', so pattern step "
                f"{step_index + 1} can never apply."
            )
            break
        links = hops[step_index]["links"]
        requested_label = step.get("edge_label")
        requested_direction = step.get("direction")
        if requested_label is None:
            continue  # wildcard label matches anything
        with_label = [link for link in links if link["edge_label"] == requested_label]
        if not with_label:
            message = (
                f"No path matched: pattern step {step_index + 1} requested "
                f"edge_label '{requested_label}' but hop {step_index + 1} "
                f"only offers: {_format_links(links)}."
            )
            break
        if requested_direction and not any(
            link["direction"] == requested_direction for link in with_label
        ):
            actual = with_label[0]["direction"]
            message = (
                f"No path matched: pattern step {step_index + 1} requested "
                f"edge_label '{requested_label}' with direction "
                f"'{requested_direction}', but '{requested_label}' only "
                f"exists as {actual}going at hop {step_index + 1}."
            )
            break
    if message is None:
        message = (
            "No path matched the full pattern (steps can match individually "
            "but not in sequence); per-hop available links are listed below."
        )

    def best_link(links: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Highest count wins; ties prefer outgoing, then label order.
        return sorted(
            links,
            key=lambda link: (
                -link.get("count", 0),
                0 if link.get("direction") == "out" else 1,
                str(link.get("edge_label")),
            ),
        )[0]

    # Suggest a corrected call: same shape as the request, labels/directions
    # taken from the real links (falling back to the best link per hop).
    suggested_steps: List[Dict[str, Any]] = []
    target_len = len(pattern) if pattern else 1
    for step_index in range(min(target_len, len(hops))):
        links = hops[step_index]["links"]
        requested = pattern[step_index] if step_index < len(pattern) else {}
        requested_label = requested.get("edge_label")
        with_label = [link for link in links if link["edge_label"] == requested_label]
        chosen = with_label[0] if with_label else best_link(links)
        suggested_steps.append(
            {
                "edge_label": chosen["edge_label"],
                "direction": chosen["direction"],
            }
        )

    return {
        "message": message,
        "available_links_per_hop": hops,
        "suggested_call": {
            "graph": graph,
            "start_id": start_id,
            "max_depth": len(suggested_steps),
            "pattern": suggested_steps,
        },
    }


def traverse(
    graph: str,
    start_id: str,
    max_depth: int = 3,
    pattern: Optional[List[Any]] = None,
    limit_paths: Optional[int] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """BFS paths from a start node, optionally filtered per hop.

    Each ``pattern`` step is either a simple edge-label string
    (e.g. ``"CONTAINS"``) or an object
    ``{"edge_label": "CONTAINS", "direction": "out" | "in"}`` for
    advanced filtering.
    """
    include = include or []
    data, error = _load_graph(graph)
    if error:
        return error

    # LLM callers often send pattern steps as bare edge-label strings
    # ("CONTAINS"); accept that shorthand, and reject anything that is
    # neither a string nor a dict with a short syntax hint.
    normalized_pattern: List[Dict[str, Any]] = []
    for step in pattern or []:
        if isinstance(step, str):
            normalized_pattern.append({"edge_label": step})
        elif isinstance(step, dict):
            normalized_pattern.append(step)
        else:
            return {
                "success": False,
                "error": (
                    f"Invalid pattern step {step!r}: use a simple "
                    'edge-label string ("CONTAINS") or an object like '
                    '{"edge_label": "CONTAINS", "direction": "out"|"in"} '
                    "for advanced filtering."
                ),
            }
    pattern = normalized_pattern

    nodes = data["nodes"]  # type: ignore[index]
    edges = data["edges"]  # type: ignore[index]
    all_values = data["all_values"]  # type: ignore[index]

    id_to_node = {n.get("id"): n for n in nodes}
    if start_id not in id_to_node:
        return {
            "success": False,
            "code": "NODE_NOT_FOUND",
            "error": f"Node not found: {start_id}",
        }
    out_adj: Dict[str, List[Dict[str, Any]]] = {}
    in_adj: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        out_adj.setdefault(e.get("source"), []).append(e)
        in_adj.setdefault(e.get("target"), []).append(e)

    def match_step(
        e: Dict[str, Any], step: Optional[Dict[str, Any]], going_out: bool
    ) -> bool:
        if step is None:
            return True
        label_ok = (step.get("edge_label") is None) or (
            e.get("label") == step.get("edge_label")
        )
        dir_req = step.get("direction")
        if dir_req == "out":
            return label_ok and going_out
        if dir_req == "in":
            return label_ok and (not going_out)
        return label_ok

    # BFS over paths
    max_depth = max(0, int(max_depth))
    paths: List[Dict[str, Any]] = []
    from collections import deque

    queue = deque()  # type: ignore[var-annotated]
    # path tuple: (node_id, depth, node_ids_list, edge_ids_list)
    queue.append((start_id, 0, [start_id], []))

    while queue:
        node_id, depth, nseq, eseq = queue.popleft()
        if depth == max_depth:
            # finalize path
            def mk_path() -> Dict[str, Any]:
                nodes_out = [
                    {
                        "id": nid,
                        "label": id_to_node.get(nid, {}).get("label", nid),
                        **(
                            {"properties": _get_properties(all_values, nid)}
                            if "properties" in include
                            else {}
                        ),
                    }
                    for nid in nseq
                ]
                edges_out = [
                    {
                        "id": e["id"],
                        "label": e.get("label", e["id"]),
                        "source": e.get("source"),
                        "target": e.get("target"),
                        **(
                            {"properties": _get_properties(all_values, e.get("id"))}
                            if "properties" in include
                            else {}
                        ),
                    }
                    for e in eseq
                ]
                return {"nodes": nodes_out, "edges": edges_out}

            paths.append(mk_path())
            if limit_paths is not None and len(paths) >= limit_paths:
                break
            continue

        # expand
        next_edges = out_adj.get(node_id, []) + in_adj.get(node_id, [])
        for e in sorted(next_edges, key=lambda x: x.get("id", "")):
            going_out = e.get("source") == node_id
            step = pattern[depth] if pattern and depth < len(pattern) else None
            if not match_step(e, step, going_out):
                continue
            next_nid = e.get("target") if going_out else e.get("source")
            if next_nid in nseq:
                continue  # acyclic per path
            queue.append((next_nid, depth + 1, nseq + [next_nid], eseq + [e]))

        if limit_paths is not None and len(paths) >= limit_paths:
            break

    # deterministic ordering
    paths.sort(key=lambda p: tuple(n.get("id", "") for n in p.get("nodes", [])))

    if not paths:
        # Empty results are almost always a pattern that does not fit the
        # graph; hand the caller a hint built from the start node's real
        # links plus a suggested valid call so it can self-correct.
        hint = _empty_traverse_hint(
            graph, start_id, max_depth, pattern, out_adj, in_adj
        )
        return {"success": True, "paths": [], "hint": hint}

    return {"success": True, "paths": paths}


# ---- API: find_connected ----


def find_connected(
    graph: str,
    id: str,
    type: str,
    edge_label: Optional[str] = None,
    direction: str = "in",
    max_depth: int = 20,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Find the nearest node of a given meta_type reachable via edges.

    BFS from *id* following edges (optionally filtered by *edge_label*)
    in the requested *direction*.  Returns the first node whose
    ``meta_type`` property matches *type* (the start node itself
    is checked first).

    If *edge_label* is ``None`` (default), all edge labels are followed.

    Cypher equivalent (direction = "in", edge_label = "L")::

        MATCH (start {{id: $id}})<-[:L*1..{max_depth}]-(target)
        WHERE target.meta_type = $type
        RETURN target LIMIT 1
    """
    include = include or []
    data, error = _load_graph(graph)
    if error:
        return error

    nodes = data["nodes"]  # type: ignore[index]
    edges = data["edges"]  # type: ignore[index]
    all_values = data["all_values"]  # type: ignore[index]

    id_to_node = {n.get("id"): n for n in nodes}

    if id not in id_to_node:
        return {
            "success": False,
            "code": "NODE_NOT_FOUND",
            "error": f"Node not found: {id}",
        }

    # Build adjacency based on direction
    adj: Dict[str, List[str]] = {}
    for e in edges:
        if edge_label is not None and e.get("label") != edge_label:
            continue
        src = e.get("source", "")
        tgt = e.get("target", "")
        if direction == "in":
            adj.setdefault(tgt, []).append(src)
        elif direction == "out":
            adj.setdefault(src, []).append(tgt)
        else:  # both
            adj.setdefault(tgt, []).append(src)
            adj.setdefault(src, []).append(tgt)

    # BFS
    from collections import deque

    queue: deque = deque()
    queue.append((id, 0))
    visited: set = set()

    while queue:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        props = _get_properties(all_values, current)
        if props.get("meta_type") == type:
            node = id_to_node.get(current, {})
            return {
                "success": True,
                "node": {
                    "id": current,
                    "label": node.get("label", current),
                    **({"properties": props} if "properties" in include else {}),
                },
            }

        if depth >= max_depth:
            continue

        for neighbor_id in sorted(adj.get(current, [])):
            if neighbor_id not in visited:
                queue.append((neighbor_id, depth + 1))

    label_desc = f" '{edge_label}'" if edge_label else ""
    return {
        "success": False,
        "code": "NO_MATCH",
        "error": (
            f"No node with meta_type '{type}' reachable from '{id}' "
            f"within {max_depth} hops via{label_desc} edges "
            f"(direction={direction})"
        ),
    }


def get_find_connected_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "find_connected",
            "description": (
                "Find the nearest node of a given type reachable by "
                "following edges in the specified direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "enum": ["origin", "meta", "data"],
                    },
                    "id": {
                        "type": "string",
                        "description": "Starting node ID.",
                    },
                    "type": {
                        "type": "string",
                        "description": "Target node meta_type.",
                    },
                    "edge_label": {
                        "type": "string",
                        "description": (
                            "Edge label to follow. "
                            "If omitted, all edge labels are followed."
                        ),
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["in", "out", "both"],
                        "default": "in",
                        "description": (
                            "'in' follows incoming edges (towards parents), "
                            "'out' follows outgoing edges (towards children), "
                            "'both' follows either direction."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 10,
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["properties"],
                        },
                    },
                },
                "required": ["graph", "id", "type"],
                "additionalProperties": False,
            },
        },
    }


def get_traverse_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "traverse",
            "description": "Bounded multi-hop traversal with optional step constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "string", "enum": ["origin", "meta", "data"]},
                    "start_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 3},
                    "pattern": {
                        "type": "array",
                        "description": (
                            "Per-hop constraints. Simple form: an "
                            'edge-label string like "CONTAINS". Advanced '
                            'form: an object like {"edge_label": '
                            '"CONTAINS", "direction": "out"|"in"}.'
                        ),
                        "items": {
                            "type": ["string", "object"],
                            "properties": {
                                "edge_label": {"type": "string"},
                                "direction": {"type": "string", "enum": ["out", "in"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "limit_paths": {"type": ["integer", "null"]},
                    "include": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["properties"]},
                    },
                },
                "required": ["graph", "start_id"],
                "additionalProperties": False,
            },
        },
    }
