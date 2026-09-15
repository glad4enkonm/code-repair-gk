from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import shutil

from verification.graph_verifier import verify_graph_update
from ._common import (
    clear_cache,
    _graph_path,
    _normalize,
)

"""
Graph Write Operations

Implements update_graph with atomic diff semantics and structural validation.
Operates on files in the current directory or graph_store/ subdirectory.

Note: This implementation performs deterministic validation and applies changes transactionally
in-memory, writing the file only on SAT. Meta-level validation (e.g., required properties) is not fully implemented;
instead, this module focuses on structural and referential integrity checks and the property-shape rules from the spec.
"""

# ---- Helpers ----


def _load_raw(graph: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    path = _graph_path(graph)
    if not path.is_file():
        return None, {
            "success": False,
            "code": "GRAPH_FILE_NOT_FOUND",
            "error": f"Graph file not found: {path.resolve()}",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
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


def _denormalize(
    store: Dict[str, Any], original_shape: Dict[str, Any]
) -> Dict[str, Any]:
    """Write back in the same shape as original file (prefer keys it already used)."""
    out: Dict[str, Any] = dict(original_shape)
    nodes_key = (
        "nodes_data"
        if "nodes_data" in original_shape
        else ("nodes" if "nodes" in original_shape else "nodes_data")
    )
    edges_key = (
        "edges_data"
        if "edges_data" in original_shape
        else ("edges" if "edges" in original_shape else "edges_data")
    )
    av_key = (
        "all_values_data"
        if "all_values_data" in original_shape
        else ("allValues" if "allValues" in original_shape else "all_values_data")
    )

    out[nodes_key] = store["nodes"]
    out[edges_key] = store["edges"]
    # Persist all_values as dict keyed by id
    out[av_key] = store["all_values"]
    return out


def _validate_properties_shape(
    props_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for item in props_list:
        if "id" not in item:
            return {
                "code": "MISSING_ID",
                "detail": "Each all_values_data item must include 'id'",
            }
        for k, v in item.items():
            if k == "id":
                continue
            if isinstance(v, (list, dict)):
                return {
                    "code": "NESTED_PROPERTIES_NOT_ALLOWED",
                    "elementId": item["id"],
                    "detail": f"Property '{k}' must be a primitive",
                }
    return None


def _validate_store_paths(props_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Properties that look like paths but are NOT store references
    _NON_STORE_PATH_KEYS = {"relative_path", "module_path"}

    def looks_like_path(k: str) -> bool:
        if k in _NON_STORE_PATH_KEYS:
            return False
        return (
            k in {"store", "code_path", "description_tex"}
            or k.endswith("_path")
            or k.endswith("_file")
            or k.endswith("_tex")
        )

    for item in props_list:
        for k, v in item.items():
            if k == "id" or not isinstance(v, str):
                continue
            if looks_like_path(k):
                if ".." in v:
                    return {
                        "code": "STORE_PATH_INVALID",
                        "elementId": item["id"],
                        "field": k,
                        "detail": "Path must not contain '..'",
                    }
                if not v.startswith("./"):
                    return {
                        "code": "STORE_PATH_INVALID",
                        "elementId": item["id"],
                        "field": k,
                        "detail": "Path must start with './'",
                    }
                rel = v[2:]
                if not (rel.startswith("store/") or rel == "store"):
                    return {
                        "code": "STORE_PATH_INVALID",
                        "elementId": item["id"],
                        "field": k,
                        "detail": "Path must be under './store/'",
                    }
    return None


def _index_by_id(lst: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {x.get("id"): x for x in lst if "id" in x}  # type: ignore[misc]


def _apply_update(
    store: Dict[str, Any],
    nodes_data: Optional[List[Dict[str, Any]]],
    edges_data: Optional[List[Dict[str, Any]]],
    av_list: Optional[List[Dict[str, Any]]],
    delete_node_ids: Optional[List[str]],
    delete_edge_ids: Optional[List[str]],
    delete_av: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply changes to a deep copy of store and return (new_store, applied_summary).
    Raises ValueError with a dict for validation issues to be reported as UNSAT.
    """
    import copy

    nodes = copy.deepcopy(store["nodes"])
    edges = copy.deepcopy(store["edges"])
    av = copy.deepcopy(store["all_values"])

    created_nodes: List[str] = []
    created_edges: List[str] = []
    updated_nodes: List[str] = []
    updated_edges: List[str] = []
    deleted_nodes: List[str] = []
    deleted_edges: List[str] = []
    props_upserted: List[str] = []
    props_deleted: List[Dict[str, Any]] = []

    node_index = _index_by_id(nodes)
    edge_index = _index_by_id(edges)

    # 1) Deletions first
    delete_node_ids = delete_node_ids or []
    delete_edge_ids = delete_edge_ids or []

    for eid in delete_edge_ids:
        if eid in edge_index:
            edges = [e for e in edges if e.get("id") != eid]
            deleted_edges.append(eid)
            edge_index.pop(eid, None)

    # Check node deletions don't leave dangling edges (must be explicitly deleted)
    if delete_node_ids:
        dangling = [
            e.get("id")
            for e in edges
            if e.get("source") in delete_node_ids or e.get("target") in delete_node_ids
        ]
        if dangling:
            raise ValueError(
                {
                    "code": "REFERENTIAL_INTEGRITY",
                    "detail": "Deleting nodes requires deleting incident edges in the same transaction",
                    "elementId": ",".join(delete_node_ids),
                }
            )
        for nid in delete_node_ids:
            if nid in node_index:
                nodes = [n for n in nodes if n.get("id") != nid]
                deleted_nodes.append(nid)
                node_index.pop(nid, None)

    # Delete all_values keys
    if delete_av:
        for item in delete_av:
            rid = item.get("id")
            keys = item.get("keys")
            if not rid:
                continue
            if rid not in av:
                continue
            if not keys:
                # remove all properties for this id
                av.pop(rid, None)
                props_deleted.append({"id": rid})
            else:
                for k in keys:
                    if k in av[rid]:
                        av[rid].pop(k)
                props_deleted.append({"id": rid, "keys": list(keys)})

    # 2) Upserts
    # nodes_data: only id and label
    if nodes_data:
        for n in nodes_data:
            nid = n.get("id")  # type: ignore[assignment]
            if not nid:
                raise ValueError(
                    {"code": "MISSING_ID", "detail": "Node must include 'id'"}
                )
            extra = {k: v for k, v in n.items() if k not in {"id", "label"}}
            if extra:
                raise ValueError(
                    {
                        "code": "FORBIDDEN_PROPERTY",
                        "elementId": nid,
                        "detail": f"Only 'id' and 'label' allowed in nodes_data; got: {list(extra.keys())}",
                    }
                )
            if nid in node_index:
                # update label only
                for i, existing in enumerate(nodes):
                    if existing.get("id") == nid:
                        nodes[i] = {
                            "id": nid,
                            "label": n.get("label", existing.get("label", nid)),
                        }
                        updated_nodes.append(nid)
                        break
            else:
                # creation
                if nid in _index_by_id(nodes):
                    raise ValueError(
                        {
                            "code": "DUPLICATE_ID",
                            "elementId": nid,
                            "detail": "Node id already exists",
                        }
                    )
                nodes.append({"id": nid, "label": n.get("label", nid)})
                created_nodes.append(nid)
                node_index[nid] = nodes[-1]

    # edges_data: only id, label, source, target
    if edges_data:
        for e in edges_data:
            eid = e.get("id")  # type: ignore[assignment]
            if not eid:
                raise ValueError(
                    {"code": "MISSING_ID", "detail": "Edge must include 'id'"}
                )
            extra = {
                k: v
                for k, v in e.items()
                if k not in {"id", "label", "source", "target"}
            }
            if extra:
                raise ValueError(
                    {
                        "code": "FORBIDDEN_PROPERTY",
                        "elementId": eid,
                        "detail": f"Only 'id','label','source','target' allowed in edges_data; got: {list(extra.keys())}",
                    }
                )
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                raise ValueError(
                    {
                        "code": "REFERENTIAL_INTEGRITY",
                        "elementId": eid,
                        "detail": "Edge must include 'source' and 'target'",
                    }
                )
            # must refer to nodes that will exist in final state
            # node_index currently includes nodes after node upserts
            if src not in node_index or tgt not in node_index:
                raise ValueError(
                    {
                        "code": "REFERENTIAL_INTEGRITY",
                        "elementId": eid,
                        "detail": "Edge source/target must exist in final state",
                    }
                )
            if eid in edge_index:
                # update label/source/target
                for i, existing in enumerate(edges):
                    if existing.get("id") == eid:
                        edges[i] = {
                            "id": eid,
                            "label": e.get("label", existing.get("label", eid)),
                            "source": src,
                            "target": tgt,
                        }
                        updated_edges.append(eid)
                        break
            else:
                if eid in _index_by_id(edges):
                    raise ValueError(
                        {
                            "code": "DUPLICATE_ID",
                            "elementId": eid,
                            "detail": "Edge id already exists",
                        }
                    )
                edges.append(
                    {
                        "id": eid,
                        "label": e.get("label", eid),
                        "source": src,
                        "target": tgt,
                    }
                )
                created_edges.append(eid)
                edge_index[eid] = edges[-1]

    # all_values_data upsert (merge semantics)
    if av_list:
        for item in av_list:
            rid = item.get("id")
            if not rid:
                raise ValueError(
                    {
                        "code": "MISSING_ID",
                        "detail": "Properties item must include 'id'",
                    }
                )

            # Check if element exists before adding properties
            if rid not in node_index and rid not in edge_index:
                raise ValueError(
                    {
                        "code": "ELEMENT_NOT_FOUND",
                        "elementId": rid,
                        "detail": f"Cannot add properties to non-existent element '{rid}'.",
                    }
                )

            props = {k: v for k, v in item.items() if k != "id"}
            if rid not in av:
                av[rid] = {}
            av[rid].update(props)
            props_upserted.append(rid)

    # Final referential integrity (no dangling edges)
    node_ids = {n.get("id") for n in nodes}
    for e in edges:
        if e.get("source") not in node_ids or e.get("target") not in node_ids:
            raise ValueError(
                {
                    "code": "REFERENTIAL_INTEGRITY",
                    "elementId": e.get("id"),
                    "detail": "Edge references non-existent node",
                }
            )

    new_store = {"nodes": nodes, "edges": edges, "all_values": av}
    summary = {
        "created": {"nodes": created_nodes, "edges": created_edges},
        "updated": {"nodes": updated_nodes, "edges": updated_edges},
        "deleted": {"nodes": deleted_nodes, "edges": deleted_edges},
        "properties": {
            "upserted": sorted(set(props_upserted)),
            "deleted": props_deleted,
        },
    }
    return new_store, summary


def update_graph(
    graph: str,
    nodes_data: Optional[List[Dict[str, Any]]] = None,
    edges_data: Optional[List[Dict[str, Any]]] = None,
    all_values_data: Optional[List[Dict[str, Any]]] = None,
    delete_node_ids: Optional[List[str]] = None,
    delete_edge_ids: Optional[List[str]] = None,
    delete_all_values_data: Optional[List[Dict[str, Any]]] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Apply a JSON diff to a single graph atomically. On any validation failure, no changes are written.

    Returns a structured SAT/UNSAT response per spec.
    """
    graph = graph.lower()
    # Load and normalize
    original, error = _load_raw(graph)
    if error:
        return {
            "status": "unsat",
            "committed": False,
            "message": error.get("error", "Failed to load graph"),
            "errors": [error],
        }
    store = _normalize(original)  # type: ignore[arg-type]

    # Validate shapes
    av_list = all_values_data or []
    if not isinstance(av_list, list):
        # If dict keyed by id given, convert to list
        tmp = []  # type: ignore[unreachable]
        for rid, props in (av_list or {}).items():
            if isinstance(props, dict):
                tmp.append({"id": rid, **props})
        av_list = tmp
    err = _validate_properties_shape(av_list)
    if err:
        return {
            "status": "unsat",
            "committed": False,
            "message": "Validation failed: 1 error",
            "errors": [err],
        }
    err = _validate_store_paths(av_list)
    if err:
        return {
            "status": "unsat",
            "committed": False,
            "message": "Validation failed: 1 error",
            "errors": [err],
        }

    # Basic checks for nodes/edges payload (only allowed fields)
    def _check_elem_list(
        lst: Optional[List[Dict[str, Any]]], allowed: set, kind: str
    ) -> Optional[Dict[str, Any]]:
        if not lst:
            return None
        for it in lst:
            eid = it.get("id", "N/A")
            for k, v in it.items():
                if k not in allowed:
                    return {
                        "code": "FORBIDDEN_PROPERTY",
                        "elementId": eid,
                        "detail": f"Only {sorted(allowed)} allowed in {kind}",
                    }
                if isinstance(v, (list, dict)):
                    return {
                        "code": "NESTED_PROPERTIES_NOT_ALLOWED",
                        "elementId": eid,
                        "detail": f"Nested values not allowed in {kind}",
                    }
        return None

    err = _check_elem_list(nodes_data, {"id", "label"}, "nodes_data")
    if err:
        return {
            "status": "unsat",
            "committed": False,
            "message": "Validation failed: 1 error",
            "errors": [err],
        }
    err = _check_elem_list(
        edges_data, {"id", "label", "source", "target"}, "edges_data"
    )
    if err:
        return {
            "status": "unsat",
            "committed": False,
            "message": "Validation failed: 1 error",
            "errors": [err],
        }

    try:
        new_store, summary = _apply_update(
            store,
            nodes_data=nodes_data,
            edges_data=edges_data,
            av_list=av_list,
            delete_node_ids=delete_node_ids,
            delete_edge_ids=delete_edge_ids,
            delete_av=delete_all_values_data,
        )
    except ValueError as ve:
        detail = ve.args[0] if ve.args else {"code": "UNKNOWN", "detail": str(ve)}
        return {
            "status": "unsat",
            "committed": False,
            "message": "Validation failed: 1 error",
            "errors": [detail],
        }

    # Commit: write back preserving original shape keys
    path = _graph_path(graph)

    if verify:
        # Create backup before writing (needed for rollback on Z3 failure)
        backup_path = Path(f"{path}.backup")
        if path.exists():
            shutil.copy2(path, backup_path)

    # Write the updated graph
    out = _denormalize(new_store, original)  # type: ignore[arg-type]
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not verify:
        clear_cache(graph)
        return {
            "status": "sat",
            "committed": True,
            "message": "Transaction committed (verification skipped)",
            "applied": {
                "created": {k: v for k, v in summary.get("created", {}).items() if v},
                "updated": {k: v for k, v in summary.get("updated", {}).items() if v},
                "deleted": {k: v for k, v in summary.get("deleted", {}).items() if v},
                "properties": summary.get("properties", {}),
            },
            "warnings": [],
        }

    # Verify the update against constraints
    try:
        graph_dir = str(path.parent) if path.parent.name else "."

        is_valid, validation_errors = verify_graph_update(graph, graph_dir=graph_dir)

        if not is_valid:
            # Validation failed - rollback the changes
            if backup_path.exists():
                shutil.move(backup_path, path)

            # Format validation errors for the response
            error_messages = []
            for error in validation_errors:
                error_messages.append(
                    {
                        "type": error.get("type", "validation_error"),
                        "message": error.get("message", "Unknown validation error"),
                    }
                )

            return {
                "status": "unsat",
                "committed": False,
                "message": "Graph validation failed after update. Changes rolled back.",
                "errors": error_messages,
            }

        # Validation passed - remove backup
        if backup_path.exists():
            backup_path.unlink()

    except Exception as e:
        # If verification fails unexpectedly, rollback and report
        if backup_path.exists():
            shutil.move(backup_path, path)

        return {
            "status": "unsat",
            "committed": False,
            "message": f"Verification system error: {str(e)}. Changes rolled back.",
            "errors": [{"type": "verification_system_error", "message": str(e)}],
        }

    applied = {
        "created": {k: v for k, v in summary.get("created", {}).items() if v},
        "updated": {k: v for k, v in summary.get("updated", {}).items() if v},
        "deleted": {k: v for k, v in summary.get("deleted", {}).items() if v},
        "properties": summary.get("properties", {}),
    }

    # Clear the cache for the updated graph
    clear_cache(graph)

    return {
        "status": "sat",
        "committed": True,
        "message": "Transaction committed and verified successfully",
        "applied": applied,
        "warnings": [],
    }


def get_update_graph_function_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "update_graph",
            "description": "Atomic batch of create/update/delete across a single graph with automatic validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {"type": "string", "enum": ["origin", "meta", "data"]},
                    "nodes_data": {"type": "array", "items": {"type": "object"}},
                    "edges_data": {"type": "array", "items": {"type": "object"}},
                    "all_values_data": {
                        "type": ["array", "object"],
                        "items": {"type": "object"},
                    },
                    "delete_node_ids": {"type": "array", "items": {"type": "string"}},
                    "delete_edge_ids": {"type": "array", "items": {"type": "string"}},
                    "delete_all_values_data": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["graph"],
                "additionalProperties": False,
            },
        },
    }
