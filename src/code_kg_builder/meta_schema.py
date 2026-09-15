"""MetaSchema — parse meta.json into a structured representation.

Reuses the extraction logic from ``verification/data_validator.py``
(``_extract_meta_schema``) for edge signatures, type properties, and
cardinality constraints.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class EdgeSignature:
    """A single source→target type pair for an EdgeType."""

    source_type: str
    target_type: str


@dataclass
class MetaSchema:
    """Structured representation of a meta.json graph."""

    node_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    property_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_signatures: Dict[str, List[EdgeSignature]] = field(default_factory=dict)
    type_properties: Dict[str, List[str]] = field(default_factory=dict)
    property_constraints: Dict[Tuple[str, str], Dict[str, Any]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, path: str) -> "MetaSchema":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "MetaSchema":
        schema = cls()
        schema._extract(data)
        return schema

    # ------------------------------------------------------------------ #
    #  Extraction
    # ------------------------------------------------------------------ #

    def _extract(self, data: Dict[str, Any]) -> None:
        nodes = data.get("nodes", data.get("nodes_data", []))
        edges = data.get("edges", data.get("edges_data", []))
        all_values = data.get("allValues", data.get("all_values_data", {}))

        # If all_values_data is a list, convert to dict keyed by id
        if isinstance(all_values, list):
            all_values = {item["id"]: item for item in all_values if "id" in item}

        # Classify nodes by meta_type
        for node in nodes:
            nid = node.get("id", "")
            props = all_values.get(nid, {})
            meta_type = props.get("meta_type", "")

            if meta_type == "NodeType":
                self.node_types[nid] = props
            elif meta_type == "EdgeType":
                self.edge_types[nid] = props
            elif meta_type == "PropertyType":
                self.property_types[nid] = props

        # Extract edge signatures and type properties from edges
        pending: Dict[str, List[Tuple[str, str]]] = {}

        for edge in edges:
            label = edge.get("label", "")
            source = edge.get("source", "")
            target = edge.get("target", "")

            if label.startswith("HAS_SOURCE"):
                pending.setdefault(source, []).append(("source", target))
            elif label.startswith("HAS_TARGET"):
                pending.setdefault(source, []).append(("target", target))
            elif label == "HAS_PROPERTY":
                self.type_properties.setdefault(source, []).append(target)
                constraint_key = edge.get("id", f"{source}-HAS_PROPERTY-{target}")
                constraints = all_values.get(constraint_key, {})
                if constraints:
                    self.property_constraints[(source, target)] = constraints

        # Pair HAS_SOURCE with HAS_TARGET by position
        for edge_type, items in pending.items():
            sources = [t for kind, t in items if kind == "source"]
            targets = [t for kind, t in items if kind == "target"]
            if not sources and not targets:
                continue
            self.edge_signatures[edge_type] = [
                EdgeSignature(source_type=s, target_type=t)
                for s, t in zip(sources, targets)
            ]

    # ------------------------------------------------------------------ #
    #  Executor helpers
    # ------------------------------------------------------------------ #

    def get_executors(self) -> Dict[str, List[str]]:
        """Return ``{executor_path: [type_names]}`` — deduplicated.

        Collects executor paths from both node_types and edge_types.
        Multiple types pointing to the same executor are grouped.
        """
        result: Dict[str, List[str]] = {}
        for type_name, props in self.node_types.items():
            path = props.get("executor")
            if path:
                result.setdefault(path, []).append(type_name)
        for type_name, props in self.edge_types.items():
            path = props.get("executor")
            if path:
                result.setdefault(path, []).append(type_name)
        return result

    # ------------------------------------------------------------------ #
    #  Validation helpers
    # ------------------------------------------------------------------ #

    def is_valid_edge_connection(
        self, edge_label: str, source_type: str, target_type: str
    ) -> bool:
        """Check if an edge connection matches any defined signature."""
        sigs = self.edge_signatures.get(edge_label, [])
        return any(
            s.source_type == source_type and s.target_type == target_type for s in sigs
        )

    def get_mandatory_properties(self, type_name: str) -> List[str]:
        """Return mandatory property names for a given type."""
        result = []
        for prop_name in self.type_properties.get(type_name, []):
            constraints = self.property_constraints.get((type_name, prop_name), {})
            if constraints.get("isMandatory", False):
                result.append(prop_name)
        return result

    def get_property_data_type(self, prop_name: str) -> str:
        """Return the dataType for a property type, defaulting to 'string'."""
        result = self.property_types.get(prop_name, {}).get("dataType", "string")
        return str(result) if result else "string"
