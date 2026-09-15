#!/usr/bin/env python3
"""
Data Graph Validator

Validates that data.json conforms to the schema defined in meta.json.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from verification.graph_parser import Graph, GraphParser


class ValidationError:
    """Represents a validation error."""

    def __init__(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        details_str = ""
        if self.details:
            details_str = "\n    " + "\n    ".join(
                f"{k}: {v}" for k, v in self.details.items()
            )
        return f"[{self.error_type}] {self.message}{details_str}"


# Edge types defined in Origin (not Meta): they are always valid labels
_BUILTIN_EDGE_TYPES = {"HAS_PROPERTY", "HAS_SOURCE", "HAS_TARGET"}


class DataValidator:
    """Validates Data Graph conformance to Meta schema."""

    def __init__(
        self,
        meta_graph: Graph,
        data_graph: Graph,
        origin_graph: Optional[Graph] = None,
    ):
        """Initialize validator.

        Args:
            meta_graph: Parsed Meta Graph (schema)
            data_graph: Parsed Data Graph to validate
            origin_graph: Parsed Origin Graph for built-in edge types
        """
        self.meta_graph = meta_graph
        self.data_graph = data_graph
        self.origin_graph = origin_graph
        self.errors: List[ValidationError] = []

        # Extract meta schema information
        self._extract_meta_schema()

    def _extract_meta_schema(self) -> None:
        """Extract schema information from Meta Graph."""
        self.node_types: set[str] = set()
        self.edge_types: set[str] = set()
        self.property_types: set[str] = set()
        self.edge_signatures: Dict[str, List[Tuple[str, str]]] = {}
        # edge_type -> [(source_type, target_type), ...]
        # Multiple signatures per edge type are supported (e.g., CONTAINS:
        # Directory→File, File→Class, Class→Method, Method→Variable).

        # Temporary storage for pairing HAS_SOURCE with HAS_TARGET
        pending: Dict[str, List[Tuple[str, str]]] = {}
        # pending[edge_type] = [('source', 'Directory'), ('target', 'File'), ...]
        self.type_properties: Dict[str, List[str]] = defaultdict(list)
        # type -> [properties]
        self.property_constraints = {}  # (type, property) -> constraints
        self.property_data_types = {}  # property_name -> data_type

        # Identify types from meta graph
        for node in self.meta_graph.nodes.values():
            node_props = self.meta_graph.all_values.get(node.id, {})
            meta_type = node_props.get("meta_type", "")

            if meta_type == "NodeType":
                self.node_types.add(node.id)
            elif meta_type == "EdgeType":
                self.edge_types.add(node.id)
            elif meta_type == "PropertyType":
                self.property_types.add(node.id)
                data_type = node_props.get("dataType", "string")
                self.property_data_types[node.id] = data_type

        # Extract edge signatures (source and target types)
        # Supports multiple signatures per edge type by pairing
        # HAS_SOURCE and HAS_TARGET edges in order of appearance.
        for edge in self.meta_graph.edges:
            if edge.label and edge.label.startswith("HAS_SOURCE"):
                pending.setdefault(edge.source, []).append(("source", edge.target))
            elif edge.label and edge.label.startswith("HAS_TARGET"):
                pending.setdefault(edge.source, []).append(("target", edge.target))
            elif edge.label == "HAS_PROPERTY":
                node_type = edge.source
                prop_type = edge.target
                self.type_properties[node_type].append(prop_type)

                # Get constraints for this property
                constraint_key = edge.id
                constraints = self.meta_graph.all_values.get(constraint_key, {})
                if constraints:
                    self.property_constraints[(node_type, prop_type)] = constraints

        # Pair HAS_SOURCE with HAS_TARGET by position to build signatures
        for edge_type, items in pending.items():
            sources = [t[1] for t in items if t[0] == "source"]
            targets = [t[1] for t in items if t[0] == "target"]
            if len(sources) != len(targets):
                self.errors.append(
                    ValidationError(
                        "MALFORMED_SIGNATURE",
                        f"EdgeType '{edge_type}' has {len(sources)} HAS_SOURCE but "
                        f"{len(targets)} HAS_TARGET edges; must be paired 1:1",
                        {"edge_type": edge_type},
                    )
                )
                continue
            self.edge_signatures[edge_type] = list(zip(sources, targets))

        # Merge built-in edge types from Origin (HAS_PROPERTY, HAS_SOURCE,
        # HAS_TARGET).  HAS_PROPERTY can connect any NodeType to any
        # PropertyType in Data.  HAS_SOURCE/HAS_TARGET are meta-level only
        # and should not appear in Data, but we still recognise them.
        self.edge_types |= _BUILTIN_EDGE_TYPES

        if self.node_types and self.property_types:
            self.edge_signatures.setdefault(
                "HAS_PROPERTY",
                [
                    (nt, pt)
                    for nt in sorted(self.node_types)
                    for pt in sorted(self.property_types)
                ],
            )

    def validate_node_types(self) -> bool:
        """Validate that all data nodes use valid types defined in meta.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking node types...")
        valid = True

        for node in self.data_graph.nodes.values():
            node_type = GraphParser.get_node_type(node, self.data_graph)

            if node_type not in self.node_types:
                self.errors.append(
                    ValidationError(
                        "INVALID_NODE_TYPE",
                        f"Node '{node.id}' uses undefined type '{node_type}'",
                        {
                            "node_id": node.id,
                            "type": node_type,
                            "valid_types": list(self.node_types),
                        },
                    )
                )
                valid = False

        if valid:
            print(f"      ✓ All {len(self.data_graph.nodes)} nodes use valid types")
        else:
            print("      ✗ Found nodes with invalid types")

        return valid

    def validate_edge_types(self) -> bool:
        """Validate that all data edges use valid types defined in meta.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking edge types...")
        valid = True

        for edge in self.data_graph.edges:
            edge_type = edge.label

            if edge_type not in self.edge_types:
                self.errors.append(
                    ValidationError(
                        "INVALID_EDGE_TYPE",
                        f"Edge '{edge.id}' uses undefined type '{edge_type}'",
                        {
                            "edge_id": edge.id,
                            "type": edge_type,
                            "valid_types": list(self.edge_types),
                        },
                    )
                )
                valid = False

        if valid:
            print(f"      ✓ All {len(self.data_graph.edges)} edges use valid types")
        else:
            print("      ✗ Found edges with invalid types")

        return valid

    def validate_edge_connections(self) -> bool:
        """Validate that edge connections match meta schema.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking edge connections...")
        valid = True

        # Build node type lookup
        node_types_map = {
            node.id: GraphParser.get_node_type(node, self.data_graph)
            for node in self.data_graph.nodes.values()
        }

        for edge in self.data_graph.edges:
            edge_type = edge.label

            # Skip if edge type is invalid (already caught in validate_edge_types)
            if edge_type not in self.edge_signatures:
                continue

            # Get actual source and target types
            actual_source_type = node_types_map.get(edge.source)
            actual_target_type = node_types_map.get(edge.target)

            # Check: (actual_source_type, actual_target_type) matches
            # at least one allowed signature for this edge type
            allowed_signatures = self.edge_signatures[edge_type]
            matched = False
            for allowed_src, allowed_tgt in allowed_signatures:
                if (
                    actual_source_type == allowed_src
                    and actual_target_type == allowed_tgt
                ):
                    matched = True
                    break

            if not matched:
                self.errors.append(
                    ValidationError(
                        "INVALID_EDGE_SIGNATURE",
                        f"Edge '{edge.id}' of type '{edge_type}' has invalid "
                        f"({actual_source_type} → {actual_target_type}); "
                        f"allowed: {allowed_signatures}",
                        {
                            "edge_id": edge.id,
                            "edge_type": edge_type,
                            "actual_source_type": actual_source_type,
                            "actual_target_type": actual_target_type,
                            "allowed_signatures": allowed_signatures,
                            "source_node": edge.source,
                            "target_node": edge.target,
                        },
                    )
                )
                valid = False

        if valid:
            print("      ✓ All edge connections match schema")
        else:
            print("      ✗ Found invalid edge connections")

        return valid

    def validate_required_properties(self) -> bool:
        """Validate that nodes have required properties.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking required properties...")
        valid = True

        for node in self.data_graph.nodes.values():
            node_type = GraphParser.get_node_type(node, self.data_graph)

            # Skip if node type is invalid
            if node_type not in self.node_types:
                continue

            # Get node properties
            node_props = self.data_graph.all_values.get(node.id, {})

            # Check each property defined for this type
            for prop_name in self.type_properties.get(node_type, []):
                constraints = self.property_constraints.get((node_type, prop_name), {})
                is_mandatory = constraints.get("isMandatory", False)

                # Convert string boolean to actual boolean if needed
                if isinstance(is_mandatory, str):
                    is_mandatory = is_mandatory.lower() == "true"

                if is_mandatory and prop_name not in node_props:
                    self.errors.append(
                        ValidationError(
                            "MISSING_REQUIRED_PROPERTY",
                            f"Node '{node.id}' of type '{node_type}' missing required property '{prop_name}'",
                            {
                                "node_id": node.id,
                                "node_type": node_type,
                                "missing_property": prop_name,
                            },
                        )
                    )
                    valid = False

        if valid:
            print("      ✓ All required properties present")
        else:
            print("      ✗ Found nodes missing required properties")

        return valid

    def validate_property_data_types(self) -> bool:
        """Validate that property values match expected data types.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking property data types...")
        valid = True

        for node in self.data_graph.nodes.values():
            node_type = GraphParser.get_node_type(node, self.data_graph)  # noqa: F841
            node_props = self.data_graph.all_values.get(node.id, {})

            for prop_name, prop_value in node_props.items():
                # Skip if property is not defined in meta
                if prop_name not in self.property_data_types:
                    continue

                expected_type = self.property_data_types[prop_name]

                # Validate based on expected type
                type_valid = True
                if expected_type == "string":
                    if not isinstance(prop_value, str):
                        type_valid = False
                elif expected_type == "integer":
                    if not isinstance(prop_value, int):
                        type_valid = False
                elif expected_type == "float":
                    if not isinstance(prop_value, (int, float)):
                        type_valid = False
                elif expected_type == "boolean":
                    if not isinstance(prop_value, bool):
                        type_valid = False
                elif expected_type == "date":
                    if not isinstance(prop_value, str):
                        type_valid = False
                    # Could add date format validation here

                if not type_valid:
                    self.errors.append(
                        ValidationError(
                            "INVALID_PROPERTY_TYPE",
                            f"Property '{prop_name}' on node '{node.id}' has wrong type",
                            {
                                "node_id": node.id,
                                "property": prop_name,
                                "expected_type": expected_type,
                                "actual_type": type(prop_value).__name__,
                                "value": str(prop_value),
                            },
                        )
                    )
                    valid = False

        if valid:
            print("      ✓ All property types match schema")
        else:
            print("      ✗ Found properties with wrong types")

        return valid

    def validate_cardinalities(self) -> bool:
        """Validate cardinality constraints for relationships.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking cardinality constraints...")
        valid = True

        # Count outgoing edges by type for each node
        edge_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for edge in self.data_graph.edges:
            if edge.label is not None:
                edge_counts[edge.source][edge.label] += 1

        # Check cardinality constraints
        for node in self.data_graph.nodes.values():
            node_type = GraphParser.get_node_type(node, self.data_graph)

            # Skip if node type is invalid
            if node_type not in self.node_types:
                continue

            # Check each edge type that can originate from this node type
            for edge_type, signatures in self.edge_signatures.items():
                # Does this edge type allow source == node_type?
                if not any(src == node_type for src, _tgt in signatures):
                    continue

                # Look for cardinality constraints on this edge type
                # Find constraint key in meta graph
                constraint_key = None
                for edge in self.meta_graph.edges:
                    if (
                        edge.source == edge_type
                        and edge.label
                        and edge.label.startswith("HAS_SOURCE")
                    ):
                        # Check if there's a constraint edge
                        for constraint_edge in self.meta_graph.edges:
                            if (
                                constraint_edge.source == node_type
                                and constraint_edge.target == edge_type
                            ):
                                constraint_key = constraint_edge.id
                                break
                        break

                if not constraint_key:
                    continue

                constraints = self.meta_graph.all_values.get(constraint_key, {})
                min_count = constraints.get("minCount")
                max_count = constraints.get("maxCount")

                actual_count = edge_counts[node.id][edge_type]

                if min_count is not None and actual_count < min_count:
                    self.errors.append(
                        ValidationError(
                            "CARDINALITY_VIOLATION_MIN",
                            f"Node '{node.id}' has too few '{edge_type}' edges",
                            {
                                "node_id": node.id,
                                "edge_type": edge_type,
                                "min_count": min_count,
                                "actual_count": actual_count,
                            },
                        )
                    )
                    valid = False

                if max_count is not None and actual_count > max_count:
                    self.errors.append(
                        ValidationError(
                            "CARDINALITY_VIOLATION_MAX",
                            f"Node '{node.id}' has too many '{edge_type}' edges",
                            {
                                "node_id": node.id,
                                "edge_type": edge_type,
                                "max_count": max_count,
                                "actual_count": actual_count,
                            },
                        )
                    )
                    valid = False

        if valid:
            print("      ✓ All cardinality constraints satisfied")
        else:
            print("      ✗ Found cardinality violations")

        return valid

    def validate_property_patterns(self) -> bool:
        """Validate property values against regex patterns.

        Returns:
            True if valid, False otherwise
        """
        print("\n   Checking validation patterns...")
        valid = True

        # Get validation patterns from meta
        patterns = {}
        for prop_name in self.property_types:
            prop_data = self.meta_graph.all_values.get(prop_name, {})
            if "validationRegex" in prop_data:
                patterns[prop_name] = prop_data["validationRegex"]

        if not patterns:
            print("      ✓ No validation patterns to check")
            return True

        import re

        for node in self.data_graph.nodes.values():
            node_props = self.data_graph.all_values.get(node.id, {})

            for prop_name, prop_value in node_props.items():
                if prop_name in patterns:
                    pattern = patterns[prop_name]
                    if isinstance(prop_value, str) and not re.match(
                        pattern, prop_value
                    ):
                        self.errors.append(
                            ValidationError(
                                "PATTERN_VALIDATION_FAILED",
                                f"Property '{prop_name}' on node '{node.id}' doesn't match pattern",
                                {
                                    "node_id": node.id,
                                    "property": prop_name,
                                    "pattern": pattern,
                                    "value": prop_value,
                                },
                            )
                        )
                        valid = False

        if valid:
            print("      ✓ All validation patterns satisfied")
        else:
            print("      ✗ Found pattern validation failures")

        return valid

    def validate(self) -> bool:
        """Perform complete validation.

        Returns:
            True if valid, False otherwise
        """
        print("\n" + "=" * 60)
        print("VALIDATING DATA GRAPH AGAINST META SCHEMA")
        print("=" * 60)

        # Run all validation checks
        checks = [
            self.validate_node_types(),
            self.validate_edge_types(),
            self.validate_edge_connections(),
            self.validate_required_properties(),
            self.validate_property_data_types(),
            self.validate_cardinalities(),
            self.validate_property_patterns(),
        ]

        is_valid = all(checks)

        # Report results
        print("\n" + "=" * 60)
        if is_valid:
            print("✓ DATA VALIDATION PASSED")
            print(
                f"   All {len(self.data_graph.nodes)} nodes and {len(self.data_graph.edges)} edges conform to schema"
            )
        else:
            print("✗ DATA VALIDATION FAILED")
            print(f"   Found {len(self.errors)} error(s):\n")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"   {error}")
            if len(self.errors) > 10:
                print(f"\n   ... and {len(self.errors) - 10} more errors")
        print("=" * 60)

        return is_valid
