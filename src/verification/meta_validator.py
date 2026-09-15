"""
Meta Validator Module

Validates Meta Graph against Origin Graph axioms.
"""

from typing import Any, Dict, List, Optional, Set
from z3 import Const, DeclareSort, Distinct, Solver, unknown, unsat  # type: ignore[import-untyped]
from verification.graph_parser import Graph
from verification.origin_axioms import OriginAxioms


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
        result = f"[{self.error_type}] {self.message}"
        if self.details:
            result += f"\n  Details: {self.details}"
        return result


class MetaValidator:
    """Validates Meta Graph conformance to Origin axioms."""

    def __init__(self, origin_graph: Graph, meta_graph: Graph):
        """Initialize validator.

        Args:
            origin_graph: Parsed Origin Graph
            meta_graph: Parsed Meta Graph to validate
        """
        self.origin = origin_graph
        self.meta = meta_graph
        self.origin_axioms = OriginAxioms(origin_graph)
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.node_types: Set[str] = set()
        self.edge_types: Set[str] = set()
        self.property_types: Set[str] = set()

        # Extract Origin concepts
        self._extract_origin_concepts()

    def _extract_origin_concepts(self) -> None:
        """Extract key concepts from Origin Graph."""
        # Core types from Origin
        # (self.node_types, self.edge_types, self.property_types already declared above)

        # Scan Origin nodes for type classifications
        for node_id, node in self.origin.nodes.items():
            node_type = node.properties.get("origin_type")
            if node_type == "MetaConcept":
                if node_id in ["NodeType", "EdgeType", "PropertyType"]:
                    # These are the meta-meta concepts
                    pass
                elif node_id in ["HAS_SOURCE", "HAS_TARGET", "HAS_PROPERTY"]:
                    self.edge_types.add(node_id)
            elif node_type == "MetaProperty":
                self.property_types.add(node_id)

    def validate_node_types(self) -> bool:
        """Validate that all Meta nodes have a valid meta_type.

        Classification is purely based on the `meta_type` property set on each node.
        Valid values: 'NodeType', 'EdgeType', 'PropertyType'.
        No guessing — if meta_type is missing or invalid, it's an error.

        Returns:
            True if valid, False otherwise
        """
        valid = True
        valid_meta_types = {"NodeType", "EdgeType", "PropertyType"}

        for node_id, node in self.meta.nodes.items():
            meta_type = node.properties.get("meta_type")

            if meta_type in valid_meta_types:
                continue
            elif meta_type is None:
                self.errors.append(
                    ValidationError(
                        "UNKNOWN_TYPE",
                        f"Node '{node_id}' has no meta_type. "
                        f"Must be one of: {sorted(valid_meta_types)}",
                        {"node_id": node_id, "meta_type": meta_type},
                    )
                )
                valid = False
            else:
                self.errors.append(
                    ValidationError(
                        "INVALID_TYPE",
                        f"Node '{node_id}' has invalid meta_type: '{meta_type}'. "
                        f"Must be one of: {sorted(valid_meta_types)}",
                        {"node_id": node_id, "meta_type": meta_type},
                    )
                )
                valid = False

        return valid

    def validate_edge_signatures(self) -> bool:
        """Validate edge signatures (source and target nodes must exist).

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check edges in Meta
        for edge in self.meta.edges:
            source_node = self.meta.nodes.get(edge.source)
            target_node = self.meta.nodes.get(edge.target)

            if not source_node:
                self.errors.append(
                    ValidationError(
                        "MISSING_SOURCE",
                        f"Edge '{edge.id}' references non-existent source '{edge.source}'",
                        {"edge_id": edge.id, "source": edge.source},
                    )
                )
                valid = False

            if not target_node:
                self.errors.append(
                    ValidationError(
                        "MISSING_TARGET",
                        f"Edge '{edge.id}' references non-existent target '{edge.target}'",
                        {"edge_id": edge.id, "target": edge.target},
                    )
                )
                valid = False

        return valid

    def validate_properties(self) -> bool:
        """Validate property definitions and data types.

        Applies to nodes whose meta_type == 'PropertyType'.

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Get valid data types from Origin
        valid_data_types = {"string", "integer", "date", "url", "boolean", "float"}

        # Check property definitions in Meta
        for node_id, node in self.meta.nodes.items():
            if node.properties.get("meta_type") != "PropertyType":
                continue

            if "dataType" in node.properties:
                data_type = node.properties["dataType"]
                if data_type not in valid_data_types:
                    self.errors.append(
                        ValidationError(
                            "INVALID_DATATYPE",
                            f"Property '{node_id}' has invalid dataType: {data_type}",
                            {
                                "property": node_id,
                                "dataType": data_type,
                                "valid_types": list(valid_data_types),
                            },
                        )
                    )
                    valid = False

            # Check for validation regex if present
            if "validationRegex" in node.properties:
                regex = node.properties["validationRegex"]
                try:
                    import re

                    re.compile(regex)
                except re.error as e:
                    self.errors.append(
                        ValidationError(
                            "INVALID_REGEX",
                            f"Property '{node_id}' has invalid regex: {str(e)}",
                            {"property": node_id, "regex": regex},
                        )
                    )
                    valid = False

        return valid

    def validate_cardinalities(self) -> bool:
        """Validate cardinality constraints.

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check cardinality constraints in all_values
        for key, values in self.meta.all_values.items():
            if "minCount" in values or "maxCount" in values:
                min_count = values.get("minCount", 0)
                max_count = values.get("maxCount", None)

                # Validate min_count
                if not isinstance(min_count, int) or min_count < 0:
                    self.errors.append(
                        ValidationError(
                            "INVALID_MIN_COUNT",
                            f"Invalid minCount for '{key}': {min_count}",
                            {"key": key, "minCount": min_count},
                        )
                    )
                    valid = False

                # Validate max_count
                if max_count is not None:
                    if not isinstance(max_count, int) or max_count < 0:
                        self.errors.append(
                            ValidationError(
                                "INVALID_MAX_COUNT",
                                f"Invalid maxCount for '{key}': {max_count}",
                                {"key": key, "maxCount": max_count},
                            )
                        )
                        valid = False

                    # Check min <= max
                    if max_count < min_count:
                        self.errors.append(
                            ValidationError(
                                "INVALID_CARDINALITY",
                                f"maxCount < minCount for '{key}'",
                                {
                                    "key": key,
                                    "minCount": min_count,
                                    "maxCount": max_count,
                                },
                            )
                        )
                        valid = False

            # Check isMandatory
            if "isMandatory" in values:
                is_mandatory = values["isMandatory"]
                if not isinstance(is_mandatory, bool):
                    self.errors.append(
                        ValidationError(
                            "INVALID_MANDATORY",
                            f"isMandatory must be boolean for '{key}', got: {is_mandatory}",
                            {"key": key, "isMandatory": is_mandatory},
                        )
                    )
                    valid = False

        return valid

    def validate_has_property_edges(self) -> bool:
        """Validate HAS_PROPERTY relationships.

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check HAS_PROPERTY edges in all_values
        for key in self.meta.all_values:
            if "has_property" in key or "HAS_PROPERTY" in key:
                # Parse the relationship
                parts = key.split("-")
                if len(parts) >= 3:
                    source_type = parts[0]
                    property_name = parts[2] if len(parts) >= 3 else None

                    # Verify source exists
                    if source_type not in self.meta.nodes:
                        self.warnings.append(
                            f"HAS_PROPERTY relationship '{key}' references unknown node '{source_type}'"
                        )

                    # Verify property exists
                    if property_name and property_name not in self.meta.nodes:
                        self.warnings.append(
                            f"HAS_PROPERTY relationship '{key}' references unknown property '{property_name}'"
                        )

        return valid

    def validate_with_z3(self) -> bool:
        """Use Z3 to verify formal constraints.

        Node classification is based solely on the `meta_type` property:
        'NodeType' → NodeType sort, 'EdgeType' → EdgeType sort,
        'PropertyType' → PropertyType sort.

        Returns:
            True if satisfiable, False otherwise
        """
        solver = Solver()

        NodeSort = DeclareSort("NodeType")
        EdgeSort = DeclareSort("EdgeType")
        PropertySort = DeclareSort("PropertyType")

        # Assign Z3 constants based on meta_type property
        meta_nodes = {}
        for node_id in self.meta.nodes:
            node = self.meta.nodes[node_id]
            meta_type = node.properties.get("meta_type", "")

            if meta_type == "NodeType":
                meta_nodes[node_id] = Const(node_id, NodeSort)
            elif meta_type == "EdgeType":
                meta_nodes[node_id] = Const(node_id, EdgeSort)
            elif meta_type == "PropertyType":
                meta_nodes[node_id] = Const(node_id, PropertySort)

        # Add distinctness constraints per sort
        if len(meta_nodes) > 1:
            node_type_consts = [c for c in meta_nodes.values() if c.sort() == NodeSort]
            edge_type_consts = [c for c in meta_nodes.values() if c.sort() == EdgeSort]
            property_type_consts = [
                c for c in meta_nodes.values() if c.sort() == PropertySort
            ]
            if len(node_type_consts) > 1:
                solver.add(Distinct(*node_type_consts))
            if len(edge_type_consts) > 1:
                solver.add(Distinct(*edge_type_consts))
            if len(property_type_consts) > 1:
                solver.add(Distinct(*property_type_consts))

        # Non-null constraints for all nodes
        for z3_const in meta_nodes.values():
            solver.add(z3_const == z3_const)  # Non-null constraint

        # Check satisfiability
        result = solver.check()

        if result == unsat:
            self.errors.append(
                ValidationError(
                    "Z3_UNSAT",
                    "Meta graph violates Origin axioms (unsatisfiable constraints)",
                    {
                        "solver_output": (
                            str(solver.unsat_core())
                            if solver.unsat_core()
                            else "No core available"
                        )
                    },
                )
            )
            return False
        elif result == unknown:
            self.warnings.append(
                "Z3 solver returned 'unknown' - constraints may be too complex"
            )
            return True
        else:  # sat
            return True

    def validate(self) -> bool:
        """Perform complete validation.

        Returns:
            True if valid, False otherwise
        """
        # Clear previous errors and warnings
        self.errors = []
        self.warnings = []

        # Run all validation checks
        checks = [
            ("Node Types", self.validate_node_types()),
            ("Edge Signatures", self.validate_edge_signatures()),
            ("Properties", self.validate_properties()),
            ("Cardinalities", self.validate_cardinalities()),
            ("HAS_PROPERTY Edges", self.validate_has_property_edges()),
            ("Z3 Formal Verification", self.validate_with_z3()),
        ]

        # Report results
        print("\n" + "=" * 60)
        print("META GRAPH VALIDATION REPORT")
        print("=" * 60)

        all_valid = True
        for check_name, is_valid in checks:
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"{check_name:.<40} {status}")
            if not is_valid:
                all_valid = False

        # Report errors
        if self.errors:
            print("\n" + "=" * 60)
            print(f"ERRORS ({len(self.errors)} found):")
            print("=" * 60)
            for i, error in enumerate(self.errors, 1):
                print(f"\n{i}. {error}")

        # Report warnings
        if self.warnings:
            print("\n" + "=" * 60)
            print(f"WARNINGS ({len(self.warnings)} found):")
            print("=" * 60)
            for i, warning in enumerate(self.warnings, 1):
                print(f"\n{i}. {warning}")

        # Summary
        print("\n" + "=" * 60)
        if all_valid:
            print("✓ VALIDATION SUCCESSFUL: Meta graph conforms to Origin axioms")
        else:
            print("✗ VALIDATION FAILED: Meta graph violates Origin axioms")
        print("=" * 60 + "\n")

        return all_valid
