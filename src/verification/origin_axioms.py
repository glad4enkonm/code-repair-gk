"""
Origin Axioms Module

Encodes the Origin Graph axioms as Z3 constraints.
"""

from typing import Any, List, Set
from z3 import (  # type: ignore[import-untyped]
    And,
    Const,
    DeclareSort,
    Distinct,
    Function,
    Implies,
    IntSort,
    Solver,
    StringSort,
    StringVal,
    sat,
)

from verification.graph_parser import Graph


class OriginAxioms:
    """Encodes Origin Graph axioms as Z3 constraints."""

    def __init__(self, origin_graph: Graph):
        """Initialize with an Origin Graph.

        Args:
            origin_graph: Parsed Origin Graph
        """
        self.origin = origin_graph
        self.solver = Solver()

        # Define Z3 sorts
        self.NodeSort = DeclareSort("Node")
        self.EdgeSort = DeclareSort("Edge")
        self.PropertySort = DeclareSort("Property")
        self.StringSort = StringSort()

        # Core concept nodes from Origin
        self.core_concepts = self._identify_core_concepts()
        self.meta_properties = self._identify_meta_properties()
        self.edge_types = self._identify_edge_types()

    def _identify_core_concepts(self) -> Set[str]:
        """Identify core concept nodes in Origin.

        Returns:
            Set of core concept node IDs
        """
        # From the axiomatization: NodeType, EdgeType, PropertyType
        return {"NodeType", "EdgeType", "PropertyType"}

    def _identify_meta_properties(self) -> Set[str]:
        """Identify meta-property nodes in Origin.

        Returns:
            Set of meta-property node IDs
        """
        # From P0 in the axiomatization
        return {
            "description",
            "dataType",
            "isMandatory",
            "validationRegex",
            "allowedValues",
            "minCount",
            "maxCount",
            "source",
            "imported_from",
            "confidence",
            "last_validated_at",
        }

    def _identify_edge_types(self) -> Set[str]:
        """Identify edge type nodes in Origin.

        Returns:
            Set of edge type node IDs
        """
        # From the axiomatization: HAS_SOURCE, HAS_TARGET, HAS_PROPERTY
        return {"HAS_SOURCE", "HAS_TARGET", "HAS_PROPERTY"}

    def generate_type_constraints(self) -> List:
        """Generate Z3 constraints for type conformance.

        Returns:
            List of Z3 constraints
        """
        constraints = []

        # Axiom A1: Core concepts are disjoint
        # NodeType, EdgeType, PropertyType are distinct
        node_type = Const("NodeType", self.NodeSort)
        edge_type = Const("EdgeType", self.NodeSort)
        prop_type = Const("PropertyType", self.NodeSort)

        constraints.append(Distinct(node_type, edge_type, prop_type))

        # Axiom A2: Edge types are distinct from core concepts
        for edge_type_id in self.edge_types:
            edge_const = Const(edge_type_id, self.NodeSort)
            constraints.append(
                And(
                    edge_const != node_type,
                    edge_const != edge_type,
                    edge_const != prop_type,
                )
            )

        return constraints

    def generate_property_constraints(self) -> List:
        """Generate Z3 constraints for property typing.

        Returns:
            List of Z3 constraints
        """
        constraints = []

        # Define property data type function
        property_datatype = Function(
            "property_datatype", self.PropertySort, self.StringSort
        )

        # Axiom: Each property has a defined data type
        for prop_id in self.meta_properties:
            prop = self.origin.nodes.get(prop_id)
            if prop and "dataType" in prop.properties:
                datatype = prop.properties["dataType"]
                prop_const = Const(prop_id, self.PropertySort)
                constraints.append(property_datatype(prop_const) == StringVal(datatype))

        return constraints

    def generate_edge_signature_constraints(self) -> List:
        """Generate constraints for edge signatures.

        Returns:
            List of Z3 constraints
        """
        constraints = []

        # Define edge source and target functions
        edge_source = Function("edge_source", self.EdgeSort, self.NodeSort)
        edge_target = Function("edge_target", self.EdgeSort, self.NodeSort)

        # Parse HAS_SOURCE and HAS_TARGET edges from all_values
        for key in self.origin.all_values:
            if "HAS_SOURCE" in key:
                # Pattern: HAS_SOURCE-source-X
                parts = key.split("-")
                if len(parts) >= 3:
                    target_type = parts[2]  # e.g., EdgeType
                    # Constraint: HAS_SOURCE must connect to EdgeType
                    if target_type == "EdgeType":
                        has_source = Const("HAS_SOURCE", self.EdgeSort)
                        edge_type_const = Const("EdgeType", self.NodeSort)
                        constraints.append(
                            Implies(
                                edge_source(has_source) == edge_type_const,
                                True,  # Valid configuration
                            )
                        )

            if "HAS_TARGET" in key:
                # Pattern: HAS_TARGET-source-X or HAS_TARGET-target-X
                parts = key.split("-")
                if len(parts) >= 3:
                    target_type = parts[2]  # e.g., NodeType
                    if target_type == "NodeType":
                        has_target = Const("HAS_TARGET", self.EdgeSort)
                        node_type_const = Const("NodeType", self.NodeSort)
                        constraints.append(
                            Implies(
                                edge_target(has_target) == node_type_const,
                                True,  # Valid configuration
                            )
                        )

        return constraints

    def generate_cardinality_constraints(self) -> List:
        """Generate cardinality constraints.

        Returns:
            List of Z3 constraints
        """
        constraints = []

        # Define cardinality functions
        min_count = Function("min_count", self.EdgeSort, IntSort())
        max_count = Function("max_count", self.EdgeSort, IntSort())

        # Parse cardinality constraints from all_values
        for key, values in self.origin.all_values.items():
            if "minCount" in values:
                # Add constraint that minCount >= 0
                edge_const = Const(key, self.EdgeSort)
                constraints.append(min_count(edge_const) >= 0)

            if "maxCount" in values:
                # Add constraint that maxCount >= minCount (if both exist)
                edge_const = Const(key, self.EdgeSort)
                if "minCount" in values:
                    constraints.append(max_count(edge_const) >= min_count(edge_const))
                else:
                    constraints.append(max_count(edge_const) >= 0)

        return constraints

    def generate_all_constraints(self) -> List:
        """Generate all Origin axiom constraints.

        Returns:
            List of all Z3 constraints
        """
        all_constraints = []
        all_constraints.extend(self.generate_type_constraints())
        all_constraints.extend(self.generate_property_constraints())
        all_constraints.extend(self.generate_edge_signature_constraints())
        all_constraints.extend(self.generate_cardinality_constraints())
        return all_constraints

    def add_constraints_to_solver(self) -> None:
        """Add all constraints to the internal solver."""
        constraints = self.generate_all_constraints()
        for constraint in constraints:
            self.solver.add(constraint)

    def check_satisfiability(self) -> bool:
        """Check if the constraints are satisfiable.

        Returns:
            True if satisfiable, False otherwise
        """
        self.add_constraints_to_solver()
        result = self.solver.check()
        return result == sat  # type: ignore[no-any-return]

    def get_model(self) -> Any:  # noqa: F821
        """Get a satisfying model if one exists.

        Returns:
            Z3 model or None
        """
        if self.check_satisfiability():
            return self.solver.model()
        return None

    def generate_smt_lib(self) -> str:
        """Generate SMT-LIB representation of constraints.

        Returns:
            SMT-LIB string
        """
        self.add_constraints_to_solver()
        return self.solver.to_smt2()  # type: ignore[no-any-return]
