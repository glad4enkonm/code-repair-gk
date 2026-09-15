#!/usr/bin/env python3
"""
Generate SMT Constraints Script

Generates SMT-LIB constraints from origin.json for formal verification.
"""

import sys
from pathlib import Path

from verification.graph_parser import GraphParser
from verification.origin_axioms import OriginAxioms


def main() -> None:
    """Generate and display SMT constraints."""

    # Path to origin file
    origin_path = Path(__file__).parent.parent / "origin.json"

    if not origin_path.exists():
        print(f"Error: Origin file not found at {origin_path}")
        sys.exit(1)

    print("=" * 60)
    print("SMT CONSTRAINT GENERATOR")
    print("Generating constraints from Origin Graph")
    print("=" * 60)

    # Parse Origin graph
    print("\n1. Parsing Origin Graph...")
    try:
        origin_graph = GraphParser.parse_graph_file(str(origin_path))
        print(f"   ✓ Loaded {len(origin_graph.nodes)} nodes")
        print(f"   ✓ Loaded {len(origin_graph.edges)} edges")
    except Exception as e:
        print(f"   ✗ Failed to parse: {e}")
        sys.exit(1)

    # Generate axioms
    print("\n2. Generating Origin Axioms...")
    axioms = OriginAxioms(origin_graph)

    # Display core concepts
    print("\n3. Core Concepts:")
    print(f"   - Core Types: {axioms.core_concepts}")
    print(f"   - Edge Types: {axioms.edge_types}")
    print(f"   - Properties: {axioms.meta_properties}")

    # Generate constraints
    print("\n4. Generating Z3 Constraints...")

    print("\n   Type Constraints:")
    type_constraints = axioms.generate_type_constraints()
    for i, constraint in enumerate(type_constraints, 1):
        print(f"     {i}. {constraint}")

    print("\n   Property Constraints:")
    prop_constraints = axioms.generate_property_constraints()
    for i, constraint in enumerate(prop_constraints, 1):
        print(f"     {i}. {constraint}")

    print("\n   Edge Signature Constraints:")
    edge_constraints = axioms.generate_edge_signature_constraints()
    for i, constraint in enumerate(edge_constraints, 1):
        print(f"     {i}. {constraint}")

    print("\n   Cardinality Constraints:")
    card_constraints = axioms.generate_cardinality_constraints()
    for i, constraint in enumerate(card_constraints, 1):
        print(f"     {i}. {constraint}")

    # Generate SMT-LIB
    print("\n5. SMT-LIB Output:")
    print("=" * 60)
    try:
        smt_lib = axioms.generate_smt_lib()
        print(smt_lib)
    except Exception as e:
        print(f"Failed to generate SMT-LIB: {e}")

    print("=" * 60)
    print("\n✓ Constraint generation complete")

    # Save to file
    output_path = Path(__file__).parent / "origin_constraints.smt2"
    try:
        with open(output_path, "w") as f:
            f.write(smt_lib)
        print(f"✓ Saved SMT-LIB to: {output_path}")
    except Exception as exc:
        print(
            f"✗ Failed to save SMT-LIB to {output_path}: {exc}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
