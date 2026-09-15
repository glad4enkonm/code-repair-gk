#!/usr/bin/env python3
"""
Main Meta Graph Verification Script

Verifies that meta.json conforms to the axioms defined in origin.json.
"""

import argparse
import sys
from pathlib import Path

from verification.graph_parser import GraphParser
from verification.origin_axioms import OriginAxioms
from verification.meta_validator import MetaValidator

_default_dir = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify meta.json against origin.json")
    parser.add_argument(
        "--origin",
        type=str,
        default=str(_default_dir / "origin.json"),
        help="Path to origin.json (default: ../origin.json)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=str(_default_dir / "meta.json"),
        help="Path to meta.json (default: ../meta.json)",
    )
    args = parser.parse_args()

    origin_path = Path(args.origin)
    meta_path = Path(args.meta)

    # Check if files exist
    if not origin_path.exists():
        print(f"Error: Origin file not found at {origin_path}")
        sys.exit(1)

    if not meta_path.exists():
        print(f"Error: Meta file not found at {meta_path}")
        sys.exit(1)

    print("=" * 60)
    print("GRAPH VERIFICATION SYSTEM")
    print("Using Z3 SMT Solver")
    print("=" * 60)
    print(f"  origin: {origin_path}")
    print(f"  meta:   {meta_path}")

    # Parse graphs
    print("\n1. Parsing Origin Graph...")
    try:
        origin_graph = GraphParser.parse_graph_file(str(origin_path))
        print(
            f"   ✓ Loaded {len(origin_graph.nodes)} nodes, {len(origin_graph.edges)} edges"
        )
        print(f"   ✓ Found {len(origin_graph.all_values)} property definitions")
    except Exception as e:
        print(f"   ✗ Failed to parse Origin Graph: {e}")
        sys.exit(1)

    print("\n2. Parsing Meta Graph...")
    try:
        meta_graph = GraphParser.parse_graph_file(str(meta_path))
        print(
            f"   ✓ Loaded {len(meta_graph.nodes)} nodes, {len(meta_graph.edges)} edges"
        )
        print(f"   ✓ Found {len(meta_graph.all_values)} property definitions")
    except Exception as e:
        print(f"   ✗ Failed to parse Meta Graph: {e}")
        sys.exit(1)

    print("\n3. Extracting Origin Axioms...")
    try:
        axioms = OriginAxioms(origin_graph)
        print(f"   ✓ Identified {len(axioms.core_concepts)} core concepts")
        print(f"   ✓ Identified {len(axioms.meta_properties)} meta-properties")
        print(f"   ✓ Identified {len(axioms.edge_types)} edge types")

        # Check Origin consistency
        print("\n4. Checking Origin Graph Consistency...")
        if axioms.check_satisfiability():
            print("   ✓ Origin axioms are satisfiable")
        else:
            print("   ✗ Origin axioms are unsatisfiable (internal inconsistency)")
            sys.exit(1)
    except Exception as e:
        print(f"   ✗ Failed to process Origin axioms: {e}")
        sys.exit(1)

    print("\n5. Validating Meta Graph against Origin Axioms...")
    try:
        validator = MetaValidator(origin_graph, meta_graph)
        is_valid = validator.validate()

        # Exit with appropriate code
        sys.exit(0 if is_valid else 1)

    except Exception as e:
        print(f"   ✗ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
