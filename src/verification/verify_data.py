#!/usr/bin/env python3
"""
Main Data Graph Verification Script

Verifies that data.json conforms to the schema defined in meta.json.
"""

import argparse
import sys
from pathlib import Path

from verification.graph_parser import GraphParser
from verification.data_validator import DataValidator

_default_dir = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify data.json against meta.json schema"
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=str(_default_dir / "meta.json"),
        help="Path to meta.json (default: ../meta.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(_default_dir / "data.json"),
        help="Path to data.json (default: ../data.json)",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    data_path = Path(args.data)

    # Check if files exist
    if not meta_path.exists():
        print(f"Error: Meta file not found at {meta_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("DATA GRAPH VERIFICATION SYSTEM")
    print("Validating data.json against meta.json schema")
    print("=" * 60)
    print(f"  meta: {meta_path}")
    print(f"  data: {data_path}")

    # Parse graphs
    print("\n1. Parsing Meta Graph (Schema)...")
    try:
        meta_graph = GraphParser.parse_graph_file(str(meta_path))
        print(
            f"   ✓ Loaded {len(meta_graph.nodes)} nodes, {len(meta_graph.edges)} edges"
        )
        print(f"   ✓ Found {len(meta_graph.all_values)} property definitions")
    except Exception as e:
        print(f"   ✗ Failed to parse Meta Graph: {e}")
        sys.exit(1)

    print("\n2. Parsing Data Graph...")
    try:
        data_graph = GraphParser.parse_graph_file(str(data_path))
        print(
            f"   ✓ Loaded {len(data_graph.nodes)} nodes, {len(data_graph.edges)} edges"
        )
        print(f"   ✓ Found {len(data_graph.all_values)} property values")
    except Exception as e:
        print(f"   ✗ Failed to parse Data Graph: {e}")
        sys.exit(1)

    print("\n3. Validating Data Graph against Meta Schema...")
    try:
        validator = DataValidator(meta_graph, data_graph)
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
