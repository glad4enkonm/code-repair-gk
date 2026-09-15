#!/usr/bin/env python3
"""
Test Verification Script

Demonstrates the verification system with test cases.
Creates test Meta graphs in memory and validates them against Origin.
"""

import argparse
import json
import sys
from pathlib import Path

from verification.graph_parser import GraphParser, Graph
from verification.meta_validator import MetaValidator

_default_dir = Path(__file__).parent.parent


def create_invalid_meta_graph() -> dict:
    """Create an intentionally invalid Meta graph for testing."""
    return {
        "nodes": [
            {"id": "InvalidNode", "label": "InvalidNode", "x": 0, "y": 0},
            {"id": "BadProperty", "label": "BadProperty", "x": 100, "y": 100},
        ],
        "edges": [
            {
                "id": "invalid-edge",
                "source": "InvalidNode",
                "target": "NonExistent",
                "label": "INVALID",
            }
        ],
        "allValues": {
            "BadProperty": {
                "dataType": "invalid_type",
                "meta_type": "PropertyType",
            },
            "InvalidNode": {"meta_type": "InvalidType"},
            "invalid-constraint": {"minCount": 5, "maxCount": 2},
        },
    }


def create_valid_meta_graph() -> dict:
    """Create a minimal valid Meta graph."""
    return {
        "nodes": [
            {"id": "Document", "label": "Document", "x": 0, "y": 0},
            {"id": "Person", "label": "Person", "x": 200, "y": 0},
            {"id": "name", "label": "name", "x": 100, "y": 200},
            {"id": "CREATED_BY", "label": "CREATED_BY", "x": 100, "y": 100},
        ],
        "edges": [
            {"id": "CREATED_BY-source", "source": "CREATED_BY", "target": "Document"},
            {"id": "CREATED_BY-target", "source": "CREATED_BY", "target": "Person"},
        ],
        "allValues": {
            "Document": {
                "description": "A document or publication",
                "meta_type": "NodeType",
            },
            "Person": {"description": "A person or author", "meta_type": "NodeType"},
            "CREATED_BY": {
                "description": "Indicates document creation relationship",
                "meta_type": "EdgeType",
            },
            "name": {
                "description": "The name of a person",
                "dataType": "string",
                "meta_type": "PropertyType",
            },
            "Person-has_property-name": {
                "isMandatory": True,
                "minCount": 1,
                "maxCount": 1,
            },
        },
    }


def test_validation(test_name: str, meta_data: dict, origin_graph: Graph) -> bool:
    """Run validation test on a Meta graph."""
    print("\n" + "#" * 60)
    print(f"# TEST: {test_name}")
    print("#" * 60)

    # Parse Meta graph
    meta_graph = GraphParser.parse_graph_data(meta_data)
    # Validate
    validator = MetaValidator(origin_graph, meta_graph)
    return bool(validator.validate())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run verification tests with Z3 solver"
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=str(_default_dir / "origin.json"),
        help="Path to origin.json (default: ../origin.json)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to meta.json for optional actual test (default: none)",
    )
    args = parser.parse_args()

    origin_path = Path(args.origin)

    if not origin_path.exists():
        print(f"Error: Origin file not found at {origin_path}")
        sys.exit(1)

    print("=" * 60)
    print("VERIFICATION SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"  origin: {origin_path}")

    # Parse Origin
    origin_graph = GraphParser.parse_graph_file(str(origin_path))
    print(f"\nLoaded Origin Graph: {len(origin_graph.nodes)} nodes")

    # Test 1: Invalid Meta graph
    result1 = test_validation(
        "Invalid Meta Graph", create_invalid_meta_graph(), origin_graph
    )

    # Test 2: Valid Meta graph
    result2 = test_validation(
        "Valid Meta Graph", create_valid_meta_graph(), origin_graph
    )

    # Test 3: Actual meta.json (optional)
    if args.meta:
        meta_path = Path(args.meta)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                actual_meta = json.load(f)
            result3 = test_validation(
                f"Actual meta.json ({meta_path})", actual_meta, origin_graph
            )
        else:
            print(f"\nWarning: Meta file not found at {meta_path}, skipping test 3")
            result3 = None
    else:
        result3 = None

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(
        f"Invalid Meta Graph: {'✗ Failed (Expected)' if not result1 else '✓ Passed (Unexpected!)'}"
    )
    print(
        f"Valid Meta Graph:   {'✓ Passed (Expected)' if result2 else '✗ Failed (Unexpected!)'}"
    )
    if result3 is not None:
        print(f"Actual meta.json:  {'✓ Passed' if result3 else '✗ Failed'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
