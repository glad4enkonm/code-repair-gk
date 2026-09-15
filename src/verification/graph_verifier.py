"""Main integration module for graph verification.

Provides a unified interface for verifying graph updates against their constraints.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from verification.graph_parser import GraphParser
from verification.origin_axioms import OriginAxioms
from verification.meta_validator import MetaValidator
from verification.data_validator import DataValidator


def verify_graph_update(
    graph_name: str, graph_dir: str = "."
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Verify a graph update based on its level in the hierarchy.

    Args:
        graph_name: Name of the graph to verify ('origin', 'meta', or 'data')
        graph_dir: Directory containing the graph JSON files (default: current directory)

    Returns:
        Tuple of (is_valid, errors) where:
        - is_valid: True if validation passed, False otherwise
        - errors: List of error dictionaries with 'type' and 'message' keys
    """
    errors: List[Dict[str, Any]] = []
    validator: Union[MetaValidator, DataValidator, None] = None

    # Determine paths
    base_path = Path(graph_dir)
    origin_path = base_path / "origin.json"
    meta_path = base_path / "meta.json"
    data_path = base_path / "data.json"

    try:
        if graph_name == "origin":
            # Origin graph is self-validating - check internal consistency
            if not origin_path.exists():
                errors.append(
                    {
                        "type": "file_not_found",
                        "message": f"Origin file not found at {origin_path}",
                    }
                )
                return False, errors

            try:
                origin_graph = GraphParser.parse_graph_file(str(origin_path))

                # Check if Origin axioms are internally consistent
                axioms = OriginAxioms(origin_graph)
                axioms.add_constraints_to_solver()

                if not axioms.check_satisfiability():
                    errors.append(
                        {
                            "type": "origin_unsatisfiable",
                            "message": "Origin axioms are internally inconsistent",
                        }
                    )
                    return False, errors

                return True, []

            except Exception as e:
                errors.append(
                    {
                        "type": "origin_validation_error",
                        "message": f"Error validating origin graph: {str(e)}",
                    }
                )
                return False, errors

        elif graph_name == "meta":
            # Validate meta against origin
            if not origin_path.exists():
                errors.append(
                    {
                        "type": "file_not_found",
                        "message": f"Origin file not found at {origin_path}",
                    }
                )
                return False, errors

            if not meta_path.exists():
                errors.append(
                    {
                        "type": "file_not_found",
                        "message": f"Meta file not found at {meta_path}",
                    }
                )
                return False, errors

            try:
                origin_graph = GraphParser.parse_graph_file(str(origin_path))
                meta_graph = GraphParser.parse_graph_file(str(meta_path))

                validator = MetaValidator(origin_graph, meta_graph)

                # Collect all validation errors
                validation_errors = []

                if not validator.validate_node_types():
                    for error in validator.errors:
                        validation_errors.append(
                            {"type": error.error_type, "message": str(error)}
                        )

                if not validator.validate_edge_signatures():
                    for error in validator.errors[len(validation_errors) :]:
                        validation_errors.append(
                            {"type": error.error_type, "message": str(error)}
                        )

                if not validator.validate_properties():
                    for error in validator.errors[len(validation_errors) :]:
                        validation_errors.append(
                            {"type": error.error_type, "message": str(error)}
                        )

                if not validator.validate_cardinalities():
                    for error in validator.errors[len(validation_errors) :]:
                        validation_errors.append(
                            {"type": error.error_type, "message": str(error)}
                        )

                if not validator.validate_has_property_edges():
                    for error in validator.errors[len(validation_errors) :]:
                        validation_errors.append(
                            {"type": error.error_type, "message": str(error)}
                        )

                # If structural validation passed, check Z3 constraints
                if not validation_errors and not validator.validate_with_z3():
                    validation_errors.append(
                        {
                            "type": "z3_unsatisfiable",
                            "message": "Meta graph constraints are unsatisfiable according to Origin axioms",
                        }
                    )

                if validation_errors:
                    return False, validation_errors

                return True, []

            except Exception as e:
                errors.append(
                    {
                        "type": "meta_validation_error",
                        "message": f"Error validating meta graph: {str(e)}",
                    }
                )
                return False, errors

        elif graph_name == "data":
            # Validate data against meta
            if not meta_path.exists():
                errors.append(
                    {
                        "type": "file_not_found",
                        "message": f"Meta file not found at {meta_path}",
                    }
                )
                return False, errors

            if not data_path.exists():
                errors.append(
                    {
                        "type": "file_not_found",
                        "message": f"Data file not found at {data_path}",
                    }
                )
                return False, errors

            try:
                origin_graph = GraphParser.parse_graph_file(str(origin_path))
                meta_graph = GraphParser.parse_graph_file(str(meta_path))
                data_graph = GraphParser.parse_graph_file(str(data_path))

                validator = DataValidator(
                    meta_graph, data_graph, origin_graph=origin_graph
                )

                # Collect all validation errors
                data_validation_errors: List[Dict[str, Any]] = []

                if not validator.validate_node_types():
                    for data_error in validator.errors:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_edge_types():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_edge_connections():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_required_properties():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_property_data_types():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_cardinalities():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if not validator.validate_property_patterns():
                    for data_error in validator.errors[len(data_validation_errors) :]:
                        data_validation_errors.append(
                            {"type": data_error.error_type, "message": str(data_error)}
                        )

                if data_validation_errors:
                    return False, data_validation_errors

                return True, []

            except Exception as e:
                errors.append(
                    {
                        "type": "data_validation_error",
                        "message": f"Error validating data graph: {str(e)}",
                    }
                )
                return False, errors

        else:
            errors.append(
                {
                    "type": "invalid_graph_name",
                    "message": f"Invalid graph name: {graph_name}. Must be 'origin', 'meta', or 'data'",
                }
            )
            return False, errors

    except Exception as e:
        errors.append(
            {
                "type": "unexpected_error",
                "message": f"Unexpected error during validation: {str(e)}",
            }
        )
        return False, errors


def format_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """
    Format validation errors for display.

    Args:
        errors: List of error dictionaries

    Returns:
        Formatted error message string
    """
    if not errors:
        return "No validation errors"

    lines = ["Graph Validation Errors:"]
    for i, error in enumerate(errors, 1):
        lines.append(f"  {i}. [{error['type']}] {error['message']}")

    return "\n".join(lines)
