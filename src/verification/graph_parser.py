"""
Graph Parser Module

Parses JSON graph representations for Origin and Meta graphs.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a graph node."""

    id: str
    label: str
    properties: Dict[str, Any]
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class Edge:
    """Represents a graph edge."""

    id: str
    source: str
    target: str
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = {}


@dataclass
class Graph:
    """Represents a complete graph."""

    nodes: Dict[str, Node]
    edges: List[Edge]
    all_values: Dict[str, Dict[str, Any]]


class GraphParser:
    """Parser for JSON graph representations."""

    @staticmethod
    def parse_graph_file(filepath: str) -> Graph:
        """Parse a JSON graph file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Graph object with parsed data
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return GraphParser.parse_graph_data(data)

    @staticmethod
    def parse_graph_data(data: Dict[str, Any]) -> Graph:
        """Parse graph data from a dictionary.

        Args:
            data: Dictionary containing graph data

        Returns:
            Graph object with parsed data
        """
        # Parse nodes
        nodes = {}
        for node_data in data.get("nodes", []):
            node = Node(
                id=node_data["id"],
                label=node_data["label"],
                x=node_data.get("x"),
                y=node_data.get("y"),
                properties={},
            )
            nodes[node.id] = node

        # Parse edges
        edges = []
        for edge_data in data.get("edges", []):
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                label=edge_data.get("label"),
                properties={},
            )
            edges.append(edge)

        # Parse allValues
        all_values = data.get("allValues", {})

        # Distribute properties to nodes and edges
        for key, values in all_values.items():
            if key in nodes:
                nodes[key].properties = values
            else:
                # Check if it's an edge property
                for edge in edges:
                    if edge.id == key:
                        edge.properties = values
                        break

        return Graph(nodes=nodes, edges=edges, all_values=all_values)

    @staticmethod
    def get_node_type(node: Node, graph: Graph) -> Optional[str]:
        """Get the type of a node from its properties.

        Args:
            node: Node to check
            graph: Graph containing the node

        Returns:
            Node type string or None
        """
        # Check for origin_type or meta_type in properties
        return node.properties.get("origin_type") or node.properties.get("meta_type")

    @staticmethod
    def get_edge_signature(edge_id: str) -> Optional[Tuple[str, str, str]]:
        """Parse edge signature from ID.

        Edge IDs in origin follow pattern: SOURCE-RELATION-TARGET

        Args:
            edge_id: Edge identifier

        Returns:
            Tuple of (source, relation, target) or None
        """
        parts = edge_id.split("-")
        if len(parts) >= 3:
            # Handle cases like HAS_PROPERTY-HAS_PROPERTY-minCount
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
            elif len(parts) == 4 and parts[1] == "HAS_PROPERTY":
                # NodeType-HAS_PROPERTY-description format
                return parts[0], "HAS_PROPERTY", parts[2]
        return None
