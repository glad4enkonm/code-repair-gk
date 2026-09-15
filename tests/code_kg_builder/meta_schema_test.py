"""Tests for MetaSchema — parses meta.json into structured representation."""

from pathlib import Path

import pytest

from code_kg_builder.meta_schema import MetaSchema

META_PATH = Path(__file__).resolve().parents[2] / "swe_bench_kg" / "meta.json"


@pytest.fixture
def schema() -> MetaSchema:
    return MetaSchema.from_file(str(META_PATH))


class TestParsing:
    def test_loads_without_error(self, schema):
        assert schema is not None

    def test_node_types(self, schema):
        assert "Directory" in schema.node_types
        assert "Class" in schema.node_types
        assert "Function" in schema.node_types
        assert len(schema.node_types) == 11

    def test_edge_types(self, schema):
        assert "CONTAINS" in schema.edge_types
        assert "CALLS" in schema.edge_types
        assert len(schema.edge_types) == 9

    def test_property_types(self, schema):
        assert "name" in schema.property_types
        assert "source_code" in schema.property_types
        assert "relative_path" in schema.property_types


class TestEdgeSignatures:
    def test_contains_multiple_signatures(self, schema):
        sigs = schema.edge_signatures["CONTAINS"]
        pairs = {(s.source_type, s.target_type) for s in sigs}
        assert ("Directory", "File") in pairs
        assert ("File", "Class") in pairs
        assert ("Class", "Method") in pairs
        assert ("Method", "Variable") in pairs
        assert ("File", "Function") in pairs
        assert ("File", "Import") in pairs
        assert ("File", "Variable") in pairs
        assert ("Class", "Variable") in pairs

    def test_calls_signature(self, schema):
        sigs = schema.edge_signatures["CALLS"]
        pairs = {(s.source_type, s.target_type) for s in sigs}
        assert ("Function", "Function") in pairs
        assert ("Method", "Method") in pairs

    def test_inherits_signature(self, schema):
        sigs = schema.edge_signatures["INHERITS"]
        assert len(sigs) == 1
        assert sigs[0].source_type == "Class"
        assert sigs[0].target_type == "Class"


class TestTypeProperties:
    def test_class_has_name(self, schema):
        assert "name" in schema.type_properties["Class"]

    def test_function_has_source_code(self, schema):
        assert "source_code" in schema.type_properties["Function"]

    def test_method_has_is_async(self, schema):
        assert "is_async" in schema.type_properties["Method"]

    def test_directory_has_relative_path(self, schema):
        assert "relative_path" in schema.type_properties["Directory"]


class TestPropertyConstraints:
    def test_class_name_mandatory(self, schema):
        constraints = schema.property_constraints.get(("Class", "name"), {})
        assert constraints.get("isMandatory") is True

    def test_class_line_start_optional(self, schema):
        constraints = schema.property_constraints.get(("Class", "line_start"), {})
        assert constraints.get("isMandatory") is False


class TestExecutors:
    def test_executor_paths_collected_from_mock(self):
        data = {
            "nodes": [
                {"id": "MyNode", "label": "MyNode"},
                {"id": "MyEdge", "label": "MyEdge"},
            ],
            "edges": [],
            "allValues": {
                "MyNode": {
                    "meta_type": "NodeType",
                    "executor": "pkg.mod:NodeExec",
                },
                "MyEdge": {
                    "meta_type": "EdgeType",
                    "executor": "pkg.mod:EdgeExec",
                },
            },
        }
        schema = MetaSchema.from_data(data)
        executors = schema.get_executors()
        assert "pkg.mod:NodeExec" in executors
        assert "MyNode" in executors["pkg.mod:NodeExec"]
        assert "pkg.mod:EdgeExec" in executors
        assert "MyEdge" in executors["pkg.mod:EdgeExec"]

    def test_executor_deduplication(self):
        data = {
            "nodes": [
                {"id": "TypeA", "label": "A"},
                {"id": "TypeB", "label": "B"},
            ],
            "edges": [],
            "allValues": {
                "TypeA": {
                    "meta_type": "NodeType",
                    "executor": "pkg.mod:Shared",
                },
                "TypeB": {
                    "meta_type": "NodeType",
                    "executor": "pkg.mod:Shared",
                },
            },
        }
        schema = MetaSchema.from_data(data)
        executors = schema.get_executors()
        assert len(executors) == 1
        assert set(executors["pkg.mod:Shared"]) == {"TypeA", "TypeB"}

    def test_executors_in_real_meta(self, schema):
        """Real meta.json now has executor properties."""
        executors = schema.get_executors()
        assert len(executors) > 0


class TestFromData:
    def test_from_dict(self):
        data = {
            "nodes": [
                {"id": "MyNode", "label": "MyNode"},
            ],
            "edges": [],
            "allValues": {
                "MyNode": {"meta_type": "NodeType", "description": "test"},
            },
        }
        schema = MetaSchema.from_data(data)
        assert "MyNode" in schema.node_types
