"""Tests for prompts.py graph schema extraction and prompt building."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from kg.prompts import extract_schema


def _write_meta_graph(path: str, nodes, edges, all_values):
    os.makedirs(path, exist_ok=True)
    meta = {"nodes": nodes, "edges": edges, "allValues": all_values}
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_extract_schema_node_types_and_properties():
    """extract_schema discovers node types, their properties, and edge types."""
    tmpdir = Path(tempfile.mkdtemp(prefix="schema_"))
    try:
        _write_meta_graph(
            str(tmpdir),
            nodes=[
                {"id": "Class", "label": "Class"},
                {"id": "File", "label": "File"},
                {"id": "Method", "label": "Method"},
                {"id": "CONTAINS", "label": "CONTAINS"},
                {"id": "CALLS", "label": "CALLS"},
                {"id": "name", "label": "name"},
                {"id": "source_code", "label": "source_code"},
                {"id": "relative_path", "label": "relative_path"},
            ],
            edges=[
                {
                    "id": "e1",
                    "label": "HAS_PROPERTY",
                    "source": "Class",
                    "target": "name",
                },
                {
                    "id": "e2",
                    "label": "HAS_PROPERTY",
                    "source": "Class",
                    "target": "source_code",
                },
                {
                    "id": "e3",
                    "label": "HAS_PROPERTY",
                    "source": "File",
                    "target": "relative_path",
                },
                {
                    "id": "e4",
                    "label": "HAS_PROPERTY",
                    "source": "Method",
                    "target": "name",
                },
                {
                    "id": "e5",
                    "label": "HAS_PROPERTY",
                    "source": "Method",
                    "target": "source_code",
                },
            ],
            all_values={
                "Class": {"meta_type": "NodeType"},
                "File": {"meta_type": "NodeType"},
                "Method": {"meta_type": "NodeType"},
                "CONTAINS": {},
                "CALLS": {},
                "name": {"meta_type": "PropertyType", "dataType": "string"},
                "source_code": {"meta_type": "PropertyType", "dataType": "string"},
                "relative_path": {"meta_type": "PropertyType", "dataType": "string"},
            },
        )

        saved_cwd = os.getcwd()
        os.chdir(str(tmpdir))
        try:
            text = extract_schema()
        finally:
            os.chdir(saved_cwd)

        assert "Class" in text
        assert "File" in text
        assert "Method" in text
        assert "name" in text
        assert "source_code" in text
        assert "relative_path" in text
        assert "CONTAINS" in text
        assert "CALLS" in text
        # Node types should appear with their properties
        assert "Class: " in text
        assert "File: " in text
        assert "Method: " in text
        # Edge types section
        assert "Edge types" in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_extract_schema_empty_graph():
    """No meta.json → returns empty string."""
    tmpdir = Path(tempfile.mkdtemp(prefix="schema_empty_"))
    try:
        saved_cwd = os.getcwd()
        os.chdir(str(tmpdir))
        try:
            assert extract_schema() == ""
        finally:
            os.chdir(saved_cwd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
