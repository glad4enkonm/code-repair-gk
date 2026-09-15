"""Tests for CallsEdgeExecutor."""

from pathlib import Path

import pytest

from code_kg_builder.context import BuildContext, NodeInfo
from code_kg_builder.executors.python.calls_edge import CallsEdgeExecutor

SAMPLE = """\
def helper():
    return 1

def caller():
    return helper()

class Service:
    def process(self):
        return helper()

    def internal(self):
        return self.process()
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "svc.py").write_text(SAMPLE)
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> BuildContext:
    c = BuildContext(repo_path=repo)
    c.python_files = [repo / "svc.py"]
    c.node_index["Function-svc.py-helper"] = NodeInfo(
        id="Function-svc.py-helper",
        label="helper",
        meta_type="Function",
        properties={"name": "helper"},
    )
    c.node_index["Function-svc.py-caller"] = NodeInfo(
        id="Function-svc.py-caller",
        label="caller",
        meta_type="Function",
        properties={"name": "caller"},
    )
    c.node_index["Method-svc.py-Service-process"] = NodeInfo(
        id="Method-svc.py-Service-process",
        label="process",
        meta_type="Method",
        properties={"name": "process"},
    )
    c.node_index["Method-svc.py-Service-internal"] = NodeInfo(
        id="Method-svc.py-Service-internal",
        label="internal",
        meta_type="Method",
        properties={"name": "internal"},
    )
    return c


class TestCallsEdgeExecutor:
    def test_resolves_bare_function_call(self, ctx):
        batches = list(CallsEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        calls = [e for e in edges if e["label"] == "CALLS"]
        targets = {(e["source"], e["target"]) for e in calls}
        # caller() calls helper()
        assert ("Function-svc.py-caller", "Function-svc.py-helper") in targets

    def test_resolves_self_method_call(self, ctx):
        batches = list(CallsEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        calls = [e for e in edges if e["label"] == "CALLS"]
        targets = {(e["source"], e["target"]) for e in calls}
        # internal() calls self.process()
        assert (
            "Method-svc.py-Service-internal",
            "Method-svc.py-Service-process",
        ) in targets

    def test_no_self_call(self, ctx):
        batches = list(CallsEdgeExecutor().run(ctx))
        edges = []
        for b in batches:
            edges.extend(b.edges_data)
        calls = [e for e in edges if e["label"] == "CALLS"]
        # No self-referencing edges
        for e in calls:
            assert e["source"] != e["target"]
