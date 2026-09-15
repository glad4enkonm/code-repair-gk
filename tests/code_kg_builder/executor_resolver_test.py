"""Tests for ExecutorResolver — resolves 'module:Class' paths via importlib."""

import pytest

from code_kg_builder.executor_resolver import (
    ExecutorResolverError,
    resolve_executor,
    resolve_executors,
)


class TestResolveExecutor:
    def test_resolves_real_class(self):
        from code_kg_builder.executors.base import BatchResult

        instance = resolve_executor("code_kg_builder.executors.base:BatchResult")
        assert isinstance(instance, BatchResult)

    def test_bad_module_raises(self):
        with pytest.raises(ExecutorResolverError, match="Cannot import"):
            resolve_executor("nonexistent.module:SomeClass")

    def test_bad_class_raises(self):
        with pytest.raises(ExecutorResolverError, match="NoSuchClass"):
            resolve_executor("code_kg_builder.executor_resolver:NoSuchClass")

    def test_missing_colon_raises(self):
        with pytest.raises(ExecutorResolverError, match="must contain"):
            resolve_executor("code_kg_builder.executor_resolver")

    def test_empty_string_raises(self):
        with pytest.raises(ExecutorResolverError):
            resolve_executor("")


class TestResolveExecutors:
    def test_resolves_multiple_unique(self):
        paths_to_types = {
            "code_kg_builder.executors.base:BatchResult": ["Class", "Function"],
        }
        instances = resolve_executors(paths_to_types)
        assert len(instances) == 1

    def test_reports_broken_path(self):
        paths_to_types = {
            "code_kg_builder.executors.base:BatchResult": ["Class"],
            "nonexistent:Broken": ["Function"],
        }
        with pytest.raises(ExecutorResolverError, match="Function"):
            resolve_executors(paths_to_types)
