"""Tests for source code extraction utilities."""

from code_kg_builder.executors.python.utils.source_extract import (
    extract_lines,
    extract_source_for_node,
)

SAMPLE = """\
import os


class Foo:
    '''Docstring.'''
    x = 1

    def bar(self):
        if True:
            return 1
        return 0
"""


def test_extract_lines_full():
    result = extract_lines(SAMPLE, 1, 16)
    assert result == SAMPLE.rstrip("\n")


def test_extract_lines_partial():
    result = extract_lines(SAMPLE, 4, 6)
    assert "class Foo" in result
    assert "x = 1" in result


def test_extract_lines_single():
    result = extract_lines(SAMPLE, 1, 1)
    assert result == "import os"


def test_extract_lines_out_of_range():
    """Lines beyond content return what exists."""
    result = extract_lines(SAMPLE, 1, 100)
    assert "return 0" in result


def test_extract_source_for_class():
    import ast

    tree = ast.parse(SAMPLE)
    cls_node = tree.body[1]  # ClassDef
    result = extract_source_for_node(SAMPLE, cls_node)
    assert "class Foo" in result
    assert "return 0" in result


def test_extract_source_for_method():
    import ast

    tree = ast.parse(SAMPLE)
    cls_node = tree.body[1]  # ClassDef
    method_node = cls_node.body[2]  # FunctionDef bar
    result = extract_source_for_node(SAMPLE, method_node)
    assert "def bar" in result
    assert "return 0" in result
    assert "class Foo" not in result
