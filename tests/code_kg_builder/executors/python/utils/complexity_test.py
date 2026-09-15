"""Tests for McCabe cyclomatic complexity."""

import ast

from code_kg_builder.executors.python.utils.complexity import cyclomatic_complexity


def parse_func(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return tree.body[0]  # type: ignore[return-value]


def test_empty_function():
    node = parse_func("def f():\n    pass\n")
    assert cyclomatic_complexity(node) == 1


def test_single_if():
    node = parse_func("def f():\n    if True:\n        pass\n")
    assert cyclomatic_complexity(node) == 2


def test_if_elif_else():
    node = parse_func(
        "def f():\n"
        "    if a:\n        pass\n"
        "    elif b:\n        pass\n"
        "    else:\n        pass\n"
    )
    assert cyclomatic_complexity(node) == 3


def test_for_loop():
    node = parse_func("def f():\n    for i in range(10):\n        pass\n")
    assert cyclomatic_complexity(node) == 2


def test_while_loop():
    node = parse_func("def f():\n    while True:\n        pass\n")
    assert cyclomatic_complexity(node) == 2


def test_except_handler():
    node = parse_func(
        "def f():\n    try:\n        pass\n    except Exception:\n        pass\n"
    )
    assert cyclomatic_complexity(node) == 2


def test_boolean_and():
    node = parse_func("def f():\n    return a and b\n")
    assert cyclomatic_complexity(node) == 2


def test_boolean_or():
    node = parse_func("def f():\n    return a or b\n")
    assert cyclomatic_complexity(node) == 2


def test_with_statement():
    node = parse_func("def f():\n    with open('f') as fh:\n        pass\n")
    assert cyclomatic_complexity(node) == 2


def test_assert_statement():
    node = parse_func("def f():\n    assert True\n")
    assert cyclomatic_complexity(node) == 2


def test_list_comprehension():
    node = parse_func("def f():\n    return [x for x in range(10)]\n")
    assert cyclomatic_complexity(node) == 2


def test_combined():
    node = parse_func(
        "def f():\n"
        "    for i in range(10):\n"
        "        if i > 5 and i < 8:\n"
        "            try:\n"
        "                pass\n"
        "            except ValueError:\n"
        "                pass\n"
    )
    # 1 (base) + 1 (for) + 1 (if) + 1 (and) + 1 (except) = 5
    assert cyclomatic_complexity(node) == 5
