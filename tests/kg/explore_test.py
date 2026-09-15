"""Tests for the exploration loop with mocked OpenAI client."""

from unittest.mock import MagicMock, patch

from kg.config import KGConfig
from kg.explore import (
    _extract_node_ids,
    _collect_node,
    _format_results,
    _parse_tool_calls,
    explore,
)


def test_extract_node_ids_from_search():
    result = {
        "success": True,
        "total": 2,
        "items": [
            {"kind": "node", "id": "File-foo.py", "label": "foo.py"},
            {"kind": "node", "id": "Class-foo.py-Bar", "label": "Bar"},
        ],
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 2
    assert nodes[0]["id"] == "File-foo.py"


def test_extract_node_ids_from_neighbors():
    result = {
        "success": True,
        "center_id": "Class-foo.py-Bar",
        "edges": [],
        "nodes": [
            {"id": "Method-foo.py-Bar-auth", "label": "auth"},
        ],
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 1
    assert nodes[0]["id"] == "Method-foo.py-Bar-auth"


def test_extract_node_ids_from_traverse():
    result = {
        "success": True,
        "paths": [
            {
                "nodes": [
                    {"id": "A", "label": "A"},
                    {"id": "B", "label": "B"},
                ],
                "edges": [{"id": "e1", "label": "CALLS"}],
            }
        ],
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 2


def test_extract_node_ids_from_get_element():
    result = {
        "success": True,
        "kind": "node",
        "id": "Function-foo.py-bar",
        "label": "bar",
        "properties": {"source_code": "def bar(): pass"},
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 1
    assert nodes[0]["properties"]["source_code"] == "def bar(): pass"


def test_extract_node_ids_failure():
    result = {"success": False, "error": "not found"}
    nodes = _extract_node_ids(result)
    assert len(nodes) == 0


def test_collect_node_deduplicates():
    touched = {}
    _collect_node(
        [
            {"id": "File-foo.py", "label": "foo.py"},
            {"id": "File-foo.py", "label": "foo.py"},  # duplicate
            {
                "id": "Class-foo.py-Bar",
                "label": "Bar",
                "properties": {"line_start": 10},
            },
        ],
        touched,
    )
    assert len(touched) == 2
    assert touched["File-foo.py"]["label"] == "foo.py"
    assert touched["Class-foo.py-Bar"]["line_start"] == 10


def test_format_results():
    results = [
        (
            "search_elements",
            {"graph": "data", "label_contains": "foo"},
            {"success": True, "total": 5, "items": []},
        ),
        (
            "get_neighbors",
            {"graph": "data", "id": "X"},
            {"success": False, "error": "ELEMENT_NOT_FOUND"},
        ),
    ]
    text = _format_results(results)
    assert "SUCCESS (search_elements)" in text
    assert "5 items" in text
    assert "ERROR (get_neighbors)" in text
    assert "ELEMENT_NOT_FOUND" in text


def test_format_results_with_source_code():
    results = [
        (
            "get_element",
            {"graph": "data", "id": "Function-a.py-foo"},
            {
                "success": True,
                "kind": "node",
                "id": "Function-a.py-foo",
                "label": "foo",
                "properties": {"source_code": "def foo(): return 42"},
            },
        ),
    ]
    text = _format_results(results)
    assert "source_code" in text
    assert "def foo" in text


def test_collect_node_merges_properties():
    """Properties from later visits should update earlier entries."""
    touched = {}
    _collect_node([{"id": "X", "label": "X"}], touched)
    _collect_node(
        [{"id": "X", "properties": {"source_code": "def x(): pass"}}], touched
    )
    assert touched["X"]["label"] == "X"
    assert touched["X"]["source_code"] == "def x(): pass"


def test_extract_node_ids_filters_edges():
    """search_elements items with kind='edge' should be filtered out."""
    result = {
        "success": True,
        "total": 2,
        "items": [
            {"kind": "node", "id": "File-foo.py", "label": "foo.py"},
            {
                "kind": "edge",
                "id": "e1",
                "label": "CALLS",
                "source": "A",
                "target": "B",
            },
        ],
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 1
    assert nodes[0]["id"] == "File-foo.py"


# ---- _parse_tool_calls tests ----

_FUNCS = {
    "search_elements": lambda **kw: None,
    "get_neighbors": lambda **kw: None,
    "traverse": lambda **kw: None,
    "get_element": lambda **kw: None,
    "find_connected": lambda **kw: None,
    "get_function_info": lambda **kw: None,
}


def test_parse_valid_calls():
    calls = [
        {"function": {"name": "search_elements", "arguments": {"graph": "data"}}},
        {
            "function": {
                "name": "get_element",
                "arguments": {"graph": "data", "id": "X"},
            }
        },
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 2
    assert graph_calls[0][0] == "search_elements"
    assert len(issues) == 0


def test_parse_string_entry():
    """String instead of dict → issue with repr of the offending value."""
    calls = [
        "search_elements(data='foo')",
        {"function": {"name": "get_element", "arguments": {"graph": "data"}}},
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 1
    assert len(issues) == 1
    assert "Expected a JSON object" in issues[0]
    assert "search_elements" in issues[0]


def test_parse_unknown_function():
    calls = [
        {"function": {"name": "nonexistent_func", "arguments": {}}},
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 0
    assert len(issues) == 1
    assert "Unknown function 'nonexistent_func'" in issues[0]
    assert "search_elements" in issues[0]
    assert "find_connected" in issues[0]


def test_parse_missing_name():
    calls = [
        {"function": {"arguments": {"graph": "data"}}},
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 0
    assert len(issues) == 1
    assert "Missing 'name'" in issues[0]


def test_parse_function_not_dict():
    calls = [
        {"function": "search_elements"},
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 0
    assert len(issues) == 1
    assert "'function' must be an object" in issues[0]


def test_parse_empty_calls():
    graph_calls, issues = _parse_tool_calls([], _FUNCS)
    assert len(graph_calls) == 0
    assert len(issues) == 0


def test_parse_mixed_valid_and_malformed():
    calls = [
        {"function": {"name": "search_elements", "arguments": {"graph": "data"}}},
        "bad_string_call",
        {"function": {"name": "get_element", "arguments": {"graph": "data"}}},
        {"function": {"name": "unknown", "arguments": {}}},
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 2
    assert len(issues) == 2
    assert any("Expected a JSON object" in s for s in issues)
    assert any("Unknown function" in s for s in issues)


def test_extract_node_ids_from_find_connected():
    """find_connected returns {"node": {...}} (singular)."""
    result = {
        "success": True,
        "node": {
            "id": "File-src/app.py",
            "label": "app.py",
            "properties": {"meta_type": "File", "relative_path": "src/app.py"},
        },
    }
    nodes = _extract_node_ids(result)
    assert len(nodes) == 1
    assert nodes[0]["id"] == "File-src/app.py"
    assert nodes[0]["properties"]["relative_path"] == "src/app.py"


def test_format_results_find_connected():
    """_format_results should handle find_connected's 'node' key."""
    results = [
        (
            "find_connected",
            {"graph": "data", "id": "Method-x.py-A-b", "type": "File"},
            {
                "success": True,
                "node": {
                    "id": "File-x.py",
                    "label": "x.py",
                    "properties": {"relative_path": "x.py", "line_count": 10},
                },
            },
        ),
    ]
    text = _format_results(results)
    assert "SUCCESS (find_connected)" in text
    assert "File-x.py" in text
    assert "relative_path" in text
    assert "x.py" in text


def test_format_results_get_function_info():
    """_format_results should render function schema from get_function_info."""
    results = [
        (
            "get_function_info",
            {"function_name": "search_elements"},
            {
                "success": True,
                "info": {
                    "type": "function",
                    "function": {
                        "name": "search_elements",
                        "description": "Search graph elements.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "graph": {
                                    "type": "string",
                                    "enum": ["origin", "meta", "data"],
                                    "description": "Graph level.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search text.",
                                },
                            },
                            "required": ["graph"],
                        },
                    },
                },
            },
        ),
    ]
    text = _format_results(results)
    assert "SUCCESS (get_function_info)" in text
    assert "search_elements" in text
    assert "graph" in text
    assert "(required)" in text
    assert "Graph level" in text


def test_parse_get_function_info_call():
    """get_function_info should be a recognized function name."""
    calls = [
        {
            "function": {
                "name": "get_function_info",
                "arguments": {"function_name": "traverse"},
            }
        },
    ]
    graph_calls, issues = _parse_tool_calls(calls, _FUNCS)
    assert len(graph_calls) == 1
    assert graph_calls[0][0] == "get_function_info"
    assert len(issues) == 0


# ---- explore() integration tests with mocked client ----


def _mock_completion(text: str):
    """Create a mock OpenAI chat completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


_SEARCH_RESULT = {
    "success": True,
    "total": 1,
    "items": [{"kind": "node", "id": "File-a.py", "label": "a.py"}],
}
_NEIGHBORS_RESULT = {
    "success": True,
    "nodes": [{"id": "Function-a.py-foo", "label": "foo"}],
}

_DEFAULT_FUNCS = {
    "search_elements": lambda **kw: _SEARCH_RESULT,
    "get_neighbors": lambda **kw: {"success": True, "nodes": []},
    "traverse": lambda **kw: {"success": True, "paths": []},
    "get_element": lambda **kw: {"success": False},
    "find_connected": lambda **kw: {"success": False},
    "get_function_info": lambda **kw: {"success": False},
}


def _run_explore(extract_side_effects, funcs=None, max_rounds=8):
    """Run explore() with mocked dependencies.

    Args:
        extract_side_effects: list of return values for extract_function_calls,
                              one per round.
        funcs: override graph funcs (defaults to _DEFAULT_FUNCS).

    Returns:
        (touched, user_messages, extract_mock)
    """
    config = KGConfig()
    config.max_rounds = max_rounds
    config.max_tool_calls = 20

    if funcs is None:
        funcs = _DEFAULT_FUNCS

    extract_mock = MagicMock(side_effect=extract_side_effects)
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_completion("ok")

    user_messages = []

    def capture_messages(**kwargs):
        for m in kwargs.get("messages", []):
            if m["role"] == "user" and m["content"] not in user_messages:
                user_messages.append(m["content"])
        return _mock_completion("ok")

    client.chat.completions.create.side_effect = capture_messages

    with patch("kg.explore._import_graph_ops", return_value=(funcs, extract_mock)):
        with patch("openai.OpenAI", return_value=client):
            with patch("os.chdir"):
                touched = explore("/fake", "bug", config)

    return touched, user_messages, extract_mock


class TestNudge:
    def test_nudge_when_no_calls_no_touches(self):
        """Model makes no calls → nudged → then searches → then finishes."""
        touched, _, extract = _run_explore(
            [
                [],  # R1: no calls, no touched → nudge
                [
                    {
                        "function": {
                            "name": "search_elements",
                            "arguments": {"graph": "data"},
                        }
                    }
                ],  # R2: search
                [],  # R3: no calls, touched → review
                [],  # R4: no calls, reviewed → break
            ]
        )

        assert len(touched) == 1
        assert "File-a.py" in touched
        assert extract.call_count == 4

    def test_nudge_only_once(self):
        """If model never explores after nudge, exploration ends."""
        touched, _, extract = _run_explore(
            [
                [],  # R1: no calls → nudge
                [],  # R2: no calls, nudged already → break
            ]
        )

        assert len(touched) == 0
        assert extract.call_count == 2


class TestCriticalReview:
    def test_review_when_no_calls_but_touched(self):
        """Model explores then stops → gets asked to critically review."""
        touched, _, extract = _run_explore(
            [
                [
                    {
                        "function": {
                            "name": "search_elements",
                            "arguments": {"graph": "data"},
                        }
                    }
                ],  # R1: search
                [],  # R2: no calls, touched → review
                [],  # R3: no calls, reviewed → break
            ]
        )

        assert len(touched) == 1
        assert extract.call_count == 3

    def test_review_only_once(self):
        """Review prompt should appear exactly once in user messages."""
        _, messages, _ = _run_explore(
            [
                [
                    {
                        "function": {
                            "name": "search_elements",
                            "arguments": {"graph": "data"},
                        }
                    }
                ],  # R1: search
                [],  # R2: no calls → review
                [],  # R3: no calls, reviewed → break
            ]
        )

        review_count = sum(1 for msg in messages if "critically review" in msg.lower())
        assert review_count == 1

    def test_review_can_trigger_more_exploration(self):
        """After review prompt, model resumes making tool calls."""
        funcs = dict(_DEFAULT_FUNCS)
        funcs["get_neighbors"] = lambda **kw: _NEIGHBORS_RESULT

        touched, _, extract = _run_explore(
            [
                [
                    {
                        "function": {
                            "name": "search_elements",
                            "arguments": {"graph": "data"},
                        }
                    }
                ],  # R1: search
                [],  # R2: no calls → review
                [
                    {
                        "function": {
                            "name": "get_neighbors",
                            "arguments": {"graph": "data", "id": "File-a.py"},
                        }
                    }
                ],  # R3: more
                [],  # R4: reviewed already → break
            ],
            funcs=funcs,
        )

        assert len(touched) == 2
        assert "File-a.py" in touched
        assert "Function-a.py-foo" in touched
