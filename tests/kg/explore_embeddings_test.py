"""Tests for embedding integration in the exploration loop (mocked server)."""

from unittest.mock import MagicMock, patch

from kg.config import KGConfig
from kg.explore import explore


def _mock_completion(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


def _embedding_config(**overrides) -> KGConfig:
    defaults = {"use_embeddings": True}
    defaults.update(overrides)
    return KGConfig(**defaults)


_DEFAULT_FUNCS = {
    "search_elements": lambda **kw: {
        "success": True,
        "total": 1,
        "items": [{"kind": "node", "id": "File-a.py", "label": "a.py"}],
    },
    "get_neighbors": lambda **kw: {"success": True, "nodes": []},
    "traverse": lambda **kw: {"success": True, "paths": []},
    "get_element": lambda **kw: {"success": False},
    "find_connected": lambda **kw: {"success": False},
    "get_function_info": lambda **kw: {"success": False},
}


def _seeds():
    return [
        {
            "id": "Method-foo.py-Auth-login",
            "label": "login",
            "meta_type": "Method",
            "name": "login",
            "relative_path": "foo.py",
            "similarity": 0.91,
        },
        {
            "id": "Class-foo.py-Auth",
            "label": "Auth",
            "meta_type": "Class",
            "name": "Auth",
            "relative_path": "foo.py",
            "similarity": 0.85,
        },
    ]


def _run_explore_embedded(extract_side_effects, config, funcs=None):
    """Run explore() with mocked client and mocked embedding helpers.

    Returns (touched, user_messages, search_tool_mock).
    """
    if funcs is None:
        funcs = dict(_DEFAULT_FUNCS)

    extract_mock = MagicMock(side_effect=extract_side_effects)
    client = MagicMock()
    user_messages = []

    def capture_messages(**kwargs):
        for m in kwargs.get("messages", []):
            if m["role"] == "user" and m["content"] not in user_messages:
                user_messages.append(m["content"])
        return _mock_completion("ok")

    client.chat.completions.create.side_effect = capture_messages

    search_tool = MagicMock(
        return_value={
            "success": True,
            "total": 1,
            "items": [
                {
                    "kind": "node",
                    "id": "Method-foo.py-Auth-login",
                    "label": "login",
                    "properties": {
                        "meta_type": "Method",
                        "similarity": 0.9,
                    },
                }
            ],
        }
    )

    top_k_mock = MagicMock(return_value=_seeds())

    with patch("kg.explore._import_graph_ops", return_value=(funcs, extract_mock)):
        with patch("openai.OpenAI", return_value=client):
            with patch("os.chdir"):
                with patch("kg.embeddings.get_top_k_for_issue", top_k_mock):
                    with patch(
                        "kg.embeddings.make_search_tool",
                        return_value=search_tool,
                    ):
                        touched = explore("/fake/graph", "bug", config)

    return touched, user_messages, search_tool


class TestExploreWithEmbeddings:
    def test_head_start_injected_into_initial_message(self):
        config = _embedding_config()
        _, user_messages, _ = _run_explore_embedded([[], []], config)
        initial = user_messages[0]
        assert "semantic similarity" in initial
        assert "Method login in foo.py (sim=0.91)" in initial
        assert "Class Auth in foo.py (sim=0.85)" in initial

    def test_search_by_embedding_dispatched_to_tool(self):
        config = _embedding_config()
        _, _, search_tool = _run_explore_embedded(
            [
                [
                    {
                        "function": {
                            "name": "search_by_embedding",
                            "arguments": {"query": "validate credentials"},
                        }
                    }
                ],
                [],
                [],
            ],
            config,
        )
        search_tool.assert_called_once_with(query="validate credentials")
        assert search_tool.return_value["success"] is True

    def test_embedding_failure_degrades_to_classic_message(self):
        config = _embedding_config()
        extract_mock = MagicMock(side_effect=[[], []])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion("ok")

        user_messages = []

        def capture_messages(**kwargs):
            for m in kwargs.get("messages", []):
                if m["role"] == "user" and m["content"] not in user_messages:
                    user_messages.append(m["content"])
            return _mock_completion("ok")

        client.chat.completions.create.side_effect = capture_messages

        funcs = dict(_DEFAULT_FUNCS)

        with patch("kg.explore._import_graph_ops", return_value=(funcs, extract_mock)):
            with patch("openai.OpenAI", return_value=client):
                with patch("os.chdir"):
                    with patch(
                        "kg.embeddings.get_top_k_for_issue",
                        side_effect=RuntimeError("server down"),
                    ):
                        touched = explore("/fake/graph", "bug", config)

        initial = user_messages[0]
        assert initial.startswith("Bug report:\nbug")
        assert "semantic similarity" not in initial
        assert "search_by_embedding" not in initial
        assert touched == {}

    def test_disabled_by_default_unchanged(self):
        config = KGConfig()
        assert config.use_embeddings is False
        extract_mock = MagicMock(side_effect=[[], []])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_completion("ok")

        user_messages = []

        def capture_messages(**kwargs):
            for m in kwargs.get("messages", []):
                if m["role"] == "user" and m["content"] not in user_messages:
                    user_messages.append(m["content"])
            return _mock_completion("ok")

        client.chat.completions.create.side_effect = capture_messages

        funcs = dict(_DEFAULT_FUNCS)

        with patch("kg.explore._import_graph_ops", return_value=(funcs, extract_mock)):
            with patch("openai.OpenAI", return_value=client):
                with patch("os.chdir"):
                    with patch("kg.embeddings.get_top_k_for_issue") as top_k_mock:
                        explore("/fake/graph", "bug", config)

        top_k_mock.assert_not_called()
        assert user_messages[0].startswith("Bug report:\nbug")
