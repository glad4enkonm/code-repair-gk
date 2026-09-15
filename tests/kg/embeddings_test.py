"""Tests for kg.embeddings — text composition, path resolution, Chroma index.

The embedding server (llama-server /v1/embeddings) is mocked at the
``_embed_texts`` seam; Chroma runs for real against tmp_path.
"""

import json
from unittest.mock import patch

import pytest

from kg.config import KGConfig
from kg.embeddings import (
    COLLECTION_NAME,
    _document_batches,
    build_embeddings,
    compose_node_text,
    embed_query,
    get_top_k_for_issue,
    make_search_tool,
    resolve_relative_paths,
    search_by_embedding,
)


def _embedding_config(**overrides) -> KGConfig:
    defaults = {"use_embeddings": True, "embedding_dim": 4}
    defaults.update(overrides)
    return KGConfig(**defaults)


AUTH_VECTOR = [1.0, 0.0, 0.0, 0.0]
DB_VECTOR = [0.0, 1.0, 0.0, 0.0]
MISC_VECTOR = [0.0, 0.0, 1.0, 0.0]


def fake_embed_texts(texts, config):
    """Deterministic embeddings keyed by substring, for similarity control."""
    vectors = []
    for text in texts:
        lower = text.lower()
        if "auth" in lower:
            vectors.append(AUTH_VECTOR)
        elif "db" in lower or "database" in lower:
            vectors.append(DB_VECTOR)
        else:
            vectors.append(MISC_VECTOR)
    return vectors


def _sample_graph() -> dict:
    return {
        "nodes": [
            {"id": "Directory-", "label": "."},
            {"id": "File-foo.py", "label": "foo.py"},
            {"id": "Class-foo.py-Auth", "label": "Auth"},
            {"id": "Method-foo.py-Auth-login", "label": "login"},
            {"id": "Function-foo.py-query_db", "label": "query_db"},
            {"id": "Import-foo.py-json", "label": "json"},
            {"id": "Exception-foo.py-ValueError", "label": "ValueError"},
        ],
        "edges": [
            {
                "id": "e1",
                "label": "CONTAINS",
                "source": "Directory-",
                "target": "File-foo.py",
            },
            {
                "id": "e2",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Class-foo.py-Auth",
            },
            {
                "id": "e3",
                "label": "CONTAINS",
                "source": "Class-foo.py-Auth",
                "target": "Method-foo.py-Auth-login",
            },
            {
                "id": "e4",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Function-foo.py-query_db",
            },
            {
                "id": "e5",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Import-foo.py-json",
            },
            {
                "id": "e6",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Exception-foo.py-ValueError",
            },
        ],
        "allValues": {
            "Directory-": {"meta_type": "Directory", "relative_path": "."},
            "File-foo.py": {
                "meta_type": "File",
                "relative_path": "foo.py",
                "line_count": 10,
            },
            "Class-foo.py-Auth": {
                "meta_type": "Class",
                "name": "Auth",
                "docstring": "auth helpers",
                "source_code": "class Auth:\n    pass",
                "line_start": 1,
                "line_end": 2,
            },
            "Method-foo.py-Auth-login": {
                "meta_type": "Method",
                "name": "login",
                "docstring": "auth login",
                "source_code": "def login(self):\n    pass",
                "line_start": 2,
                "line_end": 3,
            },
            "Function-foo.py-query_db": {
                "meta_type": "Function",
                "name": "query_db",
                "source_code": "def query_db():\n    pass",
                "line_start": 5,
                "line_end": 6,
            },
            "Import-foo.py-json": {
                "meta_type": "Import",
                "name": "json",
                "module_path": "json",
                "imported_name": "json",
                "line_number": 1,
            },
            "Exception-foo.py-ValueError": {
                "meta_type": "Exception",
                "name": "ValueError",
            },
        },
    }


def _write_graph(graph_dir) -> str:
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "data.json").write_text(json.dumps(_sample_graph()))
    return str(graph_dir)


class TestComposeNodeText:
    def test_method_with_docstring(self):
        config = _embedding_config()
        node = {"id": "Method-foo.py-Auth-login", "label": "login"}
        props = {
            "meta_type": "Method",
            "name": "login",
            "docstring": "auth login",
            "source_code": "def login(self):\n    pass",
        }
        text = compose_node_text(node, props, "foo.py", config)
        assert text.startswith("Method login in foo.py")
        assert "auth login" in text
        assert "def login(self):" in text

    def test_function_without_docstring(self):
        config = _embedding_config()
        node = {"id": "Function-foo.py-query_db", "label": "query_db"}
        props = {
            "meta_type": "Function",
            "name": "query_db",
            "source_code": "def query_db():\n    pass",
        }
        text = compose_node_text(node, props, "foo.py", config)
        assert text.startswith("Function query_db in foo.py")
        assert "def query_db():" in text

    def test_source_truncated(self):
        config = _embedding_config(embedding_max_source_chars=10)
        node = {"id": "Function-a.py-f", "label": "f"}
        source = "def f():\n    " + "x" * 100
        props = {"meta_type": "Function", "name": "f", "source_code": source}
        text = compose_node_text(node, props, "a.py", config)
        assert source[:10] in text
        assert source not in text

    def test_file_uses_relative_path(self):
        config = _embedding_config()
        node = {"id": "File-foo.py", "label": "foo.py"}
        props = {"meta_type": "File", "relative_path": "foo.py"}
        assert compose_node_text(node, props, "foo.py", config) == "File foo.py"

    def test_variable(self):
        config = _embedding_config()
        node = {"id": "Variable-foo.py-x", "label": "x"}
        props = {"meta_type": "Variable", "name": "x", "value": "1"}
        assert (
            compose_node_text(node, props, "foo.py", config)
            == "Variable x = 1 in foo.py"
        )

    def test_import(self):
        config = _embedding_config()
        node = {"id": "Import-foo.py-json", "label": "json"}
        props = {
            "meta_type": "Import",
            "module_path": "json",
            "imported_name": "json",
        }
        text = compose_node_text(node, props, "foo.py", config)
        assert text == "Import json from json in foo.py"

    def test_decorator(self):
        config = _embedding_config()
        node = {"id": "Decorator-foo.py-cached", "label": "cached"}
        props = {"meta_type": "Decorator", "name": "cached"}
        assert (
            compose_node_text(node, props, "foo.py", config)
            == "Decorator cached in foo.py"
        )

    def test_directory(self):
        config = _embedding_config()
        node = {"id": "Directory-", "label": "."}
        props = {"meta_type": "Directory", "relative_path": "src"}
        assert compose_node_text(node, props, "", config) == "Directory src"

    def test_unknown_type_generic_fallback(self):
        config = _embedding_config()
        node = {"id": "Weird-x", "label": "x"}
        props = {"meta_type": "Weird", "name": "x"}
        assert compose_node_text(node, props, "a.py", config) == "Weird x in a.py"


class TestResolveRelativePaths:
    def test_maps_nodes_to_files(self, tmp_path):
        graph_dir = _write_graph(tmp_path / "g")
        paths = resolve_relative_paths(graph_dir)
        assert paths["Method-foo.py-Auth-login"] == "foo.py"
        assert paths["Class-foo.py-Auth"] == "foo.py"
        assert paths["Function-foo.py-query_db"] == "foo.py"
        assert paths["Import-foo.py-json"] == "foo.py"
        assert paths["File-foo.py"] == "foo.py"

    def test_handles_cycle_without_hanging(self, tmp_path):
        graph_dir = tmp_path / "cyclic"
        graph_dir.mkdir()
        data = {
            "nodes": [
                {"id": "File-a.py", "label": "a.py"},
                {"id": "Class-a.py-A", "label": "A"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "label": "CONTAINS",
                    "source": "File-a.py",
                    "target": "Class-a.py-A",
                },
                {
                    "id": "e2",
                    "label": "CONTAINS",
                    "source": "Class-a.py-A",
                    "target": "File-a.py",
                },
            ],
            "allValues": {
                "File-a.py": {"meta_type": "File", "relative_path": "a.py"},
                "Class-a.py-A": {"meta_type": "Class", "name": "A"},
            },
        }
        (graph_dir / "data.json").write_text(json.dumps(data))
        paths = resolve_relative_paths(str(graph_dir))
        assert paths["Class-a.py-A"] == "a.py"


class TestEmbedQuery:
    def test_applies_query_prefix(self):
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts") as embed_mock:
            embed_mock.return_value = [[0.0, 0.0, 0.0, 1.0]]
            vector = embed_query("hello", config)
        assert vector == [0.0, 0.0, 0.0, 1.0]
        embed_mock.assert_called_once_with(["search_code_query: hello"], config)


class TestTokenGuards:
    def test_embed_query_truncates_to_doc_char_limit(self):
        config = _embedding_config(embedding_max_doc_chars=100)
        with patch("kg.embeddings._embed_texts") as embed_mock:
            embed_mock.return_value = [[0.0, 0.0, 0.0, 0.0]]
            embed_query("x" * 500, config)
        sent_text = embed_mock.call_args[0][0][0]
        assert sent_text.startswith("search_code_query: ")
        assert len(sent_text) == len("search_code_query: ") + 100

    def test_document_batches_respects_token_budget(self):
        # ~20, 5 and 20 estimated tokens with a 30-token budget
        config = _embedding_config(embedding_batch_size=10, embedding_ctx_tokens=30)
        documents = ["a" * 60, "b" * 15, "c" * 60]
        batches = list(_document_batches(documents, config))
        assert batches == [["a" * 60, "b" * 15], ["c" * 60]]

    def test_document_batches_respects_count_limit(self):
        config = _embedding_config(embedding_batch_size=2, embedding_ctx_tokens=100000)
        documents = ["a", "b", "c"]
        batches = list(_document_batches(documents, config))
        assert batches == [["a", "b"], ["c"]]

    def test_build_truncates_giant_node_text(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = tmp_path / "giant"
        graph_dir.mkdir()
        giant_value = "x" * 500000
        data = {
            "nodes": [
                {"id": "File-big.py", "label": "big.py"},
                {"id": "Variable-big.py-data", "label": "data"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "label": "CONTAINS",
                    "source": "File-big.py",
                    "target": "Variable-big.py-data",
                }
            ],
            "allValues": {
                "File-big.py": {"meta_type": "File", "relative_path": "big.py"},
                "Variable-big.py-data": {
                    "meta_type": "Variable",
                    "name": "data",
                    "value": giant_value,
                },
            },
        }
        (graph_dir / "data.json").write_text(json.dumps(data))
        config = _embedding_config(embedding_max_doc_chars=200)
        with patch(
            "kg.embeddings._embed_texts", side_effect=fake_embed_texts
        ) as embed_mock:
            build_embeddings(str(graph_dir), config)
        sent_texts = [args[0][0] for args, _ in embed_mock.call_args_list]
        assert sent_texts, "expected at least one embed call"
        document_prefix_len = len("search_code_document: ")
        for text in sent_texts:
            assert len(text) <= document_prefix_len + 200


class TestBuildAndSearch:
    def test_build_skips_structural_nodes(self, tmp_path):
        chromadb = pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
        collection = chromadb.PersistentClient(
            path=str(tmp_path / "g" / "chroma_db")
        ).get_collection(COLLECTION_NAME)
        assert collection.count() == 5
        assert "Exception-foo.py-ValueError" not in collection.get()["ids"]

    def test_build_applies_doc_prefix(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts) as m:
            build_embeddings(graph_dir, config)
        embedded_texts = [args[0][0] for args, _ in m.call_args_list]
        assert embedded_texts, "expected at least one embed call"
        for text in embedded_texts:
            assert text.startswith("search_code_document: ")

    def test_search_returns_nearest_with_properties(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
            result = search_by_embedding("database connection", graph_dir, config)
        assert result["success"] is True
        assert result["total"] >= 1
        top = result["items"][0]
        assert top["id"] == "Function-foo.py-query_db"
        assert top["kind"] == "node"
        assert top["properties"]["meta_type"] == "Function"
        assert top["properties"]["similarity"] >= 0.99

    def test_search_with_meta_type_filter(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
            result = search_by_embedding(
                "user authentication", graph_dir, config, meta_type="Method"
            )
        assert result["success"] is True
        assert result["items"]
        for item in result["items"]:
            assert item["properties"]["meta_type"] == "Method"

    def test_build_is_cached(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts) as m:
            build_embeddings(graph_dir, config)
            build_embeddings(graph_dir, config)
        assert m.call_count == 1

    def test_model_mismatch_triggers_rebuild(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts) as m:
            build_embeddings(graph_dir, _embedding_config())
            build_embeddings(
                graph_dir, _embedding_config(embedding_model="other-model")
            )
        assert m.call_count == 2

    def test_search_without_index_fails_gracefully(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = search_by_embedding("q", str(empty_dir), _embedding_config())
        assert result["success"] is False
        assert "error" in result

    def test_make_search_tool_dispatches(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        tool = make_search_tool(graph_dir, config)
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
            result = tool(query="database connection", graph="data")
        assert result["success"] is True
        assert result["items"][0]["id"] == "Function-foo.py-query_db"

    def test_get_top_k_for_issue_sorted(self, tmp_path):
        pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
            seeds = get_top_k_for_issue(
                "user authentication fails", graph_dir, config, k=3
            )
        assert len(seeds) == 3
        similarities = [s["similarity"] for s in seeds]
        assert similarities == sorted(similarities, reverse=True)
        assert seeds[0]["id"] in {"Class-foo.py-Auth", "Method-foo.py-Auth-login"}
        assert seeds[0]["relative_path"] == "foo.py"

    def test_build_stamps_node_count_and_keeps_model(self, tmp_path):
        chromadb = pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
        collection = chromadb.PersistentClient(
            path=str(tmp_path / "g" / "chroma_db")
        ).get_collection(COLLECTION_NAME)
        metadata = collection.metadata or {}
        assert metadata.get("node_count") == collection.count()
        assert metadata.get("embedding_model") == config.embedding_model

    def test_partial_index_rebuilds_on_next_run(self, tmp_path):
        chromadb = pytest.importorskip("chromadb")
        graph_dir = _write_graph(tmp_path / "g")
        config = _embedding_config()
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts):
            build_embeddings(graph_dir, config)
        # Simulate an interrupted build: drop the completion stamp while
        # keeping the rows and the matching model name
        client = chromadb.PersistentClient(path=str(tmp_path / "g" / "chroma_db"))
        client.get_collection(COLLECTION_NAME).modify(
            metadata={"embedding_model": config.embedding_model}
        )
        with patch("kg.embeddings._embed_texts", side_effect=fake_embed_texts) as m:
            build_embeddings(graph_dir, config)
        assert m.call_count == 1

    def test_get_top_k_for_issue_raises_on_missing_index(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError):
            get_top_k_for_issue("q", str(empty_dir), _embedding_config(), k=3)
