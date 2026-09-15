"""Tests for the cross-graph embedding cache (git blob-SHA keyed).

The embedding server is mocked at the ``_embed_texts`` seam (same as
embeddings_test.py); Chroma runs for real against tmp_path; the git
side uses real temporary repositories on disk.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kg.config import KGConfig
from kg.embeddings import (
    _blob_map,
    _cache_lookup,
    _cache_store,
    _get_collection,
    _fingerprint,
    _node_cache_key,
    build_embeddings,
    release_graph_client,
)

# Chroma runs for real against tmp_path and the git side builds real
# temporary repositories on disk — the whole module is slow-marked.
pytestmark = pytest.mark.slow


def _embedding_config(**overrides) -> KGConfig:
    defaults = {"use_embeddings": True, "embedding_dim": 4}
    defaults.update(overrides)
    return KGConfig(**defaults)


def _counting_embed(sent):
    """fake _embed_texts that records every document sent to the server."""

    def embed(texts, config):
        sent.extend(texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    return embed


def _git(*args, cwd):
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_with_history(path: Path):
    """Two commits: foo.py stable, bar.py changes; link.py is a symlink."""
    path.mkdir(parents=True)
    _git("init", "-q", cwd=path)
    (path / "foo.py").write_text("def alpha():\n    return 1\n")
    (path / "bar.py").write_text("def beta():\n    return 1\n")
    (path / "link.py").symlink_to("foo.py")
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", "c1", cwd=path)
    commit1 = _git("rev-parse", "HEAD", cwd=path)
    (path / "bar.py").write_text("def beta():\n    return 2\n")
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", "c2", cwd=path)
    commit2 = _git("rev-parse", "HEAD", cwd=path)
    return commit1, commit2


BETA_V1 = "def beta():\n    return 1\n"
BETA_V2 = "def beta():\n    return 2\n"


def _graph_data(beta_source: str) -> dict:
    return {
        "nodes": [
            {"id": "File-foo.py", "label": "foo.py"},
            {"id": "Function-foo.py-alpha", "label": "alpha"},
            {"id": "File-bar.py", "label": "bar.py"},
            {"id": "Function-bar.py-beta", "label": "beta"},
        ],
        "edges": [
            {
                "id": "e1",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Function-foo.py-alpha",
            },
            {
                "id": "e2",
                "label": "CONTAINS",
                "source": "File-bar.py",
                "target": "Function-bar.py-beta",
            },
        ],
        "allValues": {
            "File-foo.py": {"meta_type": "File", "relative_path": "foo.py"},
            "Function-foo.py-alpha": {
                "meta_type": "Function",
                "name": "alpha",
                "source_code": "def alpha():\n    return 1\n",
            },
            "File-bar.py": {"meta_type": "File", "relative_path": "bar.py"},
            "Function-bar.py-beta": {
                "meta_type": "Function",
                "name": "beta",
                "source_code": beta_source,
            },
        },
    }


def _write_graph(graph_dir: Path, beta_source: str) -> str:
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "data.json").write_text(json.dumps(_graph_data(beta_source)))
    return str(graph_dir)


def _cache_env(tmp_path):
    """Repo with history + config + graph dirs for commit1 and commit2.

    Graph dirs live under different parents so their *names* both parse
    as ``acme__lib__<commit>`` while remaining distinct directories.
    """
    commit1, commit2 = _repo_with_history(tmp_path / "repos" / "acme__lib")
    config = _embedding_config(
        repo_cache_dir=str(tmp_path / "repos"),
        embedding_cache_dir=str(tmp_path / "cache"),
    )
    graphs = tmp_path / "graphs"
    g1 = _write_graph(graphs / "x" / f"acme__lib__{commit1}", BETA_V1)
    g2 = _write_graph(graphs / "y" / f"acme__lib__{commit2}", BETA_V2)
    return config, g1, g2, commit1, commit2


class TestBlobMap:
    def test_parses_regular_blobs_and_skips_symlinks(self, tmp_path):
        commit1, _ = _repo_with_history(tmp_path / "repos" / "acme__lib")
        config = _embedding_config(repo_cache_dir=str(tmp_path / "repos"))
        blobs = _blob_map("acme__lib", commit1, config)
        assert set(blobs) == {"foo.py", "bar.py"}
        assert all(len(sha) == 40 for sha in blobs.values())

    def test_unknown_commit_returns_empty(self, tmp_path):
        _repo_with_history(tmp_path / "repos" / "acme__lib")
        config = _embedding_config(repo_cache_dir=str(tmp_path / "repos"))
        assert _blob_map("acme__lib", "0" * 40, config) == {}

    def test_missing_repo_dir_returns_empty(self, tmp_path):
        config = _embedding_config(repo_cache_dir=str(tmp_path / "repos"))
        assert _blob_map("nope__nope", "1" * 40, config) == {}


class TestCacheKeys:
    def test_blob_and_doc_keys_distinct_and_stable(self):
        config = _embedding_config()
        blob1 = _node_cache_key(config, "Node-1", "doc text", "a" * 40)
        blob2 = _node_cache_key(config, "Node-1", "doc text", "a" * 40)
        doc1 = _node_cache_key(config, "Node-1", "doc text")
        assert blob1 == blob2
        assert blob1 != doc1

    def test_fingerprint_changes_on_config_change(self):
        base = _fingerprint(_embedding_config())
        changed = _fingerprint(_embedding_config(embedding_max_source_chars=99))
        assert base != changed


class TestCacheStore:
    def test_duplicate_keys_deduped(self):
        # chroma's upsert rejects duplicate ids in one call; graphs can
        # contain distinct nodes with identical composed text
        class RecordingCollection:
            def __init__(self):
                self.calls = []

            def upsert(self, ids=None, embeddings=None, metadatas=None):
                self.calls.append((ids, embeddings, metadatas))

        collection = RecordingCollection()
        keys = ["dupe", "a", "dupe", "b", "dupe"]
        vectors = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        assert _cache_store(collection, keys, vectors, "repo") is True
        assert len(collection.calls) == 1
        ids, embeddings, metadatas = collection.calls[0]
        assert ids == ["dupe", "a", "b"]
        assert embeddings == [[0.0], [1.0], [3.0]]
        assert len(metadatas) == 3


class TestEmbeddingCache:
    def test_same_commit_second_graph_embeds_nothing(self, tmp_path):
        pytest.importorskip("chromadb")
        config, g1, _, commit1, _ = _cache_env(tmp_path)
        g3 = _write_graph(tmp_path / "graphs" / "z" / f"acme__lib__{commit1}", BETA_V1)
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(g1, config)
            first = len(sent)
            build_embeddings(g3, config)
        assert first == 4
        assert sent[first:] == []

    def test_changed_file_reembeds_only_its_nodes(self, tmp_path):
        pytest.importorskip("chromadb")
        config, g1, g2, _, _ = _cache_env(tmp_path)
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(g1, config)
            del sent[:]
            build_embeddings(g2, config)
        assert len(sent) == 2
        assert all("bar.py" in text for text in sent)

    def test_unparseable_identity_falls_back_to_doc_keys(self, tmp_path):
        pytest.importorskip("chromadb")
        config, _, _, _, _ = _cache_env(tmp_path)
        plain1 = _write_graph(tmp_path / "graphs" / "p1", BETA_V1)
        plain2 = _write_graph(tmp_path / "graphs" / "p2", BETA_V1)
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(plain1, config)
            del sent[:]
            build_embeddings(plain2, config)
        assert sent == []

    def test_model_mismatch_recreates_cache(self, tmp_path):
        pytest.importorskip("chromadb")
        config, g1, _, commit1, _ = _cache_env(tmp_path)
        g3 = _write_graph(tmp_path / "graphs" / "z" / f"acme__lib__{commit1}", BETA_V1)
        other_model = _embedding_config(
            repo_cache_dir=config.repo_cache_dir,
            embedding_cache_dir=config.embedding_cache_dir,
            embedding_model="other-model",
        )
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(g1, config)
            del sent[:]
            build_embeddings(g3, other_model)
        assert len(sent) == 4

    def test_fingerprint_change_invalidates(self, tmp_path):
        pytest.importorskip("chromadb")
        config, g1, _, commit1, _ = _cache_env(tmp_path)
        g3 = _write_graph(tmp_path / "graphs" / "z" / f"acme__lib__{commit1}", BETA_V1)
        changed_config = _embedding_config(
            repo_cache_dir=config.repo_cache_dir,
            embedding_cache_dir=config.embedding_cache_dir,
            embedding_max_source_chars=99,
        )
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(g1, config)
            del sent[:]
            build_embeddings(g3, changed_config)
        assert len(sent) == 4

    def test_cache_disabled_by_default(self, tmp_path):
        pytest.importorskip("chromadb")
        commit1, _ = _repo_with_history(tmp_path / "repos" / "acme__lib")
        config = _embedding_config(repo_cache_dir=str(tmp_path / "repos"))
        g1 = _write_graph(tmp_path / "graphs" / "x" / f"acme__lib__{commit1}", BETA_V1)
        g3 = _write_graph(tmp_path / "graphs" / "z" / f"acme__lib__{commit1}", BETA_V1)
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(g1, config)
            first = len(sent)
            build_embeddings(g3, config)
        assert first == 4
        assert len(sent) == 8

    def test_wrong_dim_entry_treated_as_miss(self):
        # Direct unit test of the dim guard: a corrupted/legacy entry
        # must be dropped, never served into a per-graph index
        class StubCollection:
            def get(self, ids=None, include=None):
                return {
                    "ids": list(ids),
                    "embeddings": [[1.0, 0.0] for _ in ids],
                }

        config = _embedding_config()
        found = _cache_lookup(StubCollection(), ["k1", "k2"], config)
        assert found == {}

    def test_lookup_dedupes_duplicate_keys(self):
        # chroma's get() rejects duplicate ids in one call, and doc-key
        # fallback yields duplicates when distinct nodes compose
        # identical documents
        received = []

        class DedupeSpyCollection:
            def get(self, ids=None, include=None):
                received.extend(ids)
                return {
                    "ids": list(ids),
                    "embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in ids],
                }

        config = _embedding_config()
        found = _cache_lookup(DedupeSpyCollection(), ["k", "dup", "dup", "k2"], config)
        assert received == ["k", "dup", "k2"]
        assert set(found) == {"k", "dup", "k2"}

    def test_reused_count_logged(self, tmp_path, capsys):
        pytest.importorskip("chromadb")
        config, g1, _, commit1, _ = _cache_env(tmp_path)
        g3 = _write_graph(tmp_path / "graphs" / "z" / f"acme__lib__{commit1}", BETA_V1)
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed([])):
            build_embeddings(g1, config)
            capsys.readouterr()
            build_embeddings(g3, config)
            captured = capsys.readouterr()
        assert "reused" in captured.out


class TestGraphClientRelease:
    def test_release_allows_reopen_and_is_idempotent(self, tmp_path):
        # chromadb keeps one open system (sqlite + segment files) per
        # path for the process lifetime; releasing must close it so long
        # runs do not exhaust the file-descriptor limit, while later
        # clients transparently re-open the on-disk data.
        pytest.importorskip("chromadb")
        config, g1, _, _, _ = _cache_env(tmp_path)
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed([])):
            build_embeddings(g1, config)
        release_graph_client(g1)
        release_graph_client(g1)
        collection = _get_collection(g1, config, create=False)
        assert collection is not None
        assert collection.count() == 4
