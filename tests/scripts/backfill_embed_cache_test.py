"""Tests for backfill-embed-cache.py (per-graph indexes -> shared cache)."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "backfill-embed-cache.py"
_spec = importlib.util.spec_from_file_location("bec", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from kg.config import KGConfig  # noqa: E402
from kg.embeddings import build_embeddings  # noqa: E402


def _embedding_config(**overrides) -> KGConfig:
    defaults = {"use_embeddings": True, "embedding_dim": 4}
    defaults.update(overrides)
    return KGConfig(**defaults)


def _counting_embed(sent):
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


BETA_V1 = "def beta():\n    return 1\n"


def _graph_data() -> dict:
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
                "source_code": BETA_V1,
            },
        },
    }


class TestBackfillEmbedCache:
    def test_backfills_and_enables_reuse(self, tmp_path, capsys):
        pytest.importorskip("chromadb")
        repo = tmp_path / "repos" / "acme__lib"
        repo.mkdir(parents=True)
        _git("init", "-q", cwd=repo)
        (repo / "foo.py").write_text("def alpha():\n    return 1\n")
        (repo / "bar.py").write_text(BETA_V1)
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "c1", cwd=repo)
        commit1 = _git("rev-parse", "HEAD", cwd=repo)

        graphs = tmp_path / "graphs"
        g1 = graphs / f"acme__lib__{commit1}"
        g1.mkdir(parents=True)
        (g1 / "data.json").write_text(json.dumps(_graph_data()))
        # A second graph dir without any index -> must be reported skipped
        empty_graph = graphs / ("acme__lib__" + "1" * 40)
        empty_graph.mkdir()
        (empty_graph / "data.json").write_text(
            '{"nodes": [], "edges": [], "allValues": {}}'
        )
        g3 = tmp_path / "elsewhere" / f"acme__lib__{commit1}"
        g3.mkdir(parents=True)
        (g3 / "data.json").write_text(json.dumps(_graph_data()))

        # Build g1 with the cache DISABLED (pre-cache legacy index)
        no_cache = _embedding_config(repo_cache_dir=str(tmp_path / "repos"))
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed([])):
            build_embeddings(str(g1), no_cache)

        config_path = tmp_path / "kg_config.json"
        config_path.write_text(json.dumps({"embedding_dim": 4}))
        cache_dir = tmp_path / "cache"
        rc = _mod.main(
            [
                "--config",
                str(config_path),
                "--graphs-dir",
                str(graphs),
                "--repos-dir",
                str(tmp_path / "repos"),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "skipped" in out

        # After backfill, an identical graph builds with zero embed calls
        cached_config = _embedding_config(
            repo_cache_dir=str(tmp_path / "repos"),
            embedding_cache_dir=str(cache_dir),
        )
        sent = []
        with patch("kg.embeddings._embed_texts", side_effect=_counting_embed(sent)):
            build_embeddings(str(g3), cached_config)
        assert sent == []
