"""Tests for precompute-embeddings.py (GPU-phase node embedding backfill)."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "precompute-embeddings.py"
_spec = importlib.util.spec_from_file_location("pe", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from kg.config import KGConfig  # noqa: E402


def _config(tmp_path, **overrides):
    defaults = {
        "split": "test",
        "use_embeddings": True,
        "output_dir": str(tmp_path),
        "graph_cache_dir": str(tmp_path / "graphs"),
        "repo_cache_dir": str(tmp_path / "repos"),
    }
    defaults.update(overrides)
    return KGConfig(**defaults)


def _instance(instance_id):
    return {
        "instance_id": instance_id,
        "repo": "astropy/astropy",
        "base_commit": instance_id,  # distinct commit => distinct graph dir
        "problem_statement": "bug",
    }


def _dataset(ids):
    dataset = MagicMock()
    dataset.__iter__ = MagicMock(return_value=iter([_instance(i) for i in ids]))
    dataset.__len__ = MagicMock(return_value=len(ids))
    return dataset


class TestParseArgs:
    def test_config_and_only(self):
        args = _mod._parse_args(["config/kg_config.json", "--only", "django__django-1"])
        assert args.config == "config/kg_config.json"
        assert args.only == "django__django-1"
        assert args.subprocess_per_instance is False

    def test_subprocess_flag(self):
        args = _mod._parse_args(["config/kg_config.json", "--subprocess-per-instance"])
        assert args.subprocess_per_instance is True


class TestIndexComplete:
    def _collection(self, count, model="nomic-embed-code-Q8_0", stamp=None):
        collection = MagicMock()
        collection.count.return_value = count
        collection.metadata = {
            "embedding_model": model,
            "node_count": stamp if stamp is not None else count,
        }
        return collection

    def test_complete_index(self, tmp_path):
        config = _config(tmp_path)
        graph_dir = tmp_path / "graphs" / "astropy__astropy__abc"
        with patch.object(
            _mod, "_get_collection", return_value=self._collection(10)
        ) as get:
            with patch.object(_mod, "release_graph_client") as release:
                assert _mod.index_complete(graph_dir, config) is True
        get.assert_called_once_with(str(graph_dir), config, create=False)
        release.assert_called_once_with(str(graph_dir))

    def test_empty_index(self, tmp_path):
        config = _config(tmp_path)
        with patch.object(_mod, "_get_collection", return_value=self._collection(0)):
            with patch.object(_mod, "release_graph_client"):
                assert _mod.index_complete(tmp_path / "g", config) is False

    def test_missing_collection(self, tmp_path):
        config = _config(tmp_path)
        with patch.object(_mod, "_get_collection", return_value=None):
            with patch.object(_mod, "release_graph_client"):
                assert _mod.index_complete(tmp_path / "g", config) is False

    def test_stale_stamp_is_incomplete(self, tmp_path):
        config = _config(tmp_path)
        with patch.object(
            _mod, "_get_collection", return_value=self._collection(10, stamp=9)
        ):
            with patch.object(_mod, "release_graph_client"):
                assert _mod.index_complete(tmp_path / "g", config) is False

    def test_foreign_model_is_incomplete(self, tmp_path):
        config = _config(tmp_path)
        with patch.object(
            _mod,
            "_get_collection",
            return_value=self._collection(10, model="other-model"),
        ):
            with patch.object(_mod, "release_graph_client"):
                assert _mod.index_complete(tmp_path / "g", config) is False


class TestMain:
    def test_parent_skips_complete_and_spawns_for_rest(self, tmp_path):
        config = _config(tmp_path)
        dataset = _dataset(["a-1", "a-2", "a-3"])

        complete = {"a-1": True, "a-2": False, "a-3": True}

        def fake_index_complete(graph_dir, config):
            instance_id = str(graph_dir).split("__")[-1]
            return complete[instance_id]

        spawn = MagicMock(return_value=True)

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with patch.object(_mod, "index_complete", fake_index_complete):
                with patch.object(_mod, "_spawn_worker", spawn):
                    _mod.main(
                        config,
                        config_path="config/kg_config.json",
                        subprocess_per_instance=True,
                    )

        assert spawn.call_count == 1
        assert spawn.call_args.args[1] == "a-2"

    def test_prescan_logs_periodic_progress(self, tmp_path, capsys):
        # Long prescans must not run silently: a progress line every
        # _PRESCAN_LOG_EVERY checked instances
        config = _config(tmp_path)
        dataset = _dataset([f"a-{i}" for i in range(60)])

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with patch.object(_mod, "index_complete", MagicMock(return_value=True)):
                with patch.object(_mod, "_spawn_worker") as spawn:
                    _mod.main(config, subprocess_per_instance=True)

        out = capsys.readouterr().out
        assert f"checked {_mod._PRESCAN_LOG_EVERY}/60 indexes" in out
        assert "checked 60/60 indexes" in out
        spawn.assert_not_called()

    def test_worker_failure_does_not_abort(self, tmp_path):
        config = _config(tmp_path)
        dataset = _dataset(["a-1", "a-2"])

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with patch.object(_mod, "index_complete", MagicMock(return_value=False)):
                with patch.object(
                    _mod, "_spawn_worker", MagicMock(return_value=False)
                ) as spawn:
                    _mod.main(config, subprocess_per_instance=True)

        assert spawn.call_count == 2


class TestWorkerMode:
    def test_only_instance_is_processed(self, tmp_path):
        config = _config(tmp_path)
        dataset = _dataset(["a-1", "a-2"])

        build_graph = MagicMock(return_value=str(tmp_path / "g"))
        build_embeddings = MagicMock()
        release = MagicMock()

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with patch.object(_mod, "build_graph", build_graph):
                with patch.object(_mod, "build_embeddings", build_embeddings):
                    with patch.object(_mod, "release_graph_client", release):
                        with patch.object(
                            _mod, "index_complete", MagicMock(return_value=False)
                        ):
                            _mod.main(config, only="a-1")

        build_graph.assert_called_once()
        build_embeddings.assert_called_once_with(str(tmp_path / "g"), config)
        release.assert_called_once_with(str(tmp_path / "g"))

    def test_complete_instance_skipped_even_in_worker_mode(self, tmp_path):
        config = _config(tmp_path)
        dataset = _dataset(["a-1"])

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with patch.object(_mod, "index_complete", MagicMock(return_value=True)):
                with patch.object(_mod, "build_graph") as build_graph:
                    _mod.main(config, only="a-1")

        build_graph.assert_not_called()

    def test_unknown_only_id_exits_nonzero(self, tmp_path):
        config = _config(tmp_path)
        dataset = _dataset(["a-1"])

        with patch.object(_mod, "load_dataset", return_value=dataset):
            with pytest.raises(SystemExit):
                _mod.main(config, only="missing-1")


class TestSpawnWorker:
    def test_command_shape(self):
        with patch.object(_mod, "subprocess") as subprocess_mod:
            subprocess_mod.run.return_value = MagicMock(returncode=0)
            ok = _mod._spawn_worker("config/kg_config.json", "a-1")

        assert ok is True
        command = subprocess_mod.run.call_args.args[0]
        assert command[-2:] == ["--only", "a-1"]
        assert "config/kg_config.json" in command

    def test_worker_failure_reported(self):
        with patch.object(_mod, "subprocess") as subprocess_mod:
            subprocess_mod.run.return_value = MagicMock(returncode=3)
            ok = _mod._spawn_worker(None, "a-1")

        assert ok is False
        command = subprocess_mod.run.call_args.args[0]
        assert "config/kg_config.json" not in command
