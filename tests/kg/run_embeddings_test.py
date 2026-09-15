"""Tests for embedding orchestration in run.py and CLI flag handling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from kg.config import KGConfig
from kg.run import _load_existing, _parse_args, main


def _instance(instance_id="astropy__astropy-1"):
    return {
        "instance_id": instance_id,
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "problem_statement": "bug",
        "patch": "",
        "test_patch": "",
        "hints_text": "",
        "created_at": "",
        "version": "",
        "FAIL_TO_PASS": "",
        "PASS_TO_PASS": "",
        "environment_setup_commit": "",
    }


class TestLoadExisting:
    def _write_progress(self, tmp_path, entries):
        progress = tmp_path / "p.progress.jsonl"
        with open(progress, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return progress

    def test_completed_rows_are_kept(self, tmp_path):
        progress = self._write_progress(
            tmp_path,
            [{"instance_id": "a", "text_inputs": "some context"}],
        )
        assert _load_existing(progress) == {"a"}

    def test_empty_text_rows_rerun(self, tmp_path):
        # Rows written when the LLM was unavailable must be re-processed
        progress = self._write_progress(
            tmp_path,
            [
                {"instance_id": "a", "text_inputs": ""},
                {"instance_id": "b", "text_inputs": "ok"},
                {"instance_id": "c"},  # no text_inputs key at all
            ],
        )
        assert _load_existing(progress) == {"b"}

    def test_malformed_lines_ignored(self, tmp_path):
        progress = tmp_path / "p.progress.jsonl"
        progress.write_text("not json\n" + json.dumps({"text_inputs": "x"}) + "\n")
        assert _load_existing(progress) == set()

    def test_missing_file(self, tmp_path):
        assert _load_existing(tmp_path / "nope.jsonl") == set()


class TestParseArgs:
    def test_default_no_embeddings(self):
        args = _parse_args([])
        assert args.config is None
        assert args.embeddings is False

    def test_embeddings_flag(self):
        args = _parse_args(["config/kg_config.json", "--embeddings"])
        assert args.config == "config/kg_config.json"
        assert args.embeddings is True

    def test_config_only(self):
        args = _parse_args(["config/kg_config.json"])
        assert args.config == "config/kg_config.json"
        assert args.embeddings is False

    def test_worker_flags_parse(self):
        args = _parse_args(
            [
                "config/kg_config.json",
                "--embeddings",
                "--only",
                "django__django-1",
                "--skip-convert",
                "--subprocess-per-instance",
            ]
        )
        assert args.only == "django__django-1"
        assert args.subprocess_per_instance is True
        assert args.skip_convert is True


class TestMainEmbeddingOrchestration:
    def _run_main(self, config):
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(return_value=iter([_instance()]))
        dataset.__len__ = MagicMock(return_value=1)

        build_embeddings_mock = MagicMock()
        explore_mock = MagicMock(return_value={})
        context_mock = MagicMock(return_value="text")

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", return_value=MagicMock()):
                with patch("kg.run.build_embeddings", build_embeddings_mock):
                    with patch("kg.run.explore", explore_mock):
                        with patch("kg.run.dispatch_context_mode", context_mock):
                            with patch("kg.run._convert_to_dataset"):
                                main(config)
        return build_embeddings_mock, explore_mock

    def test_build_embeddings_called_when_enabled(self, tmp_path):
        config = KGConfig(
            use_embeddings=True,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )
        build_embeddings_mock, _ = self._run_main(config)
        build_embeddings_mock.assert_called_once()

    def test_build_embeddings_not_called_when_disabled(self, tmp_path):
        config = KGConfig(
            use_embeddings=False,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )
        build_embeddings_mock, _ = self._run_main(config)
        build_embeddings_mock.assert_not_called()

    def test_embedding_failure_does_not_fail_instance(self, tmp_path):
        config = KGConfig(
            use_embeddings=True,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(return_value=iter([_instance()]))
        dataset.__len__ = MagicMock(return_value=1)

        explore_mock = MagicMock(return_value={})

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", return_value=MagicMock()):
                with patch(
                    "kg.run.build_embeddings",
                    side_effect=RuntimeError("embedding server down"),
                ):
                    with patch("kg.run.explore", explore_mock):
                        with patch("kg.run.dispatch_context_mode", MagicMock()):
                            with patch("kg.run._convert_to_dataset"):
                                main(config)

        explore_mock.assert_called_once()

    def test_output_name_distinguishes_embedding_runs(self, tmp_path):
        config = KGConfig(
            use_embeddings=True,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(return_value=iter([_instance()]))
        dataset.__len__ = MagicMock(return_value=1)

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", return_value=MagicMock()):
                with patch("kg.run.build_embeddings", MagicMock()):
                    with patch("kg.run.explore", MagicMock(return_value={})):
                        with patch("kg.run.dispatch_context_mode", MagicMock()):
                            with patch("kg.run._convert_to_dataset"):
                                main(config)

        progress_files = list(tmp_path.glob("*.progress.jsonl"))
        assert progress_files
        assert "__embed" in progress_files[0].name


class TestSubprocessPerInstance:
    def _config(self, tmp_path):
        return KGConfig(
            use_embeddings=True,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )

    def _dataset(self, ids):
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(
            return_value=iter([_instance(instance_id=i) for i in ids])
        )
        dataset.__len__ = MagicMock(return_value=len(ids))
        return dataset

    def test_parent_spawns_one_worker_per_instance(self, tmp_path):
        config = self._config(tmp_path)
        dataset = self._dataset(["a-1", "a-2"])

        run_mock = MagicMock(return_value=MagicMock(returncode=0))
        build_graph_mock = MagicMock()

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", build_graph_mock):
                with patch("kg.run.subprocess.run", run_mock):
                    with patch("kg.run._convert_to_dataset"):
                        main(
                            config,
                            config_path="config/kg_config.json",
                            subprocess_per_instance=True,
                        )

        assert run_mock.call_count == 2
        build_graph_mock.assert_not_called()
        for call, expected_id in zip(run_mock.call_args_list, ["a-1", "a-2"]):
            cmd = call.args[0]
            assert "--only" in cmd
            assert cmd[cmd.index("--only") + 1] == expected_id
            assert "--skip-convert" in cmd
            assert "--embeddings" in cmd
            assert "config/kg_config.json" in cmd

    def test_failed_worker_does_not_abort_the_run(self, tmp_path):
        config = self._config(tmp_path)
        dataset = self._dataset(["a-1", "a-2"])

        run_mock = MagicMock(
            side_effect=[MagicMock(returncode=1), MagicMock(returncode=0)]
        )

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", MagicMock()):
                with patch("kg.run.subprocess.run", run_mock):
                    with patch("kg.run._convert_to_dataset"):
                        main(
                            config,
                            subprocess_per_instance=True,
                        )

        assert run_mock.call_count == 2

    def test_skip_convert_skips_dataset_conversion(self, tmp_path):
        config = self._config(tmp_path)
        dataset = self._dataset(["a-1"])

        convert_mock = MagicMock()
        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", MagicMock()):
                with patch("kg.run.subprocess.run", MagicMock()):
                    with patch("kg.run._convert_to_dataset", convert_mock):
                        main(
                            config,
                            subprocess_per_instance=True,
                            convert=False,
                        )

        convert_mock.assert_not_called()


class TestOnlyMode:
    def _config(self, tmp_path):
        return KGConfig(
            use_embeddings=True,
            output_dir=str(tmp_path),
            graph_cache_dir=str(tmp_path / "graphs"),
            repo_cache_dir=str(tmp_path / "repos"),
        )

    def _dataset(self, ids):
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(
            return_value=iter([_instance(instance_id=i) for i in ids])
        )
        dataset.__len__ = MagicMock(return_value=len(ids))
        return dataset

    def test_only_matching_instance_is_processed(self, tmp_path):
        config = self._config(tmp_path)
        dataset = self._dataset(["a-1", "a-2"])

        build_graph_mock = MagicMock(return_value=str(tmp_path / "g"))

        with patch("kg.run.load_dataset", return_value=dataset):
            with patch("kg.run.build_graph", build_graph_mock):
                with patch("kg.run.build_embeddings", MagicMock()):
                    with patch("kg.run.explore", MagicMock(return_value={})):
                        with patch(
                            "kg.run.dispatch_context_mode",
                            MagicMock(return_value="text"),
                        ):
                            with patch("kg.run.release_graph_client", MagicMock()):
                                with patch("kg.run._convert_to_dataset"):
                                    main(config, only="a-1", convert=False)

        build_graph_mock.assert_called_once()
        progress = list(tmp_path.glob("*.progress.jsonl"))[0]
        written = [json.loads(line)["instance_id"] for line in open(progress)]
        assert written == ["a-1"]

    def test_unknown_only_id_exits_nonzero(self, tmp_path):
        config = self._config(tmp_path)
        dataset = self._dataset(["a-1"])

        with pytest.raises(SystemExit):
            with patch("kg.run.load_dataset", return_value=dataset):
                main(config, only="missing-1", convert=False)
