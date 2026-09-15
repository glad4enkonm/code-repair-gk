"""Tests for build_graphs — deduplication and parallel worker assignment."""

from unittest.mock import patch

from kg.build_graphs import build_all
from kg.build_graphs import _unique_graphs
from kg.config import KGConfig


def _make_instance(repo: str, commit: str, iid: str) -> dict:
    return {"repo": repo, "base_commit": commit, "instance_id": iid}


class TestDeduplication:
    def test_unique_graphs_deduplicates(self):
        """Two instances sharing (repo, base_commit) produce one entry."""
        dataset = [
            _make_instance("django/django", "abc123", "django-1"),
            _make_instance("django/django", "abc123", "django-2"),
            _make_instance("sympy/sympy", "def456", "sympy-1"),
        ]
        result = _unique_graphs(dataset)
        assert len(result) == 2
        assert ("django/django", "abc123", "django-1") in result
        assert ("sympy/sympy", "def456", "sympy-1") in result

    def test_unique_graphs_sorted(self):
        """Output should be sorted by (repo, base_commit) for stable assignment."""
        dataset = [
            _make_instance("z/z", "c9", "z-1"),
            _make_instance("a/a", "c1", "a-1"),
            _make_instance("m/m", "c5", "m-1"),
        ]
        result = _unique_graphs(dataset)
        repos = [r for r, _, _ in result]
        assert repos == ["a/a", "m/m", "z/z"]

    def test_unique_graphs_keeps_first_instance_id(self):
        """First instance_id wins for display when duplicates exist."""
        dataset = [
            _make_instance("a/a", "c1", "first"),
            _make_instance("a/a", "c1", "second"),
        ]
        result = _unique_graphs(dataset)
        assert result == [("a/a", "c1", "first")]

    def test_duplicate_pairs_built_once(self):
        """Two instances sharing (repo, base_commit) build only one graph."""
        dataset = [
            _make_instance("django/django", "abc123", "django-1"),
            _make_instance("django/django", "abc123", "django-2"),
            _make_instance("sympy/sympy", "def456", "sympy-1"),
        ]
        config = KGConfig()

        built = []

        def fake_build_graph(repo, commit, cfg):
            built.append((repo, commit))
            return f"/fake/{repo}__{commit}"

        with patch("kg.build_graphs.build_graph", side_effect=fake_build_graph):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config)

        assert len(built) == 2
        assert ("django/django", "abc123") in built
        assert ("sympy/sympy", "def456") in built

    def test_all_unique_pairs_built(self):
        """All unique pairs should be built exactly once."""
        dataset = [
            _make_instance("a/a", "c1", "a-1"),
            _make_instance("b/b", "c2", "b-1"),
            _make_instance("c/c", "c3", "c-1"),
        ]
        config = KGConfig()

        built = []

        with patch(
            "kg.build_graphs.build_graph",
            side_effect=lambda r, c, cfg: built.append((r, c)),
        ):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config)

        assert len(built) == 3


class TestWorkerAssignment:
    def test_single_worker_gets_all(self):
        """worker_id=0, num_workers=1 should process all pairs."""
        dataset = [_make_instance("a/a", f"c{i}", f"a-{i}") for i in range(5)]
        config = KGConfig()

        built = []

        with patch(
            "kg.build_graphs.build_graph",
            side_effect=lambda r, c, cfg: built.append((r, c)),
        ):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config, worker_id=0, num_workers=1)

        assert len(built) == 5

    def test_workers_get_disjoint_subsets(self):
        """Two workers should process disjoint halves."""
        dataset = [_make_instance("a/a", f"c{i}", f"a-{i}") for i in range(6)]
        config = KGConfig()

        w0_built = []
        w1_built = []

        def make_recorder(built_list):
            def recorder(r, c, cfg):
                built_list.append((r, c))

            return recorder

        with patch("kg.build_graphs.build_graph", side_effect=make_recorder(w0_built)):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config, worker_id=0, num_workers=2)

        with patch("kg.build_graphs.build_graph", side_effect=make_recorder(w1_built)):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(
                    config=KGConfig(), dataset=dataset, worker_id=1, num_workers=2
                )

        assert len(w0_built) == 3
        assert len(w1_built) == 3
        w0_set = set(w0_built)
        w1_set = set(w1_built)
        assert w0_set.isdisjoint(w1_set)
        assert len(w0_set | w1_set) == 6

    def test_no_worker_builds_duplicates(self):
        """A worker should never build the same (repo, commit) twice."""
        dataset = [
            _make_instance("a/a", "c1", "a-1"),
            _make_instance("a/a", "c1", "a-2"),
            _make_instance("a/a", "c2", "a-3"),
            _make_instance("a/a", "c1", "a-4"),
        ]
        config = KGConfig()

        built = []

        with patch(
            "kg.build_graphs.build_graph",
            side_effect=lambda r, c, cfg: built.append((r, c)),
        ):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config, worker_id=0, num_workers=1)

        assert built.count(("a/a", "c1")) == 1
        assert built.count(("a/a", "c2")) == 1


class TestWorkerRepoCache:
    def test_separate_repo_cache_when_parallel(self):
        """num_workers > 1 should set per-worker repo_cache_dir."""
        dataset = [_make_instance("a/a", "c1", "a-1")]
        config = KGConfig()
        original_cache = config.repo_cache_dir

        with patch("kg.build_graphs.build_graph"):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config, worker_id=2, num_workers=4)

        assert config.repo_cache_dir == f"{original_cache}_w2"

    def test_shared_repo_cache_when_single(self):
        """num_workers=1 should not modify repo_cache_dir."""
        dataset = [_make_instance("a/a", "c1", "a-1")]
        config = KGConfig()
        original_cache = config.repo_cache_dir

        with patch("kg.build_graphs.build_graph"):
            with patch("kg.build_graphs.ensure_repo_cloned"):
                build_all(dataset, config, worker_id=0, num_workers=1)

        assert config.repo_cache_dir == original_cache
