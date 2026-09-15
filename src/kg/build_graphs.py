import os
import shutil
from pathlib import Path

from .config import KGConfig


def _repo_key(repo: str) -> str:
    return repo.replace("/", "__")


def _graph_key(repo: str, base_commit: str) -> str:
    return f"{_repo_key(repo)}__{base_commit}"


def ensure_repo_cloned(repo: str, config: KGConfig) -> Path:
    repo_dir = Path(config.repo_cache_dir) / _repo_key(repo)
    if not repo_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        url = config.mirror_url.format(repo=_repo_key(repo))
        print(f"Cloning {repo} from {url} ...", flush=True)
        from git import Repo

        Repo.clone_from(url, repo_dir)
        print("  cloned", flush=True)
    else:
        print(f"  Using cached clone at {repo_dir}", flush=True)
    return repo_dir


def _checkout(repo_dir: Path, base_commit: str):
    from git import Repo

    repo = Repo(str(repo_dir))
    repo.git.clean("-fd")
    repo.git.checkout("--force", base_commit)


def _init_graph_dir(graph_dir: Path, config: KGConfig):
    """Initialize graph_dir with schema + empty data.json (flat layout).

    The proven layout (matching swe_bench_kg/) puts all three JSON files
    at the graph_dir root.  After chdir(graph_dir), _graph_path finds
    them in CWD, and verify_graph_update gets graph_dir="." which reads
    ./origin.json, ./meta.json, ./data.json consistently.
    """
    for name in ("origin.json", "meta.json"):
        src = os.path.join(config.meta_schema_dir, name)
        shutil.copy2(src, graph_dir / name)

    data_path = graph_dir / "data.json"
    if not data_path.exists():
        data_path.write_text(
            '{"nodes": [], "edges": [], "allValues": {}}\n',
            encoding="utf-8",
        )


def build_graph(repo: str, base_commit: str, config: KGConfig) -> Path:
    graph_dir = Path(config.graph_cache_dir) / _graph_key(repo, base_commit)
    data_file = graph_dir / "data.json"
    if data_file.exists():
        import json

        with open(data_file) as f:
            existing = json.load(f)
        if existing.get("nodes"):
            print(f"  Graph cached at {graph_dir}", flush=True)
            return graph_dir

    graph_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = ensure_repo_cloned(repo, config)
    _checkout(repo_dir, base_commit)
    _init_graph_dir(graph_dir, config)

    from code_kg_builder.orchestrator import Orchestrator

    saved_cwd = os.getcwd()
    try:
        orch = Orchestrator(
            repo_path=repo_dir,
            meta_path=str(graph_dir / "meta.json"),
            graph_dir=str(graph_dir),
        )
        summary = orch.run()
        print(
            f"  Graph built: {summary.total_nodes} nodes, "
            f"{summary.total_edges} edges",
            flush=True,
        )
        if summary.errors:
            for err in summary.errors:
                print(f"  ERROR: {err}", flush=True)
    finally:
        os.chdir(saved_cwd)

    return graph_dir


def _unique_graphs(dataset) -> list:
    """Return sorted (repo, base_commit, instance_id) tuples, deduplicated
    by (repo, base_commit). Multiple instances sharing the same commit
    produce a single graph build entry.
    """
    unique_pairs: dict = {}
    for instance in dataset:
        key = (instance["repo"], instance["base_commit"])
        if key not in unique_pairs:
            unique_pairs[key] = instance["instance_id"]
    return [(repo, commit, iid) for (repo, commit), iid in sorted(unique_pairs.items())]


def build_all(
    dataset,
    config: KGConfig,
    worker_id: int = 0,
    num_workers: int = 1,
):
    """Build graphs for unique (repo, base_commit) pairs (skips cached).

    When num_workers > 1, each worker processes a disjoint subset of unique
    graphs. repo_cache_dir is per-worker to avoid git checkout conflicts.
    """
    if num_workers > 1:
        config.repo_cache_dir = f"{config.repo_cache_dir}_w{worker_id}"

    all_graphs = _unique_graphs(dataset)
    assigned_graphs = [
        graph for i, graph in enumerate(all_graphs) if i % num_workers == worker_id
    ]

    print(
        f"Worker {worker_id}/{num_workers}: {len(assigned_graphs)} unique graphs "
        f"(out of {len(all_graphs)} total)",
        flush=True,
    )

    seen_repos: set = set()
    for repo, base_commit, instance_id in assigned_graphs:
        print(f"\n[{instance_id}] {repo} @ {base_commit[:8]}", flush=True)

        if repo not in seen_repos:
            ensure_repo_cloned(repo, config)
            seen_repos.add(repo)

        try:
            build_graph(repo, base_commit, config)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
