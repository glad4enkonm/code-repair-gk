#!/usr/bin/env python3
"""Precompute KG node embeddings for unfinished instances (GPU phase).

Builds every remaining instance's graph embedding index into the shared
blob-keyed cache WITHOUT running the LLM exploration loop. Meant to run
against a llama-server whose preset loads only the embed model fully on
GPU (docker/models-embed-gpu.ini); the main chain afterwards turns every
embedding build into pure cache hits and runs at explore speed.

Completeness predicate mirrors backfill-embed-cache.py: a per-graph
collection counts as done when it is non-empty, its model matches the
active config, and its node_count stamp equals the live count. Safe to
re-run after a crash or a GPU cold-reset: completed indexes are skipped.

Instances share nothing (graph dirs, shared cache, no progress rows), so
--subprocess-per-instance runs each instance in a fresh python that exits
after finishing — chroma/glibc memory returns to the OS and nothing
accumulates (same leak fix as kg.run).

Usage:
    python3 scripts/precompute-embeddings.py [config] [--only INSTANCE_ID]
                                           [--subprocess-per-instance]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from kg.build_graphs import build_graph  # noqa: E402
from kg.config import KGConfig  # noqa: E402
from kg.embeddings import (  # noqa: E402
    _get_collection,
    build_embeddings,
    release_graph_client,
)

# Prescan progress cadence: one line per N checked indexes so long
# scans never run silently
_PRESCAN_LOG_EVERY = 25


def graph_dir_for(repo: str, base_commit: str, config: KGConfig) -> Path:
    return Path(config.graph_cache_dir) / (
        f"{repo.replace('/', '__')}__{base_commit}"
    )


def index_complete(graph_dir, config: KGConfig) -> bool:
    """True when the per-graph index is complete for this config.

    Mirrors backfill-embed-cache.py's eligibility check: non-empty
    collection, matching embedding model, and a node_count stamp equal
    to the live count (the stamp is written only after a complete
    insert).
    """
    collection = _get_collection(str(graph_dir), config, create=False)
    if collection is None or collection.count() == 0:
        release_graph_client(str(graph_dir))
        return False
    metadata = collection.metadata or {}
    complete = (
        metadata.get("embedding_model") == config.embedding_model
        and metadata.get("node_count") == collection.count()
    )
    release_graph_client(str(graph_dir))
    return complete


def precompute_instance(instance, config: KGConfig) -> bool:
    """Build graph + embedding index for one instance (no LLM involved)."""
    instance_id = instance["instance_id"]
    try:
        print(f"\n[{instance_id}] {instance['repo']}", flush=True)
        graph_dir = build_graph(instance["repo"], instance["base_commit"], config)
        build_embeddings(str(graph_dir), config)
        release_graph_client(str(graph_dir))
        return True
    except Exception:
        import traceback

        print(f"  FAILED: {instance_id}", flush=True)
        traceback.print_exc()
        return False


def _spawn_worker(config_path, instance_id) -> bool:
    """Run one instance in a fresh `--only` subprocess (leak-proof)."""
    command = [sys.executable, str(Path(__file__).resolve())]
    if config_path:
        command.append(config_path)
    command += ["--only", instance_id]

    print(f"\n[{instance_id}] worker subprocess", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(
            f"  FAILED: {instance_id} (worker exit {result.returncode})",
            flush=True,
        )
        return False
    return True


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Precompute per-graph KG node embeddings into the shared "
            "cache (GPU phase; no LLM exploration)"
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to kg_config.json (defaults: KGConfig())",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Process only this instance_id (worker mode)",
    )
    parser.add_argument(
        "--subprocess-per-instance",
        action="store_true",
        help="Spawn a fresh python per instance so memory returns to the OS",
    )
    return parser.parse_args(argv)


def main(
    config: KGConfig,
    config_path: str | None = None,
    only: str | None = None,
    subprocess_per_instance: bool = False,
):
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=config.split)
    print(
        f"Dataset: princeton-nlp/SWE-bench_Lite/{config.split}"
        f" — {len(dataset)} instances"
    )
    instances = list(dataset)

    if only and not any(i["instance_id"] == only for i in instances):
        raise SystemExit(f"instance {only} not found in split {config.split}")

    if not only:
        checked = 0
        complete_count = 0
        for instance in instances:
            complete_count += index_complete(
                graph_dir_for(instance["repo"], instance["base_commit"], config),
                config,
            )
            checked += 1
            if checked % _PRESCAN_LOG_EVERY == 0:
                print(
                    f"  prescan: checked {checked}/{len(instances)} indexes"
                    f" ({complete_count} complete)",
                    flush=True,
                )
        print(
            f"  prescan: checked {len(instances)}/{len(instances)} indexes"
            f" ({complete_count} complete)",
            flush=True,
        )
        print(
            f"Index complete for {complete_count}/{len(instances)};"
            " embedding the rest"
        )

    for instance in tqdm(instances, desc="Precompute embeddings"):
        instance_id = instance["instance_id"]
        if only and instance_id != only:
            continue
        graph_dir = graph_dir_for(
            instance["repo"], instance["base_commit"], config
        )
        if index_complete(graph_dir, config):
            continue

        if subprocess_per_instance:
            ok = _spawn_worker(config_path, instance_id)
        else:
            ok = precompute_instance(instance, config)
        if ok:
            print(f"  [{instance_id}] embedding index ready", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    args = _parse_args()
    config = KGConfig.from_file(args.config) if args.config else KGConfig()
    main(
        config,
        config_path=args.config,
        only=args.only,
        subprocess_per_instance=args.subprocess_per_instance,
    )
