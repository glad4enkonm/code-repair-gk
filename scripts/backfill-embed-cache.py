#!/usr/bin/env python3
"""Backfill existing per-graph embedding indexes into the shared cache.

Walks the graph cache dir, reads every complete per-graph Chroma index,
and upserts its vectors into the cross-graph cache collection keyed by
git blob SHA (document-hash fallback). Graphs whose stamped model
differs from the active config, or whose index is incomplete, are
skipped.

Usage:
    python3 scripts/backfill-embed-cache.py [--config config/kg_config.json]
                                           [--graphs-dir PATH]
                                           [--repos-dir PATH]
                                           [--cache-dir PATH]
                                           [--limit N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg.config import KGConfig  # noqa: E402
from kg.embeddings import (  # noqa: E402
    _blob_map,
    _cache_collection,
    _cache_store,
    _get_collection,
    _node_cache_key,
    _parse_graph_identity,
    release_graph_client,
)

_PAGE_SIZE = 5000


def backfill_graph(graph_dir: Path, config: KGConfig, cache_collection):
    """Upsert one graph's vectors into the shared cache.

    Returns (status, vectors, blob_keyed, doc_keyed) with status
    "ok" or "store_failed", or None when the graph has no usable
    index (missing, empty, incomplete, or foreign model).
    """
    collection = _get_collection(str(graph_dir), config, create=False)
    if collection is None or collection.count() == 0:
        return None
    meta = collection.metadata or {}
    if meta.get("embedding_model") != config.embedding_model:
        return None
    # node_count is stamped only after a complete insert; a missing or
    # mismatched stamp means a partial or legacy index — skip it
    if meta.get("node_count") != collection.count():
        return None

    ids: list = []
    embeddings: list = []
    documents: list = []
    paths: list = []
    offset = 0
    while True:
        page = collection.get(
            limit=_PAGE_SIZE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        if not page["ids"]:
            break
        ids.extend(page["ids"])
        # embeddings may be a numpy 2D array — never an "or" operand
        page_embeddings = page.get("embeddings")
        if page_embeddings is not None:
            embeddings.extend(page_embeddings)
        documents.extend(page.get("documents") or [])
        paths.extend(
            (item or {}).get("relative_path", "")
            for item in (page.get("metadatas") or [])
        )
        offset += len(page["ids"])

    repo_key, commit = _parse_graph_identity(str(graph_dir))
    blob_map = _blob_map(repo_key, commit, config) if repo_key else {}
    keys: list = []
    blob_keyed = 0
    for node_id, document, path in zip(ids, documents, paths):
        sha = blob_map.get(path or "")
        if sha:
            blob_keyed += 1
        keys.append(_node_cache_key(config, node_id, document, sha))
    status = "ok" if _cache_store(cache_collection, keys, embeddings, repo_key) else (
        "store_failed"
    )
    return status, len(keys), blob_keyed, len(keys) - blob_keyed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill per-graph embedding indexes into the shared cache"
    )
    parser.add_argument(
        "--config",
        default="config/kg_config.json",
        help="KGConfig JSON (embedding model/dims/limits)",
    )
    parser.add_argument(
        "--graphs-dir", help="Override graph cache dir (default: config)"
    )
    parser.add_argument(
        "--repos-dir", help="Override source repo clones dir (default: config)"
    )
    parser.add_argument(
        "--cache-dir", help="Override embedding cache dir (default: config)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N graphs (0 = all)"
    )
    args = parser.parse_args(argv)

    if Path(args.config).exists():
        config = KGConfig.from_file(args.config)
    else:
        config = KGConfig()
    if args.graphs_dir:
        config.graph_cache_dir = args.graphs_dir
    if args.repos_dir:
        config.repo_cache_dir = args.repos_dir
    if args.cache_dir:
        config.embedding_cache_dir = args.cache_dir
    if not config.embedding_cache_dir:
        parser.error(
            "embedding cache disabled — set embedding_cache_dir (config or --cache-dir)"
        )

    graphs_root = Path(config.graph_cache_dir)
    if not graphs_root.is_dir():
        print(f"graphs dir not found: {graphs_root}", file=sys.stderr)
        return 1

    cache_collection = _cache_collection(config, create=True)
    graph_dirs = sorted(
        path
        for path in graphs_root.iterdir()
        if path.is_dir() and (path / "data.json").exists()
    )

    total_vectors = total_blob = total_doc = skipped = processed = failed = 0
    for index, graph_dir in enumerate(graph_dirs, 1):
        if args.limit and index > args.limit:
            break
        result = backfill_graph(graph_dir, config, cache_collection)
        # chromadb holds one open system per path for the process
        # lifetime; release it or hundreds of graphs exhaust the
        # file-descriptor limit (sqlite CANTOPEN)
        release_graph_client(str(graph_dir))
        if result is None:
            print(f"{graph_dir.name}: skipped (missing/incomplete/foreign index)")
            skipped += 1
            continue
        status, vectors, blob_keyed, doc_keyed = result
        if status == "store_failed":
            print(f"{graph_dir.name}: store FAILED ({vectors} vectors lost)", flush=True)
            failed += 1
            continue
        print(
            f"{graph_dir.name}: {vectors} vectors "
            f"({blob_keyed} blob-keyed, {doc_keyed} doc-keyed)",
            flush=True,
        )
        processed += 1
        total_vectors += vectors
        total_blob += blob_keyed
        total_doc += doc_keyed

    print(
        f"\nBackfilled {processed} graphs ({skipped} skipped, {failed} failed): "
        f"{total_vectors} vectors, {total_blob} blob-keyed, {total_doc} doc-keyed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
