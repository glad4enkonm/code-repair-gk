"""Embedding index over KG nodes — llama-server /v1/embeddings + Chroma.

Opt-in feature (``config.use_embeddings``). The embedding server is any
llama.cpp ``llama-server --embedding`` instance exposing an
OpenAI-compatible ``/v1/embeddings`` endpoint (separate port from the
chat LLM). Vectors are persisted per graph in ``chroma_db/`` next to
``data.json``.

An optional shared cross-graph cache (``config.embedding_cache_dir``)
keys vectors by git blob SHA of the node's source file, so repos seen
at multiple SWE-bench commits only embed changed files; the rest are
served from the cache.

nomic-embed-code task prefixes are applied consistently:
documents get ``search_code_document: `` and queries get
``search_code_query: `` — mixing them up silently degrades similarity.
"""

import hashlib
import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import KGConfig

COLLECTION_NAME = "kg_nodes"
CACHE_COLLECTION_NAME = "kg_node_vectors"
DOCUMENT_PREFIX = "search_code_document: "
QUERY_PREFIX = "search_code_query: "

# Bump when compose_node_text or the cache key scheme changes so stale
# cache entries are never served under a new composition rule
_COMPOSER_VERSION = "v1"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Regular file modes; symlinks (120000) and submodules (160000) excluded
_BLOB_MODES = ("100644", "100755")

# Structural nodes with no standalone semantic value. Directory nodes
# are skipped because their text is only a path, and every child node's
# composed text already carries the file path.
DEFAULT_SKIP_NODE_TYPES = ("Exception", "TypeAnnotation", "Dependency", "Directory")


def _load_data(graph_dir: str) -> dict:
    with open(Path(graph_dir) / "data.json", encoding="utf-8") as f:
        return json.load(f)


def _embed_texts(texts: List[str], config: KGConfig) -> List[List[float]]:
    """Embed a batch of pre-prefixed texts via the embedding server."""
    from openai import OpenAI

    client = OpenAI(base_url=config.embedding_api_base, api_key=config.api_key)
    response = client.embeddings.create(model=config.embedding_model, input=list(texts))
    return [item.embedding for item in response.data]


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3 chars/token) for code text."""
    return max(1, (len(text) + 2) // 3)


def _document_batches(documents: List[str], config: KGConfig) -> List[List[str]]:
    """Split documents into batches limited by count AND token budget.

    A batch is flushed when adding the next document would exceed
    ``embedding_batch_size`` items or ``embedding_ctx_tokens`` estimated
    tokens (the embedding server context). A single oversized document
    is still yielded as its own batch (truncation elsewhere should make
    that impossible, but never drop nodes silently).
    """
    if not documents:
        return []
    max_count = max(1, config.embedding_batch_size)
    budget = max(1, config.embedding_ctx_tokens)
    batches: List[List[str]] = []
    batch: List[str] = []
    used = 0
    for document in documents:
        cost = _estimate_tokens(document)
        if batch and (len(batch) >= max_count or used + cost > budget):
            batches.append(batch)
            batch, used = [], 0
        batch.append(document)
        used += cost
    if batch:
        batches.append(batch)
    return batches


def _truncate_text(text: str, config: KGConfig) -> str:
    """Clamp a composed document/query to ``embedding_max_doc_chars``."""
    limit = config.embedding_max_doc_chars
    return text[:limit] if limit else text


def embed_query(text: str, config: KGConfig) -> List[float]:
    """Embed a natural-language query (query-side prefix, truncated)."""
    prefixed = QUERY_PREFIX + _truncate_text(text, config)
    return _embed_texts([prefixed], config)[0]


def compose_node_text(
    node: dict, props: dict, relative_path: str, config: KGConfig
) -> str:
    """Build the text representation of a node for embedding."""
    meta_type = props.get("meta_type", "Node")
    name = props.get("name") or node.get("label") or node.get("id", "")

    if meta_type in ("Function", "Method", "Class"):
        parts = [f"{meta_type} {name} in {relative_path}"]
        docstring = props.get("docstring")
        if docstring:
            parts.append(str(docstring))
        source_code = props.get("source_code")
        if source_code:
            limit = config.embedding_max_source_chars
            parts.append(str(source_code)[:limit] if limit else str(source_code))
        return "\n".join(parts)
    if meta_type == "File":
        return f"File {props.get('relative_path') or relative_path}"
    if meta_type == "Variable":
        return f"Variable {name} = {props.get('value', '')} in {relative_path}"
    if meta_type == "Import":
        module_path = props.get("module_path", "")
        imported_name = props.get("imported_name", name)
        return f"Import {imported_name} from {module_path} in {relative_path}"
    if meta_type == "Decorator":
        return f"Decorator {name} in {relative_path}"
    if meta_type == "Directory":
        return f"Directory {props.get('relative_path') or relative_path}"
    return f"{meta_type} {name} in {relative_path}"


def resolve_relative_paths(graph_dir: str) -> Dict[str, str]:
    """Map every node id to its file's relative_path via CONTAINS edges.

    Pure in-memory traversal of data.json (no CWD-dependent graph API).
    CONTAINS direction: source (File/Class/Directory) contains target.
    """
    data = _load_data(graph_dir)
    all_values = data.get("allValues", {})
    file_paths: Dict[str, str] = {}
    for node in data.get("nodes", []):
        props = all_values.get(node["id"], {})
        if props.get("meta_type") == "File":
            file_paths[node["id"]] = (
                props.get("relative_path") or node.get("label") or node["id"]
            )

    parents: Dict[str, List[str]] = {}
    for edge in data.get("edges", []):
        if edge.get("label") != "CONTAINS":
            continue
        parents.setdefault(edge["target"], []).append(edge["source"])

    resolved = dict(file_paths)
    for node in data.get("nodes", []):
        node_id = node["id"]
        if node_id in resolved:
            continue
        queue = deque(parents.get(node_id, []))
        seen = {node_id}
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if current in file_paths:
                resolved[node_id] = file_paths[current]
                break
            queue.extend(parents.get(current, []))
        resolved.setdefault(node_id, "")
    return resolved


def _chroma_path(graph_dir: str) -> str:
    return str(Path(graph_dir).resolve() / "chroma_db")


def _get_client(graph_dir: str):
    import chromadb

    return chromadb.PersistentClient(path=_chroma_path(graph_dir))


def _get_collection(graph_dir: str, config: KGConfig, create: bool):
    client = _get_client(graph_dir)
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": config.embedding_model,
            },
        )
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def release_graph_client(graph_dir: str) -> None:
    """Close the per-graph chroma system so long runs do not exhaust fds.

    chromadb keeps one open system (sqlite + segment files) per path for
    the whole process lifetime; runs that touch hundreds of graphs hit
    the file-descriptor limit (sqlite CANTOPEN). Refcounting cannot be
    relied on here because every ``PersistentClient`` construction
    increments it, so the registry entry is force-dropped instead.
    Safe once the graph's embedding work is done: later clients
    transparently re-open the on-disk data.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        from chromadb.config import Settings

        settings = Settings(
            is_persistent=True, persist_directory=_chroma_path(graph_dir)
        )
        identifier = SharedSystemClient._get_identifier_from_settings(settings)
        system = SharedSystemClient._identifier_to_system.pop(identifier, None)
        if system is not None:
            system.stop()
    except Exception as exc:
        import sys

        print(f"[warn] graph client release failed ({exc!r})", file=sys.stderr)


def _skip_types(config: KGConfig) -> set:
    return set(config.embedding_skip_node_types or DEFAULT_SKIP_NODE_TYPES)


def _fingerprint(config: KGConfig) -> str:
    """Composition settings that determine every node's document text.

    Any change here (prefix, truncation limits, skip types, composer
    version) produces a different fingerprint, so cache entries keyed
    under the old fingerprint are never served.
    """
    skip = config.embedding_skip_node_types or DEFAULT_SKIP_NODE_TYPES
    return "|".join(
        [
            _COMPOSER_VERSION,
            DOCUMENT_PREFIX,
            str(config.embedding_max_doc_chars),
            str(config.embedding_max_source_chars),
            ",".join(sorted(skip)),
        ]
    )


def _node_cache_key(
    config: KGConfig, node_id: str, document: str, blob_sha: Optional[str] = None
) -> str:
    """Content-identity cache key for one node's embedding.

    Primary (blob) key: the git blob SHA of the node's file plus the
    node id — identical file content across commits hits. Fallback
    (doc) key: hash of the composed document — used when no blob is
    known (no file path, unparseable graph identity, ls-tree failure).
    """
    identity = f"{_fingerprint(config)}\0{config.embedding_model}"
    if blob_sha:
        identity += f"\0blob\0{blob_sha}\0{node_id}"
    else:
        doc_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
        identity += f"\0doc\0{doc_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _parse_graph_identity(graph_dir: str) -> Tuple[str, str]:
    """(repo_key, commit) from a graph dir named ``{repo}__{commit}``.

    Returns ("", "") when the name does not end in a 40-hex commit.
    """
    name = Path(graph_dir).name
    repo_key, sep, commit = name.rpartition("__")
    if sep and _COMMIT_SHA_RE.match(commit):
        return repo_key, commit
    return "", ""


def _blob_map(repo_key: str, commit: str, config: KGConfig) -> Dict[str, str]:
    """Map repo-relative path -> git blob SHA at ``commit``.

    Reads the already-cloned source repo (repo_cache_dir); regular
    files only. Returns {} when the repo or commit is unavailable —
    callers then fall back to document-hash cache keys.
    """
    if not repo_key or not commit:
        return {}
    repo_dir = Path(config.repo_cache_dir) / repo_key
    if not repo_dir.is_dir():
        return {}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "ls-tree", "-r", "-z", commit],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    blobs: Dict[str, str] = {}
    for entry in proc.stdout.split(b"\0"):
        if not entry:
            continue
        meta, tab, raw_path = entry.partition(b"\t")
        if not tab:
            continue
        parts = meta.split(b" ")
        if len(parts) < 3:
            continue
        mode, objtype, sha = parts[0], parts[1], parts[2]
        if mode.decode("ascii", "replace") in _BLOB_MODES and objtype == b"blob":
            blobs[raw_path.decode("utf-8", "replace")] = sha.decode("ascii", "replace")
    return blobs


def _cache_collection(config: KGConfig, create: bool = False):
    """Shared cross-graph vector cache collection; None when disabled.

    Recreated from scratch whenever fingerprint, model, or vector
    dimension no longer match the active config, mirroring the
    per-graph rebuild guard in build_embeddings.
    """
    if not config.embedding_cache_dir:
        return None
    import chromadb

    client = chromadb.PersistentClient(
        path=str(Path(config.embedding_cache_dir) / "chroma_db")
    )
    expected = {
        "fingerprint": _fingerprint(config),
        "embedding_model": config.embedding_model,
        "dim": config.embedding_dim,
    }
    try:
        existing = client.get_collection(CACHE_COLLECTION_NAME)
    except Exception:
        existing = None
    if existing is not None:
        meta = existing.metadata or {}
        if all(meta.get(key) == value for key, value in expected.items()):
            return existing
        if not create:
            return None
        client.delete_collection(CACHE_COLLECTION_NAME)
    if not create:
        return None
    return client.create_collection(name=CACHE_COLLECTION_NAME, metadata=dict(expected))


def _cache_lookup(collection, keys: List[str], config: KGConfig):
    """Exact-key lookup: {key: vector} for hits with the expected dim.

    Duplicate keys are deduplicated per chunk: chroma's get() rejects
    duplicate ids in one call, and doc-key fallback produces them
    whenever distinct nodes compose identical documents.
    """
    found: Dict[str, List[float]] = {}
    try:
        for start in range(0, len(keys), 5000):
            chunk_keys = keys[start : start + 5000]
            unique_keys = list(dict.fromkeys(chunk_keys))
            result = collection.get(ids=unique_keys, include=["embeddings"])
            for key, vector in zip(result.get("ids", []), result.get("embeddings", [])):
                # chromadb may hand back numpy arrays — never use them
                # in a boolean context ("not vector" raises ValueError)
                if vector is None or len(vector) != config.embedding_dim:
                    continue
                if not isinstance(vector, list):
                    vector = (
                        vector.tolist()
                        if hasattr(vector, "tolist")
                        else [float(x) for x in vector]
                    )
                found[key] = vector
    except Exception as exc:
        print(
            f"  [warn] embedding cache lookup failed ({exc!r}); "
            "embedding everything",
            flush=True,
        )
        return {}
    return found


def _cache_store(
    collection, keys: List[str], vectors: List[List[float]], repo_key: str
) -> bool:
    """Upsert freshly embedded vectors into the shared cache.

    Duplicate keys within a chunk are deduplicated (first vector wins):
    chroma's upsert rejects duplicate ids in one call, and graphs can
    contain distinct nodes whose composed documents are identical.
    """
    try:
        for start in range(0, len(keys), 5000):
            chunk_keys = keys[start : start + 5000]
            chunk_vectors = vectors[start : start + len(chunk_keys)]
            seen = set()
            unique_keys: List[str] = []
            unique_vectors: List[List[float]] = []
            for key, vector in zip(chunk_keys, chunk_vectors):
                if key in seen:
                    continue
                seen.add(key)
                unique_keys.append(key)
                unique_vectors.append(vector)
            collection.upsert(
                ids=unique_keys,
                embeddings=unique_vectors,
                metadatas=[{"repo_key": repo_key or "unknown"}] * len(unique_keys),
            )
        return True
    except Exception as exc:
        print(
            f"  [warn] embedding cache store failed ({exc!r}); continuing without",
            flush=True,
        )
        return False


def build_embeddings(graph_dir: str, config: KGConfig) -> None:
    """Compose, embed and store all embeddable nodes. Cached per model."""
    chroma_dir = Path(graph_dir) / "chroma_db"
    existing = _get_collection(graph_dir, config, create=False)
    if (
        existing is not None
        and existing.count() > 0
        and (existing.metadata or {}).get("embedding_model") == config.embedding_model
    ):
        # node_count is stamped only after a complete insert (see below);
        # a missing or mismatched stamp means a partial or legacy index
        # — rebuild instead of reusing it
        if (existing.metadata or {}).get("node_count") == existing.count():
            print(
                f"  Embeddings cached at {chroma_dir} "
                f"({existing.count()} nodes, model match) — skipping rebuild",
                flush=True,
            )
            return
    if existing is not None:
        _get_client(graph_dir).delete_collection(COLLECTION_NAME)

    data = _load_data(graph_dir)
    all_values = data.get("allValues", {})
    node_paths = resolve_relative_paths(graph_dir)
    skip = _skip_types(config)
    print(f"  Composing node texts for embedding ({chroma_dir})...", flush=True)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []
    paths: List[str] = []
    for node in data.get("nodes", []):
        node_id = node["id"]
        props = all_values.get(node_id, {})
        if props.get("meta_type") in skip:
            continue
        ids.append(node_id)
        documents.append(
            DOCUMENT_PREFIX
            + _truncate_text(
                compose_node_text(node, props, node_paths.get(node_id, ""), config),
                config,
            )
        )
        metadatas.append(
            {
                "meta_type": props.get("meta_type") or "Unknown",
                "name": str(props.get("name") or node.get("label", "")),
                "label": str(node.get("label", "")),
                "relative_path": node_paths.get(node_id, "") or "",
            }
        )
        paths.append(node_paths.get(node_id, "") or "")

    collection = _get_collection(graph_dir, config, create=True)
    if not ids:
        print(
            f"  Embedding index empty (no embeddable nodes): {chroma_dir}", flush=True
        )
        return

    # --- Cross-graph cache: skip embedding content we already have ---
    cache_collection = None
    try:
        cache_collection = _cache_collection(config, create=True)
    except Exception as exc:
        print(
            f"  [warn] embedding cache unavailable ({exc!r}); " "embedding everything",
            flush=True,
        )

    cache_keys: List[str] = []
    cached: Dict[str, List[float]] = {}
    repo_key = ""
    if cache_collection is not None:
        try:
            repo_key, commit = _parse_graph_identity(graph_dir)
            blob_map = _blob_map(repo_key, commit, config)
            cache_keys = [
                _node_cache_key(config, node_id, document, blob_map.get(path))
                for node_id, document, path in zip(ids, documents, paths)
            ]
            cached = _cache_lookup(cache_collection, cache_keys, config)
        except Exception:
            import traceback

            traceback.print_exc()
            cache_keys, cached = [], {}

    if cache_keys:
        miss_positions = [i for i, key in enumerate(cache_keys) if key not in cached]
    else:
        miss_positions = list(range(len(ids)))
    miss_documents = [documents[i] for i in miss_positions]

    new_vectors: List[List[float]] = []
    batches = _document_batches(miss_documents, config)
    total_batches = len(batches)
    for batch_index, batch in enumerate(batches, 1):
        new_vectors.extend(_embed_texts(batch, config))
        print(
            f"\r  Embedding {len(new_vectors)}/{len(miss_documents)} nodes"
            f" (batch {batch_index}/{total_batches})",
            end="",
            flush=True,
        )
    if miss_documents:
        print()

    if cache_collection is not None and miss_positions:
        _cache_store(
            cache_collection,
            [cache_keys[i] for i in miss_positions],
            new_vectors,
            repo_key,
        )

    # Assemble cached + fresh vectors in original node order
    vectors: List[List[float]] = []
    fresh_index = 0
    for key in cache_keys:
        if key in cached:
            vectors.append(cached[key])
        else:
            vectors.append(new_vectors[fresh_index])
            fresh_index += 1
    if not cache_keys:
        vectors = list(new_vectors)

    reused_count = len(cache_keys) - len(miss_positions)
    if reused_count:
        print(
            f"  Cache: reused {reused_count} of {len(ids)} node embeddings",
            flush=True,
        )

    # Chroma caps a single add at ~5461 records; chunk the final insert
    add_chunk = 5000
    for start in range(0, len(ids), add_chunk):
        end = start + add_chunk
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
        )
    # Stamp completion marker so interrupted builds are detected above.
    # modify() replaces metadata wholesale, so carry over any existing
    # keys — but newer chromadb forbids "hnsw:" keys in modify() (even
    # unchanged); the distance function set at creation is preserved in
    # the collection configuration and needs no re-stamping here.
    completion_metadata = {
        key: value
        for key, value in (collection.metadata or {}).items()
        if not key.startswith("hnsw:")
    }
    completion_metadata.update(
        {
            "embedding_model": config.embedding_model,
            "node_count": len(ids),
        }
    )
    collection.modify(metadata=completion_metadata)
    print(f"  Embedded {len(ids)} nodes -> {chroma_dir}", flush=True)


def search_by_embedding(
    query: str,
    graph_dir: str,
    config: KGConfig,
    meta_type: Optional[str] = None,
    n_results: Optional[int] = None,
) -> dict:
    """Semantic search over indexed nodes (search_elements-compatible shape)."""
    collection = _get_collection(graph_dir, config, create=False)
    if collection is None or collection.count() == 0:
        return {
            "success": False,
            "error": (
                f"embedding index not found at {_chroma_path(graph_dir)} "
                "— run build_embeddings first"
            ),
        }
    limit = min(n_results or config.embedding_search_k, collection.count())
    where = {"meta_type": meta_type} if meta_type else None
    vector = embed_query(query, config)
    result = collection.query(query_embeddings=[vector], n_results=limit, where=where)
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    items = []
    for node_id, distance, meta in zip(ids, distances, metadatas):
        meta = meta or {}
        properties = {k: v for k, v in meta.items() if k != "label"}
        properties["similarity"] = round(1.0 - distance, 4)
        items.append(
            {
                "kind": "node",
                "id": node_id,
                "label": meta.get("label", node_id),
                "properties": properties,
            }
        )
    return {"success": True, "total": len(items), "items": items}


def get_top_k_for_issue(
    problem_statement: str, graph_dir: str, config: KGConfig, k: int
) -> List[dict]:
    """One-shot head start: embed the issue, return top-K node summaries.

    Raises RuntimeError when the index is missing (callers degrade).
    """
    collection = _get_collection(graph_dir, config, create=False)
    if collection is None or collection.count() == 0:
        raise RuntimeError(f"embedding index not found at {_chroma_path(graph_dir)}")
    vector = embed_query(problem_statement, config)
    result = collection.query(
        query_embeddings=[vector], n_results=min(k, collection.count())
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    seeds = []
    for node_id, distance, meta in zip(ids, distances, metadatas):
        meta = meta or {}
        seeds.append(
            {
                "id": node_id,
                "label": meta.get("label", ""),
                "meta_type": meta.get("meta_type", ""),
                "name": meta.get("name", ""),
                "relative_path": meta.get("relative_path", ""),
                "similarity": round(1.0 - distance, 4),
            }
        )
    seeds.sort(key=lambda s: s["similarity"], reverse=True)
    return seeds


def make_search_tool(graph_dir: str, config: KGConfig):
    """Build the search_by_embedding callable exposed to the LLM loop."""

    def search_by_embedding_tool(
        query: Optional[str] = None,
        meta_type: Optional[str] = None,
        n_results: Optional[int] = None,
        **_ignored,
    ):
        if not query:
            return {"success": False, "error": "'query' is required"}
        return search_by_embedding(
            str(query), graph_dir, config, meta_type=meta_type, n_results=n_results
        )

    search_by_embedding_tool.__name__ = "search_by_embedding"
    return search_by_embedding_tool
