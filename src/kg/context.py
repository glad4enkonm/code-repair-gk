import logging
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

from .config import KGConfig

logger = logging.getLogger(__name__)


@contextmanager
def _graph_cwd(graph_dir: str) -> Iterator[None]:
    """Context manager: chdir into graph_dir, restore on exit."""
    saved = os.getcwd()
    os.chdir(graph_dir)
    try:
        yield
    finally:
        os.chdir(saved)


def _rank_files(touched: Dict[str, dict], graph_dir: str) -> List[str]:
    """Map touched graph nodes to files via CONTAINS edges.

    Uses ``find_connected`` to traverse the graph structure (not ID
    parsing).  For each touched node, finds the nearest ancestor of
    ``meta_type == "File"`` and reads its ``relative_path`` property.
    """
    from tool.functions.graph_file_ops.read import find_connected

    with _graph_cwd(graph_dir):
        file_freq: Dict[str, int] = {}
        for nid in touched:
            result = find_connected(
                graph="data",
                id=nid,
                type="File",
                edge_label="CONTAINS",
                include=["properties"],
            )
            if not result.get("success"):
                continue
            props = result["node"].get("properties", {})
            path = props.get("relative_path")
            if path and path.endswith(".py"):
                file_freq[path] = file_freq.get(path, 0) + 1
        return sorted(file_freq, key=lambda path: file_freq[path], reverse=True)


def _read_file(repo_dir: str, rel_path: str) -> str | None:
    full = os.path.join(repo_dir, rel_path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _read_readmes(repo_dir: str) -> Dict[str, str]:
    """Read README files from the repo root."""
    readmes: Dict[str, str] = {}
    for entry in os.listdir(repo_dir):
        if entry.lower().startswith("readme"):
            full = os.path.join(repo_dir, entry)
            if os.path.isfile(full):
                try:
                    with open(full, encoding="utf-8", errors="replace") as f:
                        readmes[entry] = f.read()
                except Exception as exc:
                    logger.warning("failed to read readme %s: %s", full, exc)
    return readmes


def read_files_within_budget(
    repo_dir: str, ranked_files: List[str], max_tokens: int
) -> Dict[str, str]:
    """Read files in order, enforcing a token budget.

    The first successfully read file is always included (even if it
    alone exceeds the budget); after that, stop at the first file that
    would overflow it. Missing files are skipped without consuming
    budget. Token estimate: ~3 chars/token (rough).
    """
    file_contents: Dict[str, str] = {}
    total_tokens = 0
    for rel_path in ranked_files:
        content = _read_file(repo_dir, rel_path)
        if content is None:
            continue
        est_tokens = len(content) // 3  # rough: ~3 chars/token
        if total_tokens + est_tokens > max_tokens and file_contents:
            break
        file_contents[rel_path] = content
        total_tokens += est_tokens
    return file_contents


def read_all_files(repo_dir: str, ranked_files: List[str]) -> Dict[str, str]:
    """Read every ranked file, ignoring any token budget.

    The token-budget cutoff dropped low-frequency files (often the
    single-touch gt file) before the downstream rerank stage could see
    them. File-level contexts therefore include every touched file in
    frequency order and let the reranker decide what matters.
    """
    file_contents: Dict[str, str] = {}
    for rel_path in ranked_files:
        content = _read_file(repo_dir, rel_path)
        if content is not None:
            file_contents[rel_path] = content
    return file_contents


def _merge_ranges(
    ranges: List[Tuple[int, int]], padding: int = 0
) -> List[Tuple[int, int]]:
    """Sort, apply ±padding, merge overlapping or adjacent ranges."""
    if not ranges:
        return []
    padded = sorted((max(1, start - padding), end + padding) for start, end in ranges)
    merged = [padded[0]]
    for start, end in padded[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _format_snippets(
    snippets_by_file: Dict[str, List[Tuple[int, int, str]]],
) -> str:
    """Format snippets with original line numbers and gap markers."""
    if not snippets_by_file:
        return ""
    parts: List[str] = []
    for filename in sorted(snippets_by_file):
        parts.append(f"[start of {filename}]")
        prev_end = 0
        for line_start, line_end, text in snippets_by_file[filename]:
            if prev_end > 0 and line_start > prev_end + 1:
                omitted = line_start - prev_end - 1
                parts.append(f"... ({omitted} lines omitted) ...")
            for offset, line in enumerate(text.splitlines()):
                parts.append(f"{line_start + offset} {line}")
            prev_end = line_end
        parts.append(f"[end of {filename}]")
    return "\n".join(parts)


def _enrich_touched(touched: Dict[str, dict], graph_dir: str) -> Dict[str, dict]:
    """Fetch missing properties from graph for Function/Method/Class nodes.

    Returns a filtered dict of nodes that have ``meta_type``,
    ``line_start``, ``line_end``, and ``source_code``.
    """
    from tool.functions.graph_file_ops.read import get_element

    with _graph_cwd(graph_dir):
        enriched: Dict[str, dict] = {}
        for nid, existing in touched.items():
            meta_type = existing.get("meta_type")
            has_lines = "line_start" in existing and "line_end" in existing
            has_source = "source_code" in existing

            if meta_type and has_lines and has_source:
                if meta_type in ("Function", "Method", "Class"):
                    enriched[nid] = existing
                continue

            result = get_element(graph="data", id=nid, include=["properties"])
            if not result.get("success"):
                continue
            props = dict(existing)
            props.update(result.get("properties", {}))

            meta_type = props.get("meta_type", "")
            if meta_type not in ("Function", "Method", "Class"):
                continue
            if "line_start" not in props or "line_end" not in props:
                continue
            enriched[nid] = props
        return enriched


def _map_nodes_to_files(enriched: Dict[str, dict], graph_dir: str) -> Dict[str, str]:
    """Map each enriched node to its parent file's relative_path."""
    from tool.functions.graph_file_ops.read import find_connected

    with _graph_cwd(graph_dir):
        node_to_file: Dict[str, str] = {}
        for nid in enriched:
            result = find_connected(
                graph="data",
                id=nid,
                type="File",
                edge_label="CONTAINS",
                include=["properties"],
            )
            if not result.get("success"):
                continue
            file_props = result["node"].get("properties", {})
            rel_path = file_props.get("relative_path")
            if rel_path and rel_path.endswith(".py"):
                node_to_file[nid] = rel_path
        return node_to_file


def _build_prompt(
    problem_statement: str,
    readmes: Dict[str, str],
    code_text: str,
) -> str:
    """Build a style-3-compatible prompt with pre-formatted code_text."""
    from swebench.inference.make_datasets.create_instance import (
        PATCH_EXAMPLE,
        make_code_text,
    )

    premise = (
        "You will be provided with a partial code base and an issue "
        "statement explaining a problem to resolve."
    )
    readmes_text = make_code_text(readmes) if readmes else ""
    example_explanation = (
        "Here is an example of a patch file. It consists of changes to "
        "the code base. It specifies the file names, the line numbers of "
        "each change, and the removed and added lines. A single patch "
        "file can contain changes to multiple files."
    )
    final_instruction = (
        "I need you to solve the provided issue by generating a single "
        "patch file that I can apply directly to this repository using "
        "git apply. Please respond with a single patch file in the "
        "format shown above."
    )
    parts = [
        premise,
        "<issue>",
        problem_statement,
        "</issue>",
        "",
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        example_explanation,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "",
        final_instruction,
        "Respond below:",
    ]
    return "\n".join(parts)


def assemble_file_level(
    touched: Dict[str, dict],
    repo: str,
    base_commit: str,
    problem_statement: str,
    graph_dir: str,
    config: KGConfig,
) -> str:
    """
    Map touched graph nodes to files, read contents, and build a style-3 prompt.
    """
    ranked_files = _rank_files(touched, graph_dir)
    if not ranked_files:
        return ""

    from .build_graphs import ensure_repo_cloned, _checkout

    repo_dir = ensure_repo_cloned(repo, config)
    _checkout(repo_dir, base_commit)

    from swebench.inference.make_datasets.create_instance import prompt_style_3

    readmes = _read_readmes(str(repo_dir))
    file_contents = read_all_files(str(repo_dir), ranked_files)

    if not file_contents:
        return ""

    instance = {
        "problem_statement": problem_statement,
        "readmes": readmes,
        "file_contents": file_contents,
    }

    return prompt_style_3(instance)


def assemble_file_snippet(
    touched: Dict[str, dict],
    repo: str,
    base_commit: str,
    problem_statement: str,
    graph_dir: str,
    config: KGConfig,
) -> str:
    """Extract relevant line ranges from files, preserving line numbers."""
    enriched = _enrich_touched(touched, graph_dir)
    if not enriched:
        return ""

    node_to_file = _map_nodes_to_files(enriched, graph_dir)
    if not node_to_file:
        return ""

    ranges_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for nid, rel_path in node_to_file.items():
        props = enriched[nid]
        ranges_by_file.setdefault(rel_path, []).append(
            (props["line_start"], props["line_end"])
        )

    sorted_files = sorted(
        ranges_by_file, key=lambda f: len(ranges_by_file[f]), reverse=True
    )

    from .build_graphs import ensure_repo_cloned, _checkout

    repo_dir = ensure_repo_cloned(repo, config)
    _checkout(repo_dir, base_commit)

    readmes = _read_readmes(str(repo_dir))

    snippets: Dict[str, List[Tuple[int, int, str]]] = {}
    total_tokens = 0

    for rel_path in sorted_files:
        merged = _merge_ranges(ranges_by_file[rel_path], config.node_context_padding)
        content = _read_file(str(repo_dir), rel_path)
        if content is None:
            continue
        file_lines = content.splitlines()
        file_snippets: List[Tuple[int, int, str]] = []
        for start, end in merged:
            start = max(1, start)
            end = min(len(file_lines), end)
            if start > end:
                continue
            text = "\n".join(file_lines[start - 1 : end])
            est_tokens = len(text) // 3
            if total_tokens + est_tokens > config.max_context_tokens and snippets:
                break
            file_snippets.append((start, end, text))
            total_tokens += est_tokens
        if file_snippets:
            snippets[rel_path] = file_snippets

    if not snippets:
        return ""

    code_text = _format_snippets(snippets)
    return _build_prompt(problem_statement, readmes, code_text)


def _collect_graph_context(touched: Dict[str, dict], graph_dir: str) -> tuple:
    """Collect node properties, edges, and file mapping in a single pass.

    Uses ``get_element`` for node properties, ``get_neighbors`` for
    edge metadata, and ``find_connected`` for file mapping (which
    handles multi-hop traversal: Method → Class → File).

    Returns ``(enriched, edge_metadata, node_to_file)``.
    """
    from tool.functions.graph_file_ops.read import (
        find_connected,
        get_element,
        get_neighbors,
    )

    with _graph_cwd(graph_dir):
        enriched: Dict[str, dict] = {}
        edge_metadata: Dict[str, List[dict]] = {}
        node_to_file: Dict[str, str] = {}

        for nid, existing in touched.items():
            # --- Node's own properties (get_element) ---
            props = dict(existing)
            needs_fetch = not (
                props.get("source_code")
                and props.get("line_start")
                and props.get("meta_type")
            )
            if needs_fetch:
                result = get_element(graph="data", id=nid, include=["properties"])
                if result.get("success"):
                    props.update(result.get("properties", {}))

            mt = props.get("meta_type", "")
            if mt not in ("Function", "Method", "Class"):
                continue
            if not props.get("source_code") or not props.get("line_start"):
                continue
            enriched[nid] = props

            # --- Edge metadata (get_neighbors) ---
            neighbors = get_neighbors(graph="data", id=nid, include=["properties"])
            edges: List[dict] = []
            if neighbors.get("success"):
                lookup = {
                    n.get("id"): n.get("properties", {})
                    for n in neighbors.get("nodes", [])
                }
                for e in neighbors.get("edges", []):
                    if e.get("source") == nid:
                        direction = "outgoing"
                        connected_id = e.get("target", "")
                    elif e.get("target") == nid:
                        direction = "incoming"
                        connected_id = e.get("source", "")
                    else:
                        continue

                    conn_props = dict(lookup.get(connected_id, {}))
                    conn_props.pop("source_code", None)
                    edges.append(
                        {
                            "edge_label": e.get("label", ""),
                            "direction": direction,
                            "connected_props": conn_props,
                        }
                    )
            edge_metadata[nid] = edges

            # --- File mapping (find_connected — handles multi-hop) ---
            fc_result = find_connected(
                graph="data",
                id=nid,
                type="File",
                edge_label="CONTAINS",
                include=["properties"],
            )
            if fc_result.get("success"):
                file_props = fc_result["node"].get("properties", {})
                rel_path = file_props.get("relative_path", "")
                if rel_path and rel_path.endswith(".py"):
                    node_to_file[nid] = rel_path

        return enriched, edge_metadata, node_to_file


def _format_node_snippets(
    snippets_by_file: Dict[str, List[dict]],
) -> str:
    """Format node snippets with metadata headers, edge info, and
    original line numbers.

    Each entry in ``snippets_by_file`` is a dict with keys:
    ``line_start``, ``line_end``, ``source_code``, ``node_props``,
    ``edges``.
    """
    if not snippets_by_file:
        return ""
    parts: List[str] = []
    for filename in sorted(snippets_by_file):
        parts.append(f"[start of {filename}]")
        prev_end = 0
        for entry in snippets_by_file[filename]:
            line_start = entry["line_start"]
            line_end = entry["line_end"]
            source = entry["source_code"]
            props = entry.get("node_props", {})
            edges = entry.get("edges", [])

            if prev_end > 0 and line_start > prev_end + 1:
                omitted = line_start - prev_end - 1
                parts.append(f"... ({omitted} lines omitted) ...")

            # Node metadata header
            meta_parts: List[str] = []
            mt = props.get("meta_type", "")
            name = props.get("name") or props.get("label", "")
            if mt and name:
                meta_parts.append(f"{mt}: {name}")
            elif name:
                meta_parts.append(name)
            if props.get("complexity") is not None:
                meta_parts.append(f"complexity={props['complexity']}")
            doc = props.get("docstring")
            if doc:
                doc_short = str(doc)[:80].replace("\n", " ")
                meta_parts.append(f'docstring="{doc_short}"')
            if meta_parts:
                parts.append(f"## {' | '.join(meta_parts)}")

            # Edge metadata
            for edge in edges:
                label = edge.get("edge_label", "")
                direction = edge.get("direction", "")
                conn = edge.get("connected_props", {})
                conn_type = conn.get("meta_type", "?")
                conn_name = (
                    conn.get("name")
                    or conn.get("relative_path")
                    or conn.get("label", "?")
                )
                conn_lines = ""
                ls = conn.get("line_start")
                le = conn.get("line_end")
                if ls is not None and le is not None:
                    conn_lines = f" (L{ls}-{le})"
                conn_complexity = ""
                cc = conn.get("complexity")
                if cc is not None:
                    conn_complexity = f", complexity={cc}"

                if direction == "outgoing":
                    parts.append(
                        f"##   --{label}--> "
                        f"{conn_type}: {conn_name}{conn_lines}{conn_complexity}"
                    )
                else:
                    parts.append(
                        f"##   <--{label}-- "
                        f"{conn_type}: {conn_name}{conn_lines}{conn_complexity}"
                    )

            # Source code with original line numbers
            for offset, line in enumerate(source.splitlines()):
                parts.append(f"{line_start + offset} {line}")

            prev_end = line_end
        parts.append(f"[end of {filename}]")
    return "\n".join(parts)


def assemble_node_source(
    touched: Dict[str, dict],
    repo: str,
    base_commit: str,
    problem_statement: str,
    graph_dir: str,
    config: KGConfig,
) -> str:
    """Pure-graph context: source_code from nodes, edge metadata as padding.

    No file reading for code — all text comes from graph properties.
    READMEs are still read from the repo (one cached checkout).
    """
    enriched, edge_metadata, node_to_file = _collect_graph_context(touched, graph_dir)
    if not enriched or not node_to_file:
        return ""

    nodes_by_file: Dict[str, List[dict]] = {}
    for nid, rel_path in node_to_file.items():
        props = enriched[nid]
        nodes_by_file.setdefault(rel_path, []).append(
            {
                "line_start": props["line_start"],
                "line_end": props["line_end"],
                "source_code": props["source_code"],
                "node_props": props,
                "edges": edge_metadata.get(nid, []),
            }
        )

    sorted_files = sorted(
        nodes_by_file, key=lambda f: len(nodes_by_file[f]), reverse=True
    )

    from .build_graphs import ensure_repo_cloned, _checkout

    repo_dir = ensure_repo_cloned(repo, config)
    _checkout(repo_dir, base_commit)

    readmes = _read_readmes(str(repo_dir))

    snippets: Dict[str, List[dict]] = {}
    total_tokens = 0

    for rel_path in sorted_files:
        nodes = sorted(nodes_by_file[rel_path], key=lambda n: n["line_start"])

        # Deduplicate: skip nodes contained in a larger node
        deduped: List[dict] = []
        for n in nodes:
            contained = any(
                e["line_start"] <= n["line_start"] and e["line_end"] >= n["line_end"]
                for e in deduped
            )
            if not contained:
                deduped = [
                    e
                    for e in deduped
                    if not (
                        n["line_start"] <= e["line_start"]
                        and n["line_end"] >= e["line_end"]
                    )
                ]
                deduped.append(n)

        file_snippets: List[dict] = []
        for n in deduped:
            est_tokens = len(n["source_code"]) // 3
            if total_tokens + est_tokens > config.max_context_tokens and snippets:
                break
            file_snippets.append(n)
            total_tokens += est_tokens

        if file_snippets:
            snippets[rel_path] = file_snippets

    if not snippets:
        return ""

    code_text = _format_node_snippets(snippets)
    return _build_prompt(problem_statement, readmes, code_text)


def dispatch_context_mode(
    touched: Dict[str, dict],
    repo: str,
    base_commit: str,
    problem_statement: str,
    graph_dir: str,
    config: KGConfig,
) -> str:
    """Assemble context based on ``config.context_mode``."""
    mode = config.context_mode
    if mode == "file_snippet":
        return assemble_file_snippet(
            touched,
            repo,
            base_commit,
            problem_statement,
            graph_dir,
            config,
        )
    if mode == "node_source":
        return assemble_node_source(
            touched,
            repo,
            base_commit,
            problem_statement,
            graph_dir,
            config,
        )
    return assemble_file_level(
        touched,
        repo,
        base_commit,
        problem_statement,
        graph_dir,
        config,
    )
