"""Graph-guided repair: SEARCH/REPLACE and node-level code repair.

Two modes:
- ``sr``: pure file-level SEARCH/REPLACE (Agentless-style)
- ``node``: graph-guided node replacement + file-level S&R fallback

Both modes produce unified diffs via :mod:`difflib` — the model never
writes diff syntax directly.
"""

import difflib
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Regex patterns ──────────────────────────────────────────────────

_SR_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)
_NODE_HEADER_RE = re.compile(r"^(.+?)\s+in\s+(.+)$")
_LINE_NUM_RE = re.compile(r"^\d+ ")


def strip_line_numbers(code_text: str) -> str:
    """Remove line number prefixes and graph metadata from code text.

    Strips:
    - Line number prefix (``NN code`` → ``code``) from all context modes
    - Graph metadata headers (``## Function: name``, ``## --CALLS-->``)
      added by node_source mode — these are not part of the actual file
    """
    result: List[str] = []
    for line in code_text.splitlines():
        if _LINE_NUM_RE.match(line):
            result.append(_LINE_NUM_RE.sub("", line, count=1))
        elif line.startswith("## "):
            continue
        else:
            result.append(line)
    return "\n".join(result)


# ── Context truncation guards ───────────────────────────────────────

# Per-file and total budgets for the <code> block sent to the LLM.
# ~360k chars ≈ 90–100k tokens: fits n_ctx=131072 with room for the
# issue statement, prompt scaffolding and the completion itself.
_MAX_FILE_CHARS = 60_000
_MAX_TOTAL_CHARS = 360_000

_TRUNCATED_NOTE = "[... {omitted} characters truncated ...]"
_STUB_NOTE = "[file content omitted to fit context budget]"

_START_OF_RE = re.compile(r"\[start of ([^\]]+)\]")


def truncate_file_sections(
    code_text: str,
    max_file_chars: int = _MAX_FILE_CHARS,
    max_total_chars: int = _MAX_TOTAL_CHARS,
    max_files: Optional[int] = None,
) -> str:
    """Cap ``[start of path]`` ... ``[end of path]`` sections to fit context.

    Guards repair prompts against oversized file contexts (a handful of
    huge files, or a rerank fallback that kept the whole pool):

    - ``max_files``: sections beyond the first N are dropped entirely
      (rerank-skip fallback repair: first sections are the top-ranked
      ones, so this keeps the heuristic top-N selection)
    - ``max_file_chars``: a single oversized file keeps its head and
      tail, with a truncation note in between
    - ``max_total_chars``: budget spent in section order; once
      exhausted, remaining files keep only their markers and a stub
      note (paths stay visible to the model)

    Text outside file sections is passed through unchanged.
    """
    matches = list(_START_OF_RE.finditer(code_text))
    if not matches:
        return code_text

    pieces: List[str] = []
    consumed = 0
    last = 0
    for position, match in enumerate(matches):
        if max_files is not None and position >= max_files:
            # Drop the section: skip to the end of its closing marker.
            end_marker = f"[end of {match.group(1)}]"
            end_pos = code_text.find(end_marker, match.end())
            last = end_pos + len(end_marker) + 1 if end_pos != -1 else len(code_text)
            continue
        pieces.append(code_text[last : match.end()])
        body_start = match.end()
        end_marker = f"[end of {match.group(1)}]"
        end_pos = code_text.find(end_marker, body_start)
        if end_pos == -1:
            # Unterminated section: treat the rest of the text as body.
            body = code_text[body_start:]
            end_marker = ""
        else:
            body = code_text[body_start:end_pos]

        allowance = min(max_file_chars, max_total_chars - consumed)
        if allowance <= len(_STUB_NOTE) + 64:
            new_body = _STUB_NOTE
        elif len(body) > allowance:
            omitted = len(body) - allowance
            half = allowance // 2
            new_body = (
                body[:half]
                + "\n"
                + _TRUNCATED_NOTE.format(omitted=omitted)
                + "\n"
                + body[len(body) - half :]
            )
            consumed += allowance
        else:
            new_body = body
            consumed += len(body)
        pieces.append(new_body)
        pieces.append(end_marker)
        if end_pos != -1:
            last = end_pos + len(end_marker)
        else:
            last = len(code_text)
    pieces.append(code_text[last:])
    return "".join(pieces)


# ── Prompt constants ────────────────────────────────────────────────

_PREMISE = (
    "You will be provided with a partial code base and an issue "
    "statement explaining a problem to resolve."
)

SR_EXAMPLE = """\
### math_utils.py
<<<<<<< SEARCH
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
=======
import math

def gcd(a, b):
    return math.gcd(a, b)
>>>>>>> REPLACE"""

NODE_EXAMPLE = """\
### calculate_gcd in math_utils.py
def calculate_gcd(a, b):
    \"\"\"Calculate GCD using math.gcd.\"\"\"
    return math.gcd(abs(a), abs(b))"""

NODE_SR_EXAMPLE = """\
### math_utils.py
<<<<<<< SEARCH
from typing import Optional
=======
from typing import Optional, List
>>>>>>> REPLACE"""


# ── Shared: diff generation ─────────────────────────────────────────


def generate_unified_diff(
    file_path: str, old_lines: List[str], new_lines: List[str]
) -> str:
    """Generate a unified diff string with ``a/`` / ``b/`` headers."""
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "".join(line + "\n" for line in diff)


def read_file_from_repo(repo_dir: str, rel_path: str) -> Optional[str]:
    """Read file content from repo working tree."""
    full = os.path.join(repo_dir, rel_path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as exc:
        logger.warning("failed to read repo file %s: %s", full, exc)
        return None


def load_graph_data(graph_dir: str) -> dict:
    """Load ``data.json`` from a graph directory."""
    with open(os.path.join(graph_dir, "data.json"), encoding="utf-8") as f:
        return json.load(f)


# ── Section splitting ──────────────────────────────────────────────


def _split_sections(raw: str) -> List[Tuple[str, str]]:
    """Split raw output into ``(header, content)`` pairs by ``###`` headers.

    The header is the text after ``### `` (without the prefix).
    """
    parts = re.split(r"^(###\s+.+)$", raw, flags=re.MULTILINE)
    sections: List[Tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()[4:].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((header, content))
    return sections


# ── sr mode: parsing ────────────────────────────────────────────────


def parse_search_replace_blocks(
    raw: str,
) -> List[Tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks from model output.

    Returns a list of ``(file_path, old_str, new_str)`` tuples.
    """
    edits: List[Tuple[str, str, str]] = []
    for header, content in _split_sections(raw):
        for match in _SR_BLOCK_RE.finditer(content):
            edits.append((header, match.group(1), match.group(2)))
    return edits


# ── sr mode: application ────────────────────────────────────────────


def apply_search_replace(
    file_lines: List[str], old_str: str, new_str: str
) -> Optional[List[str]]:
    """Apply a SEARCH/REPLACE edit to ``file_lines``.

    Returns the new list of lines if the replacement is successful,
    or ``None`` if ``old_str`` is not found or is not unique.
    """
    old_lines = old_str.splitlines()
    replacement_lines = new_str.splitlines()
    n = len(old_lines)

    if n == 0:
        return None

    positions: List[int] = []
    for i in range(len(file_lines) - n + 1):
        if file_lines[i : i + n] == old_lines:
            positions.append(i)

    if len(positions) == 0:
        return None
    if len(positions) > 1:
        return None

    pos = positions[0]
    return file_lines[:pos] + replacement_lines + file_lines[pos + n :]


# ── sr mode: prompt ─────────────────────────────────────────────────


def build_sr_prompt(problem_statement: str, code_text: str) -> str:
    """Build a SEARCH/REPLACE repair prompt.

    ``code_text`` should be the content between ``<code>`` and
    ``</code>`` tags from the original prompt (includes readmes + code).
    Line numbers and graph metadata are stripped so the model can
    reproduce code verbatim in SEARCH blocks.
    """
    clean_code = strip_line_numbers(code_text)
    instruction = (
        "I need you to solve the provided issue. For each file that "
        "needs changes, start with a line beginning with ### followed "
        "by the file path exactly as shown in the [start of ...] "
        "markers, then provide one or more search and replace blocks. "
        "The SEARCH section must exactly match existing code in the "
        "file. A single response can contain changes to multiple files."
    )

    parts = [
        _PREMISE,
        "<issue>",
        problem_statement,
        "</issue>",
        "",
        "<code>",
        clean_code,
        "</code>",
        "",
        "Here is an example of a search and replace block:",
        "<example>",
        SR_EXAMPLE,
        "</example>",
        "",
        instruction,
        "Respond below:",
    ]
    return "\n".join(parts)


# ── node mode: parsing ──────────────────────────────────────────────


def parse_node_edits(
    raw: str,
) -> List[Tuple[str, str, str]]:
    """Parse node-level edits from model output.

    Returns a list of ``(node_name, file_path, new_code)`` tuples.
    Only parses sections that are NOT SEARCH/REPLACE blocks.
    """
    edits: List[Tuple[str, str, str]] = []
    for header, content in _split_sections(raw):
        if "<<<<<<< SEARCH" in content:
            continue
        match = _NODE_HEADER_RE.match(header)
        if match:
            name = match.group(1).strip()
            file_path = match.group(2).strip()
            edits.append((name, file_path, content))
    return edits


def parse_repair_output(
    raw: str,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Parse both node edits and S&R blocks from model output.

    Returns ``(node_edits, sr_edits)``.
    """
    return parse_node_edits(raw), parse_search_replace_blocks(raw)


# ── node mode: graph index ──────────────────────────────────────────


def build_node_index(graph_data: dict) -> Dict[str, dict]:
    """Build a ``name|file_path`` lookup index from raw graph data.

    Only Function/Method/Class nodes with ``source_code`` and
    ``line_start`` are included. File path is resolved via CONTAINS
    edge traversal (handles multi-hop: Method -> Class -> File).
    """
    all_values = graph_data.get("allValues", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    file_paths: Dict[str, str] = {}
    for node in nodes:
        nid = node["id"]
        props = all_values.get(nid, {})
        if props.get("meta_type") == "File":
            rp = props.get("relative_path", "")
            if rp:
                file_paths[nid] = rp

    parent_map: Dict[str, str] = {}
    for edge in edges:
        if edge.get("label") == "CONTAINS":
            parent_map[edge["target"]] = edge["source"]

    def resolve_file(nid: str) -> Optional[str]:
        current: Optional[str] = nid
        visited: set = set()
        while current is not None and current not in visited:
            visited.add(current)
            if current in file_paths:
                return file_paths[current]
            current = parent_map.get(current)
        return None

    index: Dict[str, dict] = {}
    for node in nodes:
        nid = node["id"]
        props = all_values.get(nid, {})
        meta_type = props.get("meta_type", "")
        if meta_type not in ("Function", "Method", "Class"):
            continue
        source_code = props.get("source_code")
        line_start = props.get("line_start")
        line_end = props.get("line_end")
        name = props.get("name") or node.get("label", "")
        if not source_code or line_start is None:
            continue

        file_path = resolve_file(nid)
        if not file_path:
            continue

        key = f"{name}|{file_path}"
        index[key] = {
            "source_code": source_code,
            "line_start": line_start,
            "line_end": line_end,
            "file_path": file_path,
            "node_id": nid,
        }

    return index


# ── node mode: application ──────────────────────────────────────────


def verify_and_replace_node(
    file_lines: List[str],
    node_info: dict,
    new_code: str,
) -> Optional[List[str]]:
    """Verify graph source matches file content, then replace lines.

    Returns the new list of lines if successful, or ``None`` on
    source drift (graph ``source_code`` does not match the file at
    the expected line range).
    """
    start = node_info["line_start"]
    end = node_info["line_end"]
    source_lines = node_info["source_code"].splitlines()

    actual = file_lines[start - 1 : end]
    if actual != source_lines:
        return None

    new_lines = new_code.splitlines()
    return file_lines[: start - 1] + new_lines + file_lines[end:]


# ── node mode: prompt ───────────────────────────────────────────────


def build_node_prompt(problem_statement: str, code_text: str) -> str:
    """Build a node-level repair prompt with both formats.

    ``code_text`` should be the content between ``<code>`` and
    ``</code>`` tags from the original prompt. Line numbers and
    graph metadata are stripped.
    """
    clean_code = strip_line_numbers(code_text)
    instruction = (
        "I need you to solve the provided issue by modifying the "
        "code. You can use two formats:\n\n"
        "1. To replace an entire function, method, or class, provide "
        "the node name and file path (exactly as shown in the "
        "[start of ...] markers), followed by the complete new code:\n\n"
        "### function_name in path/to/file.py\n"
        "{complete new code}\n\n"
        "2. For smaller edits such as imports or module-level changes, "
        "use search and replace:\n\n"
        "### path/to/file.py\n"
        "<<<<<<< SEARCH\n"
        "{exact original code}\n"
        "=======\n"
        "{new code}\n"
        ">>>>>>> REPLACE"
    )

    parts = [
        _PREMISE,
        "<issue>",
        problem_statement,
        "</issue>",
        "",
        "<code>",
        clean_code,
        "</code>",
        "",
        "Example (format 1 — replace a function):",
        NODE_EXAMPLE,
        "",
        "Example (format 2 — search and replace):",
        NODE_SR_EXAMPLE,
        "",
        instruction,
        "Respond below:",
    ]
    return "\n".join(parts)


# ── Patch synthesis ─────────────────────────────────────────────────


def synthesize_sr_patch(
    repo_dir: str,
    sr_edits: List[Tuple[str, str, str]],
) -> str:
    """Apply S&R edits to files and produce a unified diff.

    Returns a concatenated diff string for all changed files.
    Edits that fail (not found, not unique) are skipped with a warning.
    """
    by_file: Dict[str, List[Tuple[str, str]]] = {}
    for file_path, old_str, new_str in sr_edits:
        by_file.setdefault(file_path, []).append((old_str, new_str))

    patches: List[str] = []
    for file_path, edits in sorted(by_file.items()):
        content = read_file_from_repo(repo_dir, file_path)
        if content is None:
            logger.warning("File not found: %s", file_path)
            continue

        old_lines = content.splitlines()
        new_lines = list(old_lines)

        for old_str, new_str in edits:
            result = apply_search_replace(new_lines, old_str, new_str)
            if result is None:
                logger.warning(
                    "S&R failed for %s (not found or not unique)",
                    file_path,
                )
                continue
            new_lines = result

        if new_lines != old_lines:
            diff = generate_unified_diff(file_path, old_lines, new_lines)
            patches.append(diff)

    return "".join(patches)


def synthesize_node_patch(
    repo_dir: str,
    node_index: Dict[str, dict],
    node_edits: List[Tuple[str, str, str]],
    sr_edits: List[Tuple[str, str, str]],
) -> str:
    """Apply node edits + S&R fallback edits, produce unified diff.

    Returns a concatenated diff string for all changed files.
    Node edits that fail (not in graph, source drift) and S&R edits
    that fail (not found, not unique) are skipped with a warning.
    """
    all_files: set = set()
    for _, file_path, _ in node_edits:
        all_files.add(file_path)
    for file_path, _, _ in sr_edits:
        all_files.add(file_path)

    patches: List[str] = []
    for file_path in sorted(all_files):
        content = read_file_from_repo(repo_dir, file_path)
        if content is None:
            logger.warning("File not found: %s", file_path)
            continue

        old_lines = content.splitlines()
        new_lines = list(old_lines)

        for node_name, node_file, new_code in node_edits:
            if node_file != file_path:
                continue
            key = f"{node_name}|{file_path}"
            node_info = node_index.get(key)
            if node_info is None:
                logger.warning(
                    "Node not found in graph: %s in %s",
                    node_name,
                    file_path,
                )
                continue
            result = verify_and_replace_node(new_lines, node_info, new_code)
            if result is None:
                logger.warning(
                    "Source mismatch for %s in %s, skipping",
                    node_name,
                    file_path,
                )
                continue
            new_lines = result

        for sr_file, old_str, new_str in sr_edits:
            if sr_file != file_path:
                continue
            result = apply_search_replace(new_lines, old_str, new_str)
            if result is None:
                logger.warning(
                    "S&R failed for %s (not found or not unique)",
                    file_path,
                )
                continue
            new_lines = result

        if new_lines != old_lines:
            diff = generate_unified_diff(file_path, old_lines, new_lines)
            patches.append(diff)

    return "".join(patches)
