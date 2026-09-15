#!/usr/bin/env python3
"""Re-rank KG-surfaced files for each SWE-bench instance with an LLM comparator.

Pipeline (KG file_level + sr context):
  1. Take all touched files per instance (``[start of <path>]`` markers in
     ``text_inputs``), together with their contents.
  2. Ask an LLM: given the issue description and two files (path + content),
     which file is more important for solving the issue?
  3. Sort the files with Python's standard ``sorted`` using the LLM as the
     comparator (``functools.cmp_to_key``); each unordered pair is asked at
     most once (pair cache).
  4. Evaluate the new ranking with the same metrics as
     compare-file-localization.py (Recall@1/3/5, Cov@5, KG ceiling) and
     compare against BM25, Agentless, and the original KG order.

Progress is checkpointed per instance to a JSONL file; re-running skips
already-processed instances.

Usage:
    python scripts/llm-rerank-files.py --split dev [--limit N] [--dry-run]
    python scripts/llm-rerank-files.py --split dev --eval-only
    python scripts/llm-rerank-files.py --split dev --csv outputs/dev-rerank.csv
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import re
import sys
from functools import cmp_to_key
from pathlib import Path

# Reuse loaders/metrics from the sibling comparison script.
_CFL_PATH = Path(__file__).resolve().parent / "compare-file-localization.py"
_cfl_spec = importlib.util.spec_from_file_location("cfl", _CFL_PATH)
cfl = importlib.util.module_from_spec(_cfl_spec)
_cfl_spec.loader.exec_module(cfl)

# Make src/kg importable for list-mode signal assembly (kg is imported
# lazily inside the assembly functions, so pairwise/eval-only paths never
# require chromadb/openai).
_SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_START_MARKER = re.compile(r"^\[start of (.+?)\]\s*$", re.MULTILINE)
_END_MARKER_TMPL = r"^\[end of {path}\]\s*$"
_DEFAULT_MAX_FILE_LINES = 400
# Thinking models spend completion budget on reasoning_content before the
# one-token FILE_A/FILE_B answer, so the cap must leave room for reasoning.
_MAX_COMPLETION_TOKENS = 2048
# Per-request thinking disable (llama.cpp chat-template kwarg; verified on
# gemma-4-26B-A4B-it-Q8_0 — drops comparator latency from ~85s to ~8s).
_NO_THINK_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
_FALLBACK_MODEL_LABEL = "model"


# ---------------------------------------------------------------------------
# Extraction: text_inputs -> [(path, content), ...]
# ---------------------------------------------------------------------------
def extract_files_with_content(text: str) -> list[tuple[str, str]]:
    """Extract (path, content) pairs from KG context markers.

    Files are returned in exploration order; READMEs are skipped.  A file
    whose ``[end of ...]`` marker is missing yields empty content.
    """
    out: list[tuple[str, str]] = []
    if not text:
        return out
    for match in _START_MARKER.finditer(text):
        path = match.group(1)
        if cfl._is_readme(path):
            continue
        end_pattern = re.compile(
            _END_MARKER_TMPL.format(path=re.escape(path)), re.MULTILINE
        )
        end_match = end_pattern.search(text, match.end())
        content = text[match.end() : end_match.start()].strip("\n") if end_match else ""
        out.append((path, content))
    return out


def truncate_content(content: str, max_lines: int) -> str:
    """Keep at most ``max_lines`` leading lines of ``content``."""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content
    return "\n".join(lines[:max_lines])


# ---------------------------------------------------------------------------
# LLM comparator
# ---------------------------------------------------------------------------
class ComparatorError(Exception):
    """Raised when the LLM never returns a parseable choice after retries."""


def parse_comparator_choice(text) -> int | None:
    """Map an LLM answer to a cmp-style result: -1 (A), 1 (B), None (garbage)."""
    if not text or not isinstance(text, str):
        return None
    stripped = text.strip().lower()
    if stripped == "a":
        return -1
    if stripped == "b":
        return 1
    has_a = "file_a" in stripped
    has_b = "file_b" in stripped
    if has_a and has_b:
        return None
    if has_a:
        return -1
    if has_b:
        return 1
    return None


_COMPARATOR_PROMPT = """\
You are given an issue description and two source files. Decide which file \
is MORE IMPORTANT for understanding and solving the issue.

Rules:
- If neither file seems relevant, still choose the one MORE RELATED to the issue.
- Never explain. Respond with EXACTLY ONE token: FILE_A or FILE_B.

ISSUE:
{issue}

FILE_A: {path_a}
{content_a}

FILE_B: {path_b}
{content_b}

Answer format examples:
Correct: FILE_A
Correct: FILE_B
Incorrect: "Neither file directly addresses the issue" (never refuse to choose)
Incorrect: "FILE_A because it handles parsing" (no explanations)
Incorrect: "FILE_A FILE_B" (exactly one token)

Which file is more important for solving the issue? One token only: \
FILE_A or FILE_B.

WARNING: Your ENTIRE response must be exactly "FILE_A" or "FILE_B" — \
one token, nothing else. Any other text is an invalid answer."""


class LLMComparator:
    """cmp(a, b)-style comparator backed by an LLM, with a pair cache."""

    def __init__(
        self,
        client,
        model: str,
        retries: int = 3,
        max_file_lines: int = _DEFAULT_MAX_FILE_LINES,
    ):
        self.client = client
        self.model = model
        self.retries = retries
        self.max_file_lines = max_file_lines
        self.n_llm_calls = 0
        self._cache: dict[tuple[str, str], int] = {}

    def _ask(self, prompt: str) -> int:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            self.n_llm_calls += 1
            if attempt == 0:
                # Plain attempt: deterministic, thinking left as deployed.
                extra_body = None
                temperature = 0.0
            elif attempt == 1:
                # Fallback 1: disable thinking — reasoning chains can
                # exhaust the completion budget before the answer token.
                extra_body = _NO_THINK_BODY
                temperature = 0.0
            else:
                # Fallback 2: thinking off + resample at higher temperature.
                extra_body = _NO_THINK_BODY
                temperature = 0.7
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": _MAX_COMPLETION_TOKENS,
            }
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            try:
                resp = self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                continue
            choice = parse_comparator_choice(resp.choices[0].message.content)
            if choice is not None:
                return choice
        detail = f" (last error: {last_error})" if last_error else ""
        raise ComparatorError(
            f"no parseable choice after {self.retries} attempts{detail}"
        )

    def compare(
        self, issue: str, file_a: tuple[str, str], file_b: tuple[str, str]
    ) -> int:
        path_a, path_b = file_a[0], file_b[0]
        if path_a == path_b:
            return 0
        key = (path_a, path_b) if path_a <= path_b else (path_b, path_a)
        if key in self._cache:
            cached = self._cache[key]
            return cached if key == (path_a, path_b) else -cached
        prompt = _COMPARATOR_PROMPT.format(
            issue=issue,
            path_a=path_a,
            content_a=truncate_content(file_a[1], self.max_file_lines),
            path_b=path_b,
            content_b=truncate_content(file_b[1], self.max_file_lines),
        )
        choice = self._ask(prompt)
        self._cache[key] = choice if key == (path_a, path_b) else -choice
        return choice


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_files(issue: str, files: list[tuple[str, str]], comparator) -> dict:
    """Sort files best-first with the comparator; on ComparatorError keep
    the original order and mark the instance failed."""
    if len(files) <= 1:
        return {"ranked": [f[0] for f in files], "failed": False, "n_comparisons": 0}

    counter = {"n": 0}

    def cmp(file_a, file_b) -> int:
        counter["n"] += 1
        return comparator.compare(issue, file_a, file_b)

    try:
        ordered = sorted(files, key=cmp_to_key(cmp))
    except ComparatorError:
        return {
            "ranked": [f[0] for f in files],
            "failed": True,
            "n_comparisons": counter["n"],
        }
    return {
        "ranked": [f[0] for f in ordered],
        "failed": False,
        "n_comparisons": counter["n"],
    }


def estimate_comparisons(n_files: int) -> int:
    """Rough comparison count for sorting n items (merge-sort bound)."""
    if n_files <= 1:
        return 0
    return max(1, int(n_files * math.log2(n_files)) - n_files + 1)


# ---------------------------------------------------------------------------
# List mode (--mode list): 1-call listwise ranking over per-file signals
# ---------------------------------------------------------------------------
_VALID_SIGNAL_TOKENS = ("freq", "sim", "sim_max", "sim_mean", "parts")
_SIGNAL_ORDER = ("freq", "sim_max", "sim_mean", "parts")


def parse_signals(spec: str) -> list[str]:
    """Parse a comma-separated signal spec into normalized signal names.

    ``sim`` expands to both aggregations (``sim_max`` + ``sim_mean``);
    ``sim_max`` / ``sim_mean`` select them individually.  Result is
    ordered canonically (freq, sim_max, sim_mean, parts) so that
    equivalent specs map to identical filenames/labels.
    """
    tokens = [token.strip() for token in (spec or "").split(",") if token.strip()]
    if not tokens:
        raise ValueError("empty --signals spec")
    expanded: set[str] = set()
    for token in tokens:
        if token not in _VALID_SIGNAL_TOKENS:
            raise ValueError(
                f"unknown signal '{token}' (valid: {', '.join(_VALID_SIGNAL_TOKENS)})"
            )
        if token == "sim":
            expanded.update(("sim_max", "sim_mean"))
        else:
            expanded.add(token)
    return [name for name in _SIGNAL_ORDER if name in expanded]


def signals_tag(signals: list[str]) -> str:
    """Filesystem-safe tag for a signal subset (filenames / eval labels)."""
    return (
        "-".join(name.replace("_", "") for name in _SIGNAL_ORDER if name in signals)
        or "none"
    )


def graph_dir_for_row(row: dict, graphs_dir: str) -> str:
    """Graph cache dir for a progress row (mirrors build_graphs._graph_key)."""
    repo_key = row.get("repo", "").replace("/", "__")
    return str(Path(graphs_dir) / f"{repo_key}__{row.get('base_commit', '')}")


def _node_sims_from_chroma(problem_statement: str, graph_dir: str, defaults: dict):
    """Map touched-node-agnostic top node ids to similarity vs the issue.

    Returns {} when the embedding index or server is unavailable — sim
    fields are then simply omitted from the prompt.
    """
    from kg.config import KGConfig
    from kg.embeddings import get_top_k_for_issue

    embedding_config = KGConfig(
        api_key=defaults.get("api_key", "any"),
        embedding_api_base=defaults.get(
            "embedding_api_base", "http://localhost:8082/v1"
        ),
        embedding_model=defaults.get("embedding_model", "nomic-embed-code-Q8_0"),
    )
    seeds = get_top_k_for_issue(problem_statement, graph_dir, embedding_config, k=500)
    return {seed["id"]: seed["similarity"] for seed in seeds}


def assemble_file_signals(
    row: dict,
    graph_dir: str,
    subparts: int = 3,
    src_chars: int = 120,
    sim_provider=None,
) -> list[dict]:
    """Assemble per-file signals (freq / sims / touched parts) for one row.

    ``sim_provider(problem_statement, graph_dir) -> {node_id: similarity}``
    supplies node similarities; ``None`` (or a provider returning {} or
    raising) means the sim fields are omitted.  Files are returned sorted
    by touch frequency, descending.
    """
    import json as _json

    from kg.embeddings import resolve_relative_paths

    touched_ids: list[str] = row.get("touched_nodes", [])
    node_paths = resolve_relative_paths(graph_dir)
    data_path = Path(graph_dir) / "data.json"
    all_values: dict = {}
    if data_path.exists():
        with open(data_path) as f:
            all_values = _json.load(f).get("allValues", {})

    # Touch frequency: every touched node votes for its containing file
    freq: dict[str, int] = {}
    for node_id in touched_ids:
        path = node_paths.get(node_id, "")
        if path.endswith(".py"):
            freq[path] = freq.get(path, 0) + 1

    # Node similarities (optional signal)
    node_sims: dict = {}
    if sim_provider is not None:
        try:
            node_sims = sim_provider(row.get("problem_statement", ""), graph_dir) or {}
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "sim provider failed for %s — sim signals omitted: %s",
                row.get("instance_id", "<unknown>"),
                exc,
            )
            node_sims = {}

    def sims_for(path: str) -> dict:
        values = [
            node_sims[node_id]
            for node_id in touched_ids
            if node_id in node_sims and node_paths.get(node_id, "") == path
        ]
        if not values:
            return {}
        top = sorted(values, reverse=True)[:3]
        return {
            "sim_max": max(values),
            "sim_mean": round(sum(top) / len(top), 4),
        }

    def parts_for(path: str) -> list[dict]:
        parts = []
        for node_id in touched_ids:  # exploration order
            if node_paths.get(node_id, "") != path:
                continue
            props = all_values.get(node_id, {})
            if not props:
                continue
            source = (props.get("source_code") or "").strip()
            if src_chars:
                source = source[:src_chars]
            parts.append(
                {
                    "meta_type": props.get("meta_type", ""),
                    "name": props.get("name") or node_id.rsplit("-", 1)[-1],
                    "docstring": (props.get("docstring") or "").strip().split("\n")[0],
                    "line_start": props.get("line_start"),
                    "line_end": props.get("line_end"),
                    "source": source.replace("\n", " | "),
                }
            )
            if len(parts) >= max(1, subparts):
                break
        return parts

    paths_sorted = sorted(freq, key=lambda p: freq[p], reverse=True)
    return [
        {
            "path": path,
            "touch_freq": freq[path],
            **sims_for(path),
            "parts": parts_for(path),
        }
        for path in paths_sorted
    ]


_LIST_PROMPT_TEMPLATE = """\
You are localizing a bug fix in a repository. Given the issue and the
candidate files surfaced by knowledge-graph exploration, rank the files
by how likely the fix must be made in them.

Rules:
- Use the per-file signals and the touched code parts as evidence.
- Return ONLY the top {top_k} files, most relevant first.
- Answer with a strict JSON array of file paths from the candidate
  list. No explanations, no other text.

ISSUE:
{issue}

CANDIDATE FILES ({n_files} total, ordered by touch frequency):

{file_blocks}

Answer: a JSON array of the top {top_k} file paths, e.g. \
["path/a.py", "path/b.py"]"""


def _render_file_block(file_signal: dict, signals: list[str]) -> str:
    """Render one candidate's signal block for the listwise prompt."""
    lines = [f"### {file_signal['path']}"]
    header = []
    if "freq" in signals:
        header.append(f"touch_freq: {file_signal.get('touch_freq', 0)}")
    if "sim_max" in signals and "sim_max" in file_signal:
        header.append(f"sim_max: {file_signal['sim_max']}")
    if "sim_mean" in signals and "sim_mean" in file_signal:
        header.append(f"sim_top3_mean: {file_signal['sim_mean']}")
    if header:
        lines.append(" | ".join(header))
    if "parts" in signals:
        for part in file_signal.get("parts", []):
            part_line = (
                f"  - {part['meta_type']} {part['name']} "
                f"(L{part['line_start']}-{part['line_end']})"
            )
            if part["docstring"]:
                part_line += f' "{part["docstring"]}"'
            lines.append(part_line)
            if part["source"]:
                lines.append(f"    src: {part['source']}")
    return "\n".join(lines)


def build_list_prompt(
    issue: str, file_signals: list[dict], signals: list[str], top_k: int
) -> str:
    """Render the listwise ranking prompt from per-file signal dicts."""
    blocks = [_render_file_block(fs, signals) for fs in file_signals]
    return _LIST_PROMPT_TEMPLATE.format(
        top_k=top_k,
        issue=issue,
        n_files=len(file_signals),
        file_blocks="\n\n".join(blocks),
    )


def parse_list_answer(text) -> list[str] | None:
    """Extract a JSON array of file paths from an LLM answer.

    Accepts raw JSON, markdown-fenced JSON, or JSON embedded in prose.
    Returns None when no valid array of strings is present.
    """
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        return None
    return parsed


class ListRanker:
    """Single-call listwise ranker backed by an LLM, pairwise-style retries."""

    def __init__(self, client, model: str, retries: int = 3):
        self.client = client
        self.model = model
        self.retries = retries
        self.n_llm_calls = 0

    def ask_ranking(self, prompt: str) -> list[str] | None:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            self.n_llm_calls += 1
            if attempt == 0:
                extra_body = None
                temperature = 0.0
            elif attempt == 1:
                extra_body = _NO_THINK_BODY
                temperature = 0.0
            else:
                extra_body = _NO_THINK_BODY
                temperature = 0.7
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": _MAX_COMPLETION_TOKENS,
            }
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            try:
                resp = self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                continue
            parsed = parse_list_answer(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        del last_error  # parity with comparator; errors surface as failure
        return None


def _chunk_prompt_signals(
    issue: str,
    prompt_signals: list[dict],
    signals: list[str],
    top_k: int,
    max_prompt_chars: int,
) -> list[list[dict]]:
    """Split freq-ordered signals into chunks whose prompts fit the cap.

    Greedy packing in frequency order: a chunk closes when the next
    block would push its rendered prompt past ``max_prompt_chars``.
    Every chunk keeps at least one file, so a single huge block still
    gets ranked on its own.
    """
    overhead = len(build_list_prompt(issue, [], signals, top_k))
    chunks: list[list[dict]] = []
    current: list[dict] = []
    used = overhead
    for file_signal in prompt_signals:
        block_len = len(_render_file_block(file_signal, signals)) + 2  # "\n\n"
        if current and used + block_len > max_prompt_chars:
            chunks.append(current)
            current = []
            used = overhead
        current.append(file_signal)
        used += block_len
    if current:
        chunks.append(current)
    return chunks


def rank_files_list(
    issue: str,
    files: list[tuple[str, str]],
    file_signals: list[dict],
    top_k: int = 5,
    engine=None,
    signals: list[str] | None = None,
    max_prompt_chars: int = 0,
) -> dict:
    """One-shot listwise ranking; on failure keep frequency order.

    ``files`` are the candidate (path, content) pairs from the KG
    context; ``file_signals`` may cover a superset of their paths.
    Unknown paths in the LLM answer are dropped; the frequency tail
    pads the ranking so every candidate stays represented.

    When ``max_prompt_chars > 0`` and the full candidate list would
    exceed the rerank model's context, candidates are ranked chunk by
    chunk (frequency order preserved) and the chunk winners merge
    round-robin before the frequency tail pads the rest. A failed
    chunk only demotes its own files; the ranking only fails when
    every chunk fails.
    """
    if len(files) <= 1:
        return {
            "ranked": [path for path, _ in files],
            "failed": False,
            "n_calls": 0,
        }
    if engine is None:
        raise ValueError("engine is required for list ranking")

    signals = signals or list(_SIGNAL_ORDER)
    signals_by_path = {fs["path"]: fs for fs in file_signals}
    freq_order = [
        path
        for path, _ in sorted(
            files,
            key=lambda f: -signals_by_path.get(f[0], {}).get("touch_freq", 0),
        )
    ]
    prompt_signals = [
        signals_by_path.get(path, {"path": path, "touch_freq": 0})
        for path in freq_order
    ]

    chunks = (
        _chunk_prompt_signals(issue, prompt_signals, signals, top_k, max_prompt_chars)
        if max_prompt_chars > 0
        else [prompt_signals]
    )

    winner_lists = []
    for chunk in chunks:
        answer = engine.ask_ranking(build_list_prompt(issue, chunk, signals, top_k))
        if answer is None:
            continue
        chunk_paths = {fs["path"] for fs in chunk}
        winners = []
        for path in answer:
            if path in chunk_paths and path not in winners:
                winners.append(path)
        winner_lists.append(winners)

    if not winner_lists:
        return {"ranked": freq_order, "failed": True, "n_calls": engine.n_llm_calls}

    # Round-robin merge of chunk winner lists (chunks follow freq order),
    # then the frequency tail pads the ranking so every candidate stays
    # represented.
    ranked = []
    for position in range(max(len(winners) for winners in winner_lists)):
        for winners in winner_lists:
            if position < len(winners) and winners[position] not in ranked:
                ranked.append(winners[position])
    ranked += [path for path in freq_order if path not in set(ranked)]
    return {
        "ranked": ranked,
        "failed": False,
        "n_calls": engine.n_llm_calls,
    }


def list_eval_method(kg_method: str, signals: list[str]) -> str:
    """Eval summary label for a list-mode rerank result."""
    return f"{kg_method}_llm_list__{signals_tag(signals)}"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_rerank(path: str) -> dict[str, list[str]]:
    """Load progress JSONL -> {instance_id: ranked_files}."""
    out: dict[str, list[str]] = {}
    progress = Path(path)
    if not progress.exists():
        return out
    with open(progress) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["instance_id"]] = row.get("ranked_files", [])
    return out


def load_rerank_models(path: str) -> set[str]:
    """Collect the distinct ``model`` values recorded in a progress file.

    Rows written before model provenance was tracked have no ``model``
    field and are not represented in the result.
    """
    models: set[str] = set()
    progress = Path(path)
    if not progress.exists():
        return models
    with open(progress) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model"):
                models.add(row["model"])
    return models


def _sanitize_model_name(model: str) -> str:
    """Make a model id filesystem-safe (e.g. ``org/name`` -> ``org-name``)."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model or "")
    return sanitized or _FALLBACK_MODEL_LABEL


def filter_rows_by_instance(rows: list[dict], instance_id: str | None) -> list[dict]:
    """Keep only ``instance_id`` (debugging aid for --instance); None = all."""
    if not instance_id:
        return rows
    return [r for r in rows if r.get("instance_id") == instance_id]


def select_rankable_rows(rows: list[dict]) -> list[dict]:
    """Completed rows only, one per instance (last attempt wins).

    KG progress files can contain aborted restart attempts (empty
    ``text_inputs`` / ``touched_nodes``) before the successful row for
    the same instance; those must never reach ranking or prompt preview.
    """
    completed: dict[str, dict] = {}
    for row in rows:
        if row.get("text_inputs"):
            completed[row["instance_id"]] = row
    return list(completed.values())


def default_progress_path(
    split: str,
    model: str,
    variant: str = "plain",
    mode: str = "pairwise",
    signals: list[str] | None = None,
) -> str:
    """Default progress file per split AND model (avoids cross-model
    checkpoint collisions when several local models share the endpoint).

    List mode additionally encodes the signal combo so ablation runs
    never share checkpoints (e.g. ``...__list__freq-parts__...``).
    """
    variant_part = "" if variant == "plain" else f"__{variant}"
    mode_part = ""
    if mode == "list":
        mode_part = f"__list__{signals_tag(signals or list(_SIGNAL_ORDER))}"
    return str(
        Path(__file__).resolve().parent.parent
        / "outputs"
        / f"llm_rerank__file_level{variant_part}{mode_part}__{split}__{_sanitize_model_name(model)}"
        ".progress.jsonl"
    )


def resolve_kg_input_path(explicit: str | None, variant: str) -> str:
    """KG progress JSONL input path: explicit override or the cfl default.

    ``explicit`` comes from ``--kg-progress-file`` (e.g. a model-separated
    tree like ``outputs/gemma26b/...`` produced by a config with its own
    ``output_dir``); when absent, fall back to the canonical path for the
    variant in the shared ``outputs/`` tree.
    """
    if explicit:
        return explicit
    return cfl.kg_progress_path("file_level", variant)


def _rank_pairwise(row: dict, files: list[tuple[str, str]], engine) -> dict:
    """Default rank_fn: pairwise comparator sort (original behavior)."""
    return rank_files(row.get("problem_statement", ""), files, engine)


def process_instances(
    rows: list[dict],
    comparator_factory,
    progress_path: str,
    limit: int = 0,
    dry_run: bool = False,
    model: str = "",
    rank_fn=None,
    spec: dict | None = None,
    calls_per_instance: int | None = None,
) -> dict:
    """Rank files per instance, checkpointing each result to ``progress_path``.

    Already-processed instances (present in the progress file) are skipped.
    ``limit`` caps how many NEW instances are processed this run (0 = all).
    ``model`` is recorded in each progress row for provenance.
    ``rank_fn(row, files, engine)`` defaults to the pairwise comparator;
    list mode passes a listwise closure (engine = ListRanker).
    ``spec`` (e.g. list mode's mode/signals/top_k) is merged into every
    progress row so results are self-describing.
    ``calls_per_instance`` overrides the pairwise comparison estimate in
    dry-run stats (list mode: exactly 1).
    Returns run statistics.
    """
    rank_fn = rank_fn or _rank_pairwise
    done = load_rerank(progress_path)
    to_process = [r for r in rows if r.get("instance_id") not in done]
    if calls_per_instance is not None:
        estimated = len(to_process) * calls_per_instance
    else:
        estimated = sum(
            estimate_comparisons(
                len(extract_files_with_content(r.get("text_inputs", "")))
            )
            for r in to_process
        )
    stats = {
        "instances": len(to_process),
        "skipped": len(rows) - len(to_process),
        "processed": 0,
        "failed": 0,
        "comparisons": 0,
        "estimated_comparisons": estimated,
    }
    if dry_run:
        return stats
    if limit > 0:
        to_process = to_process[:limit]

    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with open(progress, "a") as out:
        for row in to_process:
            files = extract_files_with_content(row.get("text_inputs", ""))
            result = rank_fn(row, files, comparator_factory())
            record = {
                "instance_id": row["instance_id"],
                "model": model,
                "ranked_files": result["ranked"],
                "failed": result["failed"],
                "n_files": len(files),
                "n_comparisons": result.get("n_comparisons", result.get("n_calls", 0)),
            }
            if spec:
                record.update(spec)
            out.write(json.dumps(record) + "\n")
            out.flush()
            stats["processed"] += 1
            stats["comparisons"] += record["n_comparisons"]
            if result["failed"]:
                stats["failed"] += 1
            status = "FAILED (kept original order)" if result["failed"] else "ok"
            print(
                f"[{row['instance_id']}] {status}: "
                f"{record['n_comparisons']} comparisons, "
                f"{len(result['ranked'])} files",
                flush=True,
            )
    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def print_summary(summary: dict[str, dict[str, float]], n_instances: int):
    # Long list-mode labels (e.g. kg_..._llm_list__freq-simmax-simmean-parts)
    # need a wider method column than the pairwise-era fixed width.
    width = max(26, max((len(m) for m in summary), default=0) + 1)
    header = (
        f"{'Method':<{width}} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} "
        f"{'Cov@5':>7} {'Ceil':>9}"
    )
    print(f"\nFile-localization comparison ({n_instances} instances)\n")
    print(header)
    print("-" * len(header))
    for method in summary:
        s = summary[method]
        ceil = f"{s['unbounded_recall']:.1%}" if method.startswith("kg_") else "—"
        print(
            f"{method:<{width}} {s['recall@1']:>8.1%} {s['recall@3']:>8.1%} "
            f"{s['recall@5']:>8.1%} {s['cov@5']:>6.1%} {ceil:>9}"
        )
    print()
    print("Ceil = unbounded recall over ALL surfaced files (non-comparable reference)")


# ---------------------------------------------------------------------------
# Config / client
# ---------------------------------------------------------------------------
def _load_defaults(config_path: str | None = None) -> dict:
    path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent.parent / "config" / "kg_config.json"
    )
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def build_client(api_base: str, api_key: str, timeout: int):
    import openai

    return openai.OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Re-rank KG-surfaced files with an LLM comparator and "
        "re-evaluate localization metrics"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="SWE-bench Lite split (default: dev)",
    )
    parser.add_argument(
        "--variant",
        choices=["plain", "embed"],
        default="plain",
        help="KG run variant to re-rank (default: plain; "
        "embed = embedding-augmented progress)",
    )
    parser.add_argument(
        "--mode",
        choices=["pairwise", "list"],
        default="pairwise",
        help="Rerank strategy: pairwise comparator sort (default) or a "
        "single listwise call over per-file signals",
    )
    parser.add_argument(
        "--signals",
        default="freq,sim,parts",
        help="List-mode signals, comma-separated subset of "
        "freq,sim,sim_max,sim_mean,parts; 'sim' = sim_max+sim_mean "
        "(default: freq,sim,parts)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="List-mode answer size, matches Recall@k windows (default: 5)",
    )
    parser.add_argument(
        "--subparts",
        type=int,
        default=3,
        help="List-mode max touched nodes shown per file (default: 3)",
    )
    parser.add_argument(
        "--src-chars",
        type=int,
        default=120,
        help="List-mode source snippet cap per touched node (default: 120)",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=100_000,
        help=(
            "List-mode prompt cap in chars; oversized candidate lists are "
            "ranked chunk by chunk and merged (0 disables chunking; "
            "default: 100000)"
        ),
    )
    parser.add_argument(
        "--graphs-dir",
        default=None,
        help="KG graph cache dir for list-mode signal assembly "
        "(default: graph_cache_dir from config)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="LLM retries per comparison before failing " "the instance (default: 3)",
    )
    parser.add_argument(
        "--max-file-lines",
        type=int,
        default=_DEFAULT_MAX_FILE_LINES,
        help="Per-file content truncation for the prompt "
        f"(default: {_DEFAULT_MAX_FILE_LINES})",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N new instances (0 = all)"
    )
    parser.add_argument(
        "--instance",
        default=None,
        help="Process only this instance_id (debugging aid)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count planned instances/comparisons, no LLM calls",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip ranking; evaluate existing progress file",
    )
    parser.add_argument(
        "--progress",
        default=None,
        help="Progress JSONL path (default: "
        "outputs/llm_rerank__file_level__<split>__"
        "<model>.progress.jsonl)",
    )
    parser.add_argument("--csv", default=None, help="Per-instance CSV output path")
    parser.add_argument(
        "--kg-progress-file",
        default=None,
        help=(
            "KG progress JSONL input override (default: "
            "compare_file_localization.kg_progress_path for the variant — "
            "the canonical outputs/ tree; set this for model-separated "
            "trees like outputs/gemma26b/)"
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--config", default=None, help="kg_config.json path")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    defaults = _load_defaults(args.config)
    model = args.model or defaults.get("model", "qwen3-coder-next-Q8_0")
    api_base = (
        args.api_base
        or os.environ.get("OPENAI_BASE_URL")
        or defaults.get("api_base", "http://localhost:8081/v1")
    )
    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or defaults.get("api_key", "any")
    )

    try:
        signals = parse_signals(args.signals) if args.mode == "list" else None
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    progress_path = args.progress or default_progress_path(
        args.split,
        model,
        args.variant,
        mode=args.mode,
        signals=signals,
    )

    # --- Identify instances for this split (same rule as cfl) ---
    kg_file_level_path = resolve_kg_input_path(args.kg_progress_file, args.variant)
    split_ids = cfl.get_split_ids(args.split)
    gt_by_id = cfl.load_gt(kg_file_level_path)
    rows = [
        r
        for r in cfl._read_jsonl(kg_file_level_path)
        if r["instance_id"] in split_ids and r["instance_id"] in gt_by_id
    ]
    rows = select_rankable_rows(rows)
    rows = filter_rows_by_instance(rows, args.instance)
    print(
        f"Split '{args.split}': {len(split_ids)} instances, "
        f"{len(rows)} with KG progress + GT"
    )
    if args.instance and not rows:
        print(f"ERROR: instance '{args.instance}' not found in this split")
        return 1
    print(f"Model: {model} @ {api_base}")

    # --- Rank (unless eval-only / dry-run) ---
    if not args.eval_only:
        if args.mode == "pairwise":

            def factory() -> LLMComparator:
                return LLMComparator(
                    client=build_client(api_base, api_key, args.timeout),
                    model=model,
                    retries=args.retries,
                    max_file_lines=args.max_file_lines,
                )

            rank_fn = None  # process_instances default: pairwise sort
            spec = None
            calls_per_instance = None
        else:
            graphs_dir = args.graphs_dir or defaults.get(
                "graph_cache_dir", "outputs/kg_graphs"
            )
            wants_sim = "sim_max" in signals or "sim_mean" in signals
            sim_provider = None
            if wants_sim:
                sim_provider = (
                    lambda problem_statement, graph_dir: _node_sims_from_chroma(
                        problem_statement, graph_dir, defaults
                    )
                )

            def factory() -> ListRanker:
                return ListRanker(
                    client=build_client(api_base, api_key, args.timeout),
                    model=model,
                    retries=args.retries,
                )

            def rank_fn_list(row, files, engine) -> dict:
                graph_dir = graph_dir_for_row(row, graphs_dir)
                file_signals = assemble_file_signals(
                    row,
                    graph_dir,
                    subparts=args.subparts,
                    src_chars=args.src_chars,
                    sim_provider=sim_provider,
                )
                return rank_files_list(
                    row.get("problem_statement", ""),
                    files,
                    file_signals,
                    top_k=args.top_k,
                    engine=engine,
                    signals=signals,
                    max_prompt_chars=args.max_prompt_chars,
                )

            rank_fn = rank_fn_list
            spec = {
                "mode": "list",
                "signals": signals,
                "top_k": args.top_k,
                "subparts": args.subparts,
                "src_chars": args.src_chars,
                "max_prompt_chars": args.max_prompt_chars,
            }
            calls_per_instance = 1

        stats = process_instances(
            rows,
            factory,
            progress_path,
            limit=args.limit,
            dry_run=args.dry_run,
            model=model,
            rank_fn=rank_fn,
            spec=spec,
            calls_per_instance=calls_per_instance,
        )
        print(
            f"\nRun stats: {stats['processed']}/{stats['instances']} processed "
            f"({stats['skipped']} skipped, {stats['failed']} failed), "
            f"{stats['comparisons']} comparisons"
        )
        if args.dry_run:
            print(f"Estimated comparisons: {stats['estimated_comparisons']}")
            if args.mode == "list" and rows:
                # Prompt preview for the first pending instance: makes any
                # signal combo inspectable without spending an LLM call.
                preview_row = rows[0]
                file_signals = assemble_file_signals(
                    preview_row,
                    graph_dir_for_row(
                        preview_row,
                        args.graphs_dir
                        or defaults.get("graph_cache_dir", "outputs/kg_graphs"),
                    ),
                    subparts=args.subparts,
                    src_chars=args.src_chars,
                    sim_provider=(
                        lambda problem_statement, graph_dir: (
                            _node_sims_from_chroma(
                                problem_statement, graph_dir, defaults
                            )
                            if ("sim_max" in signals or "sim_mean" in signals)
                            else None
                        )
                    ),
                )
                print(
                    "\n--- list-mode prompt preview "
                    f"({preview_row['instance_id']}) ---"
                )
                print(
                    build_list_prompt(
                        preview_row.get("problem_statement", ""),
                        file_signals,
                        signals,
                        args.top_k,
                    )
                )
                print("--- end preview ---")
            return 0

    # --- Evaluate ---
    rerank = load_rerank(progress_path)
    covered = sum(1 for r in rows if r["instance_id"] in rerank)
    print(f"\nEvaluating: rerank progress covers {covered}/{len(rows)} instances")
    provenance = load_rerank_models(progress_path)
    if provenance:
        print(f"Rerank models in progress file: {', '.join(sorted(provenance))}")

    instance_ids = sorted(r["instance_id"] for r in rows)
    kg_method = "kg_file_level"
    if args.variant != "plain":
        kg_method = f"kg_{args.variant}_file_level"
    rerank_method = f"{kg_method}_llm"
    if args.mode == "list":
        rerank_method = list_eval_method(kg_method, signals)
    methods_data = {
        "bm25": cfl.load_bm25(cfl.BM25_RETRIEVAL),
        "agentless": cfl.load_agentless(cfl.AGENTLESS_PATHS[args.split]),
        kg_method: cfl.load_kg(kg_file_level_path),
        rerank_method: rerank,
    }
    all_rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}
    for method in methods_data:
        method_rows = cfl.evaluate_method(
            methods_data[method], gt_by_id, instance_ids, method
        )
        all_rows.extend(method_rows)
        summary[method] = cfl.aggregate(method_rows)

    print_summary(summary, len(instance_ids))
    if args.csv:
        cfl.write_csv(all_rows, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
