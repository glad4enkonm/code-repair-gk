"""Oracle context assembly: ground-truth files as the context.

Oracle mode measures the repair ceiling in isolation: the file set
comes from the gold patch (perfect localization), while prompt format,
token budget, and all downstream inference/eval stages stay identical
to the KG ``file_level`` mode. ``Δ(oracle, file_level)`` then isolates
the localization gap.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

from .config import KGConfig

logger = logging.getLogger(__name__)

_GT_FILE_RE = re.compile(r"(?m)^\+\+\+ b/(.+)$")


def extract_gt_files(patch: str) -> List[str]:
    """Extract deduplicated file paths from ``+++ b/`` patch headers.

    Order follows first appearance in the patch.
    """
    seen: set = set()
    gt_files: List[str] = []
    for path in _GT_FILE_RE.findall(patch):
        if path not in seen:
            seen.add(path)
            gt_files.append(path)
    return gt_files


@dataclass
class OracleContext:
    """Result of oracle context assembly."""

    text: str
    gt_files: List[str]
    dropped_files: List[str]


def _style3(instance: dict) -> str:
    """Wrap swebench ``prompt_style_3`` (lazy import — heavy dependency)."""
    from swebench.inference.make_datasets.create_instance import prompt_style_3

    return prompt_style_3(instance)


def assemble_oracle_context(
    repo: str,
    base_commit: str,
    problem_statement: str,
    patch: str,
    config: KGConfig,
) -> OracleContext:
    """Build a style-3 prompt containing exactly the GT files.

    Identical format to ``assemble_file_level`` — only the file
    selection differs. GT files that are missing from the checkout or
    dropped by the token budget are reported in ``dropped_files``
    (visible degradation, never silent).
    """
    gt_files = extract_gt_files(patch)
    if not gt_files:
        return OracleContext(text="", gt_files=[], dropped_files=[])

    from .build_graphs import _checkout, ensure_repo_cloned
    from .context import _read_readmes, read_files_within_budget

    repo_dir = ensure_repo_cloned(repo, config)
    _checkout(repo_dir, base_commit)

    file_contents = read_files_within_budget(
        str(repo_dir), gt_files, config.max_context_tokens
    )
    dropped_files = [f for f in gt_files if f not in file_contents]
    if dropped_files:
        logger.warning("Oracle dropped GT files: %s", dropped_files)

    if not file_contents:
        return OracleContext(text="", gt_files=gt_files, dropped_files=dropped_files)

    readmes = _read_readmes(str(repo_dir))
    text = _style3(
        {
            "problem_statement": problem_statement,
            "readmes": readmes,
            "file_contents": file_contents,
        }
    )
    return OracleContext(text=text, gt_files=gt_files, dropped_files=dropped_files)
