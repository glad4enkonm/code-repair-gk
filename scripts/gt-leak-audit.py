"""GT-leakage audit for repair prompts (SWE-bench experiment).

Question answered: do any *gold-patch added lines* appear in the prompt, and
if so, do they all already exist in the base-commit corpus?

A gold-added line found in the prompt but ABSENT from the base-commit corpus
would mean future (post-fix) code leaked into the prompt. A matched line that
exists in the corpus is benign (the line predates the fix).

Inputs:
    --predictions  jsonl with ``instance_id`` + ``text`` (the stored prompt);
                   works for the KG sr jsonl and the BM25 predictions jsonl
    --kg-progress  KG progress jsonl whose rows carry the gold ``patch``
    --corpus-dir   per-instance ``<instance_id>/documents.jsonl`` base-commit
                   corpus (BM25 ``file_name_and_contents_indexes`` layout:
                   ``{"id": path, "contents": "path\\n<file>"}``)
    --output       result json path (also printed as a summary)

Semantics:
    - gold added lines = ``+`` lines of the unified gold patch (``+++``
      headers excluded), stripped; lines shorter than ``--min-line-chars``
      (default 8) are ignored as noise
    - matched  = added lines whose stripped text occurs in the prompt text
    - leaked   = matched lines absent from EVERY file of the base corpus
    - missing corpus / missing patch are reported as such (never silently
      skipped, never counted as leaks)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("gt-leak-audit")

_TRUNCFIX_NOTE = "run: python scripts/gt-leak-audit.py --predictions ... "


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_added_lines(patch: str, min_line_chars: int) -> list[str]:
    """Stripped added lines of a unified diff, noise-filtered, deduped."""
    added = []
    seen = set()
    for line in patch.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            stripped = line[1:].strip()
            if len(stripped) >= min_line_chars and stripped not in seen:
                seen.add(stripped)
                added.append(stripped)
    return added


def corpus_text_for_instance(corpus_dir: str, instance_id: str) -> str | None:
    """Concatenated base-commit file contents; None if the corpus dir is absent."""
    documents_path = Path(corpus_dir) / instance_id / "documents.jsonl"
    if not documents_path.exists():
        return None
    parts = []
    for row in read_jsonl(str(documents_path)):
        parts.append(row.get("contents", ""))
    return "\n".join(parts)


def run_audit(
    predictions_path: str,
    kg_progress_path: str,
    corpus_dir: str,
    min_line_chars: int = 8,
) -> dict:
    """Run the leakage audit and return the result dict (also written by main)."""
    predictions = read_jsonl(predictions_path)
    patches = {
        row["instance_id"]: row.get("patch", "")
        for row in read_jsonl(kg_progress_path)
    }

    result = {
        "predictions_path": predictions_path,
        "kg_progress_path": kg_progress_path,
        "corpus_dir": corpus_dir,
        "min_line_chars": min_line_chars,
        "totals": {
            "instances": 0,
            "matched_lines": 0,
            "leaked_lines": 0,
            "patch_missing_instances": 0,
            "corpus_missing_instances": 0,
        },
        "instances": {},
    }

    for row in predictions:
        instance_id = row["instance_id"]
        result["totals"]["instances"] += 1
        patch = patches.get(instance_id, "")
        if not patch:
            result["totals"]["patch_missing_instances"] += 1
            logger.warning("no gold patch for %s in %s", instance_id, kg_progress_path)
            result["instances"][instance_id] = {"patch_missing": True}
            continue

        prompt_text = row.get("text", "")
        added = gold_added_lines(patch, min_line_chars)
        matched = [line for line in added if line in prompt_text]

        corpus_text = corpus_text_for_instance(corpus_dir, instance_id)
        record = {"matched": matched, "leaked": [], "corpus_missing": False}
        if corpus_text is None:
            result["totals"]["corpus_missing_instances"] += 1
            record["corpus_missing"] = True
            logger.warning("base-commit corpus missing for %s under %s", instance_id, corpus_dir)
        else:
            record["leaked"] = [line for line in matched if line not in corpus_text]

        result["totals"]["matched_lines"] += len(matched)
        result["totals"]["leaked_lines"] += len(record["leaked"])
        result["instances"][instance_id] = record

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=_TRUNCFIX_NOTE)
    parser.add_argument("--predictions", required=True, help="predictions jsonl with instance_id+text")
    parser.add_argument("--kg-progress", required=True, help="KG progress jsonl with gold patch")
    parser.add_argument("--corpus-dir", required=True, help="file_name_and_contents_indexes dir")
    parser.add_argument("--min-line-chars", type=int, default=8)
    parser.add_argument("--output", required=True, help="result json path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_audit(
        args.predictions,
        args.kg_progress,
        args.corpus_dir,
        args.min_line_chars,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    totals = result["totals"]
    print(
        f"instances={totals['instances']} matched_lines={totals['matched_lines']} "
        f"leaked_lines={totals['leaked_lines']} "
        f"patch_missing={totals['patch_missing_instances']} "
        f"corpus_missing={totals['corpus_missing_instances']}"
    )
    print(f"result written to {args.output}")
    return 1 if totals["leaked_lines"] else 0


if __name__ == "__main__":
    sys.exit(main())
