import json
import logging
import subprocess
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

from .build_graphs import build_graph
from .config import KGConfig
from .context import dispatch_context_mode
from .embeddings import build_embeddings, release_graph_client
from .explore import explore

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="KG localization pipeline over SWE-bench_Lite"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to kg_config.json (defaults: KGConfig())",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help=(
            "Enable embedding-augmented exploration "
            "(requires llama-server /v1/embeddings on embedding_api_base)"
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Process only this instance_id (worker mode used by "
            "--subprocess-per-instance)"
        ),
    )
    parser.add_argument(
        "--subprocess-per-instance",
        action="store_true",
        help=(
            "Spawn a fresh python subprocess per instance so all RAM "
            "(chroma systems, glibc arenas) returns to the OS on exit — "
            "long runs otherwise grow monotonically"
        ),
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the final progress->dataset conversion (worker mode)",
    )
    return parser.parse_args(argv)


def _load_existing(progress_file: Path) -> set:
    """Ids of instances whose exploration already produced real output.

    Rows with empty ``text_inputs`` are treated as incomplete (e.g. the
    LLM was unavailable during that run) and re-processed on resume.
    """
    if not progress_file.exists():
        return set()
    ids = set()
    with open(progress_file) as f:
        for line_no, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping malformed progress line %d in %s: %s",
                    line_no,
                    progress_file,
                    exc,
                )
                continue
            if not entry.get("instance_id"):
                continue
            if entry.get("text_inputs"):
                ids.add(entry["instance_id"])
    return ids


def _process_instance(instance, config: KGConfig, progress_file: Path) -> bool:
    """Build graph, embed, explore and append one progress row.

    Returns True when the row was written; exceptions are contained so a
    single bad instance never aborts the run.
    """
    instance_id = instance["instance_id"]
    try:
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem_statement = instance["problem_statement"]

        print(f"\n[{instance_id}] {repo}", flush=True)

        # K1: Build graph
        graph_dir = build_graph(repo, base_commit, config)

        # K1b: Build embedding index (opt-in; failure is non-fatal —
        # explore() degrades to the classic loop)
        if config.use_embeddings:
            try:
                build_embeddings(str(graph_dir), config)
            except Exception:
                import traceback

                print(
                    "  [warn] embedding build failed; " "exploring without embeddings",
                    flush=True,
                )
                traceback.print_exc()

        # K2a: Explore
        touched = explore(str(graph_dir), problem_statement, config)

        # K2b: Assemble context
        text = dispatch_context_mode(
            touched, repo, base_commit, problem_statement, str(graph_dir), config
        )

        # K2c: Release the per-graph chroma system — chromadb keeps
        # one open system per path for the process lifetime and
        # 300-instance runs would exhaust the file-descriptor limit
        release_graph_client(str(graph_dir))

        entry = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "problem_statement": problem_statement,
            "patch": instance.get("patch", ""),
            "test_patch": instance.get("test_patch", ""),
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance.get("created_at", ""),
            "version": instance.get("version", ""),
            "FAIL_TO_PASS": instance.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": instance.get("PASS_TO_PASS", ""),
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
            "text_inputs": text,
            "touched_nodes": list(touched.keys()),
        }

        with open(progress_file, "a") as f:
            print(json.dumps(entry, ensure_ascii=False), file=f, flush=True)

        return True

    except Exception:
        import traceback

        print(f"  FAILED: {instance_id}", flush=True)
        traceback.print_exc()
        return False


def _spawn_worker(config_path: str | None, config: KGConfig, instance_id: str) -> bool:
    """Run one instance in a fresh `kg.run --only` subprocess.

    The worker appends the progress row itself; process exit returns all
    memory (chroma rust segments, glibc arenas) to the OS, so nothing
    accumulates across instances.
    """
    command = [sys.executable, "-m", "kg.run"]
    if config_path:
        command.append(config_path)
    if config.use_embeddings:
        command.append("--embeddings")
    command += ["--only", instance_id, "--skip-convert"]

    print(f"\n[{instance_id}] worker subprocess", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"  FAILED: {instance_id} (worker exit {result.returncode})", flush=True)
        return False
    return True


def main(
    config: KGConfig,
    config_path: str | None = None,
    only: str | None = None,
    subprocess_per_instance: bool = False,
    convert: bool = True,
):
    split = config.split
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    print(f"Dataset: princeton-nlp/SWE-bench_Lite/{split} — {len(dataset)} instances")
    # Materialize once: the only-id guard below consumes an iteration, and
    # HF datasets are cheap to list (a few hundred small dicts)
    instances = list(dataset)

    if only and not any(i["instance_id"] == only for i in instances):
        raise SystemExit(f"instance {only} not found in split {split}")

    output_name = f"data__swe-bench_lite__{config.prompt_style}__fs-kg__k-auto"
    if config.use_embeddings:
        output_name += "__embed"
    output_dir = Path(config.output_dir) / output_name
    progress_file = Path(str(output_dir) + ".progress.jsonl")
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(progress_file)
    if existing:
        print(f"Resuming: {len(existing)} already processed, will skip")

    for instance in tqdm(instances, desc=f"KG pipeline on {split}"):
        instance_id = instance["instance_id"]
        if instance_id in existing:
            continue
        if only and instance_id != only:
            continue

        if subprocess_per_instance:
            ok = _spawn_worker(config_path, config, instance_id)
        else:
            ok = _process_instance(instance, config, progress_file)
        if ok:
            existing.add(instance_id)

    # Convert progress file to HF DatasetDict
    if convert:
        _convert_to_dataset(progress_file, output_dir, config.split)

    print(f"\nDone. Dataset saved to {output_dir}")


def _convert_to_dataset(progress_file: Path, output_dir: Path, split: str = "test"):
    """Convert progress.jsonl to HF DatasetDict (BM25-compatible schema)."""

    def _mk_entry(instance):
        return {
            "instance_id": instance["instance_id"],
            "repo": instance.get("repo", ""),
            "base_commit": instance.get("base_commit", ""),
            "problem_statement": instance.get("problem_statement", ""),
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance.get("created_at", ""),
            "patch": "\n".join(["<patch>", instance.get("patch", ""), "</patch>"]),
            "test_patch": instance.get("test_patch", ""),
            "version": instance.get("version", ""),
            "FAIL_TO_PASS": instance.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": instance.get("PASS_TO_PASS", ""),
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
            "text": (instance.get("text_inputs") or "").strip() + "\n\n",
        }

    columns = [
        "instance_id",
        "text",
        "repo",
        "base_commit",
        "problem_statement",
        "hints_text",
        "created_at",
        "patch",
        "test_patch",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
    ]

    records = []
    with open(progress_file) as f:
        for line in f:
            instance = json.loads(line)
            records.append(_mk_entry(instance))

    ds = Dataset.from_list([{k: r[k] for k in columns} for r in records])
    dataset_dict = DatasetDict({split: ds})
    dataset_dict.save_to_disk(str(output_dir))
    print(f"  Saved dataset with {len(records)} entries to {output_dir}")


if __name__ == "__main__":
    args = _parse_args()
    config = KGConfig.from_file(args.config) if args.config else KGConfig()
    if args.embeddings:
        config.use_embeddings = True
    main(
        config,
        config_path=args.config,
        only=args.only,
        subprocess_per_instance=args.subprocess_per_instance,
        convert=not args.skip_convert,
    )
