"""CLI entry point for the knowledge graph builder.

Usage::

    python -m code_kg_builder \\
        --repo /path/to/repo \\
        --meta swe_bench_kg/meta.json \\
        --data swe_bench_kg/data.json \\
        --graph-dir swe_bench_kg
"""

import sys
from pathlib import Path

import click

from code_kg_builder.orchestrator import Orchestrator


@click.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True),
    help="Path to the repository to analyze.",
)
@click.option(
    "--meta",
    default="swe_bench_kg/meta.json",
    type=click.Path(exists=True),
    help="Path to meta.json.",
)
@click.option(
    "--data", default="swe_bench_kg/data.json", help="Path to data.json (output)."
)
@click.option(
    "--origin",
    default="swe_bench_kg/origin.json",
    type=click.Path(exists=True),
    help="Path to origin.json.",
)
@click.option(
    "--graph-dir",
    default=None,
    help="Directory containing the 3 JSON files (auto from --meta).",
)
@click.option(
    "--max-files", default=None, type=int, help="Limit number of files (debugging)."
)
@click.option(
    "--no-verify", is_flag=True, default=False, help="Skip final Z3 verification."
)
def main(
    repo: str,
    meta: str,
    data: str,
    origin: str,
    graph_dir: str | None,
    max_files: int | None,
    no_verify: bool,
) -> None:
    """Build a data-level knowledge graph from a Python repository."""
    if graph_dir is None:
        graph_dir = str(Path(meta).parent)

    click.echo(f"Repository:  {repo}")
    click.echo(f"Meta:        {meta}")
    click.echo(f"Graph dir:   {graph_dir}")

    orchestrator = Orchestrator(
        repo_path=Path(repo),
        meta_path=meta,
        graph_dir=graph_dir,
        max_files=max_files,
        run_verification=not no_verify,
    )

    summary = orchestrator.run()

    click.echo(str(summary))

    if summary.errors:
        click.echo(f"\n{len(summary.errors)} error(s) recorded.", err=True)

    if summary.verification_passed is False:
        click.echo("\nVerification FAILED.", err=True)
        sys.exit(1)

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
