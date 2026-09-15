"""Orchestrator — main pipeline coordinator.

Reads the Meta schema, resolves executors, runs them in phase order,
commits batches atomically, and verifies the result.

Usage::

    orchestrator = Orchestrator(
        repo_path=Path("/path/to/repo"),
        meta_path="swe_bench_kg/meta.json",
        graph_dir="swe_bench_kg",
    )
    summary = orchestrator.run()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from code_kg_builder.context import BuildContext
from code_kg_builder.executor_resolver import resolve_executors
from code_kg_builder.executors.base import BatchResult, Executor
from code_kg_builder.graph_writer import GraphWriter
from code_kg_builder.meta_schema import MetaSchema


@dataclass
class BuildSummary:
    """Summary of a build run."""

    total_nodes: int = 0
    total_edges: int = 0
    skipped_edges: int = 0
    committed_batches: int = 0
    rejected_batches: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)
    verification_passed: Optional[bool] = None
    verification_errors: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASSED" if self.verification_passed else "FAILED"
        lines = [
            f"Build Summary ({status})",
            f"  Nodes committed:    {self.total_nodes}",
            f"  Edges committed:    {self.total_edges}",
            f"  Edges skipped:      {self.skipped_edges}",
            f"  Batches committed:  {self.committed_batches}",
            f"  Batches rejected:   {self.rejected_batches}",
            f"  Errors:             {len(self.errors)}",
        ]
        if self.errors:
            lines.append("  Error details:")
            for err in self.errors[:10]:
                lines.append(f"    {err['source']}: {err['message']}")
        return "\n".join(lines)


class Orchestrator:
    """Coordinates the full build pipeline."""

    def __init__(
        self,
        repo_path: Path,
        meta_path: str = "swe_bench_kg/meta.json",
        graph_dir: str = "swe_bench_kg",
        max_files: Optional[int] = None,
        run_verification: bool = True,
        verify_commits: bool = False,
        batch_size: int = 50,
    ):
        self.repo_path = repo_path.resolve()
        self.meta_path = str(Path(meta_path).resolve())
        self.graph_dir = str(Path(graph_dir).resolve())
        self.max_files = max_files
        self.run_verification = run_verification
        self.verify_commits = verify_commits
        self.batch_size = batch_size

    def run(self) -> BuildSummary:
        """Execute the full build pipeline. Returns a summary."""
        summary = BuildSummary()
        old_cwd = os.getcwd()

        try:
            os.chdir(self.graph_dir)
            return self._run_inner(summary)
        finally:
            os.chdir(old_cwd)

    def _run_inner(self, summary: BuildSummary) -> BuildSummary:

        # 1. Load Meta schema
        schema = MetaSchema.from_file(self.meta_path)

        # 2. Resolve executors
        executor_paths = schema.get_executors()
        if not executor_paths:
            summary.errors.append(
                {
                    "source": "orchestrator",
                    "message": "No executors found in meta schema. "
                    "Add 'executor' property to NodeType/EdgeType entries.",
                }
            )
            return summary

        try:
            instances = resolve_executors(executor_paths)
        except Exception as exc:
            summary.errors.append(
                {
                    "source": "orchestrator",
                    "message": f"Executor resolution failed: {exc}",
                }
            )
            return summary

        # 3. Sort by phase
        sorted_executors: List[Executor] = sorted(
            instances.values(), key=lambda e: getattr(e, "phase", 0)
        )

        # 4. Prepare context
        ctx = BuildContext(
            repo_path=self.repo_path,
            meta_schema=schema,
            graph_dir=self.graph_dir,
        )

        if self.max_files is not None:
            sorted_executors = sorted_executors  # max_files handled per executor

        writer = GraphWriter()

        # 5. Run executors
        for executor in sorted_executors:
            source = type(executor).__name__
            try:
                for batch in self._batched(executor.run(ctx), self.batch_size):
                    self._commit_batch(writer, batch, ctx, summary, source)
            except Exception as exc:
                ctx.log_error(source, str(exc))
                summary.errors.append({"source": source, "message": str(exc)})

        # 6. Collect context errors
        for err in ctx.errors:
            summary.errors.append(
                {
                    "source": err.source,
                    "message": err.message,
                }
            )

        # 7. Verify
        if self.run_verification:
            try:
                from code_kg_builder.build_verifier import BuildVerifier

                verifier = BuildVerifier(graph_dir=self.graph_dir)
                is_valid, errors = verifier.verify()
                summary.verification_passed = is_valid
                summary.verification_errors = errors
            except Exception as exc:
                summary.verification_passed = False
                summary.verification_errors = [
                    {
                        "type": "verification_error",
                        "message": str(exc),
                    }
                ]

        return summary

    @staticmethod
    def _batched(
        batches: Iterator[BatchResult], batch_size: int
    ) -> Iterator[BatchResult]:
        """Yield merged batches, accumulating up to batch_size per yield."""
        accumulator = None
        count = 0
        for batch in batches:
            if batch.is_empty:
                continue
            accumulator = batch if accumulator is None else accumulator.merge(batch)
            count += 1
            if count >= batch_size:
                yield accumulator
                accumulator = None
                count = 0
        if accumulator is not None:
            yield accumulator

    def _filter_invalid_edges(  # TODO: revrite in a good Z3 validation
        self,
        batch: BatchResult,
        ctx: BuildContext,
        summary: BuildSummary,
        source: str,
    ) -> BatchResult:
        """Filter out edges whose (source_type, target_type) does not match
        any signature in the meta schema.

        Nodes are always kept.  Edges with unresolvable source/target types
        are kept (cannot validate without type info).
        """
        if not batch.edges_data or not ctx.meta_schema:
            return batch

        valid_sigs: Dict[str, set] = {}
        for etype, sigs in ctx.meta_schema.edge_signatures.items():
            valid_sigs[etype] = {(s.source_type, s.target_type) for s in sigs}

        type_map: Dict[str, str] = {}
        for nid, info in ctx.node_index.items():
            type_map[nid] = info.meta_type
        for av in batch.all_values_data:
            av_id = av.get("id")
            if av_id:
                mtype = av.get("meta_type") or av.get("origin_type")
                if mtype:
                    type_map[av_id] = mtype

        kept_edges = []
        for edge in batch.edges_data:
            edge_type = edge.get("label", "")
            src_id = edge.get("source", "")
            tgt_id = edge.get("target", "")

            sigs = valid_sigs.get(edge_type)
            if sigs is None:
                kept_edges.append(edge)
                continue

            src_type = type_map.get(src_id)
            tgt_type = type_map.get(tgt_id)
            if src_type is None or tgt_type is None:
                kept_edges.append(edge)
                continue

            if (src_type, tgt_type) in sigs:
                kept_edges.append(edge)
            else:
                summary.skipped_edges += 1

        batch.edges_data = kept_edges
        return batch

    def _commit_batch(
        self,
        writer: GraphWriter,
        batch: BatchResult,
        ctx: BuildContext,
        summary: BuildSummary,
        source: str,
    ):
        """Commit a (possibly merged) batch and update summary + context."""
        batch = self._filter_invalid_edges(batch, ctx, summary, source)
        result = writer.commit(batch, verify=self.verify_commits)
        if result.get("status") == "sat":
            summary.committed_batches += 1
            summary.total_nodes += len(batch.nodes_data)
            summary.total_edges += len(batch.edges_data)
            ctx.update_index(batch.nodes_data, batch.all_values_data)
        else:
            summary.rejected_batches += 1
            summary.errors.append(
                {
                    "source": source,
                    "message": result.get("message", "Unknown error"),
                }
            )
