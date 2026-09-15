"""GraphWriter — wraps ``update_graph()`` for batched atomic commits.

Each call to :meth:`commit` sends a :class:`~code_kg_builder.executors.base.BatchResult`
to ``update_graph(graph="data", ...)`` which applies the batch atomically with
Z3 verification.  On failure (UNSAT), nothing is written and the error details
are returned for logging.
"""

from typing import Any, Dict

from code_kg_builder.executors.base import BatchResult


class GraphWriter:
    """Thin wrapper around ``tool.functions.graph_file_ops.update_graph``."""

    def commit(self, batch: BatchResult, verify: bool = True) -> Dict[str, Any]:
        """Commit a batch to the data graph.

        Returns the structured SAT/UNSAT response from ``update_graph``.
        """
        from tool.functions.graph_file_ops import update_graph

        result = update_graph(
            graph="data",
            nodes_data=batch.nodes_data or None,
            edges_data=batch.edges_data or None,
            all_values_data=batch.all_values_data or None,
            verify=verify,
        )

        return result
