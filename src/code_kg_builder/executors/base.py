"""Base class and batch data structure for all executors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

if TYPE_CHECKING:
    from code_kg_builder.context import BuildContext


@dataclass
class BatchResult:
    """A batch of graph elements to be committed atomically via update_graph().

    Attributes:
        nodes_data: Each dict has at least ``{"id", "label"}``.
        edges_data: Each dict has at least
            ``{"id", "source", "target", "label"}``.
        all_values_data: Each dict has at least ``{"id"}`` plus flat
            primitive key-value pairs.
        errors: Non-fatal errors encountered while building this batch.
    """

    nodes_data: List[Dict[str, Any]] = field(default_factory=list)
    edges_data: List[Dict[str, Any]] = field(default_factory=list)
    all_values_data: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def merge(self, other: "BatchResult") -> "BatchResult":
        """Return a new BatchResult combining both batches."""
        return BatchResult(
            nodes_data=self.nodes_data + other.nodes_data,
            edges_data=self.edges_data + other.edges_data,
            all_values_data=self.all_values_data + other.all_values_data,
            errors=self.errors + other.errors,
        )

    @property
    def is_empty(self) -> bool:
        return not (self.nodes_data or self.edges_data or self.all_values_data)


class Executor(ABC):
    """Base class for all executors.

    An executor processes repository source code and yields BatchResult
    objects.  Each batch is committed atomically via ``update_graph()``.
    If a batch fails (UNSAT or exception) it is logged and skipped --
    previously committed batches are preserved.
    """

    #: Execution ordering. Lower runs first.
    #:   0 = filesystem / structural
    #:   1 = AST nodes
    #:   2 = edge resolution
    phase: int = 0

    @abstractmethod
    def run(self, ctx: "BuildContext") -> Iterator[BatchResult]:
        """Process the repository and yield batches for atomic commits."""
        raise NotImplementedError
