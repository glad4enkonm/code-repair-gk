"""BuildContext — shared state passed to every executor during the build."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from code_kg_builder.meta_schema import MetaSchema


@dataclass
class NodeInfo:
    """Cached info about a committed node, for edge resolution."""

    id: str
    label: str
    meta_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEntry:
    """An error logged during the build."""

    source: str
    message: str


@dataclass
class BuildContext:
    """Shared state passed to every executor during the build.

    Attributes:
        repo_path: Root directory of the repository being analysed.
        graph_dir: Directory containing origin.json / meta.json / data.json.
        node_index: ``id`` -> :class:`NodeInfo` for every committed node.
        python_files: Discovered ``.py`` files (populated by FilesystemExecutor).
        ast_cache: ``Path`` -> ``ast.Module`` (populated lazily by executors).
        errors: Accumulated errors from all executors.
    """

    repo_path: Path
    meta_schema: Optional["MetaSchema"] = None
    graph_dir: str = "."
    node_index: Dict[str, NodeInfo] = field(default_factory=dict)
    python_files: List[Path] = field(default_factory=list)
    ast_cache: Dict[Path, Any] = field(default_factory=dict)
    errors: List[ErrorEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

    def log_error(self, source: str, message: str) -> None:
        """Record a non-fatal error."""
        self.errors.append(ErrorEntry(source=source, message=message))

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    # ------------------------------------------------------------------ #
    #  Node index helpers
    # ------------------------------------------------------------------ #

    def update_index(
        self, nodes_data: List[Dict[str, Any]], all_values_data: List[Dict[str, Any]]
    ) -> None:
        """Update the in-memory node index after a successful commit.

        Args:
            nodes_data: Node dicts (``{"id", "label"}``).
            all_values_data: Property dicts (``{"id", "meta_type", ...}``).
        """
        # Build a lookup for all_values by id
        av_lookup: Dict[str, Dict[str, Any]] = {}
        for av in all_values_data:
            av_id = av.get("id")
            if av_id:
                av_lookup[av_id] = av

        for node in nodes_data:
            nid = node.get("id", "")
            props = av_lookup.get(nid, {})
            meta_type = props.get("meta_type", "")
            self.node_index[nid] = NodeInfo(
                id=nid,
                label=node.get("label", ""),
                meta_type=meta_type,
                properties=dict(props),
            )

    def get_nodes_by_type(self, meta_type: str) -> List[NodeInfo]:
        """Return all committed nodes of a given meta_type."""
        return [
            info for info in self.node_index.values() if info.meta_type == meta_type
        ]

    def find_node(self, meta_type: str, **filters: Any) -> Optional[NodeInfo]:
        """Find the first node matching type and property filters."""
        for info in self.node_index.values():
            if info.meta_type != meta_type:
                continue
            if all(info.properties.get(k) == v for k, v in filters.items()):
                return info
        return None

    # ------------------------------------------------------------------ #
    #  AST cache helpers
    # ------------------------------------------------------------------ #

    def get_ast(self, file_path: Path) -> Any:
        """Return cached AST for *file_path*, parsing if necessary.

        Raises ``SyntaxError`` if the file cannot be parsed.
        """
        import ast

        if file_path not in self.ast_cache:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            self.ast_cache[file_path] = ast.parse(source, filename=str(file_path))
        return self.ast_cache[file_path]
