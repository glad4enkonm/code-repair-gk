"""BuildVerifier — wraps verify_graph_update() for post-build verification."""

from typing import Any, Dict, List, Tuple


class BuildVerifier:
    """Verify a data graph against its meta schema using existing verification."""

    def __init__(self, graph_dir: str = "."):
        self.graph_dir = graph_dir

    def verify(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Run verification on the data graph.

        Returns (is_valid, errors).
        """
        from verification.graph_verifier import verify_graph_update

        return verify_graph_update("data", graph_dir=self.graph_dir)

    def report(self) -> str:
        """Return a human-readable verification report."""
        from verification.graph_verifier import format_validation_errors

        is_valid, errors = self.verify()
        if is_valid:
            return "Data graph verification: PASSED"
        return format_validation_errors(errors)
