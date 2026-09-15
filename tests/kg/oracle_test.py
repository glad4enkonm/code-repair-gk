"""Tests for oracle.py: GT-file context assembly (oracle mode)."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from kg.config import KGConfig
from kg.oracle import assemble_oracle_context, extract_gt_files

# ---------------------------------------------------------------------------
# extract_gt_files
# ---------------------------------------------------------------------------

_TWO_FILE_PATCH = """\
diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,3 @@
 def core():
+    pass
     return 1
diff --git a/pkg/util.py b/pkg/util.py
--- a/pkg/util.py
+++ b/pkg/util.py
@@ -1,2 +1,2 @@
 def util():
-    return 2
+    return 3
"""

_DUPLICATED_FILE_PATCH = """\
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-x
+y
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -2 +2 @@
-z
+w
"""


def test_extract_gt_files_preserves_order():
    assert extract_gt_files(_TWO_FILE_PATCH) == ["pkg/core.py", "pkg/util.py"]


def test_extract_gt_files_dedupes():
    assert extract_gt_files(_DUPLICATED_FILE_PATCH) == ["pkg/core.py"]


def test_extract_gt_files_empty_patch():
    assert extract_gt_files("") == []


def test_extract_gt_files_no_markers():
    assert extract_gt_files("random text\nno headers here\n") == []


def test_extract_gt_files_ignores_a_side_headers():
    """Only ``+++ b/`` (new side) counts, not ``--- a/`` (old side)."""
    assert extract_gt_files("--- a/old.py\n+++ b/new.py\n@@\n") == ["new.py"]


# ---------------------------------------------------------------------------
# Helpers for assemble_oracle_context
# ---------------------------------------------------------------------------


def _setup_repo(tmp: Path) -> Path:
    """Temp repo with README + two GT-able source files."""
    repo_dir = tmp / "repo"
    (repo_dir / "pkg").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# demo\n", encoding="utf-8")
    (repo_dir / "pkg" / "core.py").write_text(
        "def core():\n    return 1\n", encoding="utf-8"
    )
    (repo_dir / "pkg" / "util.py").write_text("x" * 3000, encoding="utf-8")
    return repo_dir


def _patch_for(*files: str) -> str:
    return "".join(f"--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n-old\n+new\n" for f in files)


def _capture_style3(instance: dict) -> str:
    """Fake prompt_style_3: exposes file selection + readmes."""
    files = ",".join(sorted(instance["file_contents"]))
    readmes = ",".join(sorted(instance["readmes"]))
    return f"PS3[{files}][{readmes}]"


def _run_oracle(repo_dir: Path, patch_text: str, config: KGConfig):
    with patch("kg.build_graphs.ensure_repo_cloned", return_value=repo_dir):
        with patch("kg.build_graphs._checkout") as mock_checkout:
            with patch("kg.oracle._style3", side_effect=_capture_style3) as mock_style3:
                result = assemble_oracle_context(
                    "org/demo", "abc123", "fix the bug", patch_text, config
                )
    return result, mock_checkout, mock_style3


# ---------------------------------------------------------------------------
# assemble_oracle_context
# ---------------------------------------------------------------------------


def test_assemble_oracle_context_selects_exactly_gt_files():
    """Prompt contains the GT files (and readmes) — nothing else."""
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_basic_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        result, _, _ = _run_oracle(
            repo_dir,
            _patch_for("pkg/core.py", "pkg/util.py"),
            KGConfig(max_context_tokens=128_000),
        )
        assert result.gt_files == ["pkg/core.py", "pkg/util.py"]
        assert result.dropped_files == []
        assert result.text == "PS3[pkg/core.py,pkg/util.py][README.md]"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_oracle_context_checks_out_base_commit():
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_checkout_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        _, mock_checkout, _ = _run_oracle(
            repo_dir, _patch_for("pkg/core.py"), KGConfig()
        )
        mock_checkout.assert_called_once_with(repo_dir, "abc123")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_oracle_context_budget_drops_big_file():
    """Small budget → big second GT file dropped and reported."""
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_budget_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        # core.py ≈ 8 tokens (24 chars), util.py = 1000 tokens (3000 chars)
        result, _, _ = _run_oracle(
            repo_dir,
            _patch_for("pkg/core.py", "pkg/util.py"),
            KGConfig(max_context_tokens=20),
        )
        assert "pkg/core.py" in result.text
        assert result.dropped_files == ["pkg/util.py"]
        assert result.gt_files == ["pkg/core.py", "pkg/util.py"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_oracle_context_first_file_always_included():
    """First file is included even when it alone exceeds the budget."""
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_first_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        # util.py = 1000 tokens, budget = 10
        result, _, _ = _run_oracle(
            repo_dir, _patch_for("pkg/util.py"), KGConfig(max_context_tokens=10)
        )
        assert result.dropped_files == []
        assert result.text == "PS3[pkg/util.py][README.md]"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_oracle_context_missing_file_reported_dropped():
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_missing_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        result, _, _ = _run_oracle(
            repo_dir,
            _patch_for("pkg/core.py", "pkg/gone.py"),
            KGConfig(max_context_tokens=128_000),
        )
        assert result.dropped_files == ["pkg/gone.py"]
        assert result.text == "PS3[pkg/core.py][README.md]"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assemble_oracle_context_empty_patch():
    tmpdir = Path(tempfile.mkdtemp(prefix="oracle_empty_"))
    try:
        repo_dir = _setup_repo(tmpdir)
        result, mock_checkout, mock_style3 = _run_oracle(repo_dir, "", KGConfig())
        assert result.text == ""
        assert result.gt_files == []
        assert result.dropped_files == []
        # No repo work, no prompt build for an empty patch
        mock_checkout.assert_not_called()
        mock_style3.assert_not_called()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
