"""
Shared utility functions for Coder Buddy.
"""

from __future__ import annotations

import difflib


def compute_unified_diff(before: str, after: str, file_name: str) -> str:
    """
    Compute a unified diff between *before* and *after*.

    Uses ``difflib.unified_diff`` with ``fromfile="a/{file_name}"`` and
    ``tofile="b/{file_name}"``.

    Args:
        before: Original source code string.
        after: Refactored (or new) source code string.
        file_name: Filename used in the diff header lines.

    Returns:
        Unified diff string, or ``""`` if *before* and *after* are identical.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{file_name}",
        tofile=f"b/{file_name}",
    )
    return "".join(diff)
