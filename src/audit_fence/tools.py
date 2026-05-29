"""Search tools with path restrictions for sandboxed audit environments."""

from __future__ import annotations

import os
from typing import Any, Callable


class SandboxedSearch:
    """Wraps a search function with path restrictions.

    The backend is any callable ``(pattern, path, **kwargs) -> str``.
    Before delegating, this wrapper checks that the ``path`` argument
    falls within the allowed directories or matches an allowed file.

    Usage::

        from audit_fence.tools import SandboxedSearch

        search = SandboxedSearch(
            backend=my_grep_function,
            allowed_dirs=["tools/"],
        )

        # Allowed:
        search("revenue", "tools/fundamental_tool_calls.json")

        # Blocked:
        search("revenue", "trace/specialist_outputs/")
        # -> "ERROR: Path 'trace/specialist_outputs/' is outside ..."

    Args:
        backend: A callable ``(pattern: str, path: str, **kwargs) -> str``
            that performs the actual search.
        allowed_dirs: List of directory prefixes that are permitted for
            search.  Paths are normalized before comparison.
        allowed_files: List of specific file paths that are permitted.
            Paths are normalized before comparison.
    """

    def __init__(
        self,
        backend: Callable[..., str],
        *,
        allowed_dirs: list[str] | None = None,
        allowed_files: list[str] | None = None,
    ):
        self._backend = backend
        self._allowed_dirs = [
            os.path.normpath(d) for d in (allowed_dirs or [])
        ]
        self._allowed_files = [
            os.path.normpath(f) for f in (allowed_files or [])
        ]

    def __call__(self, pattern: str, path: str = "", **kwargs: Any) -> str:
        """Execute the search, checking path restrictions first.

        Args:
            pattern: Search pattern to pass to the backend.
            path: File or directory path to search in.
            **kwargs: Additional keyword arguments forwarded to backend.

        Returns:
            Backend result string, or an ERROR string if the path is
            outside the allowed sandbox.
        """
        if not self._is_allowed(path):
            return (
                f"ERROR: Path '{path}' is outside the allowed search "
                f"sandbox. Allowed directories: {self._allowed_dirs}, "
                f"allowed files: {self._allowed_files}."
            )
        return self._backend(pattern, path, **kwargs)

    def _is_allowed(self, path: str) -> bool:
        """Check whether *path* falls within allowed dirs/files."""
        # No restrictions configured → everything passes
        if not self._allowed_dirs and not self._allowed_files:
            return True

        if not path:
            # Empty path with restrictions → block
            return not (self._allowed_dirs or self._allowed_files)

        norm = os.path.normpath(path)

        # Block path traversal attempts
        if ".." in norm.split(os.sep):
            return False

        # Check allowed files (exact match)
        for allowed in self._allowed_files:
            if norm == allowed:
                return True

        # Check allowed directories (prefix match)
        for allowed_dir in self._allowed_dirs:
            # norm starts with allowed_dir, or norm IS allowed_dir
            if norm == allowed_dir:
                return True
            if norm.startswith(allowed_dir + os.sep):
                return True

        return False
