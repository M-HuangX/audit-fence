"""Search tools: sandboxed paths and ripgrep backend for audit environments."""

from __future__ import annotations

import os
import shutil
import subprocess
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


class RipgrepBackend:
    """Ready-to-use ripgrep search backend for audit-fence.

    Wraps the ``rg`` CLI binary via subprocess.  Compose with
    :class:`SandboxedSearch` for path restrictions and
    ``@fence.track`` for history recording::

        from audit_fence.tools import RipgrepBackend, SandboxedSearch

        grep = RipgrepBackend(root="./trace/")

        # Optional: restrict search to specific directories
        sandboxed = SandboxedSearch(backend=grep, allowed_dirs=["tools/"])

        # Track results in fence history
        search = fence.wrap_tool(sandboxed, role="search")

    Requires ``rg`` (ripgrep) to be installed on the system.
    No Python dependencies are added — the zero-dependency promise
    is maintained.

    Args:
        root: Base directory for searches.  All ``path`` arguments
            in :meth:`__call__` are resolved relative to this root.
        max_matches: Maximum output lines before truncation
            (default 200).
        rg_path: Explicit path to the ``rg`` binary.  If not provided,
            discovered via ``shutil.which("rg")``.
    """

    def __init__(
        self,
        root: str = ".",
        *,
        max_matches: int = 200,
        rg_path: str | None = None,
    ):
        self._root = os.path.abspath(root)
        self._max_matches = max_matches
        if rg_path is not None:
            self._rg_path = rg_path
        else:
            self._rg_path = self._find_rg()

    @staticmethod
    def _find_rg() -> str:
        """Find ripgrep binary on system PATH."""
        rg = shutil.which("rg")
        if rg:
            return rg
        raise FileNotFoundError(
            "ripgrep (rg) not found. Install with: "
            "apt install ripgrep / brew install ripgrep / "
            "pip install ripgrep"
        )

    def __call__(
        self,
        pattern: str,
        path: str = "",
        *,
        context: int = 0,
        case_insensitive: bool = True,
    ) -> str:
        """Search for a regex pattern in files under root.

        Compatible with :class:`SandboxedSearch` as a backend —
        the ``(pattern, path, **kwargs)`` signature matches.

        Args:
            pattern: Regex pattern to search for.  Supports full
                regex syntax (``"62\\\\.25"``, ``"revenue|earnings"``).
            path: Relative path within root (file or directory).
                Empty string searches the entire root.
            context: Lines of surrounding context (0–10, default 0).
            case_insensitive: Case insensitive search (default True).

        Returns:
            Formatted results as ``file:line:content`` text with
            paths relative to *root*, or an ``"ERROR: ..."`` string
            on failure.
        """
        if not pattern or not pattern.strip():
            return "ERROR: pattern cannot be empty."

        search_path = (
            os.path.join(self._root, path) if path else self._root
        )
        if not os.path.exists(search_path):
            return f"ERROR: path not found: {path or '.'}"

        context = max(0, min(context, 10))

        args: list[str] = [
            self._rg_path, "--no-heading", "--with-filename",
            "--max-columns", "500", "-n",
        ]

        if case_insensitive:
            args.append("-i")

        if context > 0:
            args.extend(["-C", str(context)])

        # Pattern starting with dash needs -e flag
        if pattern.startswith("-"):
            args.extend(["-e", pattern])
        else:
            args.append(pattern)

        args.append(str(search_path))

        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            return (
                "ERROR: search timed out (15s). "
                "Narrow your pattern or path."
            )
        except Exception as e:
            return f"ERROR: ripgrep failed: {e}"

        # rg exit codes: 0=matches, 1=no matches, 2=error
        if result.returncode == 1:
            return f"No matches found for pattern: {pattern}"
        if result.returncode == 2:
            return f"ERROR: regex error — {result.stderr.strip()}"

        lines = result.stdout.strip().split("\n")

        # Convert absolute paths to root-relative
        output_lines = [self._relativize(line) for line in lines]

        if len(output_lines) > self._max_matches:
            output_lines = output_lines[:self._max_matches]
            output_lines.append(
                f"\n... truncated at {self._max_matches} lines. "
                "Narrow your pattern or path."
            )

        return "\n".join(output_lines)

    def _relativize(self, line: str) -> str:
        """Convert absolute paths in rg output to root-relative."""
        colon_idx = line.find(":")
        if colon_idx <= 0:
            return line
        file_part = line[:colon_idx]
        rest = line[colon_idx:]

        try:
            abs_path = os.path.abspath(file_part)
            if abs_path.startswith(self._root + os.sep):
                rel = os.path.relpath(abs_path, self._root)
                return rel + rest
        except (ValueError, OSError):
            pass
        return line
