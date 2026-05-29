"""FenceGroup: convenience container for managing multiple named Fence instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fence import Fence, SearchRecord


class FenceGroup:
    """Container for managing multiple named Fence instances.

    Provides named access, bulk operations, and topology helpers.
    Not required -- you can create and link Fence instances directly.

    Usage::

        group = FenceGroup()
        fund = group.create("fundamental")
        tech = group.create("technical")
        core = group.create("core")
        core.link(fund, tech)  # core can cite both specialists
    """

    def __init__(self) -> None:
        self._fences: dict[str, Fence] = {}

    def create(self, name: str, **kwargs: Any) -> Fence:
        """Create a named fence and register it in this group.

        Args:
            name: Unique name for the fence within this group.
            **kwargs: Additional keyword arguments passed to
                :class:`Fence.__init__` (e.g. ``min_evidence_length``,
                ``history_window``).

        Returns:
            The newly created :class:`Fence`.

        Raises:
            ValueError: If a fence with the given name already exists.
        """
        if name in self._fences:
            raise ValueError(f"Fence '{name}' already exists in this group")
        fence = Fence(name=name, **kwargs)
        self._fences[name] = fence
        return fence

    def __getitem__(self, name: str) -> Fence:
        """Get a fence by name.

        Raises:
            KeyError: If no fence with the given name exists.
        """
        return self._fences[name]

    def __contains__(self, name: object) -> bool:
        """Check if a fence with the given name exists."""
        return name in self._fences

    def get(self, name: str, default: Fence | None = None) -> Fence | None:
        """Get a fence by name, returning *default* if not found."""
        return self._fences.get(name, default)

    @property
    def fences(self) -> dict[str, Fence]:
        """All registered fences, by name."""
        return dict(self._fences)

    @property
    def all_rejections(self) -> list[dict]:
        """Rejections from all fences, sorted by timestamp."""
        all_rej: list[dict] = []
        for fence in self._fences.values():
            all_rej.extend(fence.rejections)
        all_rej.sort(key=lambda r: r.get("timestamp", 0))
        return all_rej

    @property
    def all_history(self) -> list[SearchRecord]:
        """Search history from all fences, sorted by timestamp."""
        records: list[SearchRecord] = []
        for fence in self._fences.values():
            records.extend(fence.history)
        records.sort(key=lambda r: r.timestamp)
        return records

    def save_log(self, path: str | Path) -> None:
        """Save all rejections from all fences to a JSONL file."""
        for fence in self._fences.values():
            fence.save_log(path)

    def reset(self) -> None:
        """Reset all fences in the group."""
        for fence in self._fences.values():
            fence.reset()

    def snapshot(self) -> dict:
        """Serialize the entire group state for persistence.

        Returns a dict that can be JSON-serialized and later restored
        with :meth:`FenceGroup.restore`.
        """
        return {
            "fences": {
                name: fence._snapshot()
                for name, fence in self._fences.items()
            },
            "links": {
                name: [u._name for u in fence._upstream if u._name]
                for name, fence in self._fences.items()
                if fence._upstream
            },
        }

    @classmethod
    def restore(cls, data: dict) -> FenceGroup:
        """Restore a FenceGroup from a snapshot dict.

        Args:
            data: A dict previously returned by :meth:`snapshot`.

        Returns:
            A new FenceGroup with all fences and links restored.
        """
        group = cls()
        for name, fence_data in data["fences"].items():
            fence = group.create(name)
            fence._restore(fence_data)
        # Restore links
        for name, upstream_names in data.get("links", {}).items():
            fence = group[name]
            for uname in upstream_names:
                if uname in group:
                    fence.link(group[uname])
        return group
