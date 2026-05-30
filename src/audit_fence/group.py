"""FenceGroup: convenience container for managing multiple named Fence instances."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .fence import Fence, SearchRecord
from .workflow import ClaimRecord


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

    @property
    def all_claims(self) -> list[ClaimRecord]:
        """All claims from all fences, sorted by timestamp."""
        claims: list[ClaimRecord] = []
        for fence in self._fences.values():
            claims.extend(fence.claims)
        claims.sort(key=lambda c: c.timestamp)
        return claims

    def trace_chain(self, claim: ClaimRecord) -> list[ClaimRecord]:
        """Follow upstream links to build the full evidence chain.

        Returns a list starting from *claim* and walking back through
        ``upstream_fence`` / ``upstream_id`` until a root (no upstream)
        is reached.  Cycle-safe (uses Python object identity).

        Example::

            chain = group.trace_chain(final_claim)
            # [final_claim, intermediate_claim, source_claim]
        """
        chain: list[ClaimRecord] = [claim]
        seen: set[int] = {id(claim)}
        current = claim
        while current.upstream_id >= 0 and current.upstream_fence:
            fence = self.get(current.upstream_fence)
            if fence is None:
                break
            upstream = next(
                (c for c in fence.claims if c.id == current.upstream_id),
                None,
            )
            if upstream is None or id(upstream) in seen:
                break
            seen.add(id(upstream))
            chain.append(upstream)
            current = upstream
        return chain

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
    def from_snapshot_manifest(
        cls,
        manifest: dict,
        *,
        document: str,
        trace_dir: str,
        per_agent_documents: dict[str, str] | None = None,
    ) -> FenceGroup:
        """Build a FenceGroup with audit topology from a snapshot manifest.

        Creates one :class:`Fence` per agent listed in the manifest, with:

        - **source** restricted to that agent's trace directory
        - **document** set from ``per_agent_documents`` override, the first
          artifact's content (if readable), or the ``document`` fallback
        - **link topology** matching the manifest's declared dependencies

        This is the bridge between production-side :class:`Snapshot` capture
        and audit-side :class:`FenceGroup` enforcement.  It takes a plain
        dict (not a ``Snapshot`` object) to avoid circular dependencies.

        Args:
            manifest: A manifest dict as produced by
                ``Snapshot.load_manifest()``.  Must contain an ``"agents"``
                key; ``"dependencies"`` is optional.
            document: The final document to audit.  Used as the fallback
                for agents that have no per-agent override and no readable
                artifact.
            trace_dir: Root trace directory on disk.  Each agent's
                ``trace_dir`` from the manifest is resolved relative to this.
            per_agent_documents: Optional mapping of
                ``{agent_name: document_text}`` to override the document
                for specific agents.

        Returns:
            A new :class:`FenceGroup` with one fence per agent, linked
            according to the manifest's dependency graph.

        Example::

            manifest = json.load(open("trace/manifest.json"))
            group = FenceGroup.from_snapshot_manifest(
                manifest,
                document=open("reports/final.md").read(),
                trace_dir="trace/",
            )
        """
        agents = manifest.get("agents", {})
        dependencies = manifest.get("dependencies", {})

        group = cls()

        for agent_name, agent_info in agents.items():
            fence = group.create(agent_name)

            # Determine the agent's trace directory on disk
            agent_trace = agent_info.get("trace_dir", f"{agent_name}/")
            agent_dir = os.path.join(trace_dir, agent_trace)

            # Set document: per-agent override > first artifact content > final document
            if per_agent_documents and agent_name in per_agent_documents:
                fence.set_document(per_agent_documents[agent_name])
            elif agent_info.get("artifacts"):
                artifact_path = os.path.join(agent_dir, agent_info["artifacts"][0])
                try:
                    with open(artifact_path, "r", encoding="utf-8") as f:
                        fence.set_document(f.read())
                except (OSError, IOError):
                    # Artifact not readable — fall back to final document
                    fence.set_document(document)
            else:
                fence.set_document(document)

            # Set source restricted to this agent's trace directory
            fence.set_source(agent_dir)

        # Build link topology from dependencies
        for agent_name, upstreams in dependencies.items():
            if agent_name not in group:
                continue
            fence = group[agent_name]
            for upstream_name in upstreams:
                if upstream_name in group:
                    fence.link(group[upstream_name])

        return group

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
