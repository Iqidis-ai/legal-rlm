"""AgentRuntime — what an agent's invoke() receives.

Provides:
  - matter_model access (read/write through brokered packets when possible)
  - llm_client for LLM-using agents (deterministic agents skip this)
  - clock for cost/latency accounting
  - persistence helpers for AgentArtifact write-through

R5 patch corrections applied:
  - Uses matter_model.memory_broker (not .broker)
  - Domain profile bound from active matter, not hardcoded "legal:1"
"""

from __future__ import annotations

import json as _json
import time as _time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .contracts import AgentArtifact, AgentInvocation


@dataclass
class BrokeredReadPacket:
    """A bounded set of read sections + the dependency manifest hash they
    came from. Written through memory_broker so reuse / freshness rules apply."""
    packet_id: str
    sections: tuple[Mapping[str, Any], ...]
    dependency_manifest_hash: str


class AgentRuntime:
    """Per-invocation runtime handed to a sub-agent's invoke().

    Bound to a single AgentInvocation so the agent does not need to
    pass matter_id, run_id, etc. through every sub-call.
    """

    def __init__(
        self,
        *,
        matter_model: Any,
        invocation: AgentInvocation,
        llm_client: Any = None,
        clock=None,
    ) -> None:
        self.matter_model = matter_model
        self.invocation = invocation
        self.llm_client = llm_client
        self.clock = clock or _time.perf_counter

    # ------------------------------------------------------------------
    # Convenience: domain-aware reads
    # ------------------------------------------------------------------

    def domain_profile_id(self) -> str:
        """Return the bound domain profile id (per R5 — not hardcoded).

        Falls back to invocation.domain_profile_id, then matter's primary
        domain composition, then empty string.
        """
        if self.invocation.domain_profile_id:
            return self.invocation.domain_profile_id
        try:
            comp = getattr(self.matter_model, "_read_matter_domain_composition", None)
            if comp is None:
                return ""
            facets, weights, primary = comp() if callable(comp) else (None, None, None)
            return str(primary or "")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Brokered I/O
    # ------------------------------------------------------------------

    def brokered_read(
        self,
        *,
        purpose: str,
        object_refs: Sequence[Mapping[str, Any]],
        policy_audience: str = "clean",
        taint_class: str = "clean",
    ) -> BrokeredReadPacket:
        """Record a dependency manifest for the agent's reads.

        For PR#3 phase 2 we record the manifest but keep section payloads
        empty — the agent reads directly from typed_evidence/extraction_slot
        stores via this runtime's `matter_model`. Future phases may inline
        typed packet sections.
        """
        broker = getattr(self.matter_model, "memory_broker", None)
        if broker is None:
            return BrokeredReadPacket(
                packet_id="",
                sections=(),
                dependency_manifest_hash="",
            )
        # Best-effort manifest recording. The exact API depends on the broker
        # build; we degrade gracefully if the surface is missing.
        try:
            manifest_hash = broker.build_output_dependency_manifest(
                purpose=purpose,
                policy_audience=policy_audience,
                taint_class=taint_class,
                object_refs=tuple(
                    (o.get("kind", ""), o.get("id", "")) for o in object_refs
                ),
                namespace_keys=(),
            )
        except Exception:
            manifest_hash = ""
        return BrokeredReadPacket(
            packet_id=manifest_hash or "",
            sections=(),
            dependency_manifest_hash=manifest_hash or "",
        )

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def write_artifacts(
        self,
        invocation_id: str,
        artifacts: Sequence[AgentArtifact],
    ) -> tuple[str, ...]:
        """Persist artifacts to agent_artifact table. Returns row ids."""
        if not artifacts:
            return ()
        from datetime import datetime, timezone
        import uuid

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        out_ids: list[str] = []
        with self.matter_model.db.transaction():
            for a in artifacts:
                aid = uuid.uuid4().hex
                payload = a.payload if isinstance(a.payload, dict) else dict(a.payload)
                self.matter_model.db.execute(
                    """INSERT INTO agent_artifact
                       (id, matter_id, invocation_id, artifact_kind, artifact_key,
                        label, payload_json, synthesis_visibility, confidence,
                        source_refs_json, typed_evidence_refs_json,
                        memory_packet_id, dependency_manifest_hash,
                        verification_state, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        aid,
                        self.matter_model.matter_id,
                        invocation_id,
                        a.artifact_kind,
                        a.artifact_key,
                        a.label,
                        _json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                        a.synthesis_visibility,
                        max(0.0, min(float(a.confidence or 0.0), 1.0)),
                        _json.dumps(list(a.source_refs)),
                        _json.dumps(list(a.typed_evidence_refs)),
                        a.memory_packet_id,
                        a.dependency_manifest_hash,
                        a.verification_state,
                        now,
                    ),
                )
                out_ids.append(aid)
        return tuple(out_ids)
