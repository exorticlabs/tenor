"""
Core Data Primitives, Episodic Timeframes & Enterprise Graph/DB Schema (node_models.py)
========================================================================================

Production-grade graph schema and relational persistence models for the 4D Spatio-Temporal
EEKG (Episodic-Entity Knowledge Graph Memory System v2).

Architecture Components:
------------------------
1. Domain Primitives & Models:
   - UpdateMode: Operational mutation policy (TIME_SLICED vs OVERWRITE).
   - EpisodeType: Classification of episodic timeframe containers (event, timeframe, conversation).
   - TimeSlice: Versioned historical attribute state with temporal validity bounds.
   - TimeframeEpisode: Segregated, chronologically ordered episodic container.
   - KnowledgeNode: Atomic slotted knowledge unit with continuous taxonomic state and dual-path mutations.
   - EntityNode: Canonical entity coordinate coordinating slotted knowledge, aliases, and episode history.
   - EntityRegistry: In-memory entity catalog with thread-safe dual-factor (regex + cosine) resolution.

2. Enterprise Persistence Engine:
   - SQLiteGraphStore: High-throughput, ACID-compliant persistence layer with WAL mode, foreign keys,
     optimized indexes, and binary serialization for NumPy vector embeddings and taxonomic manifolds.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Domain Enumerations & Custom Exceptions
# ============================================================================

class UpdateMode(str, Enum):
    """
    Defines the mutation policy for an attribute slot on an EntityNode.
    
    - TIME_SLICED: Versioned update preserving historical lineage. Previous value is
      pushed into an immutable TimeSlice record with closed [valid_from, valid_to] bounds.
    - OVERWRITE: Complete destructive in-place replacement. Any prior historical slices
      for this slot are permanently purged, and the current value is overwritten.
    """
    TIME_SLICED = "TIME_SLICED"
    OVERWRITE = "OVERWRITE"


class EntityScope(str, Enum):
    """
    3-Tier Hierarchical Scoping for Entities in the Knowledge Graph:
    - GLOBAL: Universal frameworks, core stack, primary architecture, and persistent actors.
    - SESSION: Bound to owner_session_id (sprint milestones, localized standards).
    - EPISODE: Bound to owner_episode_id (local scratch variables, transient sub-episode files).
    """
    GLOBAL = "GLOBAL"
    SESSION = "SESSION"
    EPISODE = "EPISODE"


class GraphModelError(Exception):
    """Base exception for all graph model and schema operations."""
    pass


class EntityNotFoundError(GraphModelError):
    """Raised when an operation targets an unmapped entity identifier."""
    pass


class ValidationError(GraphModelError):
    """Raised when incoming data violates model invariants."""
    pass


# ============================================================================
# Serialization & Numerical Utility Functions
# ============================================================================

def extract_tokens(text: str) -> Set[str]:
    """Extracts alphanumeric tokens with length >= 2 without regex."""
    clean = "".join(c if (c.isalnum() or c in "_-.") else " " for c in text.lower())
    return {tok for tok in clean.split() if len(tok) >= 2}


def text_contains_phrase(query: str, phrase: str) -> bool:
    """Checks exact whole-word or phrase containment without regex."""
    p = phrase.lower().strip()
    if not p:
        return False
    q = query.lower()
    if p in q:
        idx = q.find(p)
        while idx != -1:
            left_ok = (idx == 0 or not q[idx - 1].isalnum())
            end_idx = idx + len(p)
            right_ok = (end_idx == len(q) or not q[end_idx].isalnum())
            if left_ok and right_ok:
                return True
            idx = q.find(p, idx + 1)
    return False


def slugify(name: str) -> str:
    """Generates an alphanumeric slug without regex."""
    s = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def epoch_to_iso(epoch: float) -> str:
    """Converts a UNIX epoch timestamp (seconds) into an ISO 8601 UTC string."""
    if epoch == float("inf"):
        return "present"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def iso_to_epoch(iso_str: str) -> float:
    """Parses an ISO 8601 string to a UNIX epoch timestamp (seconds)."""
    if iso_str.lower() in {"present", "infinity", "inf"}:
        return float("inf")
    clean_str = iso_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_str).timestamp()


def parse_date_string(date_str: str) -> Optional[float]:
    """Parses various date and month formats into a UNIX epoch timestamp (seconds). Zero regexes."""
    if not date_str or not isinstance(date_str, str):
        return None
    clean = date_str.strip()
    for sfx in ("st", "nd", "rd", "th"):
        for d in "0123456789":
            clean = clean.replace(f"{d}{sfx}", d)
    for fmt in (
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B-%d-%Y",
        "%b-%d-%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%B %Y",
        "%b %Y",
    ):
        try:
            dt = datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    return None


def extract_dates_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts explicit date occurrences via spaCy DATE NER.
    Zero regexes.
    """
    if not text or not isinstance(text, str):
        return []
    from .nlp_extractor import get_spacy_nlp
    doc = get_spacy_nlp()(text)
    found: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "DATE":
            raw = ent.text.strip()
            if raw.lower() not in seen:
                seen.add(raw.lower())
                ep = parse_date_string(raw)
                if ep is not None:
                    found.append({
                        "raw": raw,
                        "epoch": ep,
                        "iso": epoch_to_iso(ep),
                    })
    return found


def compute_relative_temporal_metadata(
    target_epoch: float,
    ref_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes structured relative temporal processing metadata for a target epoch
    relative to a reference (current / evaluation) epoch. Zero regexes.

    Produces relative phrases such as:
      - 'today', 'yesterday', '2 days ago', 'last week', '3 weeks ago',
      - 'last month', '2 months ago', 'last year', '2 years ago'
      - or future: 'tomorrow', 'next week', 'next month', 'next year'
    """
    if target_epoch is None or target_epoch <= 0:
        return {}

    now_epoch = ref_epoch if (ref_epoch is not None and ref_epoch > 0) else time.time()
    try:
        target_dt = datetime.fromtimestamp(target_epoch, tz=timezone.utc)
        ref_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return {}

    delta_days = (ref_dt.date() - target_dt.date()).days
    delta_months = (ref_dt.year - target_dt.year) * 12 + (ref_dt.month - target_dt.month)
    delta_years = ref_dt.year - target_dt.year

    if delta_days == 0:
        phrase = "today"
        bucket = "TODAY"
    elif delta_days == 1:
        phrase = "yesterday"
        bucket = "YESTERDAY"
    elif 2 <= delta_days <= 6:
        phrase = f"{delta_days} days ago"
        bucket = "THIS_WEEK" if target_dt.isocalendar()[1] == ref_dt.isocalendar()[1] else "PAST_DAYS"
    elif 7 <= delta_days < 14:
        phrase = "last week"
        bucket = "LAST_WEEK"
    elif 14 <= delta_days < 28:
        weeks = delta_days // 7
        phrase = f"{weeks} weeks ago"
        bucket = "PAST_WEEKS"
    elif delta_months == 1 and 28 <= delta_days < 60:
        phrase = "last month"
        bucket = "LAST_MONTH"
    elif 2 <= delta_months < 12:
        phrase = f"{delta_months} months ago"
        bucket = "PAST_MONTHS"
    elif delta_years == 1 and delta_months >= 12:
        phrase = "last year"
        bucket = "LAST_YEAR"
    elif delta_years >= 2:
        phrase = f"{delta_years} years ago"
        bucket = "PAST_YEARS"
    elif delta_days == -1:
        phrase = "tomorrow"
        bucket = "TOMORROW"
    elif -6 <= delta_days <= -2:
        phrase = f"in {abs(delta_days)} days"
        bucket = "NEXT_DAYS"
    elif -13 <= delta_days <= -7:
        phrase = "next week"
        bucket = "NEXT_WEEK"
    elif delta_months == -1:
        phrase = "next month"
        bucket = "NEXT_MONTH"
    elif delta_years == -1:
        phrase = "next year"
        bucket = "NEXT_YEAR"
    elif delta_days < 0:
        phrase = f"in {abs(delta_days)} days"
        bucket = "FUTURE"
    else:
        phrase = "past"
        bucket = "PAST"

    return {
        "relative_phrase": phrase,
        "relative_bucket": bucket,
        "delta_days": int(delta_days),
        "delta_weeks": round(delta_days / 7.0, 1),
        "delta_months": int(delta_months),
        "delta_years": int(delta_years),
        "target_date_iso": target_dt.strftime("%Y-%m-%d"),
        "ref_date_iso": ref_dt.strftime("%Y-%m-%d"),
    }


def ndarray_to_bytes(arr: Optional[np.ndarray]) -> Optional[bytes]:
    """Serializes a NumPy float array to raw bytes for BLOB storage."""
    if arr is None:
        return None
    arr_clean = np.ascontiguousarray(arr, dtype=np.float32)
    return arr_clean.tobytes()


def bytes_to_ndarray(data: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserializes raw bytes from BLOB storage into a 1D NumPy float32 array."""
    if not data:
        return None
    return np.frombuffer(data, dtype=np.float32).copy()


# ============================================================================
# Core Graph Model Primitives
# ============================================================================

@dataclass
class TimeSlice:
    """
    An immutable, versioned historical slice of knowledge.
    
    Attributes:
        slice_id: Unique identifier for this historical record.
        node_id: Foreign key to parent KnowledgeNode.
        entity_id: Canonical entity identifier.
        aspect: Attribute slot key (e.g., 'primary_database', 'deadline').
        value: The value valid during this time window.
        valid_from: Epoch timestamp when this value became active.
        valid_to: Epoch timestamp when this value was superseded.
        taxonomic_state: Continuous tau coordinate vector in [-1.0, +1.0]^K.
        episode_id: Optional reference to the originating TimeframeEpisode.
        mutation_reason: Human/LLM rationale for why the state evolved.
    """
    slice_id: str
    node_id: str
    entity_id: str
    aspect: str
    value: Any
    valid_from: float
    valid_to: float
    taxonomic_state: Optional[np.ndarray] = None
    episode_id: Optional[str] = None
    mutation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.valid_to < self.valid_from:
            raise ValidationError(
                f"Invalid temporal bounds: valid_to ({self.valid_to}) < valid_from ({self.valid_from})"
            )
        if self.taxonomic_state is not None and not isinstance(self.taxonomic_state, np.ndarray):
            self.taxonomic_state = np.array(self.taxonomic_state, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        duration_days = round((self.valid_to - self.valid_from) / 86400.0, 2) if self.valid_to < float("inf") else None
        return {
            "slice_id": self.slice_id,
            "node_id": self.node_id,
            "entity_id": self.entity_id,
            "aspect": self.aspect,
            "value": self.value,
            "valid_from": self.valid_from,
            "valid_from_iso": epoch_to_iso(self.valid_from),
            "valid_to": self.valid_to,
            "valid_to_iso": epoch_to_iso(self.valid_to),
            "duration_days": duration_days,
            "timeframe_label": f"[{epoch_to_iso(self.valid_from)[:10]} to {epoch_to_iso(self.valid_to)[:10]} ({duration_days} days)]" if duration_days else f"[From {epoch_to_iso(self.valid_from)[:10]}]",
            "taxonomic_state": self.taxonomic_state.tolist() if self.taxonomic_state is not None else None,
            "episode_id": self.episode_id,
            "mutation_reason": self.mutation_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeSlice:
        tau = np.array(data["taxonomic_state"], dtype=np.float32) if data.get("taxonomic_state") is not None else None
        return cls(
            slice_id=data["slice_id"],
            node_id=data["node_id"],
            entity_id=data["entity_id"],
            aspect=data["aspect"],
            value=data["value"],
            valid_from=float(data["valid_from"]),
            valid_to=float(data["valid_to"]),
            taxonomic_state=tau,
            episode_id=data.get("episode_id"),
            mutation_reason=data.get("mutation_reason"),
        )


@dataclass
class TimeframeEpisode:
    """
    Segregated, chronologically ordered episodic container.
    
    Represents an event, interaction turn, or macro period (e.g. 'College Days', 'Night Party',
    'Incident #402') connecting multiple entities and assertions.
    """
    episode_id: str
    name: Optional[str] = None
    start_epoch: float = field(default_factory=time.time)
    end_epoch: Optional[float] = None
    summary: Optional[str] = None
    connected_entity_ids: Set[str] = field(default_factory=set)
    connected_episode_ids: Set[str] = field(default_factory=set)
    assertion_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"Episode @ {epoch_to_iso(self.start_epoch)}"
        if self.end_epoch is not None and self.end_epoch < self.start_epoch:
            raise ValidationError(
                f"Episode '{self.episode_id}' has end_epoch < start_epoch ({self.end_epoch} < {self.start_epoch})"
            )
        if not isinstance(self.connected_entity_ids, set):
            self.connected_entity_ids = set(self.connected_entity_ids)
        if not isinstance(self.connected_episode_ids, set):
            self.connected_episode_ids = set(self.connected_episode_ids)

    @property
    def entities(self) -> Set[str]:
        """Convenience property alias for connected_entity_ids."""
        return self.connected_entity_ids

    def add_entity(self, entity_id: str) -> None:
        """Binds an entity to this episode."""
        self.connected_entity_ids.add(entity_id)

    def add_connected_episode(self, episode_id: str) -> None:
        """Binds a linked episode to this episode."""
        if episode_id != self.episode_id:
            self.connected_episode_ids.add(episode_id)

    def add_assertion(self, node_id: str) -> None:
        """Binds a knowledge assertion created or updated during this episode."""
        if node_id not in self.assertion_ids:
            self.assertion_ids.append(node_id)

    def close(self, end_epoch: Optional[float] = None) -> None:
        """Closes the episode duration."""
        self.end_epoch = end_epoch if end_epoch is not None else time.time()
        if self.end_epoch < self.start_epoch:
            self.end_epoch = self.start_epoch

    def contains_epoch(self, epoch: float) -> bool:
        """Checks if a given epoch falls within this episode."""
        if epoch < self.start_epoch:
            return False
        if self.end_epoch is not None and epoch > self.end_epoch:
            return False
        return True

    def compute_relative_time(self, ref_epoch: Optional[float] = None) -> Dict[str, Any]:
        """Calculates and stores relative temporal metadata relative to reference epoch."""
        rel_meta = compute_relative_temporal_metadata(self.start_epoch, ref_epoch=ref_epoch)
        self.metadata["relative_time"] = rel_meta
        return rel_meta

    def to_dict(self, ref_epoch: Optional[float] = None) -> Dict[str, Any]:
        if "relative_time" not in self.metadata or ref_epoch is not None:
            self.compute_relative_time(ref_epoch=ref_epoch)
        rel_time = self.metadata.get("relative_time", {})
        return {
            "episode_id": self.episode_id,
            "name": self.name,
            "start_epoch": self.start_epoch,
            "start_iso": epoch_to_iso(self.start_epoch),
            "end_epoch": self.end_epoch,
            "end_iso": epoch_to_iso(self.end_epoch) if self.end_epoch is not None else None,
            "summary": self.summary,
            "connected_entity_ids": sorted(list(self.connected_entity_ids)),
            "connected_episode_ids": sorted(list(self.connected_episode_ids)),
            "assertion_ids": list(self.assertion_ids),
            "metadata": dict(self.metadata),
            "relative_time": dict(rel_time),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeframeEpisode:
        return cls(
            episode_id=data["episode_id"],
            name=data.get("name"),
            start_epoch=float(data["start_epoch"]),
            end_epoch=float(data["end_epoch"]) if data.get("end_epoch") is not None else None,
            summary=data.get("summary"),
            connected_entity_ids=set(data.get("connected_entity_ids", [])),
            connected_episode_ids=set(data.get("connected_episode_ids", [])),
            assertion_ids=list(data.get("assertion_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class KnowledgeNode:
    """
    An atomic attribute or factual slot anchored to an EntityNode.
    
    Maintains the authoritative current_value accessible in O(1), and optionally
    an audit trail of historical TimeSlice objects when mutated under TIME_SLICED policy.
    """
    node_id: str
    entity_id: str
    aspect: str
    current_value: Any
    active_since: float = field(default_factory=time.time)
    taxonomic_state: Optional[np.ndarray] = None
    current_episode_id: Optional[str] = None
    history: List[TimeSlice] = field(default_factory=list)
    access_count: int = 1
    last_activated: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.taxonomic_state is not None and not isinstance(self.taxonomic_state, np.ndarray):
            self.taxonomic_state = np.array(self.taxonomic_state, dtype=np.float32)

    def mutate(
        self,
        new_value: Any,
        mode: UpdateMode,
        occurred_at: float,
        episode_id: Optional[str] = None,
        tau: Optional[np.ndarray] = None,
        reason: Optional[str] = None,
    ) -> Optional[TimeSlice]:
        """
        Executes a state transition under the specified UpdateMode.
        
        Returns:
            The created TimeSlice if mode == TIME_SLICED, else None.
        """
        created_slice: Optional[TimeSlice] = None

        if mode == UpdateMode.TIME_SLICED:
            # Pathway 1: Versioned historical transition
            # Close previous value into a historical slice
            slice_end = max(occurred_at, self.active_since)
            created_slice = TimeSlice(
                slice_id=f"ts_{self.node_id}_{int(self.active_since)}_{uuid.uuid4().hex[:6]}",
                node_id=self.node_id,
                entity_id=self.entity_id,
                aspect=self.aspect,
                value=self.current_value,
                valid_from=self.active_since,
                valid_to=slice_end,
                taxonomic_state=self.taxonomic_state.copy() if self.taxonomic_state is not None else None,
                episode_id=self.current_episode_id,
                mutation_reason=reason,
            )
            self.history.append(created_slice)
            # Maintain chronological ordering of history
            self.history.sort(key=lambda s: s.valid_from)

        elif mode == UpdateMode.OVERWRITE:
            # Pathway 2: Destructive complete replacement (forget previous states)
            self.history.clear()

        # Update current active state
        self.current_value = new_value
        self.active_since = occurred_at
        self.current_episode_id = episode_id
        if tau is not None:
            self.taxonomic_state = np.array(tau, dtype=np.float32)
        self.last_activated = occurred_at
        self.access_count += 1

        return created_slice

    def get_active(self) -> Any:
        """Returns the presently active truth in O(1)."""
        self.access_count += 1
        self.last_activated = time.time()
        return self.current_value

    def get_at_epoch(self, query_epoch: float) -> Optional[Any]:
        """
        Resolves historical state at an arbitrary epoch timestamp.
        Returns None if no record existed at that epoch or if overwritten.
        """
        if query_epoch >= self.active_since:
            return self.current_value

        # Search past time slices in reverse chronological order
        for past_slice in reversed(self.history):
            if past_slice.valid_from <= query_epoch < past_slice.valid_to:
                return past_slice.value

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity_id": self.entity_id,
            "aspect": self.aspect,
            "current_value": self.current_value,
            "active_since": self.active_since,
            "active_since_iso": epoch_to_iso(self.active_since),
            "valid_from": self.active_since,
            "valid_from_iso": epoch_to_iso(self.active_since),
            "valid_to": float("inf"),
            "valid_to_iso": "present",
            "timeframe_label": f"[Active since {epoch_to_iso(self.active_since)[:10]}]",
            "taxonomic_state": self.taxonomic_state.tolist() if self.taxonomic_state is not None else None,
            "current_episode_id": self.current_episode_id,
            "history": [s.to_dict() for s in self.history],
            "access_count": self.access_count,
            "last_activated": self.last_activated,
            "last_activated_iso": epoch_to_iso(self.last_activated),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeNode:
        tau = np.array(data["taxonomic_state"], dtype=np.float32) if data.get("taxonomic_state") is not None else None
        history = [TimeSlice.from_dict(s) for s in data.get("history", [])]
        return cls(
            node_id=data["node_id"],
            entity_id=data["entity_id"],
            aspect=data["aspect"],
            current_value=data["current_value"],
            active_since=float(data["active_since"]),
            taxonomic_state=tau,
            current_episode_id=data.get("current_episode_id"),
            history=history,
            access_count=int(data.get("access_count", 1)),
            last_activated=float(data.get("last_activated", data["active_since"])),
        )


@dataclass
class EntityNode:
    """
    The primary addressable entity in the knowledge graph.
    
    Coordinates canonical identity, surface aliases, slotted knowledge attributes,
    scope (GLOBAL, SESSION, EPISODE), and references to connected episodic timeframes.
    """
    entity_id: str
    canonical_name: str
    entity_type: str = "concept"
    aliases: Set[str] = field(default_factory=set)
    slotted_knowledge: Dict[str, KnowledgeNode] = field(default_factory=dict)
    connected_episode_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    scope: EntityScope = EntityScope.SESSION
    owner_session_id: Optional[str] = None
    owner_episode_id: Optional[str] = None
    reference_count: int = 1
    sub_episode_ids: Set[str] = field(default_factory=set)
    session_ids: Set[str] = field(default_factory=set)
    has_factual_anchor: bool = False
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.aliases, set):
            self.aliases = set(self.aliases)
        # Ensure canonical name is always registered as an alias
        self.aliases.add(self.canonical_name.strip())
        if not isinstance(self.sub_episode_ids, set):
            self.sub_episode_ids = set(self.sub_episode_ids)
        if not isinstance(self.session_ids, set):
            self.session_ids = set(self.session_ids)
        if isinstance(self.scope, str):
            try:
                self.scope = EntityScope(self.scope)
            except ValueError:
                self.scope = EntityScope.GLOBAL

    def add_alias(self, alias: str) -> None:
        """Registers a surface form alias."""
        clean = alias.strip()
        if clean:
            self.aliases.add(clean)

    def matches_alias(self, mention: str) -> bool:
        """Checks if a string matches the canonical name, ID, or any alias."""
        clean = mention.strip().lower()
        if self.canonical_name.lower() == clean or self.entity_id.lower() == clean:
            return True
        for a in self.aliases:
            if a.lower() == clean:
                return True
        return False

    def get_active_slots(self) -> Dict[str, Any]:
        """Returns a dictionary of all active aspect -> value pairs."""
        return {aspect: kn.get_active() for aspect, kn in self.slotted_knowledge.items()}

    def get_slot_value(self, aspect: str, default: Any = None) -> Any:
        """Returns the current active value for a slot aspect, or default if not set."""
        kn = self.slotted_knowledge.get(aspect.strip().lower())
        return kn.get_active() if kn is not None else default

    def link_episode(self, episode_id: str) -> None:
        """Links an episode to this entity in chronological insertion order."""
        if episode_id not in self.connected_episode_ids:
            self.connected_episode_ids.append(episode_id)
        self.last_updated = time.time()

    def set_slot(
        self,
        aspect: str,
        value: Any,
        mode: UpdateMode = UpdateMode.TIME_SLICED,
        occurred_at: Optional[float] = None,
        episode_id: Optional[str] = None,
        tau: Optional[np.ndarray] = None,
        reason: Optional[str] = None,
        slot_id: Optional[str] = None,
    ) -> KnowledgeNode:
        """
        Transactionally assigns or mutates an attribute slot or arbitrary data entry on this entity.
        Supports any size and number of data entries (single facts, whole files, code, or structured blobs).
        """
        now = time.time() if occurred_at is None else occurred_at
        aspect_clean = aspect.strip().lower()
        key = slot_id.strip().lower() if slot_id else aspect_clean
        node_id = f"kn_{self.entity_id}_{key}"

        # Factual anchor: receiving a knowledge slot or data entry anchors the entity permanently
        self.has_factual_anchor = True

        if key not in self.slotted_knowledge:
            # Genesis assignment for this slot / data entry
            kn = KnowledgeNode(
                node_id=node_id,
                entity_id=self.entity_id,
                aspect=aspect_clean,
                current_value=value,
                active_since=now,
                taxonomic_state=tau,
                current_episode_id=episode_id,
                history=[],
                last_activated=now,
            )
            self.slotted_knowledge[key] = kn
        else:
            # Mutation on existing slot / entry
            kn = self.slotted_knowledge[key]
            kn.mutate(
                new_value=value,
                mode=mode,
                occurred_at=now,
                episode_id=episode_id,
                tau=tau,
                reason=reason,
            )

        if episode_id:
            self.link_episode(episode_id)

        self.last_updated = now
        return kn

    def get_slot_value(self, aspect: str, default: Any = None) -> Any:
        """Returns the current active value for a slot aspect or key, or default if not set."""
        aspect_clean = aspect.strip().lower()
        if aspect_clean in self.slotted_knowledge:
            return self.slotted_knowledge[aspect_clean].get_active()
        for kn in self.slotted_knowledge.values():
            if kn.aspect == aspect_clean:
                return kn.get_active()
        return default

    def get_all_entries(self) -> List[KnowledgeNode]:
        """Returns all knowledge nodes / data entries stored under this entity."""
        return list(self.slotted_knowledge.values())

    def get_active(self, aspect: str) -> Optional[Any]:
        """O(1) retrieval of current authoritative slot value."""
        return self.get_slot_value(aspect, default=None)

    def get_history(self, aspect: str) -> List[TimeSlice]:
        """Returns the chronological history of past time slices for an aspect."""
        aspect_clean = aspect.strip().lower()
        kn = self.slotted_knowledge.get(aspect_clean)
        return list(kn.history) if kn is not None else []

    def get_at_epoch(self, aspect: str, query_epoch: float) -> Optional[Any]:
        """Resolves historical attribute value valid at query_epoch."""
        aspect_clean = aspect.strip().lower()
        kn = self.slotted_knowledge.get(aspect_clean)
        return kn.get_at_epoch(query_epoch) if kn is not None else None

    def get_active_slots_with_timeframe(self) -> List[Dict[str, Any]]:
        """
        Returns all active knowledge slots enriched with complete temporal validity,
        ISO timestamp intervals, duration active, and historical mutation history.
        """
        results = []
        for aspect, kn in self.slotted_knowledge.items():
            results.append({
                "aspect": kn.aspect,
                "current_value": kn.current_value,
                "active_since": kn.active_since,
                "active_since_iso": epoch_to_iso(kn.active_since),
                "valid_from": kn.active_since,
                "valid_from_iso": epoch_to_iso(kn.active_since),
                "valid_to": float("inf"),
                "valid_to_iso": "present",
                "timeframe_label": f"[Active since {epoch_to_iso(kn.active_since)[:10]}]",
                "mutation_count": len(kn.history),
                "history": [s.to_dict() for s in kn.history],
                "current_episode_id": kn.current_episode_id,
            })
        return results

    def match_aspects_and_slices(
        self,
        query_text: str,
        query_vec: Optional[np.ndarray] = None,
        embedder: Any = None,
        query_epoch: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs dual regex + semantic match checking on 'aspect' ("what"), slot values,
        and historical time slices, alongside exact date and timeframe validity verification.
        
        Evaluates:
        1. Aspect ("what") regex & token matching: e.g. \b{aspect}\b in query.
        2. Value regex & token matching: e.g. exact strings like "March 18, 2024", "DeepL API v2", "PostgreSQL 14".
        3. Semantic cosine similarity: query vector vs slot representation ("{canonical_name} {aspect}: {value}").
        4. Temporal timeframe calculation: checks if dates or epochs in query align with valid_from/valid_to,
           and computes duration in days/weeks.
        """
        q_lower = query_text.lower().strip()
        q_tokens = extract_tokens(q_lower)
        dates_in_query = extract_dates_from_text(query_text)
        q_date_epochs = [d["epoch"] for d in dates_in_query]

        matches: List[Dict[str, Any]] = []

        for aspect_key, kn in self.slotted_knowledge.items():
            aspect_name = kn.aspect.lower()
            val_str = str(kn.current_value).lower()
            
            # 1. Lexical Check on "aspect" ("what") without regex
            aspect_tokens = extract_tokens(aspect_name)
            aspect_hit = text_contains_phrase(q_lower, aspect_name) or text_contains_phrase(q_lower, aspect_name.replace("_", " "))
            aspect_overlap = len(q_tokens & aspect_tokens) / max(len(aspect_tokens), 1) if aspect_tokens else 0.0

            # 2. Lexical Check on current value without regex
            val_tokens = extract_tokens(val_str)
            val_hit = text_contains_phrase(q_lower, val_str) if len(val_str) > 2 else False
            val_overlap = len(q_tokens & val_tokens) / max(len(val_tokens), 1) if val_tokens else 0.0

            s_lex = 0.0
            if aspect_hit and val_hit:
                s_lex = 1.0
            elif aspect_hit:
                s_lex = 0.88 + 0.10 * val_overlap
            elif val_hit:
                s_lex = 0.85 + 0.10 * aspect_overlap
            elif aspect_overlap > 0:
                s_lex = 0.40 + 0.35 * aspect_overlap
            elif val_overlap > 0:
                s_lex = 0.35 + 0.30 * val_overlap

            # 3. Check historical time slices for previous values or superseded aspects
            hist_matches = []
            for past_slice in kn.history:
                p_aspect = past_slice.aspect.lower()
                p_val_str = str(past_slice.value).lower()
                p_aspect_tokens = extract_tokens(p_aspect)
                p_val_tokens = extract_tokens(p_val_str)
                p_overlap = len(q_tokens & (p_aspect_tokens | p_val_tokens))
                p_hit = p_overlap > 0 or text_contains_phrase(q_lower, p_aspect) or (len(p_val_str) > 2 and text_contains_phrase(q_lower, p_val_str))
                
                # Check if query date falls within past slice timeframe
                t_hit = False
                for q_ep in q_date_epochs:
                    if past_slice.valid_from <= q_ep <= past_slice.valid_to:
                        t_hit = True
                        break
                
                # Include slice if slice matches, date matches, or parent slot has lexical/semantic match
                if p_hit or t_hit or aspect_overlap > 0 or val_overlap > 0 or s_lex > 0.20:
                    dur_days = round((past_slice.valid_to - past_slice.valid_from) / 86400.0, 2)
                    hist_matches.append({
                        "aspect": past_slice.aspect,
                        "value": past_slice.value,
                        "valid_from": past_slice.valid_from,
                        "valid_from_iso": epoch_to_iso(past_slice.valid_from),
                        "valid_to": past_slice.valid_to,
                        "valid_to_iso": epoch_to_iso(past_slice.valid_to),
                        "duration_days": dur_days,
                        "mutation_reason": past_slice.mutation_reason,
                    })

            # 4. Semantic Match Check on Aspect + Value
            s_sim = 0.0
            if query_vec is not None and embedder is not None:
                try:
                    slot_repr = f"{self.canonical_name} {kn.aspect}: {kn.current_value}"
                    slot_vec = embedder.embed_text(slot_repr)
                    norm_q = np.linalg.norm(query_vec)
                    norm_s = np.linalg.norm(slot_vec)
                    if norm_q > 1e-9 and norm_s > 1e-9:
                        s_sim = max(0.0, float(np.dot(query_vec / norm_q, slot_vec / norm_s)))
                except Exception:
                    pass

            # 5. Temporal / Timeframe Relevance Check
            s_temporal = 0.0
            cur_valid_from = kn.active_since
            cur_valid_from_iso = epoch_to_iso(cur_valid_from)

            for q_ep in q_date_epochs:
                if q_ep >= cur_valid_from:
                    s_temporal = max(s_temporal, 0.90)
                else:
                    for p in hist_matches:
                        if p["valid_from"] <= q_ep <= p["valid_to"]:
                            s_temporal = max(s_temporal, 0.95)

            if query_epoch is not None:
                if query_epoch >= cur_valid_from:
                    s_temporal = max(s_temporal, 0.50)

            # Combined score for this slot
            fused_score = max(
                s_lex,
                0.50 * s_lex + 0.35 * s_sim + 0.15 * s_temporal,
                0.85 * s_sim if s_sim > 0.65 else 0.0,
                0.80 if s_temporal > 0.85 and (aspect_overlap > 0 or val_overlap > 0) else 0.0,
            )

            if fused_score > 0.15 or hist_matches:
                now_t = time.time() if query_epoch is None else query_epoch
                days_active = round(max(0.0, now_t - cur_valid_from) / 86400.0, 2)
                matches.append({
                    "entity_id": self.entity_id,
                    "canonical_name": self.canonical_name,
                    "aspect": kn.aspect,
                    "value": kn.current_value,
                    "valid_from": cur_valid_from,
                    "valid_from_iso": cur_valid_from_iso,
                    "valid_to": float("inf"),
                    "valid_to_iso": "present",
                    "days_active": days_active,
                    "timeframe_label": f"[Active since {cur_valid_from_iso[:10]}]",
                    "history_slices": hist_matches,
                    "lexical_score": round(float(s_lex), 4),
                    "semantic_score": round(float(s_sim), 4),
                    "temporal_score": round(float(s_temporal), 4),
                    "score": round(float(fused_score), 4),
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": sorted(list(self.aliases)),
            "slotted_knowledge": {k: v.to_dict() for k, v in self.slotted_knowledge.items()},
            "connected_episode_ids": list(self.connected_episode_ids),
            "importance": self.importance,
            "scope": self.scope.value if isinstance(self.scope, EntityScope) else str(self.scope),
            "owner_session_id": self.owner_session_id,
            "owner_episode_id": self.owner_episode_id,
            "reference_count": self.reference_count,
            "sub_episode_ids": sorted(list(self.sub_episode_ids)),
            "session_ids": sorted(list(self.session_ids)),
            "has_factual_anchor": self.has_factual_anchor or len(self.slotted_knowledge) > 0,
            "created_at": self.created_at,
            "created_at_iso": epoch_to_iso(self.created_at),
            "last_updated": self.last_updated,
            "last_updated_iso": epoch_to_iso(self.last_updated),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityNode:
        slotted = {
            k: KnowledgeNode.from_dict(v)
            for k, v in data.get("slotted_knowledge", {}).items()
        }
        scope_val = data.get("scope", "GLOBAL")
        try:
            scope_enum = EntityScope(scope_val)
        except ValueError:
            scope_enum = EntityScope.GLOBAL

        return cls(
            entity_id=data["entity_id"],
            canonical_name=data["canonical_name"],
            entity_type=data.get("entity_type", "concept"),
            aliases=set(data.get("aliases", [])),
            slotted_knowledge=slotted,
            connected_episode_ids=list(data.get("connected_episode_ids", [])),
            importance=float(data.get("importance", 0.5)),
            scope=scope_enum,
            owner_session_id=data.get("owner_session_id"),
            owner_episode_id=data.get("owner_episode_id"),
            reference_count=int(data.get("reference_count", 1)),
            sub_episode_ids=set(data.get("sub_episode_ids", [])),
            session_ids=set(data.get("session_ids", [])),
            has_factual_anchor=bool(data.get("has_factual_anchor", len(slotted) > 0)),
            created_at=float(data.get("created_at", time.time())),
            last_updated=float(data.get("last_updated", time.time())),
        )


# ============================================================================
# Thread-Safe In-Memory Entity Registry with Dual-Factor Matching
# ============================================================================

class EntityRegistry:
    """
    Thread-safe canonical entity registry with hierarchical 3-tier scoping,
    resolution-first linking, and dual-factor (lexical + dense semantic) resolution.
    Zero hardcoded word dictionaries: uses linguistic POS and dependency relations.
    """

    @staticmethod
    def clean_mention_name(raw_name: str) -> Optional[str]:
        """
        Pure grammatical mention cleaner:
        Strips leading determiners, numbers, and pronouns using spaCy POS and universal dependency tags.
        Rejects candidates that lack a substantive PROPN or NOUN head or consist entirely of stopwords.
        """
        if not raw_name or not raw_name.strip():
            return None

        # Quick pre-filter for non-text / markup
        stripped = raw_name.strip(" ,.:;\"'()[]{}<>`~*#\n\r\t")
        if len(stripped) < 2 or len(stripped) > 50:
            return None

        try:
            from .nlp_extractor import get_spacy_nlp
        except ImportError:
            from tenor.nlp_extractor import get_spacy_nlp
        nlp = get_spacy_nlp()
        doc = nlp(stripped)
        tokens = list(doc)

        # Grammatically strip leading determiners, numbers, pronouns, or stopwords
        while tokens and (tokens[0].is_stop or tokens[0].like_num or tokens[0].pos_ in {"DET", "NUM", "PRON"} or tokens[0].dep_ in {"det", "nummod"}):
            tokens.pop(0)

        if not tokens:
            return None

        # Check if substantive noun/propn or uppercase named technical identifier
        is_named_or_tech = (
            any(c.isupper() for c in stripped[1:])
            or any(c in stripped for c in ("_", "/", "."))
            or stripped[0].isupper()
        )
        has_substantive = any(t.pos_ in {"PROPN", "NOUN"} and not t.is_stop for t in tokens) or is_named_or_tech
        if not has_substantive:
            return None

        clean = " ".join([t.text for t in tokens]).strip(" ,.:;\"'()[]{}<>`~*#\n\r\t")
        if len(clean) < 2 or len(clean) > 50:
            return None

        # Structural Entity Typology: must be proper noun, technical identifier, compound concept, or capitalized concept noun
        is_propn = any(t.pos_ == "PROPN" for t in tokens)
        is_tech = any(c in clean for c in ("_", "/", ".")) or any(c.isupper() for c in clean[1:]) or any(c.isdigit() for c in clean)
        is_capitalized_concept = clean[0].isupper() and any(t.pos_ in ("NOUN", "PROPN") and not t.is_stop for t in tokens)
        content_tokens = [t for t in tokens if not t.is_stop and not t.like_num and t.pos_ in ("NOUN", "PROPN", "ADJ")]
        is_compound = len(content_tokens) >= 2

        if not (is_propn or is_tech or is_compound or is_capitalized_concept):
            return None

        return clean

    def __init__(self, alpha: float = 0.50, match_threshold: float = 0.75):
        self._lock = threading.RLock()
        self.alpha = alpha
        self.match_threshold = match_threshold
        
        # entity_id -> EntityNode
        self.entities: Dict[str, EntityNode] = {}
        # normalized_alias -> entity_id
        self.alias_map: Dict[str, str] = {}
        # token -> Set[entity_id] (inverted token index for O(1) candidate pruning)
        self.token_map: Dict[str, Set[str]] = {}
        # entity_id -> normalized unit vector (R^D)
        self.entity_vectors: Dict[str, np.ndarray] = {}

    def is_visible(
        self,
        entity: EntityNode,
        active_session_id: Optional[str] = None,
        active_episode_id: Optional[str] = None,
    ) -> bool:
        """
        Hierarchical Visibility:
        - GLOBAL: universally visible across all sessions and episodes.
        - SESSION: visible within matching owner_session_id.
        - EPISODE: visible within same session across sub-episodes for recurrence linking.
        """
        if entity.scope == EntityScope.GLOBAL:
            return True
        if entity.scope == EntityScope.SESSION:
            return active_session_id is None or entity.owner_session_id == active_session_id
        if entity.scope == EntityScope.EPISODE:
            return active_session_id is None or entity.owner_session_id == active_session_id
        return True

    def _record_reference(
        self,
        entity: EntityNode,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """Records entity mention and executes automatic scope promotion."""
        entity.reference_count += 1
        if episode_id:
            entity.sub_episode_ids.add(episode_id)
            # Automatic Promotion: EPISODE -> SESSION if seen across >= 2 sub-episodes
            if entity.scope == EntityScope.EPISODE and len(entity.sub_episode_ids) >= 2:
                entity.scope = EntityScope.SESSION
                entity.owner_session_id = session_id
                logger.info(f"Promoted entity '{entity.canonical_name}' EPISODE -> SESSION (seen across {len(entity.sub_episode_ids)} sub-episodes)")

        if session_id:
            entity.session_ids.add(session_id)
            # Automatic Promotion: SESSION -> GLOBAL if seen across >= 2 sessions
            if len(entity.session_ids) >= 2:
                entity.scope = EntityScope.GLOBAL
                logger.info(f"Promoted entity '{entity.canonical_name}' to GLOBAL (seen across {len(entity.session_ids)} sessions)")

    def is_admissible(self, entity: EntityNode) -> bool:
        """
        Two-Tier Entity Creation Gating:
        - Rule A (Factual Anchor): Holds a knowledge slot, code, schema, config, or fact -> Admitted immediately.
        - Rule B (Recurrence Anchor): Recurs across >= 2 sub-episodes or >= 3 turns -> Admitted.
        - Rule C (Singleton Action Drop): Mentioned once with 0 assertions / slots -> Rejected as transient noise.
        """
        if entity.has_factual_anchor or len(entity.slotted_knowledge) > 0:
            return True
        if len(entity.sub_episode_ids) >= 2 or entity.reference_count >= 3:
            return True
        return False

    def prune_inadmissible(self) -> List[str]:
        """Prunes entities that fail creation gating rules (Rule C singleton drop)."""
        pruned_ids = []
        with self._lock:
            to_remove = [eid for eid, ent in self.entities.items() if not self.is_admissible(ent)]
            for eid in to_remove:
                ent = self.entities.pop(eid)
                self.entity_vectors.pop(eid, None)
                for a in ent.aliases:
                    if self.alias_map.get(a.lower()) == eid:
                        del self.alias_map[a.lower()]
                pruned_ids.append(eid)
        return pruned_ids

    def _index_alias(self, alias_str: str, entity_id: str) -> None:
        """Indexes an alias and its constituent tokens."""
        clean = alias_str.lower().strip()
        if not clean:
            return
        self.alias_map[clean] = entity_id
        for tok in clean.split():
            if len(tok) >= 2:
                if tok not in self.token_map:
                    self.token_map[tok] = set()
                self.token_map[tok].add(entity_id)

    def register_entity(self, entity: EntityNode, vector: Optional[np.ndarray] = None) -> EntityNode:
        """Registers an entity and indexes its aliases."""
        with self._lock:
            self.entities[entity.entity_id] = entity
            for alias in entity.aliases:
                self._index_alias(alias, entity.entity_id)
            if vector is not None:
                norm = np.linalg.norm(vector)
                unit_vec = (vector / norm) if norm > 1e-9 else vector
                self.entity_vectors[entity.entity_id] = np.array(unit_vec, dtype=np.float32)
            return entity

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Direct O(1) lookup by canonical ID."""
        with self._lock:
            return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[EntityNode]:
        """Looks up an entity by canonical name or alias (case-insensitive)."""
        with self._lock:
            eid = self.alias_map.get(name.lower().strip())
            if eid:
                return self.entities.get(eid)
            return None

    def find_entities_in_text(self, text: str, min_len: int = 3) -> List[Tuple[EntityNode, float]]:
        """
        High-performance grep-style matching that identifies which known entities and aliases
        are present directly within the query/text, avoiding heuristic extraction.
        """
        if not text or not text.strip():
            return []

        text_lower = text.lower()
        # Fast stopword set for filtering bare single-word structural tokens
        structural_stopwords = {
            "what", "when", "where", "how", "why", "which", "who", "whom",
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "to", "from", "in", "on", "at", "by", "for", "with", "about",
            "into", "through", "during", "before", "after", "above", "below",
            "can", "could", "would", "should", "will", "shall", "may", "might",
            "did", "do", "does", "have", "has", "had", "tell", "me", "you",
            "my", "your", "our", "their", "there", "here", "this", "that",
            "and", "or", "but", "if", "order", "items", "item", "so", "as"
        }

        # Tokenize query without regex
        query_words = extract_tokens(text_lower)
        candidate_eids = set()
        with self._lock:
            for w in query_words:
                if w not in structural_stopwords and w in self.token_map:
                    candidate_eids.update(self.token_map[w])

            matched_results: List[Tuple[EntityNode, float]] = []
            seen_eids = set()

            for eid in candidate_eids:
                if eid in seen_eids or eid not in self.entities:
                    continue
                ent = self.entities[eid]
                best_score = 0.0

                for alias in ent.aliases:
                    a_lower = alias.lower().strip()
                    if a_lower in structural_stopwords or len(a_lower) < min_len:
                        continue
                    # Whole word containment check without regex
                    if text_contains_phrase(text_lower, a_lower):
                        score = 1.0 if len(a_lower.split()) > 1 else 0.85
                        if score > best_score:
                            best_score = score

                if best_score > 0.0:
                    matched_results.append((ent, best_score))
                    seen_eids.add(eid)

            # Sort by match confidence and alias length
            matched_results.sort(key=lambda x: (x[1], len(x[0].canonical_name)), reverse=True)
            return matched_results

    def compute_lexical_score(self, mention: str, entity: EntityNode) -> float:
        """
        Factor 1: Computes progressive 3-tier lexical matching score S_lex in [0.0, 1.0]:
        - Tier 1: Full word / substring containment (e.g., 'postgres' in 'postgresql').
        - Tier 2: Partial multi-word overlap / token intersection (e.g., 'postgres' in 'postgres 14').
        - Tier 3: Relative / fuzzy n-gram matching (character 3-grams).
        - Plus aspect and historical time-slice lexical matching.
        """
        m_norm = mention.lower().strip()
        if not m_norm:
            return 0.0

        scores = [0.0]
        mention_tokens = extract_tokens(m_norm)
        m_ngrams = set(m_norm[i:i+3] for i in range(len(m_norm) - 2)) if len(m_norm) >= 3 else set()

        for alias in entity.aliases:
            a_norm = alias.lower().strip()
            if not a_norm:
                continue

            # --- Tier 1: Full Word / Substring Containment ---
            if m_norm == a_norm:
                scores.append(1.0)
            elif m_norm in a_norm or a_norm in m_norm:
                len_ratio = min(len(m_norm), len(a_norm)) / max(len(m_norm), len(a_norm), 1)
                scores.append(0.85 + 0.12 * len_ratio)

            # --- Tier 2: Partial Multi-Word Overlap (Token Intersection) ---
            alias_tokens = extract_tokens(a_norm)
            common_tokens = mention_tokens & alias_tokens
            if common_tokens:
                jaccard = len(common_tokens) / max(len(mention_tokens | alias_tokens), 1)
                subset_bonus = 0.85 if (mention_tokens.issubset(alias_tokens) or alias_tokens.issubset(mention_tokens)) else 0.70
                scores.append(min(0.92, subset_bonus + 0.15 * jaccard))

            # --- Tier 3: Relative / Fuzzy N-Gram Match ---
            if m_ngrams and len(a_norm) >= 3:
                a_ngrams = set(a_norm[i:i+3] for i in range(len(a_norm) - 2))
                if a_ngrams:
                    ngram_sim = len(m_ngrams & a_ngrams) / max(len(m_ngrams | a_ngrams), 1)
                    if ngram_sim >= 0.40:
                        scores.append(0.55 + 0.30 * ngram_sim)

        # 4. Aspect ("what") and slot value lexical matching in slotted_knowledge (without regex)
        for aspect_key, kn in entity.slotted_knowledge.items():
            asp_clean = kn.aspect.lower()
            val_clean = str(kn.current_value).lower()

            if text_contains_phrase(m_norm, asp_clean) or text_contains_phrase(m_norm, asp_clean.replace("_", " ")):
                scores.append(0.88)
            else:
                asp_toks = extract_tokens(asp_clean.replace("_", " "))
                c = mention_tokens & asp_toks
                if c:
                    scores.append(0.65 + 0.20 * (len(c) / max(len(asp_toks), 1)))

            if len(val_clean) > 2:
                if text_contains_phrase(m_norm, val_clean):
                    scores.append(0.85)
                elif val_clean in m_norm or m_norm in val_clean:
                    scores.append(0.78)

            for past_slice in kn.history:
                p_asp = past_slice.aspect.lower()
                p_val = str(past_slice.value).lower()
                if text_contains_phrase(m_norm, p_asp):
                    scores.append(0.82)
                if len(p_val) > 2 and (text_contains_phrase(m_norm, p_val) or p_val in m_norm):
                    scores.append(0.80)

        return max(scores)

    def compute_similarity_score(self, query_vec: Optional[np.ndarray], entity_id: str) -> float:
        """
        Factor 2: Computes dense cosine similarity S_sim in [-1.0, 1.0].
        """
        if query_vec is None or entity_id not in self.entity_vectors:
            return 0.0
        e_vec = self.entity_vectors[entity_id]
        dot = float(np.dot(query_vec, e_vec))
        return max(-1.0, min(1.0, dot))

    def resolve_or_create(
        self,
        mention: str,
        embedding: Optional[np.ndarray] = None,
        entity_type: str = "concept",
        scope: EntityScope = EntityScope.SESSION,
        owner_session_id: Optional[str] = None,
        owner_episode_id: Optional[str] = None,
    ) -> Tuple[Optional[EntityNode], bool]:
        """
        Resolution-First Entity Resolution:
        1. Strips quantifiers ("several strategies" -> "strategies").
        2. Discards abstract discourse meta-nouns.
        3. Checks active visible scope for existing entities via exact alias match or token containment.
        4. Matches existing -> increments references, tracks sub_episode/session, triggers automatic promotion.
        5. No match -> registers new entity with specified scope.
        
        Returns:
            Tuple[Optional[EntityNode], was_created (bool)]
        """
        clean_mention = self.clean_mention_name(mention)
        if not clean_mention:
            return None, False

        m_lower = clean_mention.lower()

        with self._lock:
            # Step 1: Fast O(1) alias cache lookup
            if m_lower in self.alias_map:
                existing_id = self.alias_map[m_lower]
                ent = self.entities.get(existing_id)
                if ent:
                    self._record_reference(ent, owner_session_id, owner_episode_id)
                    return ent, False

            # Step 2: Token containment / substring check against existing visible entities
            # e.g., "PostgreSQL" in "PostgreSQL 14" -> match to existing canonical entity
            for alias_key, eid in self.alias_map.items():
                ent = self.entities.get(eid)
                if ent and self.is_visible(ent, owner_session_id, owner_episode_id):
                    if (m_lower in alias_key or alias_key in m_lower) and min(len(m_lower), len(alias_key)) >= 3:
                        ent.add_alias(clean_mention)
                        self._index_alias(clean_mention, ent.entity_id)
                        self._record_reference(ent, owner_session_id, owner_episode_id)
                        return ent, False

            # Normalize query vector if present
            q_unit = None
            if embedding is not None:
                norm = np.linalg.norm(embedding)
                q_unit = (embedding / norm) if norm > 1e-9 else embedding

            # Multi-tier candidate pruning via inverted token map
            cand_ids: Set[str] = set()
            m_tokens = m_lower.split()
            for tok in m_tokens:
                if tok in self.token_map:
                    cand_ids.update(self.token_map[tok])
                elif len(tok) >= 3:
                    for known_tok, eids in self.token_map.items():
                        if tok in known_tok or known_tok in tok:
                            cand_ids.update(eids)

            visible_candidates = [
                self.entities[cid] for cid in cand_ids
                if cid in self.entities and self.is_visible(self.entities[cid], owner_session_id, owner_episode_id)
            ]

            # If no direct token overlap but registry is small, evaluate all visible entities
            if not visible_candidates:
                if len(self.entities) <= 200:
                    visible_candidates = [
                        e for e in self.entities.values()
                        if self.is_visible(e, owner_session_id, owner_episode_id)
                    ]

            # Fast path if strictly no visible candidates exist across registry
            if not visible_candidates:
                slug = slugify(m_lower)
                canonical_id = f"{entity_type}:{slug}" if slug else f"entity_{uuid.uuid4().hex[:8]}"
                if canonical_id in self.entities:
                    canonical_id = f"{canonical_id}_{uuid.uuid4().hex[:4]}"
                new_entity = EntityNode(
                    entity_id=canonical_id,
                    canonical_name=clean_mention,
                    entity_type=entity_type,
                    aliases={clean_mention},
                    scope=scope,
                    owner_session_id=owner_session_id,
                    owner_episode_id=owner_episode_id,
                    reference_count=1,
                    sub_episode_ids={owner_episode_id} if owner_episode_id else set(),
                    session_ids={owner_session_id} if owner_session_id else set(),
                )
                self.register_entity(new_entity, vector=embedding)
                return new_entity, True

            # Step 3: Progressive 3-Tier Lexical + Vector Similarity Evaluation
            best_score = -1.0
            best_entity: Optional[EntityNode] = None

            for entity in visible_candidates:
                s_lex = self.compute_lexical_score(clean_mention, entity)
                s_sim = self.compute_similarity_score(q_unit, entity.entity_id) if q_unit is not None else 0.0

                s_fused = self.alpha * s_lex + (1.0 - self.alpha) * max(0.0, s_sim)
                s_match = max(s_fused, s_lex if s_lex >= 0.78 else 0.0, s_sim if s_sim >= 0.82 else 0.0)
                if s_match > best_score:
                    best_score = s_match
                    best_entity = entity

            # Step 4: Decision Boundary
            if best_entity is not None and best_score >= self.match_threshold:
                best_entity.add_alias(clean_mention)
                self._index_alias(clean_mention, best_entity.entity_id)
                self._record_reference(best_entity, owner_session_id, owner_episode_id)
                return best_entity, False

            # Step 5: Create new canonical entity with assigned scope
            slug = slugify(m_lower)
            canonical_id = f"{entity_type}:{slug}" if slug else f"entity_{uuid.uuid4().hex[:8]}"
            if canonical_id in self.entities:
                canonical_id = f"{canonical_id}_{uuid.uuid4().hex[:4]}"

            # Rule: Initially SESSION scope; promoted to GLOBAL ONLY when len(session_ids) >= 2
            sess_set = {owner_session_id} if owner_session_id else set()
            new_scope = EntityScope.GLOBAL if len(sess_set) >= 2 else EntityScope.SESSION
            new_entity = EntityNode(
                entity_id=canonical_id,
                canonical_name=clean_mention,
                entity_type=entity_type,
                aliases={clean_mention},
                scope=new_scope,
                owner_session_id=owner_session_id,
                owner_episode_id=owner_episode_id,
                reference_count=1,
                sub_episode_ids={owner_episode_id} if owner_episode_id else set(),
                session_ids=sess_set,
            )
            self.register_entity(new_entity, vector=q_unit)
            return new_entity, True

    def search_entities(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> List[Tuple[EntityNode, float]]:
        """Dual-factor high-speed candidate-pruned search across registered entities."""
        with self._lock:
            q_unit = None
            if query_vector is not None:
                norm = np.linalg.norm(query_vector)
                q_unit = (query_vector / norm) if norm > 1e-9 else query_vector

            # Step 1: Candidate identification via inverted token_map + vectorized matrix
            candidate_ids: Set[str] = set()

            # A. Inverted token lookup (O(1))
            q_tokens = list(extract_tokens(query.lower()))
            for tok in q_tokens:
                if tok in self.token_map:
                    candidate_ids.update(self.token_map[tok])

            # B. Vectorized similarity top-k via single matrix multiply (0.1ms)
            vector_sim_map: Dict[str, float] = {}
            if q_unit is not None and self.entity_vectors:
                e_ids = list(self.entity_vectors.keys())
                e_vecs = [self.entity_vectors[eid] for eid in e_ids]
                if e_vecs:
                    mat = np.array(e_vecs)
                    sims = np.dot(mat, q_unit)
                    top_v_idx = np.argsort(sims)[::-1][:top_k * 4]
                    for idx in top_v_idx:
                        s_val = float(sims[idx])
                        vector_sim_map[e_ids[idx]] = s_val
                        if s_val > 0.15:
                            candidate_ids.add(e_ids[idx])

            # Fallback if no candidates found
            if not candidate_ids:
                candidates_to_score = list(self.entities.values())[:top_k]
            else:
                candidates_to_score = [self.entities[eid] for eid in candidate_ids if eid in self.entities]

            scored_candidates: List[Tuple[EntityNode, float]] = []
            for entity in candidates_to_score:
                s_lex = self.compute_lexical_score(query, entity)
                s_sim = vector_sim_map.get(entity.entity_id)
                if s_sim is None and q_unit is not None:
                    s_sim = self.compute_similarity_score(q_unit, entity.entity_id)
                elif s_sim is None:
                    s_sim = 0.0

                score = self.alpha * s_lex + (1.0 - self.alpha) * max(0.0, s_sim)
                if score > 0.05:
                    scored_candidates.append((entity, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[:top_k]

    def query_entities_multi_aspect(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        query_epoch: Optional[float] = None,
        top_k: int = 10,
        embedder: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Deep multi-aspect search across all entities and their slotted knowledge.
        Evaluates 'aspect' ("what"), slot values, historical time slices,
        and temporal timeframe validity using both regex and semantic vector matching.
        """
        with self._lock:
            all_slot_matches: List[Dict[str, Any]] = []
            q_unit = None
            if query_vector is not None:
                norm = np.linalg.norm(query_vector)
                q_unit = (query_vector / norm) if norm > 1e-9 else query_vector

            # Fast candidate pruning: only inspect entities with token overlap or high vector similarity
            q_tokens = extract_tokens(query.lower())
            candidate_eids: Set[str] = set()
            for tok in q_tokens:
                if tok in self.token_map:
                    candidate_eids.update(self.token_map[tok])

            if q_unit is not None:
                # Add top vector-similar entities
                sim_cands = self.search_entities(query, query_vector=query_vector, top_k=20)
                for ent, _ in sim_cands:
                    candidate_eids.add(ent.entity_id)

            candidates = [self.entities[eid] for eid in candidate_eids if eid in self.entities]
            if not candidates:
                candidates = list(self.entities.values())[:30]

            for entity in candidates:
                slot_matches = entity.match_aspects_and_slices(
                    query_text=query,
                    query_vec=q_unit,
                    embedder=None,  # Lexical and slot attributes checked instantly without thousands of redundant neural passes
                    query_epoch=query_epoch,
                )
                if slot_matches:
                    all_slot_matches.extend(slot_matches)

            all_slot_matches.sort(key=lambda x: x["score"], reverse=True)
            return all_slot_matches[:top_k]


# ============================================================================
# Enterprise SQLite Database Storage Engine
# ============================================================================

class SQLiteGraphStore:
    """
    ACID-compliant relational database persistence store for the EEKG graph schema.
    
    Provides:
    - Normalized relational schema for Entities, Aliases, Episodes, KnowledgeNodes, and TimeSlices.
    - Foreign key integrity and cascading deletes.
    - WAL journal mode with connection pooling per-thread.
    - Complete BLOB storage for vector embeddings and taxonomic state coordinates.
    - High-speed indexed range querying for time-travel queries.
    """

    DDL_STATEMENTS = [
        # 1. Canonical Entities (Unified with Surface Aliases, Scopes & Vector Embeddings)
        """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            entity_type TEXT NOT NULL DEFAULT 'concept',
            aliases_json TEXT NOT NULL DEFAULT '[]',
            vector_blob BLOB,
            dim INTEGER DEFAULT 896,
            importance REAL NOT NULL DEFAULT 0.5,
            scope TEXT NOT NULL DEFAULT 'GLOBAL',
            owner_session_id TEXT,
            owner_episode_id TEXT,
            created_at REAL NOT NULL,
            last_updated REAL NOT NULL,
            metadata_json TEXT
        );
        """,
        # 2. Episodic Timeframes (Continuous Intervals)
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            start_epoch REAL NOT NULL,
            end_epoch REAL,
            summary TEXT,
            metadata_json TEXT
        );
        """,
        # 3. Episode-Entity Junction
        """
        CREATE TABLE IF NOT EXISTS episode_entities (
            episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
            entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            PRIMARY KEY (episode_id, entity_id)
        );
        """,
        # 4. Slotted Knowledge Nodes
        """
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            node_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            aspect TEXT NOT NULL,
            current_value_json TEXT NOT NULL,
            active_since REAL NOT NULL,
            current_episode_id TEXT REFERENCES episodes(episode_id) ON DELETE SET NULL,
            taxonomic_state_blob BLOB,
            access_count INTEGER NOT NULL DEFAULT 1,
            last_activated REAL NOT NULL
        );
        """,
        # 5. Immutable Versioned Time Slices (History)
        """
        CREATE TABLE IF NOT EXISTS time_slices (
            slice_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
            entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            aspect TEXT NOT NULL,
            value_json TEXT NOT NULL,
            valid_from REAL NOT NULL,
            valid_to REAL NOT NULL,
            episode_id TEXT REFERENCES episodes(episode_id) ON DELETE SET NULL,
            taxonomic_state_blob BLOB,
            mutation_reason TEXT
        );
        """,
        # 6. Episode-Assertion Junction
        """
        CREATE TABLE IF NOT EXISTS episode_assertions (
            episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
            node_id TEXT NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
            PRIMARY KEY (episode_id, node_id)
        );
        """,
        # 7. Entity-to-Entity Semantic Relations
        """
        CREATE TABLE IF NOT EXISTS entity_relations (
            relation_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            episode_id TEXT REFERENCES episodes(episode_id) ON DELETE SET NULL,
            created_at REAL NOT NULL
        );
        """,
        # 8. Persistent Agent Tasks (Work DAG)
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            root_goal TEXT,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            dependencies_json TEXT NOT NULL DEFAULT '[]',
            entities_json TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            completed_at REAL,
            result_summary TEXT,
            metadata_json TEXT
        );
        """,
        # 9. Persistent Task Execution Steps (Work Log, Modified Files, Terminal Commands, Errors)
        """
        CREATE TABLE IF NOT EXISTS task_steps (
            step_id TEXT PRIMARY KEY,
            task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
            action TEXT NOT NULL,
            observation TEXT,
            modified_files_json TEXT NOT NULL DEFAULT '[]',
            terminal_commands_json TEXT NOT NULL DEFAULT '[]',
            error_json TEXT,
            is_milestone INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL,
            metadata_json TEXT
        );
        """,
        # 10. Granular Dialogue Turns & Code Chunks (Atomic Conversational Memory)
        """
        CREATE TABLE IF NOT EXISTS dialogue_turns (
            turn_id TEXT PRIMARY KEY,
            episode_id TEXT REFERENCES episodes(episode_id) ON DELETE CASCADE,
            turn_number INTEGER,
            speaker TEXT,
            text TEXT NOT NULL,
            epoch REAL,
            vector_blob BLOB
        );
        """,
        # High-Performance Query Indexes
        "CREATE INDEX IF NOT EXISTS idx_kn_entity_aspect ON knowledge_nodes(entity_id, aspect);",
        "CREATE INDEX IF NOT EXISTS idx_ts_node_valid ON time_slices(node_id, valid_from, valid_to);",
        "CREATE INDEX IF NOT EXISTS idx_ts_entity_aspect ON time_slices(entity_id, aspect);",
        "CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_epoch, end_epoch);",
        "CREATE INDEX IF NOT EXISTS idx_ep_assertions_ep ON episode_assertions(episode_id);",
        "CREATE INDEX IF NOT EXISTS idx_ep_assertions_node ON episode_assertions(node_id);",
        "CREATE INDEX IF NOT EXISTS idx_ent_rel_src ON entity_relations(source_id);",
        "CREATE INDEX IF NOT EXISTS idx_ent_rel_tgt ON entity_relations(target_id);",
        "CREATE INDEX IF NOT EXISTS idx_ent_rel_rel ON entity_relations(relation);",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);",
        "CREATE INDEX IF NOT EXISTS idx_task_steps_task ON task_steps(task_id);",
        "CREATE INDEX IF NOT EXISTS idx_task_steps_time ON task_steps(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_entities_scope ON entities(scope, owner_session_id, owner_episode_id);",
        "CREATE INDEX IF NOT EXISTS idx_dialogue_turns_ep ON dialogue_turns(episode_id);",
        "CREATE INDEX IF NOT EXISTS idx_dialogue_turns_num ON dialogue_turns(turn_number);",
        "CREATE INDEX IF NOT EXISTS idx_dialogue_turns_epoch ON dialogue_turns(epoch);",
    ]

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL and foreign keys."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_database(self) -> None:
        """Initializes tables, constraints, and indexes."""
        conn = self._get_connection()
        with conn:
            for statement in self.DDL_STATEMENTS:
                conn.execute(statement)

            # Automatic backwards-compatible schema migration
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(entities);")
                cols = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()]
                if cols and "scope" not in cols:
                    conn.execute("ALTER TABLE entities ADD COLUMN scope TEXT NOT NULL DEFAULT 'GLOBAL';")
                if cols and "owner_session_id" not in cols:
                    conn.execute("ALTER TABLE entities ADD COLUMN owner_session_id TEXT;")
                if cols and "owner_episode_id" not in cols:
                    conn.execute("ALTER TABLE entities ADD COLUMN owner_episode_id TEXT;")
            except Exception as e:
                logger.debug(f"Schema migration check: {e}")

    def close(self) -> None:
        """Closes the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------------
    # Entity Persistence Operations
    # ------------------------------------------------------------------------

    def upsert_entity(self, entity: EntityNode, vector: Optional[np.ndarray] = None) -> None:
        """Persists an EntityNode, its surface aliases, scope, and its optional embedding directly in entities table."""
        conn = self._get_connection()
        aliases_list = sorted(list(entity.aliases))
        aliases_json = json.dumps(aliases_list)
        v_blob = ndarray_to_bytes(vector) if vector is not None else None
        v_dim = len(vector) if vector is not None else None
        scope_str = entity.scope.value if isinstance(entity.scope, EntityScope) else str(entity.scope)
        meta_json = json.dumps({
            "reference_count": entity.reference_count,
            "sub_episode_ids": sorted(list(entity.sub_episode_ids)),
            "session_ids": sorted(list(entity.session_ids)),
            "has_factual_anchor": entity.has_factual_anchor or len(entity.slotted_knowledge) > 0,
        })

        with conn:
            # 1. Upsert unified entity row (canonical info + aliases JSON + vector BLOB + scope)
            conn.execute(
                """
                INSERT INTO entities (
                    entity_id, canonical_name, entity_type, aliases_json,
                    vector_blob, dim, importance, scope, owner_session_id, owner_episode_id,
                    created_at, last_updated, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    entity_type = excluded.entity_type,
                    aliases_json = excluded.aliases_json,
                    vector_blob = COALESCE(excluded.vector_blob, entities.vector_blob),
                    dim = COALESCE(excluded.dim, entities.dim),
                    importance = excluded.importance,
                    scope = excluded.scope,
                    owner_session_id = excluded.owner_session_id,
                    owner_episode_id = excluded.owner_episode_id,
                    last_updated = excluded.last_updated,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    entity.entity_id,
                    entity.canonical_name,
                    entity.entity_type,
                    aliases_json,
                    v_blob,
                    v_dim,
                    entity.importance,
                    scope_str,
                    entity.owner_session_id,
                    entity.owner_episode_id,
                    entity.created_at,
                    entity.last_updated,
                    meta_json,
                ),
            )

            # 2. Upsert connected episodes (safely selecting from existing episodes)
            for ep_id in entity.connected_episode_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO episode_entities (episode_id, entity_id)
                    SELECT episode_id, ? FROM episodes WHERE episode_id = ?;
                    """,
                    (entity.entity_id, ep_id),
                )

    def load_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Loads a complete EntityNode with its scope, slotted knowledge, and historical time slices."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM entities WHERE entity_id = ?;", (entity_id,)).fetchone()
        if not row:
            return None

        # Load aliases directly from unified column
        aliases: Set[str] = set()
        if "aliases_json" in row.keys() and row["aliases_json"]:
            try:
                aliases = set(json.loads(row["aliases_json"]))
            except Exception:
                aliases = set()
        aliases.add(row["canonical_name"])

        # Load episodes connected
        ep_rows = conn.execute(
            "SELECT episode_id FROM episode_entities WHERE entity_id = ? ORDER BY episode_id;",
            (entity_id,),
        ).fetchall()
        connected_eps = [r["episode_id"] for r in ep_rows]

        # Load slotted knowledge
        kn_rows = conn.execute("SELECT * FROM knowledge_nodes WHERE entity_id = ?;", (entity_id,)).fetchall()
        slotted_knowledge: Dict[str, KnowledgeNode] = {}

        for kn_row in kn_rows:
            node_id = kn_row["node_id"]
            # Load time slices for this knowledge node
            ts_rows = conn.execute(
                "SELECT * FROM time_slices WHERE node_id = ? ORDER BY valid_from ASC;",
                (node_id,),
            ).fetchall()

            history = [
                TimeSlice(
                    slice_id=ts["slice_id"],
                    node_id=ts["node_id"],
                    entity_id=ts["entity_id"],
                    aspect=ts["aspect"],
                    value=json.loads(ts["value_json"]),
                    valid_from=ts["valid_from"],
                    valid_to=ts["valid_to"],
                    taxonomic_state=bytes_to_ndarray(ts["taxonomic_state_blob"]),
                    episode_id=ts["episode_id"],
                    mutation_reason=ts["mutation_reason"],
                )
                for ts in ts_rows
            ]

            prefix = f"kn_{entity_id}_"
            slot_key = node_id[len(prefix):] if node_id.startswith(prefix) else node_id
            slotted_knowledge[slot_key] = KnowledgeNode(
                node_id=node_id,
                entity_id=entity_id,
                aspect=kn_row["aspect"],
                current_value=json.loads(kn_row["current_value_json"]),
                active_since=kn_row["active_since"],
                taxonomic_state=bytes_to_ndarray(kn_row["taxonomic_state_blob"]),
                current_episode_id=kn_row["current_episode_id"],
                history=history,
                access_count=kn_row["access_count"],
                last_activated=kn_row["last_activated"],
            )

        scope_val = row["scope"] if "scope" in row.keys() else "GLOBAL"
        try:
            scope_enum = EntityScope(scope_val)
        except ValueError:
            scope_enum = EntityScope.GLOBAL

        owner_session_id = row["owner_session_id"] if "owner_session_id" in row.keys() else None
        owner_episode_id = row["owner_episode_id"] if "owner_episode_id" in row.keys() else None

        ref_count = 1
        sub_eps: Set[str] = set()
        sess_ids: Set[str] = set()
        has_anchor = len(slotted_knowledge) > 0

        if "metadata_json" in row.keys() and row["metadata_json"]:
            try:
                m_data = json.loads(row["metadata_json"])
                ref_count = int(m_data.get("reference_count", 1))
                sub_eps = set(m_data.get("sub_episode_ids", []))
                sess_ids = set(m_data.get("session_ids", []))
                has_anchor = bool(m_data.get("has_factual_anchor", has_anchor))
            except Exception:
                pass

        return EntityNode(
            entity_id=row["entity_id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            aliases=aliases,
            slotted_knowledge=slotted_knowledge,
            connected_episode_ids=connected_eps,
            importance=row["importance"],
            scope=scope_enum,
            owner_session_id=owner_session_id,
            owner_episode_id=owner_episode_id,
            reference_count=ref_count,
            sub_episode_ids=sub_eps,
            session_ids=sess_ids,
            has_factual_anchor=has_anchor,
            created_at=row["created_at"],
            last_updated=row["last_updated"],
        )

    # ------------------------------------------------------------------------
    # Knowledge Node & TimeSlice Persistence Operations
    # ------------------------------------------------------------------------

    def upsert_knowledge_node(self, node: KnowledgeNode) -> None:
        """Persists a KnowledgeNode and flushes its history slices."""
        conn = self._get_connection()
        val_json = json.dumps(node.current_value)
        tau_blob = ndarray_to_bytes(node.taxonomic_state)

        with conn:
            # Safely verify episode existence to respect foreign key constraint
            ep_id = node.current_episode_id
            if ep_id is not None:
                ep_row = conn.execute("SELECT 1 FROM episodes WHERE episode_id = ?;", (ep_id,)).fetchone()
                if not ep_row:
                    ep_id = None

            conn.execute(
                """
                INSERT INTO knowledge_nodes (
                    node_id, entity_id, aspect, current_value_json, active_since,
                    current_episode_id, taxonomic_state_blob, access_count, last_activated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    current_value_json = excluded.current_value_json,
                    active_since = excluded.active_since,
                    current_episode_id = excluded.current_episode_id,
                    taxonomic_state_blob = excluded.taxonomic_state_blob,
                    access_count = excluded.access_count,
                    last_activated = excluded.last_activated;
                """,
                (
                    node.node_id,
                    node.entity_id,
                    node.aspect,
                    val_json,
                    node.active_since,
                    ep_id,
                    tau_blob,
                    node.access_count,
                    node.last_activated,
                ),
            )

            # Persist historical slices
            for s in node.history:
                s_json = json.dumps(s.value)
                s_tau_blob = ndarray_to_bytes(s.taxonomic_state)
                s_ep_id = s.episode_id
                if s_ep_id is not None:
                    s_ep_row = conn.execute("SELECT 1 FROM episodes WHERE episode_id = ?;", (s_ep_id,)).fetchone()
                    if not s_ep_row:
                        s_ep_id = None

                conn.execute(
                    """
                    INSERT INTO time_slices (
                        slice_id, node_id, entity_id, aspect, value_json,
                        valid_from, valid_to, episode_id, taxonomic_state_blob, mutation_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(slice_id) DO NOTHING;
                    """,
                    (
                        s.slice_id,
                        s.node_id,
                        s.entity_id,
                        s.aspect,
                        s_json,
                        s.valid_from,
                        s.valid_to,
                        s_ep_id,
                        s_tau_blob,
                        s.mutation_reason,
                    ),
                )

    # ------------------------------------------------------------------------
    # Episode Persistence Operations
    # ------------------------------------------------------------------------

    def upsert_episode(self, episode: TimeframeEpisode) -> None:
        """Persists an episodic container and its entity participant junction rows."""
        conn = self._get_connection()
        meta_json = json.dumps(episode.metadata)

        with conn:
            conn.execute(
                """
                INSERT INTO episodes (episode_id, name, start_epoch, end_epoch, summary, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    name = excluded.name,
                    start_epoch = excluded.start_epoch,
                    end_epoch = excluded.end_epoch,
                    summary = excluded.summary,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    episode.episode_id,
                    episode.name,
                    episode.start_epoch,
                    episode.end_epoch,
                    episode.summary,
                    meta_json,
                ),
            )

            # Insert connected entities (safely selecting from existing entities to respect foreign keys)
            for eid in episode.connected_entity_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO episode_entities (episode_id, entity_id)
                    SELECT ?, entity_id FROM entities WHERE entity_id = ?;
                    """,
                    (episode.episode_id, eid),
                )

            # Insert bound assertions (safely selecting from existing knowledge_nodes to respect foreign keys)
            for nid in episode.assertion_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO episode_assertions (episode_id, node_id)
                    SELECT ?, node_id FROM knowledge_nodes WHERE node_id = ?;
                    """,
                    (episode.episode_id, nid),
                )

    def load_episode(self, episode_id: str) -> Optional[TimeframeEpisode]:
        """Loads a TimeframeEpisode with all connected entity references and assertion IDs."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?;", (episode_id,)).fetchone()
        if not row:
            return None

        e_rows = conn.execute(
            "SELECT entity_id FROM episode_entities WHERE episode_id = ?;",
            (episode_id,),
        ).fetchall()
        entities = {r["entity_id"] for r in e_rows}

        a_rows = conn.execute(
            "SELECT node_id FROM episode_assertions WHERE episode_id = ? ORDER BY node_id;",
            (episode_id,),
        ).fetchall()
        assertions = [r["node_id"] for r in a_rows]

        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return TimeframeEpisode(
            episode_id=row["episode_id"],
            name=row["name"],
            start_epoch=row["start_epoch"],
            end_epoch=row["end_epoch"],
            summary=row["summary"],
            connected_entity_ids=entities,
            assertion_ids=assertions,
            metadata=meta,
        )

    def load_all_episodes(self) -> List[TimeframeEpisode]:
        """Loads all TimeframeEpisodes from SQLite in chronological order."""
        conn = self._get_connection()
        ep_rows = conn.execute("SELECT episode_id FROM episodes ORDER BY start_epoch ASC;").fetchall()
        episodes = []
        for r in ep_rows:
            ep = self.load_episode(r["episode_id"])
            if ep is not None:
                episodes.append(ep)
        return episodes

    # ------------------------------------------------------------------------
    # Entity Relation Persistence Operations
    # ------------------------------------------------------------------------

    def upsert_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        episode_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> str:
        """Persists a directional semantic relation between two entities."""
        conn = self._get_connection()
        t = created_at if created_at is not None else time.time()
        clean_rel = relation.strip()
        rel_id = f"rel_{source_id}_{target_id}_{clean_rel}".lower().replace(" ", "_").replace(":", "_")
        with conn:
            conn.execute(
                """
                INSERT INTO entity_relations (relation_id, source_id, target_id, relation, weight, episode_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relation_id) DO UPDATE SET
                    weight = excluded.weight,
                    episode_id = COALESCE(excluded.episode_id, entity_relations.episode_id);
                """,
                (rel_id, source_id, target_id, clean_rel, weight, episode_id, t),
            )
        return rel_id

    def load_entity_relations(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Loads semantic relations for a specific entity or all entities."""
        conn = self._get_connection()
        if entity_id:
            rows = conn.execute(
                "SELECT * FROM entity_relations WHERE source_id = ? OR target_id = ? ORDER BY created_at ASC;",
                (entity_id, entity_id),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM entity_relations ORDER BY created_at ASC;").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------------
    # Bulk Registry Synchronization
    # ------------------------------------------------------------------------

    def load_entity_vector(self, entity_id: str) -> Optional[np.ndarray]:
        """Loads dense embedding vector for an entity directly from the unified entities table."""
        conn = self._get_connection()
        row = conn.execute("SELECT vector_blob FROM entities WHERE entity_id = ?;", (entity_id,)).fetchone()
        if row and row["vector_blob"]:
            return bytes_to_ndarray(row["vector_blob"])
        return None

    def populate_registry(self, registry: EntityRegistry) -> None:
        """Loads all entities, aliases, and embeddings from SQLite into the in-memory registry."""
        conn = self._get_connection()
        e_rows = conn.execute("SELECT entity_id, vector_blob FROM entities;").fetchall()
        for r in e_rows:
            entity = self.load_entity(r["entity_id"])
            if entity is not None:
                vec = bytes_to_ndarray(r["vector_blob"]) if r["vector_blob"] else None
                registry.register_entity(entity, vector=vec)

    def flush_registry(self, registry: EntityRegistry, only_admissible: bool = True) -> None:
        """Persists the state of an in-memory EntityRegistry to SQLite, enforcing creation gating."""
        with registry._lock:
            for entity in registry.entities.values():
                if only_admissible and not registry.is_admissible(entity):
                    continue
                vec = registry.entity_vectors.get(entity.entity_id)
                self.upsert_entity(entity, vector=vec)
                for kn in entity.slotted_knowledge.values():
                    self.upsert_knowledge_node(kn)

    # ------------------------------------------------------------------------
    # Persistent Work DAG & Task Execution Steps (Layer 4 Memory Persistence)
    # ------------------------------------------------------------------------

    def upsert_task(
        self,
        task_id: str,
        description: str,
        status: str = "PENDING",
        dependencies: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
        created_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        result_summary: Optional[str] = None,
        root_goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persists or updates an agent task in the SQLite Work DAG."""
        conn = self._get_connection()
        deps_json = json.dumps(list(dependencies or []))
        ents_json = json.dumps(list(entities or []))
        meta_json = json.dumps(metadata or {})
        t_create = created_at if created_at is not None else time.time()

        with conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, root_goal, description, status, dependencies_json,
                    entities_json, created_at, completed_at, result_summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    root_goal = COALESCE(excluded.root_goal, tasks.root_goal),
                    description = excluded.description,
                    status = excluded.status,
                    dependencies_json = excluded.dependencies_json,
                    entities_json = excluded.entities_json,
                    completed_at = COALESCE(excluded.completed_at, tasks.completed_at),
                    result_summary = COALESCE(excluded.result_summary, tasks.result_summary),
                    metadata_json = excluded.metadata_json;
                """,
                (
                    task_id,
                    root_goal,
                    description,
                    str(status),
                    deps_json,
                    ents_json,
                    t_create,
                    completed_at,
                    result_summary,
                    meta_json,
                ),
            )

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Loads a single task by ID from the SQLite Work DAG."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,)).fetchone()
        if not row:
            return None
        return {
            "task_id": row["task_id"],
            "root_goal": row["root_goal"],
            "description": row["description"],
            "status": row["status"],
            "dependencies": json.loads(row["dependencies_json"]),
            "entities": json.loads(row["entities_json"]),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "result_summary": row["result_summary"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        }

    def load_all_tasks(self) -> List[Dict[str, Any]]:
        """Loads all agent tasks from the SQLite Work DAG in creation order."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM tasks ORDER BY created_at ASC;").fetchall()
        results = []
        for r in rows:
            results.append({
                "task_id": r["task_id"],
                "root_goal": r["root_goal"],
                "description": r["description"],
                "status": r["status"],
                "dependencies": json.loads(r["dependencies_json"]),
                "entities": json.loads(r["entities_json"]),
                "created_at": r["created_at"],
                "completed_at": r["completed_at"],
                "result_summary": r["result_summary"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            })
        return results

    def record_task_step(
        self,
        action: str,
        observation: str = "",
        task_id: Optional[str] = None,
        modified_files: Optional[Sequence[str]] = None,
        terminal_commands: Optional[Sequence[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        is_milestone: bool = False,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
    ) -> str:
        """
        Persists an agent execution step (actions, observations, touched files,
        terminal commands, and errors) into the task_steps SQLite table.
        """
        conn = self._get_connection()
        s_id = step_id or f"step_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        t = timestamp if timestamp is not None else time.time()
        files_json = json.dumps(list(modified_files or []))
        cmds_json = json.dumps(list(terminal_commands or []))
        err_json = json.dumps(error) if error else None
        meta_json = json.dumps(metadata or {})

        with conn:
            # Validate task_id reference to respect foreign key constraint
            valid_task_id = None
            if task_id:
                t_row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?;", (task_id,)).fetchone()
                if t_row:
                    valid_task_id = task_id

            conn.execute(
                """
                INSERT INTO task_steps (
                    step_id, task_id, action, observation, modified_files_json,
                    terminal_commands_json, error_json, is_milestone, timestamp, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    task_id = excluded.task_id,
                    action = excluded.action,
                    observation = excluded.observation,
                    modified_files_json = excluded.modified_files_json,
                    terminal_commands_json = excluded.terminal_commands_json,
                    error_json = excluded.error_json,
                    is_milestone = excluded.is_milestone,
                    timestamp = excluded.timestamp,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    s_id,
                    valid_task_id,
                    action,
                    observation,
                    files_json,
                    cmds_json,
                    err_json,
                    1 if is_milestone else 0,
                    t,
                    meta_json,
                ),
            )
        return s_id

    def load_task_steps(
        self,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Loads historical task steps from SQLite in chronological order."""
        conn = self._get_connection()
        if task_id:
            rows = conn.execute(
                "SELECT * FROM task_steps WHERE task_id = ? ORDER BY timestamp ASC LIMIT ?;",
                (task_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task_steps ORDER BY timestamp ASC LIMIT ?;",
                (limit,),
            ).fetchall()

        results = []
        for r in rows:
            results.append({
                "step_id": r["step_id"],
                "task_id": r["task_id"],
                "action": r["action"],
                "observation": r["observation"],
                "modified_files": json.loads(r["modified_files_json"]),
                "terminal_commands": json.loads(r["terminal_commands_json"]),
                "error": json.loads(r["error_json"]) if r["error_json"] else None,
                "is_milestone": bool(r["is_milestone"]),
                "timestamp": r["timestamp"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            })
        return results

    # ------------------------------------------------------------------------
    # Granular Conversational Turns Persistence (Atomic Dialogue Memory)
    # ------------------------------------------------------------------------

    def upsert_turn(
        self,
        turn_id: str,
        episode_id: Optional[str],
        turn_number: Optional[int],
        speaker: str,
        text: str,
        epoch: Optional[float] = None,
        vector: Optional[np.ndarray] = None,
    ) -> None:
        """Persists an individual conversation turn or code snippet with its embedding vector."""
        conn = self._get_connection()
        v_blob = ndarray_to_bytes(vector) if vector is not None else None
        t_epoch = epoch if epoch is not None else time.time()
        with conn:
            conn.execute(
                """
                INSERT INTO dialogue_turns (turn_id, episode_id, turn_number, speaker, text, epoch, vector_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    turn_number = excluded.turn_number,
                    speaker = excluded.speaker,
                    text = excluded.text,
                    epoch = excluded.epoch,
                    vector_blob = COALESCE(excluded.vector_blob, dialogue_turns.vector_blob);
                """,
                (turn_id, episode_id, turn_number, speaker, text, t_epoch, v_blob),
            )

    def upsert_turns_batch(self, turns: List[Dict[str, Any]]) -> None:
        """Efficiently batch-inserts or updates dialogue turns in a single atomic transaction."""
        if not turns:
            return
        conn = self._get_connection()
        rows_to_insert = []
        for t in turns:
            v = t.get("vector")
            v_blob = ndarray_to_bytes(v) if v is not None else None
            rows_to_insert.append((
                str(t["turn_id"]),
                t.get("episode_id"),
                t.get("turn_number"),
                t.get("speaker", ""),
                t.get("text", ""),
                float(t.get("epoch") or time.time()),
                v_blob,
            ))
        with conn:
            conn.executemany(
                """
                INSERT INTO dialogue_turns (turn_id, episode_id, turn_number, speaker, text, epoch, vector_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    turn_number = excluded.turn_number,
                    speaker = excluded.speaker,
                    text = excluded.text,
                    epoch = excluded.epoch,
                    vector_blob = COALESCE(excluded.vector_blob, dialogue_turns.vector_blob);
                """,
                rows_to_insert,
            )

    def load_all_turns(self) -> List[Dict[str, Any]]:
        """Loads all dialogue turns with their vector embeddings from SQLite in chronological order."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM dialogue_turns ORDER BY epoch ASC, turn_number ASC;").fetchall()
        results = []
        for r in rows:
            vec = bytes_to_ndarray(r["vector_blob"]) if r["vector_blob"] else None
            results.append({
                "turn_id": r["turn_id"],
                "episode_id": r["episode_id"],
                "turn_number": r["turn_number"],
                "speaker": r["speaker"],
                "text": r["text"],
                "epoch": r["epoch"],
                "vector": vec,
            })
        return results

    def load_turns_for_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """Loads all dialogue turns associated with a specific episode container."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM dialogue_turns WHERE episode_id = ? ORDER BY epoch ASC, turn_number ASC;",
            (episode_id,),
        ).fetchall()
        results = []
        for r in rows:
            vec = bytes_to_ndarray(r["vector_blob"]) if r["vector_blob"] else None
            results.append({
                "turn_id": r["turn_id"],
                "episode_id": r["episode_id"],
                "turn_number": r["turn_number"],
                "speaker": r["speaker"],
                "text": r["text"],
                "epoch": r["epoch"],
                "vector": vec,
            })
        return results

