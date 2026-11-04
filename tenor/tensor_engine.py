"""
4D Spatio-Temporal Tensor Engine & Signed Plasticity (tensor_engine.py)
=====================================================================

Mathematical and topological core of EEKG v2:
- Dynamic, signed associative attention tensors in [-1.0, 1.0].
- Online signed Hebbian synaptic plasticity with diminishing saturation.
- Speaker attribution and anti-contamination isolation.
- Inverse Hub Frequency (IHF) dampening for hairball immunity.
- 4D continuous spatio-temporal spreading activation (Personalized PageRank).
- Dual-factor + topological tri-channel retrieval fusion.
- Contradiction and mutation attenuation.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .embedder import QwenSemanticEmbedder, cosine_similarity, l2_normalize
from .node_models import EntityNode, TimeframeEpisode

logger = logging.getLogger(__name__)


class SpatioTemporalTensor:
    """
    Continuous 4D dynamic attention tensor and episodic incidence engine.
    """

    def __init__(
        self,
        embedder: Optional[QwenSemanticEmbedder] = None,
        default_eta: float = 0.40,
        prune_threshold: float = 0.015,
        alpha_pagerank: float = 0.65,
        time_lambda: float = 0.001,
        time_tau_decay: float = 86400.0,
    ) -> None:
        self.embedder = embedder or QwenSemanticEmbedder()
        self.default_eta = default_eta
        self.prune_threshold = prune_threshold
        self.alpha_pagerank = alpha_pagerank
        self.time_lambda = time_lambda
        self.time_tau_decay = time_tau_decay

        # Physical Tensor Storage
        # Symmetric edge key: (min(u, v), max(u, v))
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        self.edge_counts: Dict[Tuple[int, int], int] = {}
        self.edge_timestamps: Dict[Tuple[int, int], float] = {}

        # Semantic E2E Relations Storage
        self.semantic_relations: List[Dict[str, Any]] = []
        self.entity_relations: Dict[Tuple[int, int], List[str]] = {}

        # Entity Indexing & Embeddings
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.entities: Dict[str, EntityNode] = {}
        self.entity_vectors: Dict[int, np.ndarray] = {}

        # Episodic Store & Incidence Matrix
        self.episodes: List[TimeframeEpisode] = []
        self.episode_id_to_idx: Dict[str, int] = {}
        self.episode_vectors: List[np.ndarray] = []
        # Incidence map: (episode_idx, entity_idx) -> weight
        self.incidence_entries: Dict[Tuple[int, int], float] = {}

        # Direct Episode-to-Episode Graph Edges: (ep_idx_1, ep_idx_2) -> weight
        self.episode_edges: Dict[Tuple[int, int], float] = {}
        self._cached_episode_adj: Optional[sp.csr_matrix] = None
        self._episode_adj_dirty: bool = True

        # Granular Conversation Turns / File Chunks Indexing
        self.turn_records: List[Dict[str, Any]] = []
        self.turn_vectors: List[np.ndarray] = []

        # Cached Sparse Matrices
        self._cached_csr: Optional[sp.csr_matrix] = None
        self._cached_incidence: Optional[sp.csr_matrix] = None
        self._is_dirty: bool = True
        self._incidence_dirty: bool = True

        # Pre-Cached Vectorized Turn Matrix & Inverted Token Index
        self._turn_mat: Optional[np.ndarray] = None
        self._turn_token_index: Optional[Dict[str, List[int]]] = None
        self._turn_idf: Optional[Dict[str, float]] = None
        self._turn_user_mask: Optional[np.ndarray] = None
        self._turn_episode_ids: Optional[List[str]] = None
        self._turn_dirty: bool = True

    def ensure_turn_index(self) -> None:
        """Precomputes contiguous float32 BLAS matrix, inverted token index, and BM25 IDF for instant retrieval."""
        if not self._turn_dirty and self._turn_mat is not None and len(self._turn_mat) == len(self.turn_vectors):
            return

        if not self.turn_vectors:
            self._turn_mat = np.empty((0, getattr(self.embedder, "dim", 896)), dtype=np.float32)
            self._turn_user_mask = np.empty(0, dtype=bool)
            self._turn_episode_ids = []
            self._turn_token_index = {}
            self._turn_idf = {}
            self._turn_dirty = False
            return

        import re
        from collections import defaultdict

        self._turn_mat = np.ascontiguousarray(self.turn_vectors, dtype=np.float32)
        self._turn_user_mask = np.array(
            [str(t.get("speaker") or "").lower() == "user" for t in self.turn_records],
            dtype=bool,
        )
        self._turn_episode_ids = [t.get("episode_id") or "" for t in self.turn_records]

        n_docs = max(1, len(self.turn_records))
        idx_map = defaultdict(list)
        for idx, t_rec in enumerate(self.turn_records):
            words = set(re.findall(r"[a-zA-Z0-9_\-\.]+", t_rec.get("text", "").lower()))
            for w in words:
                idx_map[w].append(idx)
        self._turn_token_index = dict(idx_map)

        # Precompute BM25 Inverse Document Frequency (IDF) for all corpus vocabulary tokens
        # Common / ubiquitous words naturally get ~0.0 weight; rare distinctive words get maximum weight
        idf_map = {}
        for w, doc_list in idx_map.items():
            df = len(doc_list)
            idf = float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
            idf_map[w] = max(0.0, idf)
        self._turn_idf = idf_map
        self._max_idf = float(np.log(1.0 + (n_docs + 0.5) / 0.5))
        self._turn_dirty = False

    # ------------------------------------------------------------------------
    # Entity & Episode Registration
    # ------------------------------------------------------------------------

    def register_entity(
        self,
        entity: EntityNode,
        vector: Optional[np.ndarray] = None,
    ) -> int:
        """Registers an EntityNode into the tensor index."""
        eid = entity.entity_id
        if eid in self.entity_to_id:
            idx = self.entity_to_id[eid]
            self.entities[eid] = entity
            if vector is not None:
                self.entity_vectors[idx] = l2_normalize(np.asarray(vector, dtype=np.float32))
            return idx

        idx = len(self.entity_to_id)
        self.entity_to_id[eid] = idx
        self.id_to_entity[idx] = eid
        self.entities[eid] = entity

        if vector is not None:
            self.entity_vectors[idx] = l2_normalize(np.asarray(vector, dtype=np.float32))
        else:
            v = self.embedder.embed_text(entity.canonical_name)
            self.entity_vectors[idx] = l2_normalize(np.asarray(v, dtype=np.float32))

        self._is_dirty = True
        self._incidence_dirty = True
        return idx

    def get_or_register_entity(
        self,
        entity_id: str,
        vector: Optional[np.ndarray] = None,
        canonical_name: Optional[str] = None,
    ) -> int:
        """Retrieves or registers an entity by ID."""
        if entity_id in self.entity_to_id:
            idx = self.entity_to_id[entity_id]
            if vector is not None and idx not in self.entity_vectors:
                self.entity_vectors[idx] = l2_normalize(np.asarray(vector, dtype=np.float32))
            return idx

        name = canonical_name or entity_id
        entity = EntityNode(entity_id=entity_id, canonical_name=name)
        return self.register_entity(entity, vector=vector)

    def _index_episode_turns(self, episode: TimeframeEpisode) -> None:
        """
        Parses and embeds individual dialogue turns or code chunks from episode metadata.
        Pinpoints exact conversational turns (e.g., Turn D1:3: 'Caroline: ...')
        and code chunks rather than only searching high-level episode summaries.
        """
        raw_data = (
            episode.metadata.get("episode_data")
            or episode.metadata.get("turns")
            or episode.metadata.get("conversation")
            or episode.metadata.get("content")
        )
        if not raw_data:
            return

        turns_to_index: List[Dict[str, Any]] = []

        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                if isinstance(item, dict):
                    turn_id = str(item.get("turn_id") or item.get("id") or f"T{i+1}")
                    spk = str(item.get("speaker") or item.get("role") or "").strip()
                    txt = str(item.get("text") or item.get("content") or item.get("value") or "").strip()
                    if txt:
                        turns_to_index.append({
                            "turn_id": turn_id,
                            "speaker": spk,
                            "text": txt,
                        })
                elif isinstance(item, str) and item.strip():
                    turns_to_index.append({
                        "turn_id": f"T{i+1}",
                        "speaker": "",
                        "text": item.strip(),
                    })

        elif isinstance(raw_data, dict):
            if "turns" in raw_data and isinstance(raw_data["turns"], list):
                for i, item in enumerate(raw_data["turns"]):
                    if isinstance(item, dict):
                        turn_id = str(item.get("turn_id") or item.get("id") or f"T{i+1}")
                        spk = str(item.get("speaker") or item.get("role") or "").strip()
                        txt = str(item.get("text") or item.get("content") or item.get("value") or "").strip()
                        if txt:
                            turns_to_index.append({"turn_id": turn_id, "speaker": spk, "text": txt})
                    elif isinstance(item, str) and item.strip():
                        turns_to_index.append({"turn_id": f"T{i+1}", "speaker": "", "text": item.strip()})
            elif "file_path" in raw_data and "content" in raw_data:
                # File / code chunking
                f_path = str(raw_data["file_path"])
                lines = str(raw_data["content"]).splitlines()
                chunk_size = 25
                for c_start in range(0, max(1, len(lines)), 20):
                    c_lines = lines[c_start:c_start + chunk_size]
                    chunk_text = f"File {f_path} (lines {c_start+1}-{c_start+len(c_lines)}):\n" + "\n".join(c_lines)
                    turns_to_index.append({
                        "turn_id": f"chunk_{c_start}",
                        "speaker": "code",
                        "text": chunk_text,
                    })

        elif isinstance(raw_data, str) and raw_data.strip():
            # Multiline transcript or formatted string
            text_str = raw_data.strip()
            # Split by lines starting with dialogue markers or Turn designations
            turn_lines = re.split(r"(?m)(?=^(?:Turn\s+[A-Za-z0-9_:-]+|\[[A-Za-z0-9_\s-]+\]:|[A-Z][a-z0-9_]+:))", text_str)
            if len(turn_lines) > 1:
                for idx, t_text in enumerate(turn_lines):
                    clean_t = t_text.strip()
                    if clean_t:
                        turns_to_index.append({
                            "turn_id": f"T{idx+1}",
                            "speaker": "",
                            "text": clean_t,
                        })
            elif len(text_str) > 300:
                # Paragraph chunks
                paragraphs = [p.strip() for p in text_str.split("\n\n") if p.strip()]
                for idx, p_text in enumerate(paragraphs):
                    turns_to_index.append({
                        "turn_id": f"P{idx+1}",
                        "speaker": "",
                        "text": p_text,
                    })
            else:
                turns_to_index.append({
                    "turn_id": "T1",
                    "speaker": "",
                    "text": text_str,
                })

        if not turns_to_index:
            return

        # Prepare formatted text representation for all turns
        formatted_turns: List[str] = []
        turn_meta: List[Tuple[str, str, str]] = []
        for t in turns_to_index:
            txt = t["text"]
            spk = t.get("speaker", "")
            tid = t.get("turn_id", "")
            
            if tid and not tid.lower().startswith("turn") and not tid.startswith("P") and not tid.startswith("chunk"):
                turn_label = f"Turn {tid}"
            else:
                turn_label = tid

            if spk and turn_label:
                formatted_turn = f"{turn_label} [{spk}]: {txt}"
            elif spk:
                formatted_turn = f"[{spk}]: {txt}"
            elif turn_label:
                formatted_turn = f"{turn_label}: {txt}"
            else:
                formatted_turn = txt

            formatted_turns.append(formatted_turn)
            turn_meta.append((tid, spk, formatted_turn))

        # Batch embed and unit-normalize all extracted turns in a single vectorized pass
        batch_vecs = self.embedder.embed_text(formatted_turns)
        if batch_vecs.ndim == 1:
            batch_vecs = batch_vecs.reshape(1, -1)

        for i, (tid, spk, formatted_turn) in enumerate(turn_meta):
            v = l2_normalize(np.asarray(batch_vecs[i], dtype=np.float32))
            self.turn_records.append({
                "turn_id": f"{episode.episode_id}_{tid}",
                "episode_id": episode.episode_id,
                "episode_name": episode.name,
                "speaker": spk,
                "text": formatted_turn,
                "epoch": episode.start_epoch,
            })
            self.turn_vectors.append(v)
            self._turn_dirty = True

    def register_turn(
        self,
        turn_id: str,
        episode_id: Optional[str],
        speaker: str,
        text: str,
        epoch: Optional[float] = None,
        vector: Optional[np.ndarray] = None,
        turn_number: Optional[int] = None,
        episode_name: Optional[str] = None,
    ) -> None:
        """Registers a granular conversation turn or code chunk directly into the memory tensor index."""
        ep_name = episode_name or ""
        if not ep_name and episode_id:
            ep_idx = self.episode_id_to_idx.get(episode_id)
            if ep_idx is not None and ep_idx < len(self.episodes):
                ep_name = getattr(self.episodes[ep_idx], "name", "")
            else:
                for ep in self.episodes:
                    if getattr(ep, "episode_id", "") == episode_id:
                        ep_name = getattr(ep, "name", "")
                        break

        v = (
            l2_normalize(np.asarray(vector, dtype=np.float32))
            if vector is not None
            else l2_normalize(np.asarray(self.embedder.embed_text(text), dtype=np.float32))
        )
        self.turn_records.append({
            "turn_id": turn_id,
            "turn_number": turn_number,
            "episode_id": episode_id,
            "episode_name": ep_name,
            "speaker": speaker,
            "text": text,
            "epoch": float(epoch or 0.0),
        })
        self.turn_vectors.append(v)
        self._turn_dirty = True

    def add_episode(
        self,
        episode: TimeframeEpisode,
        vector: Optional[np.ndarray] = None,
    ) -> int:
        """Registers a TimeframeEpisode into the episodic store and embeds constituent turns/chunks."""
        ep_id = episode.episode_id
        if ep_id in self.episode_id_to_idx:
            idx = self.episode_id_to_idx[ep_id]
            self.episodes[idx] = episode
            if vector is not None:
                self.episode_vectors[idx] = l2_normalize(np.asarray(vector, dtype=np.float32))
            self._index_episode_turns(episode)
            return idx

        idx = len(self.episodes)
        self.episodes.append(episode)
        self.episode_id_to_idx[ep_id] = idx

        if vector is not None:
            v = l2_normalize(np.asarray(vector, dtype=np.float32))
        else:
            text = f"{episode.name} {episode.summary or ''}".strip()
            v = l2_normalize(np.asarray(self.embedder.embed_text(text), dtype=np.float32))
        self.episode_vectors.append(v)

        # Wire incidence from episode entities
        for eid in episode.entities:
            ent_idx = self.get_or_register_entity(eid)
            self.incidence_entries[(idx, ent_idx)] = 1.0

        # Compute relative temporal processing metadata
        if hasattr(episode, "compute_relative_time"):
            ref_e = getattr(self, "_reference_epoch", None) or episode.start_epoch
            episode.compute_relative_time(ref_epoch=ref_e)

        # Index individual dialogue turns and chunks
        self._index_episode_turns(episode)

        self._incidence_dirty = True
        self._episode_adj_dirty = True
        return idx

    def sync_relative_temporal_metadata(self, ref_epoch: Optional[float] = None) -> None:
        """
        Synchronizes structured relative temporal metadata across all indexed episodes and sessions
        relatively to the reference evaluation / current date.
        """
        if ref_epoch is None:
            ep_epochs = [float(ep.start_epoch) for ep in self.episodes if getattr(ep, "start_epoch", None) is not None and float(ep.start_epoch) > 0]
            turn_epochs = [float(t.get("epoch") or 0.0) for t in self.turn_records if float(t.get("epoch") or 0.0) > 0]
            candidates = ep_epochs + turn_epochs
            ref_epoch = max(candidates) if candidates else time.time()

        self._reference_epoch = float(ref_epoch)
        for ep in self.episodes:
            if hasattr(ep, "compute_relative_time"):
                ep.compute_relative_time(ref_epoch=self._reference_epoch)

    # ------------------------------------------------------------------------
    # Signed Hebbian Synaptic Plasticity & Speaker Attribution
    # ------------------------------------------------------------------------

    def record_interaction(
        self,
        entities: Sequence[str],
        polarity: float = 0.0,
        timestamp: Optional[float] = None,
        episode_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        listener_id: Optional[str] = None,
        eta: Optional[float] = None,
    ) -> None:
        """
        Applies signed Hebbian synaptic updates and wires episodic incidence.
        Implements speaker isolation to prevent opinion contamination:
        - Content entities clique updated at polarity p.
        - Speaker -> content entities updated at polarity p.
        - Listener is strictly excluded from associative edges.
        - Incidence: H[ep, speaker]=1.0, H[ep, content]=1.0, H[ep, listener]=0.2.
        """
        if not entities and not speaker_id:
            return

        t = timestamp if timestamp is not None else time.time()
        learning_rate = eta if eta is not None else self.default_eta
        p = float(np.clip(polarity, -1.0, 1.0))

        # Register all entities
        ent_indices: List[int] = [self.get_or_register_entity(eid) for eid in entities]
        speaker_idx = self.get_or_register_entity(speaker_id) if speaker_id else None
        listener_idx = self.get_or_register_entity(listener_id) if listener_id else None

        # Determine edge pairs
        edge_pairs: List[Tuple[int, int]] = []
        if speaker_idx is not None or listener_idx is not None:
            # Separate content entities from speaker/listener
            content_indices = [
                idx for idx in ent_indices
                if idx != speaker_idx and idx != listener_idx
            ]
            # 1. Intra-content full clique
            for i in range(len(content_indices)):
                for j in range(i + 1, len(content_indices)):
                    edge_pairs.append((content_indices[i], content_indices[j]))
            # 2. Speaker -> content entities
            if speaker_idx is not None:
                for c_idx in content_indices:
                    edge_pairs.append((speaker_idx, c_idx))
            # 3. Listener is STRICTLY EXCLUDED from edge formation
        else:
            # Standard full clique across all mentioned entities
            for i in range(len(ent_indices)):
                for j in range(i + 1, len(ent_indices)):
                    edge_pairs.append((ent_indices[i], ent_indices[j]))

        # Apply Signed Hebbian Plasticity
        for u, v in edge_pairs:
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            w_t = self.edge_weights.get(key, 0.0)

            if p >= 0.0:
                # Positive valence / affinity growth (saturating toward +1.0)
                # Baseline potentiation of 0.3 * eta ensures co-occurrence is preserved
                delta_w = learning_rate * max(p, 0.3) * (1.0 - w_t)
            else:
                # Negative valence / suppressive barrier (accelerating toward -1.0)
                delta_w = learning_rate * p * (1.0 + w_t)

            w_next = float(np.clip(w_t + delta_w, -1.0, 1.0))
            self.edge_weights[key] = w_next
            self.edge_counts[key] = self.edge_counts.get(key, 0) + 1
            self.edge_timestamps[key] = t

        if edge_pairs:
            self._is_dirty = True

        # Wire incidence matrix if episode is present
        if episode_id is not None and episode_id in self.episode_id_to_idx:
            ep_idx = self.episode_id_to_idx[episode_id]
            for idx in ent_indices:
                weight = 0.2 if (listener_idx is not None and idx == listener_idx) else 1.0
                self.incidence_entries[(ep_idx, idx)] = weight
            if speaker_idx is not None:
                self.incidence_entries[(ep_idx, speaker_idx)] = 1.0
            if listener_idx is not None:
                self.incidence_entries[(ep_idx, listener_idx)] = 0.2
            self._incidence_dirty = True

    def record_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        episode_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records an explicit directional semantic relationship between two entities,
        and applies an associative update to the underlying 4D attention tensor.
        """
        src_idx = self.get_or_register_entity(source_id)
        tgt_idx = self.get_or_register_entity(target_id)
        t = timestamp if timestamp is not None else time.time()
        clean_rel = relation.strip()

        rel_record = {
            "source_id": source_id,
            "target_id": target_id,
            "source_idx": src_idx,
            "target_idx": tgt_idx,
            "relation": clean_rel,
            "weight": weight,
            "episode_id": episode_id,
            "timestamp": t,
        }
        self.semantic_relations.append(rel_record)

        pair_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
        if pair_key not in self.entity_relations:
            self.entity_relations[pair_key] = []
        if clean_rel not in self.entity_relations[pair_key]:
            self.entity_relations[pair_key].append(clean_rel)

        # Wire associative edge in the attention tensor
        self.record_interaction(
            entities=[source_id, target_id],
            polarity=weight,
            timestamp=t,
            episode_id=episode_id,
        )

    def get_relations_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Returns all semantic relations where entity_id is source or target."""
        return [
            r for r in self.semantic_relations
            if r["source_id"] == entity_id or r["target_id"] == entity_id
        ]

    # ------------------------------------------------------------------------
    # Inverse Hub Frequency (IHF) Adjacency Construction
    # ------------------------------------------------------------------------

    def build_ihf_adjacency_matrix(self, prune_threshold: Optional[float] = None) -> sp.csr_matrix:
        """
        Builds the sparse IHF-dampened adjacency matrix W_ihf in R^(N x N).
        Dampens high-degree hub nodes and prunes weak synapses.
        """
        if not self._is_dirty and self._cached_csr is not None:
            return self._cached_csr

        n = len(self.entity_to_id)
        if n == 0:
            empty = sp.csr_matrix((0, 0), dtype=np.float32)
            self._cached_csr = empty
            self._is_dirty = False
            return empty

        thresh = prune_threshold if prune_threshold is not None else self.prune_threshold

        # Step 1: Compute absolute degree of each entity
        degrees = np.zeros(n, dtype=np.float32)
        for (u, v), w in self.edge_weights.items():
            if u < n and v < n:
                abs_w = abs(w)
                degrees[u] += abs_w
                degrees[v] += abs_w

        # Step 2: Compute Inverse Hub Frequency
        # ihf(i) = ln(1.0 + N / (deg(i) + 1.0))
        ihf = np.log(1.0 + float(n) / (degrees + 1.0)).astype(np.float32)

        # Step 3: Populate directed dampened edges
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for (u, v), w in self.edge_weights.items():
            if u >= n or v >= n:
                continue

            # Direction u -> v
            norm_uv = (w / max(degrees[u], 1e-6)) * ihf[v]
            if abs(norm_uv) >= thresh:
                rows.append(u)
                cols.append(v)
                data.append(float(norm_uv))

            # Direction v -> u
            norm_vu = (w / max(degrees[v], 1e-6)) * ihf[u]
            if abs(norm_vu) >= thresh:
                rows.append(v)
                cols.append(u)
                data.append(float(norm_vu))

        if rows:
            csr = sp.csr_matrix(
                (data, (rows, cols)),
                shape=(n, n),
                dtype=np.float32,
            )
        else:
            csr = sp.csr_matrix((n, n), dtype=np.float32)

        self._cached_csr = csr
        self._is_dirty = False
        return csr

    def get_incidence_matrix(self) -> sp.csr_matrix:
        """Builds and caches the sparse episodic incidence matrix H in R^(M x N)."""
        if not self._incidence_dirty and self._cached_incidence is not None:
            return self._cached_incidence

        m = len(self.episodes)
        n = len(self.entity_to_id)
        if m == 0 or n == 0:
            empty = sp.csr_matrix((m, n), dtype=np.float32)
            self._cached_incidence = empty
            self._incidence_dirty = False
            return empty

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for (ep_idx, ent_idx), weight in self.incidence_entries.items():
            if ep_idx < m and ent_idx < n:
                rows.append(ep_idx)
                cols.append(ent_idx)
                data.append(float(weight))

        csr = sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)
        self._cached_incidence = csr
        self._incidence_dirty = False
        return csr

    def get_episode_adjacency_matrix(self) -> sp.csr_matrix:
        """
        Constructs and caches the direct Episode-to-Episode transition matrix W_ep in R^(M x M).
        Wires two fundamental episodic relationships:
        1. Temporal / Chronological Adjacency: Continuous temporal proximity decay across sequential episodes.
        2. Knowledge Slice Lineage: Direct causal links between episodes that mutate or reference the same historical TimeSlice.
        """
        if not self._episode_adj_dirty and self._cached_episode_adj is not None:
            return self._cached_episode_adj

        m = len(self.episodes)
        if m == 0:
            empty = sp.csr_matrix((0, 0), dtype=np.float32)
            self._cached_episode_adj = empty
            self._episode_adj_dirty = False
            return empty

        # 1. Temporal Sequential Adjacency
        sorted_ep_indices = sorted(range(m), key=lambda idx: self.episodes[idx].start_epoch)
        for i in range(len(sorted_ep_indices) - 1):
            idx_a = sorted_ep_indices[i]
            idx_b = sorted_ep_indices[i + 1]
            ep_a = self.episodes[idx_a]
            ep_b = self.episodes[idx_b]
            dt = abs(ep_b.start_epoch - ep_a.start_epoch)
            # 7-day half-life decay: exp(-dt / tau)
            w_time = float(np.exp(-dt / (7.0 * 86400.0)))
            if w_time > 0.05:
                pair_key1 = (idx_a, idx_b)
                pair_key2 = (idx_b, idx_a)
                self.episode_edges[pair_key1] = max(self.episode_edges.get(pair_key1, 0.0), w_time)
                self.episode_edges[pair_key2] = max(self.episode_edges.get(pair_key2, 0.0), w_time)
                ep_a.add_connected_episode(ep_b.episode_id)
                ep_b.add_connected_episode(ep_a.episode_id)

        # 2. Knowledge Slice Lineage Edges (shared TimeSlice mutation chains)
        reg_entities = getattr(self, "entities", {})
        if hasattr(self, "registry") and hasattr(self.registry, "entities"):
            reg_entities = self.registry.entities or reg_entities

        for ent in reg_entities.values():
            sk = getattr(ent, "slotted_knowledge", {})
            if isinstance(sk, dict):
                for kn in sk.values():
                    slice_ep_ids = []
                    for s in getattr(kn, "history", []):
                        ep_ref = getattr(s, "episode_id", None)
                        if ep_ref:
                            slice_ep_ids.append(ep_ref)
                    owner_ep = getattr(kn, "owner_episode_id", None)
                    if owner_ep:
                        slice_ep_ids.append(owner_ep)

                    slice_indices = [self.episode_id_to_idx[eid] for eid in slice_ep_ids if eid in self.episode_id_to_idx]
                    for x in range(len(slice_indices)):
                        for y in range(x + 1, len(slice_indices)):
                            idx_x, idx_y = slice_indices[x], slice_indices[y]
                            if idx_x != idx_y:
                                self.episode_edges[(idx_x, idx_y)] = 1.0
                                self.episode_edges[(idx_y, idx_x)] = 1.0
                                self.episodes[idx_x].add_connected_episode(self.episodes[idx_y].episode_id)
                                self.episodes[idx_y].add_connected_episode(self.episodes[idx_x].episode_id)

        rows = [p[0] for p in self.episode_edges.keys() if p[0] < m and p[1] < m]
        cols = [p[1] for p in self.episode_edges.keys() if p[0] < m and p[1] < m]
        data = [v for p, v in self.episode_edges.items() if p[0] < m and p[1] < m]

        if rows and cols:
            adj = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
            col_sums = np.asarray(adj.sum(axis=0)).ravel()
            col_sums[col_sums == 0] = 1.0
            inv_col = sp.diags(1.0 / col_sums, dtype=np.float32)
            stochastic_adj = adj.dot(inv_col)
        else:
            stochastic_adj = sp.csr_matrix((m, m), dtype=np.float32)

        self._cached_episode_adj = stochastic_adj
        self._episode_adj_dirty = False
        return stochastic_adj

    # ------------------------------------------------------------------------
    # 4D Spreading Activation (Topological Personalized PageRank)
    # ------------------------------------------------------------------------

    def spreading_activation_4d(
        self,
        seed_entities: Optional[Dict[str, float] | Sequence[str]] = None,
        seed_episodes: Optional[Dict[str, float] | Sequence[str]] = None,
        query_vector: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        t_now: Optional[float] = None,
        diffusion_steps: int = 2,
        alpha: Optional[float] = None,
        lambda_time: Optional[float] = None,
        tau_decay: Optional[float] = None,
    ) -> np.ndarray:
        """
        Traverses the 4D graph via Personalized PageRank power iteration:
        Stage 1: Multi-factor seed activation a_0 grounded in:
          - Factor 1: Seed entity presence in query.
          - Factor 1.5: Episode Seed Injection via Transposed Incidence Matrix H^T.
          - Factor 2: Dense semantic similarity to query vector (continuous scaling).
          - Factor 3: Entity data relevance (slotted knowledge aspects, values, and past time-slices).
        Stage 2: Continuous spatio-temporal diffusion modulated by elapsed epoch decay.
        """
        n = len(self.entity_to_id)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        now = t_now if t_now is not None else time.time()
        retention = alpha if alpha is not None else self.alpha_pagerank
        decay_rate = lambda_time if lambda_time is not None else self.time_lambda
        decay_period = tau_decay if tau_decay is not None else self.time_tau_decay

        # Stage 1: Build seed activation vector a_0 in R^N
        a_0 = np.zeros(n, dtype=np.float32)

        # Factor 1: Seed Entity Matches (Lexical / Working Memory Bias)
        if seed_entities:
            if isinstance(seed_entities, dict):
                for ent_key, weight in seed_entities.items():
                    if ent_key in self.entity_to_id:
                        idx = self.entity_to_id[ent_key]
                        a_0[idx] += float(weight)
                    else:
                        # Check canonical name / alias match
                        for eid, ent in self.entities.items():
                            if ent.matches_alias(ent_key):
                                idx = self.entity_to_id[eid]
                                a_0[idx] += float(weight)
            else:
                for ent_key in seed_entities:
                    if ent_key in self.entity_to_id:
                        idx = self.entity_to_id[ent_key]
                        a_0[idx] += 1.0
                    else:
                        for eid, ent in self.entities.items():
                            if ent.matches_alias(ent_key):
                                idx = self.entity_to_id[eid]
                                a_0[idx] += 1.0

        # Factor 1.5: Episode Seed Projection via Transposed Incidence Matrix H^T
        m = len(self.episodes)
        e_0 = np.zeros(m, dtype=np.float32)
        if seed_episodes:
            if isinstance(seed_episodes, dict):
                for ep_key, weight in seed_episodes.items():
                    ep_idx = self.episode_id_to_idx.get(ep_key)
                    if ep_idx is not None and ep_idx < m:
                        e_0[ep_idx] = max(e_0[ep_idx], float(weight))
                    else:
                        for idx_e, ep_obj in enumerate(self.episodes):
                            if ep_obj.episode_id == ep_key or ep_obj.name.startswith(str(ep_key)):
                                e_0[idx_e] = max(e_0[idx_e], float(weight))
            else:
                for ep_key in seed_episodes:
                    ep_idx = self.episode_id_to_idx.get(ep_key)
                    if ep_idx is not None and ep_idx < m:
                        e_0[ep_idx] = 1.0
                    else:
                        for idx_e, ep_obj in enumerate(self.episodes):
                            if ep_obj.episode_id == ep_key or ep_obj.name.startswith(str(ep_key)):
                                e_0[idx_e] = 1.0

            if m > 0 and n > 0 and np.any(e_0 > 0):
                H = self.get_incidence_matrix()
                a_from_ep = np.asarray(H.T.dot(e_0), dtype=np.float32).ravel()
                max_ep_proj = float(np.max(a_from_ep)) if len(a_from_ep) > 0 else 0.0
                if max_ep_proj > 1e-6:
                    a_0 += (a_from_ep / max_ep_proj) * 1.5

        # Factor 2: Dense Semantic Similarity (Continuous soft scaling, no hard cutoff)
        if query_vector is not None and self.entity_vectors:
            q_norm = l2_normalize(np.asarray(query_vector, dtype=np.float32))
            for idx, e_vec in self.entity_vectors.items():
                if idx < n:
                    cos_sim = float(np.dot(q_norm, e_vec))
                    if cos_sim > 0.20:
                        a_0[idx] += float(max(0.0, cos_sim - 0.15) ** 1.5)

        # Stage 1.5: Dynamic Traversal & Spectral Shannon Entropy Formulation
        # Zero regex, zero heuristics, zero LLM calls.
        # Derived purely from spectral dispersion of seed activation energy across graph manifold.
        a0_pos = np.maximum(a_0, 0.0)
        sum_a0 = float(np.sum(a0_pos))
        if sum_a0 > 1e-12:
            p = a0_pos / sum_a0
            # Normalized Shannon Entropy H_bar in [0.0, 1.0]
            # H = - sum(p * ln(p)) / ln(N)
            nonzero_p = p[p > 1e-12]
            shannon_h = -float(np.sum(nonzero_p * np.log(nonzero_p)))
            max_h = float(math.log(max(n, 2)))
            h_bar = float(np.clip(shannon_h / max_h, 0.0, 1.0))
        else:
            h_bar = 0.50

        # Dynamic Traversal Parameters (derived mathematically from spectral dispersion):
        # - Low entropy (sharp/factual query): 1 hop, high restart retention (0.92), narrow focus
        # - High entropy (diffuse/relational query): 3-5 hops, lower retention (0.70), deep multi-hop diffusion
        if diffusion_steps is None or diffusion_steps <= 0:
            adaptive_hops = int(np.clip(np.floor(1.0 + 4.0 * h_bar), 1, 5))
        else:
            adaptive_hops = diffusion_steps

        if alpha is None:
            adaptive_retention = float(np.clip(0.95 - 0.28 * h_bar, 0.67, 0.95))
        else:
            adaptive_retention = alpha

        # Dynamic residual convergence threshold & chunk scaling factor
        convergence_tol = float(np.clip(1e-3 * (1.0 - 0.85 * h_bar), 1e-4, 1e-3))
        chunk_scale = float(np.clip(1.0 + 1.6 * h_bar, 1.0, 3.0))

        # Seed Normalization
        max_a0 = float(np.max(a_0))
        if max_a0 > 1e-6:
            a_0 = a_0 / max_a0
        else:
            # Fallback: uniform restart if no seeds match
            a_0 = np.ones(n, dtype=np.float32) / float(n)

        # Stage 2: Continuous Spatio-Temporal Diffusion
        W_ihf = self.build_ihf_adjacency_matrix()

        # Build time-decayed effective adjacency matrix
        if W_ihf.nnz > 0 and len(self.edge_timestamps) > 0:
            W_eff = W_ihf.copy().tolil()
            for (u, v), t_edge in self.edge_timestamps.items():
                if u < n and v < n:
                    dt = max(0.0, now - t_edge)
                    decay = float(math.exp(-decay_rate * (dt / max(decay_period, 1.0))))
                    if W_eff[u, v] != 0:
                        W_eff[u, v] *= decay
                    if W_eff[v, u] != 0:
                        W_eff[v, u] *= decay
            W_trans = W_eff.tocsr().T
        else:
            W_trans = W_ihf.T

        # Normalize to column-stochastic transition matrix to ensure energy conservation
        col_sums = np.asarray(W_trans.sum(axis=0)).ravel()
        col_sums[col_sums == 0] = 1.0
        inv_col = sp.diags(1.0 / col_sums, dtype=np.float32)
        W_stochastic = W_trans.dot(inv_col)

        # Personalized PageRank Power Iteration with Dynamic Relative Residual Stopping
        a_curr = a_0.copy()
        actual_hops = 0
        residual = 0.0
        for _ in range(adaptive_hops):
            diffused = W_stochastic.dot(a_curr)
            a_next = adaptive_retention * diffused + (1.0 - adaptive_retention) * a_0
            norm_curr = float(np.linalg.norm(a_curr, ord=1))
            diff_norm = float(np.linalg.norm(a_next - a_curr, ord=1))
            residual = diff_norm / max(norm_curr, 1e-6)
            a_curr = a_next
            actual_hops += 1
            if residual < convergence_tol:
                break

        # Compute joint episode activation from diffused entity state + episode seeds
        if m > 0:
            H = self.get_incidence_matrix()
            a_pos = np.maximum(a_curr, 0.0).astype(np.float32)
            deg_episodes = np.asarray(H.sum(axis=1)).ravel()
            dampeners = np.sqrt(np.maximum(deg_episodes, 1.0)).astype(np.float32)
            e_from_graph = np.asarray(H.dot(a_pos) / dampeners, dtype=np.float32).ravel()
            max_efg = float(np.max(e_from_graph)) if len(e_from_graph) > 0 else 0.0
            if max_efg > 1e-6:
                e_from_graph = e_from_graph / max_efg
            self.last_episode_activation = 0.60 * e_from_graph + 0.40 * e_0
        else:
            self.last_episode_activation = np.zeros(0, dtype=np.float32)

        # Record traversal metadata for downstream dynamic memory chunk assembly
        self.last_traversal_meta = {
            "entropy": round(h_bar, 4),
            "hops_traversed": actual_hops,
            "max_hops": adaptive_hops,
            "retention_alpha": round(adaptive_retention, 4),
            "relative_residual": round(residual, 6),
            "chunk_scale": round(chunk_scale, 3),
        }

        return a_curr

    # ------------------------------------------------------------------------
    # Episode Scoring & Tri-Channel Retrieval Fusion
    # ------------------------------------------------------------------------

    def score_episodes_from_activation(self, activation: np.ndarray) -> np.ndarray:
        """
        Projects entity activation onto the episodic incidence matrix H
        with square-root degree dampening to prevent dense episode bias.
        """
        m = len(self.episodes)
        if m == 0 or len(activation) == 0:
            return np.zeros(m, dtype=np.float32)

        H = self.get_incidence_matrix()
        act_pos = np.maximum(activation, 0.0).astype(np.float32)

        # H @ max(a, 0)
        raw_scores = H.dot(act_pos)

        # Sub-linear degree dampening: sqrt(max(deg, 1.0))
        deg_episodes = np.asarray(H.sum(axis=1)).ravel()
        dampeners = np.sqrt(np.maximum(deg_episodes, 1.0)).astype(np.float32)

        graph_scores = raw_scores / dampeners
        return np.asarray(graph_scores, dtype=np.float32)

    def query_episodes(
        self,
        query_text: str,
        top_k: int = 5,
        query_vector: Optional[np.ndarray] = None,
        t_now: Optional[float] = None,
        seed_entities: Optional[Dict[str, float] | Sequence[str]] = None,
        seed_episodes: Optional[Dict[str, float] | Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tri-Channel Retrieval Fusion:
        1. Dense Semantic Similarity (weight = 0.50)
        2. Bipartite Dynamic Attention Diffusion & Episode Seeding (weight = 0.50)
        """
        m = len(self.episodes)
        if m == 0:
            return []

        now = t_now if t_now is not None else time.time()
        q_vec = (
            l2_normalize(np.asarray(query_vector, dtype=np.float32))
            if query_vector is not None
            else l2_normalize(np.asarray(self.embedder.embed_text(query_text), dtype=np.float32))
        )

        # Channel 1: Dense Semantic Similarity (Qwen 896-dim)
        dense_scores = np.zeros(m, dtype=np.float32)
        if self.episode_vectors:
            for i, ep_vec in enumerate(self.episode_vectors):
                sim = float(np.dot(q_vec, ep_vec))
                dense_scores[i] = max(0.0, sim)

        # Channel 2: Topological Graph Activation with Bipartite Episode Seeding
        activation = self.spreading_activation_4d(
            seed_entities=seed_entities,
            seed_episodes=seed_episodes,
            query_vector=q_vec,
            t_now=now,
        )
        graph_scores = self.score_episodes_from_activation(activation)
        max_graph = float(np.max(graph_scores)) if len(graph_scores) > 0 else 0.0
        if max_graph > 1e-6:
            norm_graph = graph_scores / max_graph
        else:
            norm_graph = graph_scores

        # Blend with joint episode activation if available
        ep_act = getattr(self, "last_episode_activation", None)
        if ep_act is not None and len(ep_act) == m and np.any(ep_act > 0):
            norm_graph = 0.50 * norm_graph + 0.50 * ep_act

        # Diffuse activation across direct Episode-to-Episode graph edges (temporal continuity + slice lineages)
        ep_adj = self.get_episode_adjacency_matrix()
        if ep_adj is not None and ep_adj.shape[0] == m and np.any(norm_graph > 0):
            ep_diffused = np.asarray(ep_adj.dot(norm_graph), dtype=np.float32).ravel()
            norm_graph = 0.70 * norm_graph + 0.30 * ep_diffused

        # Channel 3: Lexical Matching across Episode Name & Summary via BM25 IDF Weighting
        # Zero hardcoded stopwords, zero arbitrary length thresholds
        lex_scores = np.zeros(m, dtype=np.float32)
        import re
        self.ensure_turn_index()
        turn_idf = getattr(self, "_turn_idf", {}) or {}
        q_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.]+", query_text.lower()))

        token_weights: Dict[str, float] = {}
        if q_tokens:
            max_idf = getattr(self, "_max_idf", 10.0)
            token_weights = {tok: turn_idf.get(tok, max_idf) for tok in q_tokens}
            total_idf = sum(token_weights.values()) or 1.0
            for i, ep in enumerate(self.episodes):
                ep_text = f"{ep.name or ''} {ep.summary or ''}".lower()
                matched_idf = sum(w for tok, w in token_weights.items() if tok in ep_text)
                if matched_idf > 0:
                    lex_scores[i] = min(1.0, float(matched_idf / total_idf) * 2.0)

        # Channel 4: Turn-to-Episode Bidirectional Spreading
        # Distinctive query content tokens (high IDF) hit exact turns via _turn_token_index.
        # These turn hits back-propagate an activation signal to their parent episode,
        # rescuing episodes with sparse or generic summaries.
        turn_spread_scores = np.zeros(m, dtype=np.float32)
        if q_tokens and hasattr(self, "_turn_token_index") and self._turn_token_index:
            for tok, w in token_weights.items():
                if w >= 3.0:  # Distinctive anchor token
                    m_turns = self._turn_token_index.get(tok, [])
                    for t_idx in m_turns:
                        if t_idx < len(self._turn_episode_ids):
                            ep_id = self._turn_episode_ids[t_idx]
                            ep_idx = self.episode_id_to_idx.get(ep_id)
                            if ep_idx is not None and ep_idx < m:
                                turn_spread_scores[ep_idx] += float(w)

        max_spread = float(np.max(turn_spread_scores)) if len(turn_spread_scores) > 0 else 0.0
        norm_spread = turn_spread_scores / max_spread if max_spread > 1e-6 else turn_spread_scores

        # Factor in direct episode seeds (temporal date range & session anchors) with 25% effective weight
        seed_ep_arr = np.zeros(m, dtype=np.float32)
        if seed_episodes:
            if isinstance(seed_episodes, dict):
                for ep_k, w in seed_episodes.items():
                    idx = self.episode_id_to_idx.get(ep_k)
                    if idx is not None and idx < m:
                        seed_ep_arr[idx] = max(seed_ep_arr[idx], float(w))
                    else:
                        for idx_e, ep_obj in enumerate(self.episodes):
                            if ep_obj.episode_id == ep_k or ep_obj.name.startswith(str(ep_k)):
                                seed_ep_arr[idx_e] = max(seed_ep_arr[idx_e], float(w))
            else:
                for ep_k in seed_episodes:
                    idx = self.episode_id_to_idx.get(ep_k)
                    if idx is not None and idx < m:
                        seed_ep_arr[idx] = 1.0
                    else:
                        for idx_e, ep_obj in enumerate(self.episodes):
                            if ep_obj.episode_id == ep_k or ep_obj.name.startswith(str(ep_k)):
                                seed_ep_arr[idx_e] = 1.0

        # Factor in relative temporal metadata matching from query
        q_lower = query_text.lower()
        rel_keywords = [
            "yesterday", "today", "tomorrow", "last week", "past week", "this week",
            "last month", "past month", "this month", "last year", "past year", "this year",
        ]
        has_rel_query = any(k in q_lower for k in rel_keywords) or ("days ago" in q_lower) or ("weeks ago" in q_lower) or ("months ago" in q_lower) or ("years ago" in q_lower)
        if has_rel_query:
            for idx_e, ep_obj in enumerate(self.episodes):
                r_meta = getattr(ep_obj, "metadata", {}).get("relative_time") or {}
                r_phrase = r_meta.get("relative_phrase", "").lower()
                r_bucket = r_meta.get("relative_bucket", "").lower()
                if r_phrase and (f" {r_phrase} " in f" {q_lower} " or f" {r_phrase}" in f" {q_lower}" or f"{r_phrase} " in f"{q_lower} "):
                    seed_ep_arr[idx_e] = max(seed_ep_arr[idx_e], 1.0)
                elif r_bucket and (f" {r_bucket.lower()} " in f" {q_lower} "):
                    seed_ep_arr[idx_e] = max(seed_ep_arr[idx_e], 0.90)

        max_dense = float(np.max(dense_scores)) if len(dense_scores) > 0 else 0.0
        norm_dense = dense_scores / max_dense if max_dense > 1e-6 else dense_scores

        max_seed_ep = float(np.max(seed_ep_arr)) if len(seed_ep_arr) > 0 else 0.0
        if max_seed_ep > 1e-6:
            norm_seed_ep = seed_ep_arr / max_seed_ep
            fused_scores = 0.25 * norm_seed_ep + 0.30 * norm_dense + 0.15 * norm_graph + 0.10 * lex_scores + 0.20 * norm_spread
        else:
            fused_scores = 0.35 * norm_dense + 0.25 * norm_graph + 0.15 * lex_scores + 0.25 * norm_spread

        top_indices = np.argsort(-fused_scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            ep = self.episodes[idx]
            results.append({
                "episode": ep.to_dict(),
                "final_score": float(fused_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "graph_score": float(norm_graph[idx]),
                "lex_score": float(lex_scores[idx]),
                "turn_spread_score": float(norm_spread[idx]),
            })

        return results

    @staticmethod
    def _extract_session_id(episode_id: str, episode_name: str = "") -> str:
        """Extracts normalized session key (e.g. 'session_83') from episode ID or human name."""
        import re
        if episode_name:
            m = re.search(r"\bSession\s*(\d+)\b", episode_name, re.IGNORECASE)
            if m:
                return f"session_{int(m.group(1))}"
        if episode_id:
            m = re.search(r"ep_0*(\d+)_", episode_id)
            if m:
                return f"session_{int(m.group(1))}"
            if episode_id.startswith("ep_"):
                return episode_id
        return episode_id or ""

    def query_conversation_turns(
        self,
        query_text: str,
        top_k: int = 5,
        query_vector: Optional[np.ndarray] = None,
        boost_episodes: Optional[Dict[str, float]] = None,
        chunk_scale: float = 1.0,
        candidate_episodes: Optional[Sequence[str]] = None,
        grounded_entities: Optional[Sequence[str]] = None,
        grounded_slots: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top matching exact dialogue turns or code chunks using graph-grounded
        hierarchical traversal:
        1. Binds query to candidate session frames and grounded entities/slots.
        2. Shortlists candidate turns within candidate episodes matching entity names/aliases or slot terms.
        3. Resolves dense semantic + lexical scores over the shortlisted candidate pool before backfilling.
        Zero domain hardcoding, zero LLM calls.
        """
        if not self.turn_vectors or not self.turn_records:
            return []

        import re

        # Natural clause splitting on standard punctuation boundaries (semicolons, question marks, newlines, conjunctions)
        # Dense embedder (all-MiniLM-L6-v2) handles semantic clause composition without arbitrary regexes or word counts
        sub_queries = [query_text]
        raw_clauses = [c.strip() for c in re.split(r"[;\n\?]|\s+and\s+|,\s*", query_text) if c.strip()]
        for c in raw_clauses:
            if c != query_text and c not in sub_queries:
                sub_queries.append(c)

        self.ensure_turn_index()
        turn_mat = self._turn_mat
        if turn_mat is None or len(turn_mat) == 0:
            return []

        dynamic_k = max(1, top_k)

        # Batch embed all subqueries that do not have precomputed vectors
        sub_vecs: List[np.ndarray] = []
        sq_to_embed = []
        sq_to_embed_indices = []
        for i, sq in enumerate(sub_queries):
            if sq == query_text and query_vector is not None:
                sub_vecs.append(l2_normalize(np.asarray(query_vector, dtype=np.float32)))
            else:
                sub_vecs.append(None)
                sq_to_embed.append(sq)
                sq_to_embed_indices.append(i)

        if sq_to_embed:
            batch_embs = self.embedder.embed_text(sq_to_embed)
            if len(sq_to_embed) == 1 and batch_embs.ndim == 1:
                batch_embs = batch_embs.reshape(1, -1)
            for j, orig_idx in enumerate(sq_to_embed_indices):
                sub_vecs[orig_idx] = l2_normalize(np.asarray(batch_embs[j], dtype=np.float32))

        # Vectorized scoring helpers
        user_mask = self._turn_user_mask
        token_index = self._turn_token_index or {}
        turn_idf = getattr(self, "_turn_idf", {}) or {}
        max_idf = getattr(self, "_max_idf", 10.0)
        boost_ep_arr = None
        if boost_episodes and self._turn_episode_ids:
            b_vals = np.zeros(len(self._turn_episode_ids), dtype=np.float32)
            for idx, ep_id in enumerate(self._turn_episode_ids):
                if ep_id in boost_episodes:
                    b_vals[idx] = 0.35 * min(1.0, float(boost_episodes[ep_id]))
            boost_ep_arr = b_vals

        # --------------------------------------------------------------------
        # Hierarchical Graph Grounding: Candidate Sessions & Anchor Slot Terms
        # --------------------------------------------------------------------
        cand_session_keys: Set[str] = set()
        if candidate_episodes:
            for ce in candidate_episodes:
                skey = self._extract_session_id(ce)
                if skey:
                    cand_session_keys.add(skey)
                ep_idx = self.episode_id_to_idx.get(ce)
                if ep_idx is not None and ep_idx < len(self.episodes):
                    ep_name = getattr(self.episodes[ep_idx], "name", "")
                    skey2 = self._extract_session_id("", ep_name)
                    if skey2:
                        cand_session_keys.add(skey2)
        if boost_episodes:
            for be in boost_episodes.keys():
                skey = self._extract_session_id(be)
                if skey:
                    cand_session_keys.add(skey)
                ep_idx = self.episode_id_to_idx.get(be)
                if ep_idx is not None and ep_idx < len(self.episodes):
                    ep_name = getattr(self.episodes[ep_idx], "name", "")
                    skey2 = self._extract_session_id("", ep_name)
                    if skey2:
                        cand_session_keys.add(skey2)

        # Knowledge Graph Registry Grounded Entities & Slots
        registry_entities = getattr(self, "entities", {})
        if hasattr(self, "registry") and hasattr(self.registry, "entities"):
            registry_entities = self.registry.entities or registry_entities

        anchor_terms: Set[str] = set()
        if grounded_entities:
            for ge in grounded_entities:
                if isinstance(ge, str) and ge.strip():
                    anchor_terms.add(ge.strip().lower())

        for ent in registry_entities.values():
            cname = getattr(ent, "canonical_name", "").lower()
            if cname and cname in query_text.lower():
                anchor_terms.add(cname)
            for a in getattr(ent, "aliases", []):
                if isinstance(a, str) and a.lower() in query_text.lower():
                    anchor_terms.add(a.lower())

        # Domain slots defined by the Knowledge Graph Registry (slotted_knowledge)
        known_slots: Set[str] = set()
        for ent in registry_entities.values():
            sk = getattr(ent, "slotted_knowledge", None)
            if isinstance(sk, dict):
                for k in sk.keys():
                    known_slots.add(str(k).lower())

        if grounded_slots:
            for gs in grounded_slots:
                if isinstance(gs, str) and gs.strip():
                    known_slots.add(gs.strip().lower())

        slot_terms: Set[str] = {s for s in known_slots if s in query_text.lower()}

        # Distinctive content words from query via BM25 IDF (zero hardcoded stopwords or length thresholds)
        q_tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", query_text.lower())
        token_idfs: Dict[str, float] = {}
        if q_tokens:
            token_idfs = {tok: turn_idf.get(tok, max_idf) for tok in q_tokens}
            for tok, idf_val in token_idfs.items():
                if idf_val >= 3.0:
                    anchor_terms.add(tok)

        # --------------------------------------------------------------------
        # Bidirectional Turn-to-Session Spreading:
        # Detect sessions outside candidate_episodes with strong anchor turn support
        # --------------------------------------------------------------------
        session_anchor_weights: Dict[str, float] = {}
        for tok, idf_val in token_idfs.items():
            if idf_val >= 3.0:
                matching_turn_idxs = token_index.get(tok, [])
                for t_idx in matching_turn_idxs:
                    if t_idx < len(self.turn_records):
                        t_rec = self.turn_records[t_idx]
                        skey = self._extract_session_id(t_rec.get("episode_id", ""), t_rec.get("episode_name", ""))
                        if skey:
                            session_anchor_weights[skey] = session_anchor_weights.get(skey, 0.0) + idf_val

        # Dynamically promote sessions that accumulate distinctive anchor evidence
        for skey, total_w in session_anchor_weights.items():
            if total_w >= 5.0:  # At least 2 distinctive anchor hits or 1 highly distinctive keyword
                cand_session_keys.add(skey)

        # Date & temporal expression parsing for direct turn attendance (both turns on that date and date in text)
        from .node_models import parse_date_string
        iso_dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query_text)
        written_dates = re.findall(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[.,]?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
            query_text,
            re.IGNORECASE,
        )
        month_year_dates = re.findall(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4})\b",
            query_text,
            re.IGNORECASE,
        )
        all_candidate_dates = iso_dates + written_dates + month_year_dates
        parsed_epochs = []
        for d_str in all_candidate_dates:
            p_ep = parse_date_string(d_str)
            if p_ep is not None and p_ep > 0:
                parsed_epochs.append(p_ep)

        t_min, t_max = None, None
        if len(parsed_epochs) >= 2:
            t_min = min(parsed_epochs)
            t_max = max(parsed_epochs) + 86400.0  # inclusive of end day
        elif len(parsed_epochs) == 1:
            t_min = parsed_epochs[0] - 86400.0
            t_max = parsed_epochs[0] + 86400.0

        query_date_phrases = [d.lower() for d in all_candidate_dates]

        # --------------------------------------------------------------------
        # Hierarchical Shortlisting of Candidate Turns
        # --------------------------------------------------------------------
        tier1_indices: List[int] = []
        tier2_indices: List[int] = []
        tier3_indices: List[int] = []
        tier4_indices: List[int] = []

        if cand_session_keys or anchor_terms or t_min is not None or query_date_phrases:
            for idx, t_rec in enumerate(self.turn_records):
                t_skey = self._extract_session_id(t_rec.get("episode_id", ""), t_rec.get("episode_name", ""))
                t_text = t_rec.get("text", "").lower()
                t_epoch = float(t_rec.get("epoch") or 0.0)

                has_anchor = any(f" {term}" in f" {t_text}" or f"{term} " in f"{t_text} " for term in anchor_terms)
                has_slot = any(f" {st}" in f" {t_text}" or f"{st} " in f"{t_text} " for st in slot_terms) if slot_terms else True
                in_session = t_skey in cand_session_keys or t_rec.get("episode_id") in cand_session_keys

                # Direct attendance: turn on that date OR date present in turn text
                on_date = (t_min is not None and t_max is not None and t_epoch > 0 and t_min <= t_epoch <= t_max)
                mentions_date = any(dp in t_text for dp in query_date_phrases) if query_date_phrases else False
                is_temporal_match = on_date or mentions_date

                if (in_session or on_date) and (has_anchor and has_slot or is_temporal_match):
                    tier1_indices.append(idx)
                elif (in_session or on_date) and (has_anchor or has_slot or is_temporal_match):
                    tier2_indices.append(idx)
                elif has_anchor or is_temporal_match:
                    tier3_indices.append(idx)
                elif in_session:
                    tier4_indices.append(idx)

            # Include candidate session context turns + anchor matching turns
            shortlisted_indices = tier1_indices + tier2_indices + tier3_indices + tier4_indices
        else:
            shortlisted_indices = []

        tier1_set = set(tier1_indices)
        tier2_set = set(tier2_indices)
        tier3_set = set(tier3_indices)

        def _score_turn_subquery(
            sq_vec: np.ndarray,
            sq: str,
            cand_pool: Optional[List[int]] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            sq_tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", sq.lower())
            token_weights = {t: turn_idf.get(t, max_idf) for t in set(sq_tokens)}
            total_query_idf = sum(token_weights.values()) or 1.0

            if cand_pool is not None and len(cand_pool) > 0:
                sub_mat = turn_mat[cand_pool]
                scores = np.asarray(sub_mat @ sq_vec, dtype=np.float32)
                np.maximum(scores, 0.0, out=scores)

                anc_weights_arr = np.zeros(len(cand_pool), dtype=np.float32)
                for i_pos, orig_idx in enumerate(cand_pool):
                    t_rec = self.turn_records[orig_idx]
                    t_text = t_rec.get("text", "").lower()
                    t_epoch = float(t_rec.get("epoch") or 0.0)

                    # Gentle structural tier tie-breaker (does not overpower semantic similarity)
                    if orig_idx in tier1_set:
                        scores[i_pos] += 0.08
                    elif orig_idx in tier2_set:
                        scores[i_pos] += 0.05
                    elif orig_idx in tier3_set:
                        scores[i_pos] += 0.02

                    # Temporal anchor boost for turns on that date or mentioning the date
                    turn_on_date = (t_min is not None and t_max is not None and t_epoch > 0 and t_min <= t_epoch <= t_max)
                    turn_mentions_date = any(dp in t_text for dp in query_date_phrases) if query_date_phrases else False
                    if turn_on_date or turn_mentions_date:
                        anc_weights_arr[i_pos] += 12.0
                        scores[i_pos] += 0.20

                    # Dynamic BM25 IDF Lexical Coverage on Question Predicate (zero stopwords or length thresholds)
                    if token_weights:
                        matched_idf = sum(w for t, w in token_weights.items() if t in t_text)
                        anc_weights_arr[i_pos] += matched_idf
                        if matched_idf > 0:
                            coverage = matched_idf / total_query_idf
                            scores[i_pos] += 0.40 * (coverage ** 1.2)

                return scores, anc_weights_arr
            else:
                scores = np.asarray(turn_mat @ sq_vec, dtype=np.float32)
                np.maximum(scores, 0.0, out=scores)

                anc_weights_arr = np.zeros(len(turn_mat), dtype=np.float32)
                if boost_ep_arr is not None:
                    scores += boost_ep_arr

                if token_weights:
                    for term, w in token_weights.items():
                        m_indices = token_index.get(term)
                        if m_indices:
                            scores[m_indices] += 0.35 * (w / total_query_idf)
                            anc_weights_arr[m_indices] += w

                return scores, anc_weights_arr

        # Step C: Resolve Query & Subqueries over Shortlisted Candidate Pool with Quota Management
        active_pool = shortlisted_indices if len(shortlisted_indices) > 0 else None
        selected_candidates: Dict[Any, Tuple[Dict[str, Any], float]] = {}

        if active_pool is not None and len(active_pool) > 0:
            if len(sub_queries) > 1:
                # Score candidate pool against each subquery clause
                clause_score_matrices: List[np.ndarray] = []
                clause_anc_matrices: List[np.ndarray] = []
                for sq_idx, sq in enumerate(sub_queries):
                    sq_vec = sub_vecs[sq_idx]
                    scs, ancs = _score_turn_subquery(sq_vec, sq, cand_pool=active_pool)
                    clause_score_matrices.append(scs)
                    clause_anc_matrices.append(ancs)

                # Combined score: Max score across clauses + multi-clause synergy bonus
                stacked = np.stack(clause_score_matrices, axis=0)  # shape: (n_clauses, n_cands)
                max_scores = np.max(stacked, axis=0)
                mean_scores = np.mean(stacked, axis=0)
                scores_arr = max_scores + 0.15 * mean_scores
                anc_arr = np.max(np.stack(clause_anc_matrices, axis=0), axis=0)
            else:
                scores_arr, anc_arr = _score_turn_subquery(sub_vecs[0], query_text, cand_pool=active_pool)

            # Build candidate tuple list: (pos, orig_idx, score, session_key, is_tier1_2, anchor_weight)
            cands = []
            for pos, orig_idx in enumerate(active_pool):
                t_rec = self.turn_records[orig_idx]
                skey = self._extract_session_id(t_rec.get("episode_id", ""), t_rec.get("episode_name", ""))
                is_grounded = (orig_idx in tier1_set or orig_idx in tier2_set)
                sc = float(scores_arr[pos])
                anc_w = float(anc_arr[pos]) if anc_arr is not None else 0.0
                cands.append((pos, orig_idx, sc, skey, is_grounded, anc_w))

            # Quota Allocation 1: Reserved slots for grounded shortlisted turns (Tiers 1 & 2)
            # Guarantees that factual answers matching query anchors cannot be crowded out by conversational chatter
            reserved_grounded_cap = max(6, int(round(dynamic_k * 0.45)))
            grounded_cands = [c for c in cands if c[4] and c[5] > 0]
            grounded_cands.sort(key=lambda x: (x[5], x[2]), reverse=True)
            for _, orig_idx, sc, _, _, _ in grounded_cands[:reserved_grounded_cap]:
                t_rec = self.turn_records[orig_idx]
                tid = t_rec.get("turn_id")
                if tid and tid not in selected_candidates:
                    selected_candidates[tid] = (t_rec, sc)

            # Quota Allocation 2: Per-Retrieved-Episode Slots
            # Guarantees that every candidate episode in cand_session_keys receives representation
            from collections import defaultdict
            session_cand_map = defaultdict(list)
            for c in cands:
                session_cand_map[c[3]].append(c)

            # In each candidate session, sort turns by composite relevance (grounded anchor mass + dense score)
            for skey in session_cand_map:
                session_cand_map[skey].sort(key=lambda x: (x[5] > 0, x[5], x[2]), reverse=True)

            for skey in cand_session_keys:
                if skey in session_cand_map:
                    for _, orig_idx, sc, _, _, _ in session_cand_map[skey][:2]:
                        t_rec = self.turn_records[orig_idx]
                        tid = t_rec.get("turn_id")
                        if tid and tid not in selected_candidates:
                            selected_candidates[tid] = (t_rec, sc)
                            break

            # Quota Allocation 3: Global Relevance Pool
            # Fill remaining quota up to dynamic_k with best remaining turns from active_pool
            cands_by_score = sorted(cands, key=lambda x: x[2], reverse=True)
            for _, orig_idx, sc, _, _, _ in cands_by_score:
                if len(selected_candidates) >= dynamic_k:
                    break
                t_rec = self.turn_records[orig_idx]
                tid = t_rec.get("turn_id")
                if tid and tid not in selected_candidates:
                    selected_candidates[tid] = (t_rec, sc)

        # Quota Allocation 4: Global Fallback Backfill (if active pool had fewer items than dynamic_k)
        if len(selected_candidates) < dynamic_k:
            fill_scores, _ = _score_turn_subquery(sub_vecs[0], query_text, cand_pool=None)
            top_fill = np.argsort(fill_scores)[::-1]
            for pos in top_fill:
                t_rec = self.turn_records[pos]
                tid = t_rec.get("turn_id")
                if tid and tid not in selected_candidates:
                    selected_candidates[tid] = (t_rec, float(fill_scores[pos]))
                if len(selected_candidates) >= dynamic_k:
                    break

        # Final Rank Ordering: Sort all selected quota turns in descending order of score
        results = [
            {**t_rec, "score": sc}
            for t_rec, sc in sorted(selected_candidates.values(), key=lambda x: x[1], reverse=True)
        ]
        return results

    # ------------------------------------------------------------------------
    # Contradiction & Mutation Attenuation
    # ------------------------------------------------------------------------

    def attenuate_entity(
        self,
        entity_id: str,
        edge_suppression: float = 0.45,
        incidence_suppression: float = 0.50,
    ) -> None:
        """
        Suppresses an obsolete or superseded concept (e.g., migration source):
        - Scales incident tensor edges by edge_suppression (default 0.45, 55% reduction).
        - Scales episodic incidence by incidence_suppression (default 0.50, 50% reduction).
        Preserves historical time-travel queryability while attenuating current retrieval.
        """
        if entity_id not in self.entity_to_id:
            return

        idx = self.entity_to_id[entity_id]

        # Suppress associative edges
        for key in list(self.edge_weights.keys()):
            u, v = key
            if u == idx or v == idx:
                self.edge_weights[key] *= edge_suppression

        # Suppress episodic incidence
        for key in list(self.incidence_entries.keys()):
            ep_idx, ent_idx = key
            if ent_idx == idx:
                self.incidence_entries[key] *= incidence_suppression

        self._is_dirty = True
        self._incidence_dirty = True
        logger.info(f"Attenuated obsolete entity '{entity_id}' in tensor engine.")

    def get_top_activated_entities(
        self,
        activation: np.ndarray,
        top_k: int = 5,
        min_activation: float = 0.05,
    ) -> List[Tuple[EntityNode, float]]:
        """Returns top activated EntityNodes from a spreading activation vector."""
        if len(activation) == 0:
            return []

        sorted_indices = np.argsort(-activation)
        results: List[Tuple[EntityNode, float]] = []

        for idx in sorted_indices:
            score = float(activation[idx])
            if score < min_activation or len(results) >= top_k:
                break
            eid = self.id_to_entity.get(int(idx))
            if eid and eid in self.entities:
                results.append((self.entities[eid], score))

        return results
