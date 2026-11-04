"""
Cognitive LLM Memory Manager & Structuring Engine (llm_manager.py)
==================================================================

Orchestrates the complete cognitive cycle for EEKG:
1. Pre-Query Candidate Memory Matching:
   - Sequence-end tokenization and clause segmentation.
   - Dual filler removal: Fast regex stopword stripping + dense embedding cluster filtering.
   - Candidate entity regex matching (\b{alias}\b) + vector cosine similarity check against stored memory.
2. LLM Decision & Data Structuring:
   - Evaluates incoming data against candidate stored memory.
   - Decides whether to UPDATE existing entities or CREATE new canonical entities.
   - Segregates data entries into discrete attribute slots.
   - Determines mutation pathways: TIME_SLICED (versioned history) vs. OVERWRITE (hard replacement).
   - Generates episodic naming, timeframe intervals, and multi-entity connections.
3. Memory Commitment & Internal Operations:
   - Commits structured entities, time slices, and episodes to EntityRegistry and SQLiteGraphStore.
   - Dispatches downstream internal operations: continuous taxonomic projection (TaxonomicField)
     and signed Hebbian attention updates (tensor_engine).
4. Grounded Query Retrieval:
   - Decomposes user questions, checks historical slice flags, and synthesizes verified answers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .embedder import QwenSemanticEmbedder, cosine_similarity, l2_normalize
from .llm_client import LLMClientError, MemoryLLMClient
from .node_models import (
    EntityNode,
    EntityRegistry,
    SQLiteGraphStore,
    TimeframeEpisode,
    TimeSlice,
    UpdateMode,
    epoch_to_iso,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pre-Retrieval Filtering & Candidate Matching
# ============================================================================

class WordFilter:
    """
    High-speed linguistic filter:
    1. Strips conversational fillers and functional stopwords via regex.
    2. Filters out semantically generic tokens via embedding cluster rejection.
    """

    # Conversational fillers, discourse markers, and high-frequency functional words
    FILLER_REGEX = re.compile(
        r"\b(?:"
        r"uh|um|like|so|actually|basically|well|i\s+mean|you\s+know|literally|just|sort\s+of|kind\s+of|"
        r"anyway|the|a|an|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|shall|should|"
        r"can|could|may|might|must|in|on|at|to|for|with|by|about|against|between|into|through|during|before|"
        r"after|above|below|from|up|down|in|out|off|over|under|again|further|then|once|here|there|when|where|"
        r"why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|too|very"
        r")\b",
        re.IGNORECASE,
    )

    # Clause and sequence end delimiters
    SEQUENCE_END_REGEX = re.compile(r"[.!?;:\n,]+")

    def __init__(self, embedder: Optional[QwenSemanticEmbedder] = None):
        self.embedder = embedder or QwenSemanticEmbedder()
        self._init_filler_clusters()

    def _init_filler_clusters(self) -> None:
        """Initializes semantic centroids of generic filler phrases."""
        generic_fillers = [
            "something like that",
            "you know what I mean",
            "as a matter of fact",
            "more or less",
            "at the end of the day",
            "general stuff and things",
        ]
        filler_vecs = np.asarray(self.embedder.embed_text(generic_fillers), dtype=np.float32)
        if filler_vecs.ndim == 2 and filler_vecs.shape[0] > 0:
            centroid = np.mean(filler_vecs, axis=0)
            self.filler_centroid = l2_normalize(centroid)
        else:
            self.filler_centroid = np.zeros(self.embedder.dim, dtype=np.float32)

    def extract_informative_sequences(self, text: str) -> List[str]:
        """
        Splits text based on sequence ends, strips regex fillers, and prunes
        tokens matching generic filler embedding clusters.
        """
        raw_clauses = self.SEQUENCE_END_REGEX.split(text)
        informative_tokens: List[str] = []

        for clause in raw_clauses:
            cleaned = self.FILLER_REGEX.sub(" ", clause).strip()
            tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]
            for token in tokens:
                if re.match(r"^[a-zA-Z0-9_\-\./]+$", token):
                    informative_tokens.append(token)

        if not informative_tokens:
            return []

        # Optional: Embedding cluster rejection for ambiguous tokens
        filtered_tokens: List[str] = []
        token_vecs = self.embedder.embed_text(informative_tokens)
        if token_vecs.ndim == 1:
            token_vecs = np.expand_dims(token_vecs, axis=0)

        for i, token in enumerate(informative_tokens):
            if np.all(self.filler_centroid == 0):
                filtered_tokens.append(token)
                continue
            sim_to_filler = cosine_similarity(token_vecs[i], self.filler_centroid)
            if sim_to_filler < 0.85:  # Retain distinct semantic tokens
                filtered_tokens.append(token)

        return filtered_tokens


class MemoryCandidateMatcher:
    """
    Pre-queries stored memory for an incoming utterance or question:
    - Matches candidate entity names and aliases via word-boundary regex.
    - Matches candidate entities via dense embedding similarity.
    - Assembles the candidate memory context to inject into the LLM prompt.
    """

    def __init__(self, registry: EntityRegistry, embedder: Optional[QwenSemanticEmbedder] = None):
        self.registry = registry
        self.embedder = embedder or QwenSemanticEmbedder()
        self.word_filter = WordFilter(embedder=self.embedder)

    def find_candidates(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the stored memory for entities matching the text via regex and vector similarity.
        """
        informative_tokens = self.word_filter.extract_informative_sequences(text)
        matched_entities: Dict[str, Tuple[EntityNode, float]] = {}

        # 1. Regex entity matching on informative tokens and full text
        with self.registry._lock:
            for entity_id, entity in self.registry.entities.items():
                s_lex = 0.0
                # Check each alias against full text and tokens
                for alias in entity.aliases:
                    alias_norm = alias.strip().lower()
                    pattern = rf"\b{re.escape(alias_norm)}\b"
                    if re.search(pattern, text, re.IGNORECASE):
                        s_lex = max(s_lex, 1.0)
                        break
                    for token in informative_tokens:
                        if token.lower() == alias_norm or re.search(pattern, token, re.IGNORECASE):
                            s_lex = max(s_lex, 0.85)
                            break

                if s_lex > 0:
                    matched_entities[entity_id] = (entity, s_lex)

        # 2. Dense similarity matching on full text embedding
        text_vec = self.embedder.embed_text(text)
        sim_results = self.registry.search_entities(text, query_vector=text_vec, top_k=top_k)

        for entity, score in sim_results:
            if entity.entity_id in matched_entities:
                prev_e, prev_score = matched_entities[entity.entity_id]
                matched_entities[entity.entity_id] = (entity, max(prev_score, score))
            else:
                matched_entities[entity.entity_id] = (entity, score)

        # 3. Assemble candidate context payloads
        candidates: List[Dict[str, Any]] = []
        sorted_matches = sorted(matched_entities.values(), key=lambda x: x[1], reverse=True)[:top_k]

        for entity, score in sorted_matches:
            # Active slots and metadata
            active_slots = {
                aspect: kn.current_value
                for aspect, kn in entity.slotted_knowledge.items()
            }
            slot_metadata = {
                aspect: {
                    "active_since": kn.active_since,
                    "active_since_iso": epoch_to_iso(kn.active_since)[:10] if kn.active_since else "present",
                }
                for aspect, kn in entity.slotted_knowledge.items()
            }
            # Recent history (last 2 slices per slot)
            recent_history = {}
            for aspect, kn in entity.slotted_knowledge.items():
                if kn.history:
                    recent_history[aspect] = [
                        {
                            "value": s.value,
                            "valid_from": s.valid_from,
                            "valid_from_iso": epoch_to_iso(s.valid_from)[:10] if s.valid_from else "unknown",
                            "valid_to": s.valid_to,
                            "valid_to_iso": epoch_to_iso(s.valid_to)[:10] if s.valid_to else "unknown",
                            "reason": s.mutation_reason,
                        }
                        for s in kn.history[-2:]
                    ]

            candidates.append({
                "entity_id": entity.entity_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "aliases": list(entity.aliases),
                "active_slots": active_slots,
                "slot_metadata": slot_metadata,
                "recent_history": recent_history,
                "connected_episodes": entity.connected_episode_ids[-3:],
                "match_score": round(score, 3),
            })

        return candidates


# ============================================================================
# Master LLM Memory Manager
# ============================================================================

class LLMMemoryManager:
    """
    Coordinates LLM-driven structured memory ingestion, candidate context pre-matching,
    entity creation/update decisions, time-slice management, and grounded query answering.
    """

    def __init__(
        self,
        registry: EntityRegistry,
        store: Optional[SQLiteGraphStore] = None,
        llm_client: Optional[MemoryLLMClient] = None,
        embedder: Optional[QwenSemanticEmbedder] = None,
        tensor_engine: Optional[Any] = None,
        taxonomic_field: Optional[Any] = None,
    ):
        self.registry = registry
        self.store = store
        self.llm_client = llm_client or MemoryLLMClient()
        self.embedder = embedder or QwenSemanticEmbedder()
        self.tensor_engine = tensor_engine
        self.taxonomic_field = taxonomic_field
        self.candidate_matcher = MemoryCandidateMatcher(registry=self.registry, embedder=self.embedder)

    # ------------------------------------------------------------------------
    # 1. Ingestion Pipeline: Pre-match -> LLM Structuring -> Commit -> Tensors
    # ------------------------------------------------------------------------

    def ingest_turn(
        self,
        raw_text: str,
        speaker: str = "user",
        epoch: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for process_incoming_data."""
        return self.process_incoming_data(raw_data=raw_text, speaker=speaker, session_epoch=epoch)

    def process_incoming_data(
        self,
        raw_data: Union[str, Dict[str, Any], List[Any]],
        speaker: str = "user",
        session_epoch: Optional[float] = None,
        data_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline:
        1. Pre-query candidate memory using sequence ends, regex filler stripping, and dense similarity.
        2. Pass incoming data + candidate memory to LLM for autonomous structuring decisions.
           - Data entries can be a single line, whole file, conversation turn, or whole episode data.
           - The LLM autonomously decides WHAT to store and WHERE to store it (entity slot vs episode container).
        3. Commit entities, slots, and episode to registry and SQLite store.
        4. Run internal downstream tensor and taxonomic updates.
        """
        t_now = time.time() if session_epoch is None else session_epoch

        # Normalize incoming data to string representation for LLM prompt
        if isinstance(raw_data, str):
            input_text = raw_data
        elif isinstance(raw_data, dict):
            if "file_path" in raw_data and "content" in raw_data:
                input_text = f"FILE: {raw_data['file_path']}\nCONTENT:\n{raw_data['content']}"
            else:
                input_text = json.dumps(raw_data, indent=2)
        elif isinstance(raw_data, list):
            # Sequence of conversation turns or items
            turn_lines = []
            for item in raw_data:
                if isinstance(item, dict):
                    role = item.get("role", "speaker").upper()
                    c = item.get("content", "")
                    turn_lines.append(f"[{role}]: {c}")
                else:
                    turn_lines.append(str(item))
            input_text = "\n\n".join(turn_lines)
        else:
            input_text = str(raw_data)

        # Step 1: Pre-query candidate matching
        candidate_memory = self.candidate_matcher.find_candidates(input_text[:2000], top_k=5)

        # Step 2: Formulate prompt for LLM structuring & decision
        system_prompt = (
            "You are the executive memory architect for an advanced cognitive memory system.\n"
            "Analyze the incoming data against the stored memory candidates.\n\n"
            "CORE ARCHITECTURAL PRINCIPLE:\n"
            "- Data entries can be a SINGLE LINE, a WHOLE FILE (e.g. SQL schema, source code, config), "
            "a CONVERSATION TURN, or WHOLE EPISODE DATA based on the need.\n"
            "- YOU decide WHAT to store (filtering out conversational fluff, throat-clearing, and noise) "
            "and WHERE to store it.\n\n"
            "STORAGE TARGETS:\n"
            "1. ENTITY-LEVEL KNOWLEDGE SLOTS (Discrete Attributes, Whole Files, Code, or Turn Assertions):\n"
            "   - If data belongs to a specific entity, store it under that entity.\n"
            "   - Use natural, self-descriptive aspect names (e.g. 'schema_sql', 'docker_compose', "
            "'guidelines', 'architecture_overview', 'configuration', 'version', 'active_role', 'code_implementation').\n"
            "   - The value can be a single line, a full file, a code snippet, a configuration object, or turn notes. "
            "Do NOT truncate vital technical details into 3-word fragments!\n"
            "   - Decide whether to UPDATE an existing stored candidate ('decision': 'UPDATE', 'target_entity_id') "
            "or CREATE a novel canonical entity ('decision': 'CREATE', 'canonical_name', 'entity_type', 'aliases'). "
            "Never create entities for conversational phrases (e.g. 've got', 'I think', 'there is') or isolated numbers.\n\n"
            "2. MUTATION POLICIES (TIME_SLICED vs. OVERWRITE):\n"
            "   - Use update_mode='TIME_SLICED' if the state is an evolution, migration, or replacement over time "
            "where history should be preserved (e.g. 'switched from MongoDB to PostgreSQL', 'upgraded React 17 to 18'). "
            "Include the rationale in 'reason'.\n"
            "   - Use update_mode='OVERWRITE' if correcting an error, setting an in-place configuration, or updating without preserving past history.\n\n"
            "3. EPISODE CONTAINER:\n"
            "   - Provide an informative episode title reflecting the actual milestone or discussion "
            "(e.g. 'Database Architecture & Migration to PostgreSQL', 'Session: Authentication Setup').\n"
            "   - Provide a comprehensive summary of the milestone.\n"
            "   - If the input contains whole episode data, multi-turn transcript, or session-level artifacts, "
            "you can store it in 'episode_data'.\n\n"
            "4. MULTI-ENTITY CONNECTIONS:\n"
            "   - Identify relationships between co-occurring or interacting entities.\n\n"
            "OUTPUT RULES:\n"
            "- Output concise, focused JSON. Keep descriptions and summaries to 1-2 short sentences.\n"
            "- Extract the 2 to 4 primary technical entities and their key attribute slots.\n\n"
            "Output strict JSON conforming to this schema:\n"
            "{\n"
            '  "episode": {\n'
            '    "name": str,\n'
            '    "summary": str,\n'
            '    "episode_data": any|null,\n'
            '    "start_epoch": float,\n'
            '    "end_epoch": float|null\n'
            "  },\n"
            '  "entities": [\n'
            '    {"decision": "UPDATE"|"CREATE", "target_entity_id": str|null, "canonical_name": str, "entity_type": str, "aliases": [str]}\n'
            "  ],\n"
            '  "slots": [\n'
            '    {"entity_name_or_id": str, "aspect": str, "value": any, "update_mode": "TIME_SLICED"|"OVERWRITE", "reason": str|null}\n'
            "  ],\n"
            '  "connections": [\n'
            '    {"source_entity": str, "target_entity": str, "relation": str}\n'
            "  ]\n"
            "}"
        )

        user_content = json.dumps({
            "current_session_epoch": t_now,
            "current_session_time_iso": epoch_to_iso(t_now),
            "incoming_data_type": data_type_hint or ("text" if isinstance(raw_data, str) else type(raw_data).__name__),
            "incoming_data": input_text,
            "speaker": speaker,
            "matched_candidate_memory": candidate_memory,
        }, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Incoming Data and Memory Context:\n{user_content}"},
        ]

        structured_data: Dict[str, Any] = {}
        max_llm_retries = 5
        last_error = None

        for attempt in range(max_llm_retries):
            try:
                response_str = self.llm_client.chat(messages, temperature=0.1, max_tokens=1800, json_output=True)
                parsed = MemoryLLMClient.clean_json_response(response_str)
                if isinstance(parsed, dict) and any(k in parsed for k in ("entities", "slots", "data_entries", "episode")):
                    structured_data = parsed
                    break
                else:
                    raise LLMClientError(f"LLM returned invalid schema without entities/slots/episode: {parsed}")
            except Exception as e:
                last_error = e
                wait_sec = min(15.0, 2.0 * (2 ** attempt)) + (attempt * 0.5)
                logger.warning(
                    f"LLM extraction attempt {attempt + 1}/{max_llm_retries} failed: {e}. "
                    f"Waiting {wait_sec:.1f}s before retry..."
                )
                time.sleep(wait_sec)

        if not structured_data:
            raise LLMClientError(
                f"LLM memory extraction failed after {max_llm_retries} attempts: {last_error}. "
                "Deterministic fallback ingestion is disabled."
            )

        # Step 3: Commit structured output to EntityRegistry & Database
        committed_report = self._commit_structured_data(structured_data, t_now, raw_data=raw_data)

        # Step 4: Run downstream internal operations (Taxonomic projection & Tensors)
        self._run_internal_operations(input_text, committed_report)

        return committed_report

    def _deterministic_fallback(
        self,
        raw_text: str,
        epoch: float,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Rule-based deterministic structuring fallback when LLM backend is offline.
        Uses FastNLPExtractor to parse entities, mutations, and slotted attributes without arbitrary truncation.
        """
        from .nlp_extractor import FastNLPExtractor

        extractor = FastNLPExtractor()
        extracted_names = extractor.extract_entities(raw_text)
        mutations = extractor.detect_mutations(raw_text)
        slotted_attrs = extractor.extract_slotted_attributes(raw_text)

        if not extracted_names:
            tokens = self.candidate_matcher.word_filter.extract_informative_sequences(raw_text)
            valid_tokens = [t for t in tokens if FastNLPExtractor.is_valid_entity(t)]
            extracted_names = [valid_tokens[0].capitalize()] if valid_tokens else ["GeneralContext"]

        entities_out = []
        seen_names = set()

        for name in extracted_names:
            clean_name = name.strip()
            if not FastNLPExtractor.is_valid_entity(clean_name) or clean_name.lower() in seen_names:
                continue
            seen_names.add(clean_name.lower())

            existing = self.registry.get_entity_by_name(clean_name)
            if existing:
                entities_out.append({
                    "decision": "UPDATE",
                    "target_entity_id": existing.entity_id,
                    "canonical_name": existing.canonical_name,
                    "entity_type": existing.entity_type,
                    "aliases": [clean_name],
                })
            else:
                entities_out.append({
                    "decision": "CREATE",
                    "target_entity_id": None,
                    "canonical_name": clean_name,
                    "entity_type": "concept",
                    "aliases": [clean_name],
                })

        # Process slotted attributes and mutations
        slots_out = []
        is_migration = bool(mutations) or bool(re.search(r"\b(switch|migrate|change|replace|move)\b", raw_text, re.I))
        mode = "TIME_SLICED" if is_migration else "OVERWRITE"

        if entities_out:
            ref_name = entities_out[0]["canonical_name"]
            slots_out.append({
                "entity_name_or_id": ref_name,
                "aspect": "content",
                "value": raw_text.strip(),
                "update_mode": mode,
                "reason": mutations[0]["reason"] if mutations else ("State migration" if is_migration else "Standard ingestion"),
            })

        for attr in slotted_attrs:
            target_ent = attr.get("entity", entities_out[0]["canonical_name"] if entities_out else "GeneralContext")
            slots_out.append({
                "entity_name_or_id": target_ent,
                "aspect": attr.get("aspect", "specification"),
                "value": attr.get("value"),
                "update_mode": mode,
                "reason": "Deterministic slot extraction",
            })

        # Inter-entity connections
        connections_out = []
        for i in range(len(entities_out)):
            for j in range(i + 1, len(entities_out)):
                connections_out.append({
                    "source_entity": entities_out[i]["canonical_name"],
                    "target_entity": entities_out[j]["canonical_name"],
                    "relation": "co_occurs_with",
                })

        return {
            "entities": entities_out,
            "slots": slots_out,
            "episode": {
                "name": f"Event @ {epoch_to_iso(epoch)}",
                "summary": raw_text[:200],
                "episode_data": raw_text if len(raw_text) > 200 else None,
                "start_epoch": epoch,
                "end_epoch": epoch,
            },
            "connections": connections_out,
        }

    def _commit_structured_data(self, data: Dict[str, Any], t_now: float, raw_data: Any = None) -> Dict[str, Any]:
        """
        Executes database and in-memory commits based on LLM structuring decisions.
        Handles arbitrary-length data entries (single line, full files, conversation turns, or episode data).
        """
        entity_name_to_id: Dict[str, str] = {}
        created_or_updated_entities: List[EntityNode] = []

        # 1. Process Entity Decisions
        for ent_info in data.get("entities", []):
            decision = str(ent_info.get("decision", "CREATE")).upper()
            target_id = ent_info.get("target_entity_id")
            canonical_name = ent_info.get("canonical_name", "Unknown").strip()
            entity_type = ent_info.get("entity_type", "concept")
            aliases = ent_info.get("aliases", [])

            # Validate entity name to prevent conversational phrases leaking
            from .nlp_extractor import FastNLPExtractor
            if not FastNLPExtractor.is_valid_entity(canonical_name):
                continue

            if decision == "UPDATE" and target_id and self.registry.get_entity(target_id):
                entity = self.registry.get_entity(target_id)
                assert entity is not None
                for alias in aliases:
                    if FastNLPExtractor.is_valid_entity(alias):
                        entity.add_alias(alias)
                entity_name_to_id[canonical_name.lower()] = entity.entity_id
                entity_name_to_id[entity.entity_id] = entity.entity_id
                created_or_updated_entities.append(entity)
            else:
                # CREATE novel entity
                v = self.embedder.embed_text(canonical_name)
                entity, _ = self.registry.resolve_or_create(canonical_name, embedding=v, entity_type=entity_type)
                if entity is not None:
                    for alias in aliases:
                        if FastNLPExtractor.is_valid_entity(alias):
                            entity.add_alias(alias)
                    entity_name_to_id[canonical_name.lower()] = entity.entity_id
                    entity_name_to_id[entity.entity_id] = entity.entity_id
                    created_or_updated_entities.append(entity)

        # 2. Process Episode Container (including episode-level data)
        ep_data = data.get("episode", {})
        ep_name = ep_data.get("name") or f"Episode @ {epoch_to_iso(t_now)}"
        ep_id = f"ep_{int(t_now)}_{abs(hash(ep_name)) % 10000}"
        start_epoch = float(ep_data.get("start_epoch") or t_now)
        end_epoch = float(ep_data["end_epoch"]) if ep_data.get("end_epoch") is not None else None

        episode_metadata = dict(ep_data.get("metadata", {}))
        if ep_data.get("episode_data") is not None:
            episode_metadata["episode_data"] = ep_data["episode_data"]
        elif raw_data is not None:
            episode_metadata["episode_data"] = raw_data

        episode = TimeframeEpisode(
            episode_id=ep_id,
            name=ep_name,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            summary=ep_data.get("summary"),
            metadata=episode_metadata,
        )

        for ent in created_or_updated_entities:
            episode.add_entity(ent.entity_id)
            ent.link_episode(episode.episode_id)

        # 3. Synchronize with Tensor Engine & SQLite Store
        if self.tensor_engine is not None:
            for ent in created_or_updated_entities:
                v = self.registry.entity_vectors.get(ent.entity_id)
                self.tensor_engine.register_entity(ent, vector=v)
            self.tensor_engine.add_episode(episode)

        if self.store:
            for ent in created_or_updated_entities:
                v = self.registry.entity_vectors.get(ent.entity_id)
                self.store.upsert_entity(ent, vector=v)
            self.store.upsert_episode(episode)

        # 4. Process Slotted Knowledge Mutations (support both 'slots' and 'data_entries')
        slots_list = data.get("slots") or data.get("data_entries", [])
        committed_slots: List[Dict[str, Any]] = []

        for slot in slots_list:
            ref = str(slot.get("entity_name_or_id") or slot.get("entity") or "").strip().lower()
            entity_id = entity_name_to_id.get(ref)

            # Fallback lookup
            if not entity_id:
                for ent in created_or_updated_entities:
                    if ent and (ent.canonical_name.lower() == ref or ent.entity_id.lower() == ref):
                        entity_id = ent.entity_id
                        break
            if not entity_id and created_or_updated_entities and created_or_updated_entities[0]:
                entity_id = created_or_updated_entities[0].entity_id

            if not entity_id:
                continue

            entity = self.registry.get_entity(entity_id)
            if not entity:
                continue

            aspect = str(slot.get("aspect") or slot.get("key") or "specification").strip()
            val = slot.get("value") if slot.get("value") is not None else slot.get("content")
            raw_mode = str(slot.get("update_mode", "TIME_SLICED")).upper()
            mode = UpdateMode.TIME_SLICED if raw_mode == "TIME_SLICED" else UpdateMode.OVERWRITE
            reason = slot.get("reason")
            slot_id = slot.get("slot_id") or slot.get("name")

            # Project continuous taxonomic state vector for this knowledge assertion
            tau_coords = None
            if self.taxonomic_field is not None:
                try:
                    assertion_text = f"{entity.canonical_name} {aspect}: {str(val)[:500]}"
                    v_assertion = self.embedder.embed_text(assertion_text)
                    tau_state = self.taxonomic_field.project_assertion(v_assertion)
                    tau_coords = tau_state.coords
                    self.taxonomic_field.update_poles_online(v_assertion, tau_state)
                except Exception as ex:
                    logger.warning(f"Taxonomic projection for slot '{aspect}' failed: {ex}")

            kn = entity.set_slot(
                aspect=aspect,
                value=val,
                mode=mode,
                occurred_at=start_epoch,
                episode_id=episode.episode_id,
                tau=tau_coords,
                reason=reason,
                slot_id=slot_id,
            )
            episode.add_assertion(kn.node_id)
            committed_slots.append(kn.to_dict())

            if self.store:
                self.store.upsert_knowledge_node(kn)

        # Re-persist episode so that all bound assertion_ids are written to episode_assertions junction
        if self.store:
            self.store.upsert_episode(episode)

        # 5. Process Multi-Entity Connections (and register semantic relations + synaptic edges)
        connections_list = data.get("connections", [])
        for conn in connections_list:
            src = conn.get("source_entity") or conn.get("source")
            tgt = conn.get("target_entity") or conn.get("target")
            rel = str(conn.get("relation") or "connected_to").strip()
            if src and tgt:
                src_id = entity_name_to_id.get(str(src).lower())
                tgt_id = entity_name_to_id.get(str(tgt).lower())
                if src_id and tgt_id and src_id != tgt_id:
                    # Persist to SQLite relational table
                    if self.store:
                        self.store.upsert_relation(
                            source_id=src_id,
                            target_id=tgt_id,
                            relation=rel,
                            weight=1.0,
                            episode_id=episode.episode_id,
                            created_at=start_epoch,
                        )
                    # Persist to TensorEngine
                    if self.tensor_engine:
                        self.tensor_engine.record_relation(
                            source_id=src_id,
                            target_id=tgt_id,
                            relation=rel,
                            weight=1.0,
                            episode_id=episode.episode_id,
                            timestamp=start_epoch,
                        )

        return {
            "episode": episode.to_dict(),
            "entities": [e.to_dict() for e in created_or_updated_entities],
            "slots": committed_slots,
            "connections": connections_list,
        }

    def _run_internal_operations(self, raw_text: str, report: Dict[str, Any]) -> None:
        """
        Executes internal downstream operations:
        - Continuous taxonomic axis projections.
        - Signed Hebbian synaptic updates and incidence matrix updates in tensor engine.
        """
        # 1. Taxonomic Projection
        tau = None
        if self.taxonomic_field is not None:
            try:
                v_text = self.embedder.embed_text(raw_text)
                tau = self.taxonomic_field.project_assertion(v_text)
                # Apply online pole drift
                self.taxonomic_field.update_poles_online(v_text, tau)
            except Exception as e:
                logger.warning(f"Taxonomic projection failed: {e}")

        # 2. Tensor Engine Plasticity & Incidence Updates
        if self.tensor_engine is not None:
            try:
                entities = [e["entity_id"] for e in report.get("entities", [])]
                eval_polarity = float(tau.evaluative) if tau is not None and hasattr(tau, "evaluative") else 0.0
                ep_id = report.get("episode", {}).get("episode_id")

                if hasattr(self.tensor_engine, "record_interaction"):
                    self.tensor_engine.record_interaction(
                        entities=entities,
                        polarity=eval_polarity,
                        episode_id=ep_id,
                    )
            except Exception as e:
                logger.warning(f"Tensor engine update failed: {e}")

    # ------------------------------------------------------------------------
    # 2. Query Pipeline: Dual-Factor Pre-Match -> Stored Graph -> LLM Answer
    # ------------------------------------------------------------------------

    def query_memory(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answers questions grounded in stored memory:
        1. Pre-queries candidate entities using regex + vector similarity.
        2. Detects historical query intention ('before', 'previously', 'used to', etc.).
        3. Retrieves active values and historical slices.
        4. Retrieves top matching episodes and conversation turns/chunks via tensor_engine.
        5. Combines entity knowledge, episodic summaries, and exact turns into verified context.
        6. Synthesizes concise, verified 1-2 sentence answer via LLM.
        """
        # Step 1: Pre-query candidates
        candidates = self.candidate_matcher.find_candidates(question, top_k=top_k)

        # Step 2: Detect historical & contradiction query intention
        is_historical = bool(re.search(r"\b(before|previously|past|earlier|used to|history|switched from|contradict|conflict)\b", question, re.I))

        # Step 3: Format entity ground truth context
        entity_context_lines: List[str] = []
        for cand in candidates:
            ent_name = cand["canonical_name"]
            slots = cand["active_slots"]
            slot_meta = cand.get("slot_metadata", {})
            if slots:
                slot_strs = [f"{aspect}={val}" for aspect, val in slots.items()]
                entity_context_lines.append(f"Entity: {ent_name} (Active: {', '.join(slot_strs)})")

            # Automatically surface both active and historical TimeSlice records whenever any slot has an entry in history, regardless of query wording
            if cand.get("recent_history"):
                for aspect, slices in cand["recent_history"].items():
                    val = slots.get(aspect)
                    active_t1 = slot_meta.get(aspect, {}).get("active_since_iso")
                    for s in slices:
                        prev_val = s["value"]
                        t0 = s.get("valid_from_iso") or (epoch_to_iso(s["valid_from"])[:10] if isinstance(s.get("valid_from"), (int, float)) else str(s.get("valid_from", "past")))
                        t1_slice = s.get("valid_to_iso") or (epoch_to_iso(s["valid_to"])[:10] if isinstance(s.get("valid_to"), (int, float)) else str(s.get("valid_to", "present")))
                        t1 = active_t1 if active_t1 and active_t1 != "present" else t1_slice
                        entity_context_lines.append(
                            f"[Contradiction Alert]: Current value is '{val}' (active since {t1}), but previous value was '{prev_val}' (valid {t0} to {t1})."
                        )
                        entity_context_lines.append(
                            f"  Historical {ent_name}.{aspect}: was '{prev_val}' (valid {t0} to {t1_slice})"
                        )

        # Step 3b: Deep Multi-Aspect ('what') and Timeframe Slot Search across Registry
        if hasattr(self.registry, "query_entities_multi_aspect"):
            try:
                q_vec = self.embedder.embed_text(question) if self.embedder else None
                aspect_matches = self.registry.query_entities_multi_aspect(
                    query=question,
                    query_vector=q_vec,
                    top_k=max(6, top_k),
                    embedder=self.embedder,
                )
                for m in aspect_matches:
                    tf_info = m.get("timeframe_label") or f"[Active since {m.get('valid_from_iso', '')[:10]}]"
                    line = f"Knowledge Slot: {m['canonical_name']} -> aspect='{m['aspect']}': {m['value']} {tf_info}"
                    if m.get("history_slices"):
                        hist_strs = [f"was '{h['value']}' {h.get('timeframe_label', '')}" for h in m["history_slices"]]
                        line += f" (History: {'; '.join(hist_strs)})"
                    if line not in entity_context_lines:
                        entity_context_lines.append(line)
                    # When a slot contains history, inject both versions into the LLM prompt
                    if m.get("history_slices"):
                        cur_val = m.get("value")
                        cur_t1 = m.get("valid_from_iso", "")[:10]
                        for h in m["history_slices"]:
                            prev_h_val = h.get("value")
                            h_t0 = h.get("valid_from_iso", "")[:10] or "past"
                            h_t1 = h.get("valid_to_iso", "")[:10] or cur_t1
                            t1_str = cur_t1 or h_t1
                            alert_line = f"[Contradiction Alert]: Current value is '{cur_val}' (active since {t1_str}), but previous value was '{prev_h_val}' (valid {h_t0} to {t1_str})."
                            if alert_line not in entity_context_lines:
                                entity_context_lines.append(alert_line)
            except Exception as ex:
                logger.warning(f"Error querying multi-aspect entities: {ex}")

        # Step 4: Retrieve Episodic Insights and Granular Conversation Turns
        episodic_context_lines: List[str] = []
        matched_episodes: List[Dict[str, Any]] = []
        matched_turns: List[Dict[str, Any]] = []

        if self.tensor_engine is not None:
            try:
                # Top matching episodes
                matched_episodes = self.tensor_engine.query_episodes(question, top_k=min(10, max(5, top_k)))
                if matched_episodes:
                    # Sort episodes strictly by start_epoch ascending for chronological order
                    matched_episodes.sort(key=lambda res: float(res.get("episode", {}).get("start_epoch") or 0.0))
                    episodic_context_lines.append("Chronological Project Milestones & Sessions:")
                    for idx, res in enumerate(matched_episodes, 1):
                        ep = res.get("episode", {})
                        ep_name = ep.get("name") or f"Session {idx}"
                        ep_sum = ep.get("summary") or ""
                        start_t = float(ep.get("start_epoch") or 0.0)
                        date_str = epoch_to_iso(start_t)[:10] if start_t > 0 else ""
                        date_prefix = f" - {date_str}" if (date_str and date_str not in ep_name) else ""
                        episodic_context_lines.append(f"{idx}. [{ep_name}{date_prefix}]: {ep_sum}")

                if hasattr(self.tensor_engine, "query_conversation_turns"):
                    cand_ep_ids = [res.get("episode", {}).get("episode_id") for res in matched_episodes if "episode" in res and res.get("episode", {}).get("episode_id")]
                    cand_ent_names = [c["canonical_name"] for c in candidates if "canonical_name" in c]
                    cand_slot_names = [m.get("aspect") for m in aspect_matches if m.get("aspect")] if 'aspect_matches' in locals() else []
                    matched_turns = self.tensor_engine.query_conversation_turns(
                        question,
                        top_k=max(15, top_k),
                        candidate_episodes=cand_ep_ids,
                        grounded_entities=cand_ent_names,
                        grounded_slots=cand_slot_names,
                    )
                    if matched_turns:
                        # Sort turns chronologically by timestamp or turn index
                        def _turn_key(t):
                            ts = t.get("timestamp")
                            if ts is not None:
                                try:
                                    return float(ts)
                                except (ValueError, TypeError):
                                    pass
                            tid = t.get("turn_id")
                            if isinstance(tid, (int, float)):
                                return float(tid)
                            elif isinstance(tid, str):
                                nums = re.findall(r"\d+", tid)
                                if nums:
                                    try:
                                        return float(nums[-1])
                                    except Exception:
                                        pass
                            return 0.0

                        matched_turns.sort(key=_turn_key)
                        episodic_context_lines.append("Exact Dialogue Turns & Evidence (Chronological):")
                        for turn in matched_turns:
                            t_text = turn.get("text", "").strip()
                            t_ep = turn.get("episode_name")
                            prefix = f"({t_ep}) " if t_ep else ""
                            # Truncate overly long turn text so the prompt context remains crisp
                            if len(t_text) > 600:
                                t_text = t_text[:600].strip() + "..."
                            if t_text:
                                episodic_context_lines.append(f"- {prefix}\"{t_text}\"")
            except Exception as ex:
                logger.warning(f"Failed to query episodes or turns in query_memory: {ex}")

        # Step 4b: General-Purpose Dynamic Timeline Memory Calculations
        from .node_models import extract_dates_from_text
        timeline_calculations: List[str] = []
        all_dates_found: List[Dict[str, Any]] = []
        seen_epochs = set()

        # Dynamic date extraction from question
        for d in extract_dates_from_text(question):
            if d["epoch"] not in seen_epochs:
                seen_epochs.add(d["epoch"])
                all_dates_found.append({"raw": d["raw"], "epoch": d["epoch"], "source": "query"})

        # Dynamic date extraction from entity facts and history
        for line in entity_context_lines:
            for d in extract_dates_from_text(line):
                if d["epoch"] not in seen_epochs:
                    seen_epochs.add(d["epoch"])
                    all_dates_found.append({"raw": d["raw"], "epoch": d["epoch"], "source": "entity"})

        # Dynamic date extraction from retrieved episodes
        for res in matched_episodes:
            ep = res.get("episode", {})
            start_t = float(ep.get("start_epoch") or 0.0)
            if start_t > 0 and start_t not in seen_epochs:
                seen_epochs.add(start_t)
                all_dates_found.append({"raw": epoch_to_iso(start_t)[:10], "epoch": start_t, "source": ep.get("name", "Episode")})

        # Calculate chronological intervals between adjacent dates
        if len(all_dates_found) >= 2:
            all_dates_found.sort(key=lambda x: x["epoch"])
            for i in range(len(all_dates_found) - 1):
                d1 = all_dates_found[i]
                d2 = all_dates_found[i + 1]
                days_diff = round(abs(d2["epoch"] - d1["epoch"]) / 86400.0)
                if 1 <= days_diff <= 3650:
                    calc_line = f"[Timeline Calculation]: from {d1['raw']} till {d2['raw']} is {days_diff} days."
                    if calc_line not in timeline_calculations:
                        timeline_calculations.append(calc_line)

            # If question explicitly references 2 or more dates, calculate the direct span
            q_dates = [d for d in all_dates_found if d["source"] == "query"]
            if len(q_dates) >= 2:
                q_d1, q_d2 = q_dates[0], q_dates[-1]
                q_days = round(abs(q_d2["epoch"] - q_d1["epoch"]) / 86400.0)
                direct_calc = f"[Direct Query Span Calculation]: from {q_d1['raw']} till {q_d2['raw']} is {q_days} days."
                if direct_calc not in timeline_calculations:
                    timeline_calculations.insert(0, direct_calc)

        # Combine Entity Facts, Timeline Calculations, Episodic Insights, and Exact Dialogue Turns
        all_blocks = []
        if entity_context_lines:
            all_blocks.append("Entity Knowledge Facts:\n" + "\n".join(entity_context_lines))
        if timeline_calculations:
            all_blocks.append("Verified Timeline Calculations:\n" + "\n".join(timeline_calculations))
        if episodic_context_lines:
            all_blocks.append("\n".join(episodic_context_lines))

        ground_context = "\n\n".join(all_blocks) if all_blocks else "No direct matching memory found."

        # Step 5: Prompt LLM for answer synthesis
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise, grounded memory assistant. Answer the user question strictly "
                    "using the provided verified memory context (including entity facts, timeline calculations, "
                    "episodic insights, and conversation turns).\n"
                    "- For ORDERING or TIMELINE questions: List all session milestones in strict chronological order with their topic/name (e.g. 1. Session 1 ..., 2. Session 2 ..., etc.).\n"
                    "- For TEMPORAL or DURATION questions: State exact dates and elapsed days using the verified timeline calculations (e.g., from Date A till Date B is N days).\n"
                    "- For CONTRADICTION questions: If contradictory or conflicting statements are present (or if there is a [Contradiction Alert]), you MUST explicitly state: 'There is contradictory information', specify both conflicting statements with their values/dates, and ask 'Which statement is correct?'.\n"
                    "- If asked to calculate or combine metrics across multiple systems/sessions, compute and state the exact total sum clearly.\n"
                    "- If historical state is requested, refer to the historical slices.\n"
                    "- If the provided context does not contain the requested information, explicitly state: 'Based on the provided chat, there is no information related to [topic].'\n"
                    "Keep answers concise, direct, and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Verified Memory Context:\n{ground_context}\n\nQuestion: {question}",
            },
        ]

        if self.llm_client.is_available():
            try:
                answer = self.llm_client.chat(messages, temperature=0.0, max_tokens=650, json_output=False)
                ans_str = answer.strip() if answer else ""
                if not ans_str:
                    ans_str = "Based on the provided chat, there is no information related to this question."
                return {
                    "answer": ans_str,
                    "candidates": candidates,
                    "episodes": matched_episodes,
                    "turns": matched_turns,
                    "ground_context": ground_context,
                }
            except Exception as e:
                logger.warning(f"LLM query synthesis failed: {e}")

        # Fallback direct formatting
        return {
            "answer": ground_context if (entity_context_lines or episodic_context_lines) else "Based on the provided chat, there is no information related to this question.",
            "candidates": candidates,
            "episodes": matched_episodes,
            "turns": matched_turns,
            "ground_context": ground_context,
        }
