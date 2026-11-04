"""
Turn Extractor & Conversational Turn Ingestion Engine (turn_extractor.py)
========================================================================

Module for conversational turn extraction, size minimization, and ingestion
into the 4D Spatio-Temporal EEKG memory architecture.

Core Capabilities:
1. Turn Ingestion Configuration (TurnIngestionConfig):
   - Supports variable turn length (turns_per_batch) deciding the number of turns
     processed per extraction/ingestion call.
   - Configurable overlap, minimization mode, entity/slot extraction, and salience gating.
2. Turn Normalization:
   - Polymorphic handling of dicts (role/content, speaker/text), tuples, raw strings,
     and structured TurnRecord objects.
3. LLM-Driven Turn Extraction & Minimization (LLMTurnExtractor):
   - Condenses dialogue turns to eliminate phatic chit-chat, conversational filler, and verbosity
     while strictly preserving technical assertions, numbers, dates, decisions, entities,
     and speaker intent (similar to advanced agent memory architectures like Mem0, Zep, A-Mem).
   - Extracts atomic factual propositions, domain entities, slotted knowledge, and episodic summaries.
   - Deterministic linguistic fallback using FastNLPExtractor and dependency grammar (Zero hardcoding).
4. Batch Turn Ingestion Pipeline:
   - Chunked batch dispatch respecting configured variable turn length.
   - Atomic multi-level persistence into SQLite (dialogue_turns, entities, knowledge_nodes, episodes)
     and in-memory tensor engine indexing (register_turn, turn_vectors, BM25 token index).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .embedder import QwenSemanticEmbedder, l2_normalize
from .llm_client import LLMClientError, MemoryLLMClient
from .nlp_extractor import FastNLPExtractor
from .node_models import (
    EntityNode,
    EntityRegistry,
    SQLiteGraphStore,
    TimeframeEpisode,
    UpdateMode,
    epoch_to_iso,
    parse_date_string,
)
from .tensor_engine import SpatioTemporalTensor

logger = logging.getLogger(__name__)


@dataclass
class TurnIngestionConfig:
    """
    Configuration for conversational turn extraction and ingestion.
    Allows dynamic and variable turn lengths per ingestion call.
    """
    turns_per_batch: int = 5
    """Number of dialogue turns processed per extraction/ingestion call (variable turn length)."""

    overlap: int = 0
    """Number of context turns to carry over from the previous batch for conversational continuity."""

    minimize_turns: bool = True
    """Whether to use LLM to distill and minimize turn sizes, stripping conversational noise."""

    compression_mode: str = "dense"
    """Compression policy: 'dense' (distill to essential propositions), 'concise' (trim filler), or 'verbatim'."""

    extract_entities_and_slots: bool = True
    """Whether to extract entities, slotted knowledge, and state mutations during turn processing."""

    extract_episode: bool = True
    """Whether to synthesize/update an episode container for the turn batch."""

    store_raw_turns: bool = True
    """Whether to store the original raw turn text alongside the minimized turn text."""

    min_salience: float = 0.0
    """Optional minimum salience score (0.0 to 1.0) required to ingest a turn."""

    max_batch_tokens: Optional[int] = None
    """Optional ceiling on total estimated tokens per batch; splits batches earlier if exceeded."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns_per_batch": self.turns_per_batch,
            "overlap": self.overlap,
            "minimize_turns": self.minimize_turns,
            "compression_mode": self.compression_mode,
            "extract_entities_and_slots": self.extract_entities_and_slots,
            "extract_episode": self.extract_episode,
            "store_raw_turns": self.store_raw_turns,
            "min_salience": self.min_salience,
            "max_batch_tokens": self.max_batch_tokens,
        }


@dataclass
class TurnRecord:
    """
    Normalized representation of a single conversational turn.
    """
    turn_id: str
    turn_number: int
    speaker: str
    text: str
    epoch: float
    minimized_text: Optional[str] = None
    episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "speaker": self.speaker,
            "text": self.text,
            "minimized_text": self.minimized_text or self.text,
            "epoch": self.epoch,
            "episode_id": self.episode_id,
            "metadata": self.metadata,
        }


def normalize_turn_input(
    raw_turn: Any,
    index: int = 1,
    default_epoch: Optional[float] = None,
    default_episode_id: Optional[str] = None,
) -> TurnRecord:
    """
    Polymorphically normalizes diverse raw turn inputs into a standardized TurnRecord.
    Supports:
    - TurnRecord instances (returned as-is or augmented)
    - Dicts: {speaker/role, text/content/utterance, epoch/timestamp, turn_id, turn_number}
    - Tuples: (speaker, text) or (speaker, text, epoch)
    - Strings: "Speaker: Text" or raw text string
    """
    t_now = default_epoch if default_epoch is not None else time.time()

    if isinstance(raw_turn, TurnRecord):
        if not raw_turn.episode_id and default_episode_id:
            raw_turn.episode_id = default_episode_id
        return raw_turn

    if isinstance(raw_turn, dict):
        speaker = str(
            raw_turn.get("speaker")
            or raw_turn.get("role")
            or raw_turn.get("user")
            or raw_turn.get("author")
            or "user"
        ).strip()
        text = str(
            raw_turn.get("text")
            or raw_turn.get("content")
            or raw_turn.get("utterance")
            or raw_turn.get("message")
            or ""
        ).strip()
        raw_epoch = raw_turn.get("epoch") or raw_turn.get("timestamp") or raw_turn.get("time")
        if raw_epoch is not None:
            if isinstance(raw_epoch, (int, float)):
                ep_val = float(raw_epoch)
            else:
                parsed = parse_date_string(str(raw_epoch))
                ep_val = parsed if parsed is not None else t_now
        else:
            ep_val = t_now + (index * 0.001)

        t_num = int(raw_turn.get("turn_number") or raw_turn.get("index") or index)
        ep_id = raw_turn.get("episode_id") or default_episode_id
        t_id = str(raw_turn.get("turn_id") or f"turn_{ep_id or 'gen'}_{t_num}")
        minimized = raw_turn.get("minimized_text")
        meta = {k: v for k, v in raw_turn.items() if k not in (
            "speaker", "role", "text", "content", "utterance", "message",
            "epoch", "timestamp", "time", "turn_id", "turn_number", "minimized_text", "episode_id"
        )}
        return TurnRecord(
            turn_id=t_id,
            turn_number=t_num,
            speaker=speaker,
            text=text,
            epoch=ep_val,
            minimized_text=minimized,
            episode_id=ep_id,
            metadata=meta,
        )

    if isinstance(raw_turn, (list, tuple)):
        spk = str(raw_turn[0]).strip() if len(raw_turn) > 0 else "user"
        txt = str(raw_turn[1]).strip() if len(raw_turn) > 1 else ""
        ep_val = t_now + (index * 0.001)
        if len(raw_turn) > 2 and raw_turn[2] is not None:
            try:
                ep_val = float(raw_turn[2])
            except (ValueError, TypeError):
                parsed = parse_date_string(str(raw_turn[2]))
                if parsed:
                    ep_val = parsed
        t_id = f"turn_{default_episode_id or 'gen'}_{index}"
        return TurnRecord(
            turn_id=t_id,
            turn_number=index,
            speaker=spk,
            text=txt,
            epoch=ep_val,
            episode_id=default_episode_id,
        )

    # String input
    str_val = str(raw_turn).strip()
    spk = "user"
    txt = str_val
    colon_idx = str_val.find(":")
    if 0 < colon_idx < 30 and "\n" not in str_val[:colon_idx]:
        potential_speaker = str_val[:colon_idx].strip()
        if re.match(r"^[A-Za-z0-9_\-\s]+$", potential_speaker):
            spk = potential_speaker
            txt = str_val[colon_idx + 1:].strip()

    t_id = f"turn_{default_episode_id or 'gen'}_{index}"
    return TurnRecord(
        turn_id=t_id,
        turn_number=index,
        speaker=spk,
        text=txt,
        epoch=t_now + (index * 0.001),
        episode_id=default_episode_id,
    )


@dataclass
class TurnExtractionResult:
    """Output bundle from LLMTurnExtractor for a batch of turns."""
    minimized_turns: List[TurnRecord]
    extracted_facts: List[str]
    entities: List[Dict[str, Any]]
    slots: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    episode_data: Dict[str, Any]
    raw_character_count: int
    minimized_character_count: int
    compression_ratio: float
    used_llm: bool


@dataclass
class TurnIngestionReport:
    """Execution summary returned after ingesting a sequence of turns."""
    total_turns_input: int
    total_batches: int
    turns_per_batch: int
    turns_stored: int
    raw_character_count: int
    minimized_character_count: int
    compression_ratio: float
    entities_committed: int
    slots_committed: int
    episodes_committed: int
    batch_reports: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns_input": self.total_turns_input,
            "total_batches": self.total_batches,
            "turns_per_batch": self.turns_per_batch,
            "turns_stored": self.turns_stored,
            "raw_character_count": self.raw_character_count,
            "minimized_character_count": self.minimized_character_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "entities_committed": self.entities_committed,
            "slots_committed": self.slots_committed,
            "episodes_committed": self.episodes_committed,
            "batch_reports": self.batch_reports,
        }


class LLMTurnExtractor:
    """
    LLM-powered turn extractor and size minimizer.
    Analyzes conversation windows (variable turn length) to:
    1. Minimize turn size by stripping conversational filler, greetings, and verbosity
       while strictly retaining all technical assertions, decisions, numbers, and facts.
    2. Extract atomic factual propositions.
    3. Extract domain entities, slots, and mutations into the EEKG graph schema.
    4. Provide deterministic linguistic fallbacks when LLM backend is offline.
    """

    def __init__(
        self,
        llm_client: Optional[MemoryLLMClient] = None,
        embedder: Optional[QwenSemanticEmbedder] = None,
        nlp_extractor: Optional[FastNLPExtractor] = None,
    ) -> None:
        self.llm_client = llm_client or MemoryLLMClient()
        self.embedder = embedder or QwenSemanticEmbedder()
        self.nlp = nlp_extractor or FastNLPExtractor()

    def extract_and_minimize(
        self,
        turns: List[TurnRecord],
        config: Optional[TurnIngestionConfig] = None,
        existing_context: Optional[str] = None,
    ) -> TurnExtractionResult:
        """
        Processes a batch of dialogue turns:
        - If config.minimize_turns is True and LLM is reachable, invokes LLM to distill turns.
        - Falls back to linguistic syntactic compression via FastNLPExtractor if offline.
        """
        cfg = config or TurnIngestionConfig()
        raw_char_len = sum(len(t.text) for t in turns)

        if not turns:
            return TurnExtractionResult(
                minimized_turns=[],
                extracted_facts=[],
                entities=[],
                slots=[],
                connections=[],
                episode_data={},
                raw_character_count=0,
                minimized_character_count=0,
                compression_ratio=1.0,
                used_llm=False,
            )

        # If minimization is explicitly disabled, return verbatim
        if not cfg.minimize_turns or cfg.compression_mode == "verbatim":
            for t in turns:
                t.minimized_text = t.text
            return self._build_verbatim_result(turns, raw_char_len)

        # Attempt LLM-driven turn extraction and minimization
        structured_data: Optional[Dict[str, Any]] = None
        used_llm = False

        if self.llm_client.is_available():
            try:
                structured_data = self._call_llm_minimizer(turns, cfg, existing_context)
                if structured_data and "minimized_turns" in structured_data:
                    used_llm = True
            except Exception as e:
                logger.warning(f"LLM turn extraction failed ({e}). Reverting to linguistic fallback.")

        # Fallback to deterministic linguistic extraction if LLM failed or offline
        if not structured_data:
            structured_data = self._deterministic_linguistic_minimizer(turns, cfg)
            used_llm = False

        # Align minimized text back to TurnRecord objects
        minimized_turn_records: List[TurnRecord] = []
        min_turns_data = structured_data.get("minimized_turns", [])
        min_text_map: Dict[int, str] = {}
        for item in min_turns_data:
            if isinstance(item, dict):
                t_idx = item.get("turn_number")
                t_txt = item.get("minimized_text") or item.get("text")
                if t_idx is not None and t_txt:
                    min_text_map[int(t_idx)] = str(t_txt).strip()

        for t in turns:
            min_txt = min_text_map.get(t.turn_number)
            if not min_txt:
                min_txt = self._linguistic_compress_single_turn(t.text)
            t.minimized_text = min_txt
            minimized_turn_records.append(t)

        min_char_len = sum(len(t.minimized_text or t.text) for t in minimized_turn_records)
        ratio = (min_char_len / raw_char_len) if raw_char_len > 0 else 1.0

        return TurnExtractionResult(
            minimized_turns=minimized_turn_records,
            extracted_facts=structured_data.get("extracted_facts", []),
            entities=structured_data.get("entities", []),
            slots=structured_data.get("slots", []),
            connections=structured_data.get("connections", []),
            episode_data=structured_data.get("episode", {}),
            raw_character_count=raw_char_len,
            minimized_character_count=min_char_len,
            compression_ratio=ratio,
            used_llm=used_llm,
        )

    def _call_llm_minimizer(
        self,
        turns: List[TurnRecord],
        cfg: TurnIngestionConfig,
        existing_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Formulates prompt and calls LLM to condense turns and extract structured knowledge."""
        system_prompt = (
            "You are an expert conversation memory architect in an advanced cognitive AI system.\n"
            "Your task is to analyze a window of dialogue turns, MINIMIZE their size, and EXTRACT structured memory.\n\n"
            "OBJECTIVES:\n"
            "1. TURN SIZE MINIMIZATION:\n"
            "   - Rewrite each dialogue turn into a dense, informative representation ('minimized_text').\n"
            "   - STRIP OUT: Conversational pleasantries, greetings (e.g. 'hi', 'how are you', 'thanks'),\n"
            "     throat-clearing (e.g. 'Sure, I would love to help you with that', 'Let me think about that'),\n"
            "     repetitive confirmations, and verbose filler phrasing.\n"
            "   - STRICTLY PRESERVE: All factual statements, system architectures, technical specifications,\n"
            "     file names, version numbers, ports, dates, requirements, design decisions, constraints, user preferences,\n"
            "     and speaker attribution. Never omit concrete numbers or named entities.\n\n"
            "2. ATOMIC PROPOSITIONS ('extracted_facts'):\n"
            "   - Extract 1-5 atomic factual sentences capturing the core knowledge asserted in this turn window.\n\n"
            "3. ENTITY & SLOT EXTRACTION:\n"
            "   - Identify domain entities to CREATE or UPDATE ('canonical_name', 'entity_type', 'decision', 'aliases').\n"
            "   - Extract attribute slots ('entity_name_or_id', 'aspect', 'value', 'update_mode'='TIME_SLICED'|'OVERWRITE', 'reason').\n"
            "   - Use TIME_SLICED if an attribute evolved or migrated (e.g. upgraded version, replaced database);\n"
            "     use OVERWRITE if setting an in-place configuration or state.\n\n"
            "4. EPISODIC SEGMENT ('episode'):\n"
            "   - Provide an informative title and a 1-2 sentence summary of this conversation segment.\n\n"
            "OUTPUT FORMAT:\n"
            "Strict JSON matching this schema:\n"
            "{\n"
            '  "minimized_turns": [\n'
            '    {"turn_number": int, "speaker": str, "minimized_text": str}\n'
            "  ],\n"
            '  "extracted_facts": [str],\n'
            '  "entities": [\n'
            '    {"decision": "CREATE"|"UPDATE", "canonical_name": str, "entity_type": str, "aliases": [str]}\n'
            "  ],\n"
            '  "slots": [\n'
            '    {"entity_name_or_id": str, "aspect": str, "value": any, "update_mode": "TIME_SLICED"|"OVERWRITE", "reason": str|null}\n'
            "  ],\n"
            '  "connections": [\n'
            '    {"source_entity": str, "target_entity": str, "relation": str}\n'
            "  ],\n"
            '  "episode": {\n'
            '    "name": str,\n'
            '    "summary": str\n'
            "  }\n"
            "}"
        )

        turns_payload = []
        for t in turns:
            turns_payload.append({
                "turn_number": t.turn_number,
                "speaker": t.speaker,
                "text": t.text,
                "time_iso": epoch_to_iso(t.epoch),
            })

        user_content = {
            "turns_window": turns_payload,
            "turns_count": len(turns),
            "compression_mode": cfg.compression_mode,
        }
        if existing_context:
            user_content["prior_context"] = existing_context[:1000]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, indent=2)},
        ]

        resp = self.llm_client.chat(messages, temperature=0.1, max_tokens=1500, json_output=True)
        parsed = MemoryLLMClient.clean_json_response(resp)
        if isinstance(parsed, dict) and "minimized_turns" in parsed:
            return parsed
        raise LLMClientError(f"LLM response missing 'minimized_turns': {parsed}")

    def _linguistic_compress_single_turn(self, text: str) -> str:
        """
        Linguistic compression fallback using syntactic clause analysis.
        Removes introductory pleasantries and keeps clauses with high syntactic content.
        Zero hardcoding rule compliant: relies entirely on spaCy POS tags, dependency parse, and entity detection.
        """
        if not text or len(text) < 20:
            return text

        doc = self.nlp.clean_text_for_parsing(text)
        from .nlp_extractor import get_spacy_nlp
        spacy_nlp = get_spacy_nlp()
        spacy_doc = spacy_nlp(doc)

        kept_sentences = []
        try:
            sents = list(spacy_doc.sents)
        except Exception:
            sents = [spacy_doc]

        for sent in sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Check if sentence contains entities, numbers, dates, or technical symbols
            has_ent = bool(getattr(sent, "ents", None))
            has_num = any(tok.like_num or tok.pos_ in ("NUM", "PROPN") for tok in sent)
            has_verb = any(tok.pos_ in ("VERB", "AUX") for tok in sent)
            has_substantive = any(tok.pos_ in ("NOUN", "PROPN") for tok in sent)

            # Keep sentence if it has substantive informational content
            if (has_ent or has_num) or (has_verb and has_substantive):
                kept_sentences.append(sent_text)

        if kept_sentences:
            compressed = " ".join(kept_sentences).strip()
            # If compression yielded something reasonable, return it; otherwise keep original
            if len(compressed) >= 10:
                return compressed
        return text.strip()

    def _deterministic_linguistic_minimizer(
        self,
        turns: List[TurnRecord],
        cfg: TurnIngestionConfig,
    ) -> Dict[str, Any]:
        """
        Deterministic fallback when LLM is unavailable:
        Applies FastNLPExtractor across the dialogue turns to minimize text
        and extract entities, mutations, and slots.
        """
        minimized_turns_data = []
        combined_text_lines = []

        for t in turns:
            min_txt = self._linguistic_compress_single_turn(t.text)
            minimized_turns_data.append({
                "turn_number": t.turn_number,
                "speaker": t.speaker,
                "minimized_text": min_txt,
            })
            combined_text_lines.append(f"[{t.speaker}]: {min_txt}")

        combined_text = "\n".join(combined_text_lines)

        # Extract entities and slots via FastNLPExtractor
        extracted_entities = self.nlp.extract_entities(combined_text)
        mutations = self.nlp.detect_mutations(combined_text)
        slotted_attrs = self.nlp.extract_slotted_attributes(combined_text)

        entities_out = []
        seen_ents = set()
        for e in extracted_entities:
            clean_e = e.strip()
            if clean_e.lower() in seen_ents:
                continue
            seen_ents.add(clean_e.lower())
            entities_out.append({
                "decision": "CREATE",
                "canonical_name": clean_e,
                "entity_type": "concept",
                "aliases": [clean_e],
            })

        slots_out = []
        is_migration = bool(mutations)
        mode = "TIME_SLICED" if is_migration else "OVERWRITE"

        for attr in slotted_attrs:
            slots_out.append({
                "entity_name_or_id": attr.get("entity", entities_out[0]["canonical_name"] if entities_out else "Context"),
                "aspect": attr.get("aspect", "specification"),
                "value": attr.get("value"),
                "update_mode": mode,
                "reason": "Linguistic slot extraction",
            })

        # Connections between adjacent co-occurring entities
        connections_out = []
        for i in range(len(entities_out) - 1):
            connections_out.append({
                "source_entity": entities_out[i]["canonical_name"],
                "target_entity": entities_out[i + 1]["canonical_name"],
                "relation": "co_occurs_with",
            })

        first_epoch = turns[0].epoch if turns else time.time()
        ep_summary = combined_text[:250] if len(combined_text) > 250 else combined_text

        return {
            "minimized_turns": minimized_turns_data,
            "extracted_facts": [t["minimized_text"] for t in minimized_turns_data if len(t["minimized_text"]) > 15][:5],
            "entities": entities_out,
            "slots": slots_out,
            "connections": connections_out,
            "episode": {
                "name": f"Turns {turns[0].turn_number}-{turns[-1].turn_number} @ {epoch_to_iso(first_epoch)}",
                "summary": ep_summary,
            },
        }

    def _build_verbatim_result(self, turns: List[TurnRecord], raw_char_len: int) -> TurnExtractionResult:
        """Returns turns verbatim without compression."""
        return TurnExtractionResult(
            minimized_turns=turns,
            extracted_facts=[],
            entities=[],
            slots=[],
            connections=[],
            episode_data={
                "name": f"Turns {turns[0].turn_number}-{turns[-1].turn_number}",
                "summary": "Verbatim conversational turns",
            },
            raw_character_count=raw_char_len,
            minimized_character_count=raw_char_len,
            compression_ratio=1.0,
            used_llm=False,
        )


class TurnIngestionPipeline:
    """
    Ingestion orchestrator managing chunked conversation batching with variable turn lengths.
    Commits minimized dialogue turns into the SpatioTemporalTensor and SQLiteGraphStore.
    """

    def __init__(
        self,
        extractor: LLMTurnExtractor,
        tensor_engine: SpatioTemporalTensor,
        registry: EntityRegistry,
        store: Optional[SQLiteGraphStore] = None,
        embedder: Optional[QwenSemanticEmbedder] = None,
        taxonomic_field: Any = None,
    ) -> None:
        self.extractor = extractor
        self.tensor_engine = tensor_engine
        self.registry = registry
        self.store = store
        self.embedder = embedder or QwenSemanticEmbedder()
        self.taxonomic_field = taxonomic_field

    def ingest_turn_sequence(
        self,
        raw_turns: Sequence[Any],
        config: Optional[TurnIngestionConfig] = None,
        episode_id: Optional[str] = None,
        episode_name: Optional[str] = None,
        session_epoch: Optional[float] = None,
    ) -> TurnIngestionReport:
        """
        Main entry point for conversational turn ingestion:
        1. Normalizes all turns into TurnRecord objects.
        2. Partitions into batches of size config.turns_per_batch (variable turn length).
        3. Calls LLMTurnExtractor to minimize turn size and extract structured entities/slots.
        4. Registers turns into tensor_engine (embeddings, token posting index) and SQLite.
        5. Commits entities and slots to EntityRegistry, taxonomic field, and database.
        6. Ensures contiguous index rebuild in tensor engine.
        """
        cfg = config or TurnIngestionConfig()
        base_epoch = session_epoch if session_epoch is not None else time.time()

        # Step 1: Normalize all turns
        normalized_turns: List[TurnRecord] = []
        for i, raw_t in enumerate(raw_turns, 1):
            record = normalize_turn_input(
                raw_t,
                index=i,
                default_epoch=base_epoch,
                default_episode_id=episode_id,
            )
            normalized_turns.append(record)

        total_input = len(normalized_turns)
        if total_input == 0:
            return TurnIngestionReport(
                total_turns_input=0,
                total_batches=0,
                turns_per_batch=cfg.turns_per_batch,
                turns_stored=0,
                raw_character_count=0,
                minimized_character_count=0,
                compression_ratio=1.0,
                entities_committed=0,
                slots_committed=0,
                episodes_committed=0,
            )

        # Step 2: Chunk into batches according to variable turn length (cfg.turns_per_batch)
        step_size = max(1, cfg.turns_per_batch - cfg.overlap) if cfg.overlap > 0 else max(1, cfg.turns_per_batch)
        batches: List[List[TurnRecord]] = []
        for b_start in range(0, total_input, step_size):
            b_end = min(total_input, b_start + cfg.turns_per_batch)
            batch = normalized_turns[b_start:b_end]
            if batch:
                batches.append(batch)
            if b_end >= total_input:
                break

        total_batches = len(batches)
        stored_turns_count = 0
        total_raw_chars = 0
        total_min_chars = 0
        total_entities_committed = 0
        total_slots_committed = 0
        total_episodes_committed = 0
        batch_reports: List[Dict[str, Any]] = []

        prior_context: Optional[str] = None

        # Step 3: Process each batch
        for b_idx, batch in enumerate(batches, 1):
            extraction_result = self.extractor.extract_and_minimize(
                turns=batch,
                config=cfg,
                existing_context=prior_context,
            )

            total_raw_chars += extraction_result.raw_character_count
            total_min_chars += extraction_result.minimized_character_count

            # Prepare Episode Container for this batch
            ep_meta = extraction_result.episode_data
            cur_ep_name = (
                episode_name
                or ep_meta.get("name")
                or f"Episode Batch {b_idx} (Turns {batch[0].turn_number}-{batch[-1].turn_number})"
            )
            cur_ep_id = episode_id or f"ep_batch_{int(base_epoch)}_{b_idx}"
            cur_ep_summary = ep_meta.get("summary") or f"Conversation turns {batch[0].turn_number} to {batch[-1].turn_number}"

            batch_episode = TimeframeEpisode(
                episode_id=cur_ep_id,
                name=cur_ep_name,
                start_epoch=batch[0].epoch,
                end_epoch=batch[-1].epoch,
                summary=cur_ep_summary,
            )

            if cfg.extract_episode:
                self.tensor_engine.add_episode(batch_episode)
                if self.store:
                    self.store.upsert_episode(batch_episode)
                total_episodes_committed += 1

            # Register Minimized Turns into Tensor Engine and SQLite
            turns_to_store_sqlite: List[Dict[str, Any]] = []

            for t_rec in extraction_result.minimized_turns:
                text_to_index = t_rec.minimized_text if cfg.minimize_turns and t_rec.minimized_text else t_rec.text

                # Optional salience gate
                if cfg.min_salience > 0.0:
                    sal = self.extractor.nlp.compute_turn_salience(text_to_index)
                    if sal < cfg.min_salience:
                        continue

                # Compute vector embedding for the turn
                t_vec = l2_normalize(np.asarray(self.embedder.embed_text(text_to_index), dtype=np.float32))

                # Register in Tensor Engine
                self.tensor_engine.register_turn(
                    turn_id=t_rec.turn_id,
                    episode_id=cur_ep_id,
                    speaker=t_rec.speaker,
                    text=text_to_index,
                    epoch=t_rec.epoch,
                    vector=t_vec,
                    turn_number=t_rec.turn_number,
                    episode_name=cur_ep_name,
                )

                turns_to_store_sqlite.append({
                    "turn_id": t_rec.turn_id,
                    "episode_id": cur_ep_id,
                    "turn_number": t_rec.turn_number,
                    "speaker": t_rec.speaker,
                    "text": text_to_index,
                    "epoch": t_rec.epoch,
                    "vector": t_vec,
                })
                stored_turns_count += 1

            # Persist turns batch to SQLite
            if self.store and turns_to_store_sqlite:
                self.store.upsert_turns_batch(turns_to_store_sqlite)

            # Process Extracted Entities and Slots
            if cfg.extract_entities_and_slots:
                ent_count, slot_count = self._commit_extracted_knowledge(
                    extraction_result=extraction_result,
                    episode=batch_episode,
                    start_epoch=batch[0].epoch,
                )
                total_entities_committed += ent_count
                total_slots_committed += slot_count

            # Maintain rolling context for subsequent batches
            prior_context = extraction_result.episode_data.get("summary") or extraction_result.minimized_turns[-1].minimized_text

            batch_reports.append({
                "batch_index": b_idx,
                "turns_count": len(batch),
                "raw_chars": extraction_result.raw_character_count,
                "minimized_chars": extraction_result.minimized_character_count,
                "compression_ratio": round(extraction_result.compression_ratio, 4),
                "used_llm": extraction_result.used_llm,
                "episode_id": cur_ep_id,
            })

        # Finalize turn index in tensor engine (rebuilds contiguous matrix & BM25 posting index)
        self.tensor_engine.ensure_turn_index()

        overall_ratio = (total_min_chars / total_raw_chars) if total_raw_chars > 0 else 1.0

        return TurnIngestionReport(
            total_turns_input=total_input,
            total_batches=total_batches,
            turns_per_batch=cfg.turns_per_batch,
            turns_stored=stored_turns_count,
            raw_character_count=total_raw_chars,
            minimized_character_count=total_min_chars,
            compression_ratio=overall_ratio,
            entities_committed=total_entities_committed,
            slots_committed=total_slots_committed,
            episodes_committed=total_episodes_committed,
            batch_reports=batch_reports,
        )

    def _commit_extracted_knowledge(
        self,
        extraction_result: TurnExtractionResult,
        episode: TimeframeEpisode,
        start_epoch: float,
    ) -> Tuple[int, int]:
        """Commits entities and slotted attributes to EntityRegistry, TensorEngine, and SQLite."""
        committed_entities: List[EntityNode] = []
        name_to_id: Dict[str, str] = {}

        for ent_info in extraction_result.entities:
            cname = str(ent_info.get("canonical_name") or "").strip()
            if not FastNLPExtractor.is_valid_entity(cname):
                continue

            etype = ent_info.get("entity_type") or "concept"
            v = self.embedder.embed_text(cname)
            node, _ = self.registry.resolve_or_create(cname, embedding=v, entity_type=etype)
            if node:
                for alias in ent_info.get("aliases", []):
                    if FastNLPExtractor.is_valid_entity(alias):
                        node.add_alias(alias)
                node.link_episode(episode.episode_id)
                episode.add_entity(node.entity_id)
                self.tensor_engine.register_entity(node, vector=v)
                if self.store:
                    self.store.upsert_entity(node, vector=v)
                committed_entities.append(node)
                name_to_id[cname.lower()] = node.entity_id
                name_to_id[node.entity_id] = node.entity_id

        slots_committed = 0
        for slot in extraction_result.slots:
            ref = str(slot.get("entity_name_or_id") or "").strip().lower()
            target_id = name_to_id.get(ref)
            if not target_id and committed_entities:
                target_id = committed_entities[0].entity_id

            if not target_id:
                continue

            entity = self.registry.get_entity(target_id)
            if not entity:
                continue

            aspect = str(slot.get("aspect") or "fact").strip()
            val = slot.get("value")
            if val is None:
                continue

            mode_str = str(slot.get("update_mode", "OVERWRITE")).upper()
            mode = UpdateMode.TIME_SLICED if mode_str == "TIME_SLICED" else UpdateMode.OVERWRITE
            reason = slot.get("reason")

            tau_coords = None
            if self.taxonomic_field is not None:
                try:
                    assertion_text = f"{entity.canonical_name} {aspect}: {str(val)[:300]}"
                    v_assertion = self.embedder.embed_text(assertion_text)
                    tau_state = self.taxonomic_field.project_assertion(v_assertion)
                    tau_coords = tau_state.coords
                except Exception:
                    pass

            kn = entity.set_slot(
                aspect=aspect,
                value=val,
                mode=mode,
                occurred_at=start_epoch,
                episode_id=episode.episode_id,
                tau=tau_coords,
                reason=reason,
            )
            episode.add_assertion(kn.node_id)
            if self.store:
                self.store.upsert_entity(entity)
            slots_committed += 1

        return len(committed_entities), slots_committed
