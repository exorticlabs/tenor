"""
Agent Memory Orchestrator & Top-Level API (orchestrator.py)
==========================================================

Top-level operational facade unifying all memory subsystems:
- Layer 1: Long-term slotted knowledge & entity registry (EntityNode, KnowledgeNode, SQLiteGraphStore)
- Layer 2: 4D dynamic attention tensor & episodic incidence matrix (tensor_engine)
- Layer 3: Continuous temporal recency, Ebbinghaus decay, and emergent taxonomic manifold (taxonomy)
- Layer 4: Volatile working memory DAG & active registers (working_memory)

Coordinates selective step ingestion with salience gating, dual-stream context retrieval
(query seeds + working memory attention injection), and token-budgeted prompt injection.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .embedder import QwenSemanticEmbedder
from .llm_client import MemoryLLMClient
from .llm_manager import LLMMemoryManager
from .nlp_extractor import FastNLPExtractor
from .node_models import EntityNode, EntityRegistry, SQLiteGraphStore, TimeframeEpisode, parse_date_string
from .taxonomy import TaxonomicField
from .tensor_engine import SpatioTemporalTensor
from .turn_extractor import LLMTurnExtractor, TurnIngestionConfig, TurnIngestionPipeline, TurnIngestionReport
from .working_memory import TaskStatus, WorkingMemoryEngine

logger = logging.getLogger(__name__)


class AgentMemoryOrchestrator:
    """
    Master coordinator and public API facade for the EEKG v2 memory system.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        registry: Optional[EntityRegistry] = None,
        store: Optional[SQLiteGraphStore] = None,
        tensor_engine: Optional[SpatioTemporalTensor] = None,
        embedder: Optional[QwenSemanticEmbedder] = None,
        taxonomic_field: Optional[TaxonomicField] = None,
        llm_client: Optional[MemoryLLMClient] = None,
        llm_manager: Optional[LLMMemoryManager] = None,
        working_memory: Optional[WorkingMemoryEngine] = None,
        nlp_extractor: Optional[FastNLPExtractor] = None,
        turn_config: Optional[TurnIngestionConfig] = None,
        turn_extractor: Optional[LLMTurnExtractor] = None,
        working_bias_weight: float = 0.45,
        salience_threshold: float = 0.35,
        session_start_epoch: Optional[float] = None,
    ) -> None:
        self.session_start_epoch = session_start_epoch if session_start_epoch is not None else time.time()
        self.working_bias_weight = working_bias_weight
        self.salience_threshold = salience_threshold

        # 1. Core Embedder
        self.embedder = embedder or QwenSemanticEmbedder()

        # 2. Persistence & Entity Registry
        self.store = store or (SQLiteGraphStore(db_path) if db_path else None)
        self.registry = registry or EntityRegistry()

        # 3. Continuous Taxonomic Field
        self.taxonomic_field = taxonomic_field or TaxonomicField(embedder=self.embedder)

        # 4. Spatio-Temporal Tensor Engine
        self.tensor_engine = tensor_engine or SpatioTemporalTensor(embedder=self.embedder)

        # 5. Volatile Working Memory & NLP Extractor
        self.working_memory = working_memory or WorkingMemoryEngine()
        self.nlp = nlp_extractor or FastNLPExtractor()

        # Warm NLP pipeline early to eliminate first-query cold-start stalls
        try:
            from .nlp_extractor import get_spacy_nlp
            _ = get_spacy_nlp()("warmup")
        except Exception:
            pass

        # Load persisted entities, episodes, relations, and tasks from store if available
        if self.store is not None:
            self._warm_from_store()
        self._sync_tensor_entities()

        # 6. LLM Client & Memory Manager
        self.llm_client = llm_client or MemoryLLMClient()
        self.llm_mgr = llm_manager or LLMMemoryManager(
            registry=self.registry,
            store=self.store,
            llm_client=self.llm_client,
            embedder=self.embedder,
            tensor_engine=self.tensor_engine,
            taxonomic_field=self.taxonomic_field,
        )

        # 7. Conversational Turn Extractor & Ingestion Pipeline
        self.turn_config = turn_config or TurnIngestionConfig()
        self.turn_extractor = turn_extractor or LLMTurnExtractor(
            llm_client=self.llm_client,
            embedder=self.embedder,
            nlp_extractor=self.nlp,
        )
        self.turn_pipeline = TurnIngestionPipeline(
            extractor=self.turn_extractor,
            tensor_engine=self.tensor_engine,
            registry=self.registry,
            store=self.store,
            embedder=self.embedder,
            taxonomic_field=self.taxonomic_field,
        )

    def _warm_from_store(self) -> None:
        """Loads existing entities, knowledge nodes, and episodes from SQLite into EntityRegistry and TensorEngine."""
        if self.store is None:
            return
        try:
            self.store.populate_registry(self.registry)
            # Load episodes into tensor engine
            episodes = self.store.load_all_episodes()
            for ep in episodes:
                self.tensor_engine.add_episode(ep)
            # Load granular dialogue turns with embeddings into tensor engine
            turns = self.store.load_all_turns()
            for t in turns:
                self.tensor_engine.register_turn(
                    turn_id=t["turn_id"],
                    episode_id=t.get("episode_id"),
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    epoch=t.get("epoch"),
                    vector=t.get("vector"),
                    turn_number=t.get("turn_number"),
                )
            # Precompute contiguous matrix and inverted token posting index
            self.tensor_engine.ensure_turn_index()

            # Load semantic relations into tensor engine
            relations = self.store.load_entity_relations()
            for r in relations:
                self.tensor_engine.record_relation(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation=r["relation"],
                    weight=float(r.get("weight", 1.0)),
                    episode_id=r.get("episode_id"),
                    timestamp=float(r.get("created_at") or time.time()),
                )

            # Load persistent tasks from Work DAG into working memory
            persisted_tasks = self.store.load_all_tasks()
            active_task_id = None
            for pt in persisted_tasks:
                from .working_memory import TaskStatus
                sub = self.working_memory.add_subtask(
                    task_id=pt["task_id"],
                    description=pt["description"],
                    dependencies=pt["dependencies"],
                    entities=pt["entities"],
                )
                sub.status = TaskStatus(pt["status"])
                sub.created_at = pt["created_at"]
                sub.completed_at = pt["completed_at"]
                sub.result_summary = pt["result_summary"]
                if sub.status == TaskStatus.IN_PROGRESS:
                    active_task_id = sub.task_id
                if pt.get("root_goal") and not self.working_memory.root_goal:
                    self.working_memory.set_root_goal(pt["root_goal"])

            self.working_memory.current_subtask_id = active_task_id

            logger.info(
                f"Warmed {len(self.registry.entities)} entities, {len(episodes)} episodes, "
                f"{len(relations)} relations, and {len(persisted_tasks)} tasks from SQLite store."
            )
        except Exception as e:
            logger.warning(f"Failed warming from SQLite store: {e}")

    def _sync_tensor_entities(self) -> None:
        """Synchronizes registry entities into the tensor engine index in vectorized batches."""
        missing_eids = []
        missing_names = []
        for eid, ent in self.registry.entities.items():
            if eid not in self.registry.entity_vectors:
                missing_eids.append(eid)
                missing_names.append(ent.canonical_name)

        if missing_names:
            batch_size = 256
            for b_idx in range(0, len(missing_names), batch_size):
                b_names = missing_names[b_idx : b_idx + batch_size]
                b_vecs = self.embedder.embed_text(b_names)
                if b_vecs.ndim == 1:
                    b_vecs = b_vecs.reshape(1, -1)
                for i, v in enumerate(b_vecs):
                    cur_eid = missing_eids[b_idx + i]
                    self.registry.entity_vectors[cur_eid] = v

        for eid, ent in self.registry.entities.items():
            vec = self.registry.entity_vectors.get(eid)
            self.tensor_engine.register_entity(ent, vector=vec)

    # ------------------------------------------------------------------------
    # Step Ingestion & Salience Gating
    # ------------------------------------------------------------------------

    def record_step(
        self,
        action: str,
        observation: str,
        modified_files: Optional[Sequence[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        is_milestone: bool = False,
        t_now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes an execution turn in the agent runtime:
        1. Working Memory State Mutation: records touched files and errors.
        2. Salience Assessment: evaluates turn importance via FastNLPExtractor.
        3. Gated Long-Term Ingestion: commits to EEKG and 4D tensor only if
           salience >= 0.35 or is_milestone is True, discarding transient noise.
        4. Mutation Attenuation: suppresses obsolete concepts in tensor engine.
        """
        now = t_now if t_now is not None else time.time()
        if hasattr(self.tensor_engine, "sync_relative_temporal_metadata"):
            self.tensor_engine.sync_relative_temporal_metadata(ref_epoch=t_now)

        # Step 1: Update Working State
        if modified_files:
            for f in modified_files:
                self.working_memory.record_file_touch(f)

        if error:
            self.working_memory.record_error(
                error_type=error.get("type", "Error"),
                message=error.get("message", "Unknown error"),
                file_path=error.get("file"),
            )
        else:
            if self.working_memory.active_error is not None:
                self.working_memory.clear_error()

        # Step 2: Turn Salience Assessment
        step_text = f"{action}\n{observation}".strip()
        mutations = self.nlp.detect_mutations(step_text)
        entities = self.nlp.extract_entities(step_text)
        salience = self.nlp.compute_turn_salience(
            step_text,
            entities=entities,
            has_mutation=bool(mutations),
        )

        report: Dict[str, Any] = {
            "salience": salience,
            "is_milestone": is_milestone,
            "mutations": mutations,
            "extracted_entities": entities,
        }

        # Step 3: Gated Long-Term Ingestion
        if salience >= self.salience_threshold or is_milestone:
            ingest_result = self.llm_mgr.ingest_turn(step_text, epoch=now)
            report["status"] = "ingested"
            report["ingest_result"] = ingest_result

            # Synchronize newly registered entities into tensor engine
            self._sync_tensor_entities()

            # Step 4: Contradiction & Mutation Attenuation
            for mut in mutations:
                obs = mut.get("obsolete")
                if obs:
                    # Look up obsolete entity in registry
                    cand = self.registry.get_entity_by_name(obs)
                    if cand:
                        self.tensor_engine.attenuate_entity(cand.entity_id)
                    else:
                        self.tensor_engine.attenuate_entity(obs)
        else:
            report["status"] = "bypassed_salience_gate"

        # Step 5: Persistent Work DAG & Step Logging in SQLite
        if self.store is not None:
            curr_task_id = self.working_memory.current_subtask_id
            if curr_task_id and curr_task_id in self.working_memory.subtasks:
                sub = self.working_memory.subtasks[curr_task_id]
                self.store.upsert_task(
                    task_id=sub.task_id,
                    description=sub.description,
                    status=sub.status.value,
                    dependencies=sub.dependencies,
                    entities=sub.entities,
                    created_at=sub.created_at,
                    completed_at=sub.completed_at,
                    result_summary=sub.result_summary,
                    root_goal=self.working_memory.root_goal,
                )

            step_id = self.store.record_task_step(
                action=action,
                observation=observation,
                task_id=curr_task_id,
                modified_files=modified_files,
                error=error,
                is_milestone=is_milestone,
                timestamp=now,
            )
            report["persisted_step_id"] = step_id

        return report

    def add_task(
        self,
        task_id: str,
        description: str,
        dependencies: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
    ) -> Any:
        """Adds a task to the Work DAG and persists it directly to SQLite."""
        sub = self.working_memory.add_subtask(
            task_id=task_id,
            description=description,
            dependencies=list(dependencies or []),
            entities=list(entities or []),
        )
        if self.store is not None:
            self.store.upsert_task(
                task_id=sub.task_id,
                description=sub.description,
                status=sub.status.value,
                dependencies=sub.dependencies,
                entities=sub.entities,
                created_at=sub.created_at,
                root_goal=self.working_memory.root_goal,
            )
        return sub

    def transition_task(
        self,
        task_id: str,
        status: Any,
        result_summary: Optional[str] = None,
    ) -> None:
        """Transitions a task's status in working memory and updates SQLite."""
        from .working_memory import TaskStatus
        stat_enum = TaskStatus(status) if isinstance(status, str) else status
        self.working_memory.transition_subtask(task_id, stat_enum, result_summary=result_summary)
        sub = self.working_memory.subtasks.get(task_id)
        if sub and self.store is not None:
            self.store.upsert_task(
                task_id=sub.task_id,
                description=sub.description,
                status=sub.status.value,
                dependencies=sub.dependencies,
                entities=sub.entities,
                created_at=sub.created_at,
                completed_at=sub.completed_at,
                result_summary=sub.result_summary,
                root_goal=self.working_memory.root_goal,
            )

    # ------------------------------------------------------------------------
    # Executive Layer 4 Facade: Tasks, Tools, Scratchpad & Workspace
    # ------------------------------------------------------------------------

    def create_task(
        self,
        goal: str,
        workspace_path: str = "",
        task_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        user_constraints: Optional[List[str]] = None,
    ) -> Any:
        """Initializes a new top-level TaskRecord and anchors active execution."""
        task = self.working_memory.create_task(
            goal=goal,
            workspace_path=workspace_path,
            task_id=task_id,
            parameters=parameters,
            acceptance_criteria=acceptance_criteria,
            user_constraints=user_constraints,
        )
        return task

    def get_active_task(self) -> Any:
        """Returns the currently active task record."""
        return self.working_memory.get_active_task()

    def set_workspace_path(self, path: str, scope: str = "task", ttl_turns: int = 5) -> None:
        """Anchors canonical project workspace directory."""
        self.working_memory.set_workspace_path(path, scope=scope, ttl_turns=ttl_turns)

    def get_workspace_path(self) -> str:
        """Returns the active workspace directory."""
        return self.working_memory.get_workspace_path()

    def record_tool_run(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        status: Any,
        intent: str = "",
        exit_code: Optional[int] = None,
        stdout_summary: str = "",
        raw_output_ref: Optional[str] = None,
        files_read: Optional[List[str]] = None,
        files_written: Optional[List[str]] = None,
        files_deleted: Optional[List[str]] = None,
        duration_ms: float = 0.0,
        error_message: Optional[str] = None,
    ) -> Any:
        """Records full operational telemetry for a tool execution step."""
        return self.working_memory.record_tool_run(
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            intent=intent,
            exit_code=exit_code,
            stdout_summary=stdout_summary,
            raw_output_ref=raw_output_ref,
            files_read=files_read,
            files_written=files_written,
            files_deleted=files_deleted,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def add_scratchpad_note(self, category: str, text: str) -> None:
        """Adds a note to a structured scratchpad compartment."""
        self.working_memory.add_scratchpad_note(category, text)

    def register_background_task(
        self,
        job_id: str,
        command: str,
        cwd: str,
        pid: Optional[int] = None,
        summary: str = "",
    ) -> Any:
        """Registers a running background subprocess."""
        return self.working_memory.register_background_task(
            job_id=job_id,
            command=command,
            cwd=cwd,
            pid=pid,
            summary=summary,
        )

    def update_background_task(
        self,
        job_id: str,
        status: Optional[str] = None,
        exit_code: Optional[int] = None,
        log_tail: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Any:
        """Updates background job status, exit code, and log tail."""
        return self.working_memory.update_background_task(
            job_id=job_id,
            status=status,
            exit_code=exit_code,
            log_tail=log_tail,
            summary=summary,
        )

    def format_working_context_block(self, max_tokens: int = 400) -> str:
        """Formats the high-density Markdown context block for prompt injection."""
        return self.working_memory.format_working_context_block(max_tokens=max_tokens)

    def ingest_data(
        self,
        raw_data: Any,
        speaker: str = "user",
        epoch: Optional[float] = None,
        data_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct multi-scale ingestion endpoint.
        Accepts a single line, a whole file, a conversation turn, or whole episode data.
        The LLM autonomously decides what to store and where to store it.
        """
        now = epoch if epoch is not None else time.time()
        result = self.llm_mgr.process_incoming_data(
            raw_data=raw_data,
            speaker=speaker,
            session_epoch=now,
            data_type_hint=data_type_hint,
        )
        self._sync_tensor_entities()
        return result

    def ingest_file(
        self,
        file_path: str,
        content: str,
        epoch: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a whole file (e.g. schema, config, documentation, code file).
        """
        return self.ingest_data(
            raw_data={"file_path": file_path, "content": content},
            speaker="system",
            epoch=epoch,
            data_type_hint="file",
        )

    def ingest_episode(
        self,
        name: str,
        content: Any,
        epoch: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests whole episode or session data (e.g. multi-turn session transcript or project milestone).
        """
        if isinstance(content, dict):
            payload = dict(content)
            payload["episode_name"] = name
        elif isinstance(content, list):
            payload = {"episode_name": name, "turns": content}
        else:
            payload = f"EPISODE: {name}\n\n{content}"

        return self.ingest_data(
            raw_data=payload,
            speaker="user",
            epoch=epoch,
            data_type_hint="episode",
        )

    def ingest_turns(
        self,
        turns: Sequence[Any],
        config: Optional[TurnIngestionConfig] = None,
        episode_id: Optional[str] = None,
        episode_name: Optional[str] = None,
        session_epoch: Optional[float] = None,
    ) -> TurnIngestionReport:
        """
        Conversational turn extractor and multi-turn batch ingestion endpoint.
        Processes dialogue turns using variable turn length batching (configured via TurnIngestionConfig).

        Execution steps:
        1. Normalizes diverse turn representations (dicts, tuples, strings, TurnRecord).
        2. Partitions turns into batches of size config.turns_per_batch (variable turn length).
        3. LLM-driven distillation: minimizes turn size by stripping conversational filler
           and fluff while preserving all facts, parameters, and entities.
        4. Ingests minimized dialogue turns into SpatioTemporalTensor and SQLite dialogue_turns.
        5. Commits domain entities and slotted knowledge to EntityRegistry and database.
        6. Synchronizes tensor entity vectors and guarantees contiguous turn search indexes.

        Parameters:
        - turns: Sequence of dicts, tuples, strings, or TurnRecord objects.
        - config: Optional TurnIngestionConfig overriding default settings (e.g. turns_per_batch).
        - episode_id: Optional episode identifier.
        - episode_name: Optional episode name/title.
        - session_epoch: Optional base timestamp for turns.
        """
        cfg = config or self.turn_config
        report = self.turn_pipeline.ingest_turn_sequence(
            raw_turns=turns,
            config=cfg,
            episode_id=episode_id,
            episode_name=episode_name,
            session_epoch=session_epoch if session_epoch is not None else time.time(),
        )
        self._sync_tensor_entities()
        return report

    def ingest_turn(
        self,
        text: str,
        speaker: str = "user",
        epoch: Optional[float] = None,
        turn_number: Optional[int] = None,
        episode_id: Optional[str] = None,
        episode_name: Optional[str] = None,
        config: Optional[TurnIngestionConfig] = None,
    ) -> TurnIngestionReport:
        """
        Convenience endpoint to ingest an individual conversational turn.
        """
        turn_dict = {
            "speaker": speaker,
            "text": text,
            "epoch": epoch,
            "turn_number": turn_number or (len(self.tensor_engine.turn_records) + 1),
            "episode_id": episode_id,
        }
        single_cfg = config or TurnIngestionConfig(
            turns_per_batch=1,
            minimize_turns=self.turn_config.minimize_turns,
            extract_entities_and_slots=self.turn_config.extract_entities_and_slots,
            extract_episode=self.turn_config.extract_episode,
        )
        return self.ingest_turns(
            turns=[turn_dict],
            config=single_cfg,
            episode_id=episode_id,
            episode_name=episode_name,
            session_epoch=epoch,
        )

    # ------------------------------------------------------------------------
    # Dual-Stream Context Retrieval
    # ------------------------------------------------------------------------

    def _extract_episode_seeds(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Dynamically extracts candidate episode seed nodes directly from query semantics and time references:
        1. Parse ISO dates (YYYY-MM-DD), month-day dates (August 1, 2024), and session indices (Session 21).
        2. Compute continuous Gaussian distance:
           delta_t = min(|ep.start_epoch - t_ref|, |ep.end_epoch - t_ref|)
           w_t = exp( - (delta_t)^2 / (2 * sigma_t^2) )  where sigma_t ~ 14 days (1,209,600 s)
           If an explicit range [t_min, t_max] is extracted:
           w_t = 1.0 if episode overlaps or falls within [t_min, t_max], decaying smoothly outside.
        3. Dense semantic resonance:
           Uses query_vector against episode summary embeddings in tensor_engine (cosine similarity).
        4. Fuses temporal and semantic activations into a normalized seed map {episode_id: activation}.
        """
        seeds: Dict[str, float] = {}
        if not hasattr(self, "tensor_engine") or self.tensor_engine is None or not self.tensor_engine.episodes:
            return seeds

        episodes = self.tensor_engine.episodes

        # Factor 1: Temporal Expression Extraction (Generalizable temporal expression parsing)
        iso_dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        written_dates = re.findall(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[.,]?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
            query,
            re.IGNORECASE,
        )
        month_year_dates = re.findall(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4})\b",
            query,
            re.IGNORECASE,
        )

        all_candidate_date_strs = iso_dates + written_dates + month_year_dates
        parsed_epochs: List[float] = []
        for d_str in all_candidate_date_strs:
            p_ep = parse_date_string(d_str)
            if p_ep is not None and p_ep > 0:
                parsed_epochs.append(p_ep)

        # Factor 2: Session index reference (e.g., Session 21, Session 21 to 40, Session 21-40)
        session_matches = re.findall(
            r"\b(?:session|phase|milestone)\s*(\d+)(?:\s*(?:to|-|through)\s*(?:session\s*)?(\d+))?\b",
            query,
            re.IGNORECASE,
        )
        target_sessions: Set[int] = set()
        for sm in session_matches:
            s_start = int(sm[0])
            if sm[1]:
                s_end = int(sm[1])
                target_sessions.update(range(min(s_start, s_end), max(s_start, s_end) + 1))
            else:
                target_sessions.add(s_start)

        # Factor 3: Dense Semantic Resonance with Episode Summaries
        q_vec = query_vector
        if q_vec is None and hasattr(self, "embedder") and self.embedder is not None:
            raw_v = self.embedder.embed_text(query)
            q_vec = np.asarray(raw_v, dtype=np.float32)
        if q_vec is not None:
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm > 1e-6:
                q_vec = q_vec / q_norm

        ep_vectors = getattr(self.tensor_engine, "episode_vectors", None)
        sigma_t = 14.0 * 86400.0  # 14-day temporal decay kernel

        for idx, ep in enumerate(episodes):
            ep_id = getattr(ep, "episode_id", f"ep_{idx}")
            ep_name = getattr(ep, "name", "")
            ep_start = float(getattr(ep, "start_epoch", 0.0) or 0.0)
            ep_end = float(getattr(ep, "end_epoch", 0.0) or ep_start or 0.0)

            score = 0.0

            # A. Temporal Gaussian / Interval matching
            if len(parsed_epochs) >= 2:
                t_min = min(parsed_epochs)
                t_max = max(parsed_epochs) + 86400.0  # inclusive of end day
                if ep_start > 0:
                    if (ep_start <= t_max and ep_end >= t_min) or (t_min <= ep_start <= t_max):
                        score = max(score, 1.0)
                    else:
                        dt = min(abs(ep_start - t_max), abs(ep_end - t_min))
                        temporal_decay = float(np.exp(- (dt ** 2) / (2.0 * (sigma_t ** 2))))
                        score = max(score, 0.90 * temporal_decay)
            elif len(parsed_epochs) == 1:
                t_target = parsed_epochs[0]
                if ep_start > 0:
                    dt = min(abs(ep_start - t_target), abs(ep_end - t_target))
                    temporal_decay = float(np.exp(- (dt ** 2) / (2.0 * (sigma_t ** 2))))
                    score = max(score, 0.95 * temporal_decay)

            # B. Session reference matching
            if target_sessions:
                s_num_match = re.search(r"\bSession\s*(\d+)\b", ep_name, re.IGNORECASE)
                if s_num_match:
                    s_num = int(s_num_match.group(1))
                    if s_num in target_sessions:
                        score = max(score, 1.0)

            # B2. Relative Temporal Processing Matching (yesterday, last month, last year, etc.)
            rel_meta = getattr(ep, "metadata", {}).get("relative_time") or {}
            if rel_meta:
                r_phrase = rel_meta.get("relative_phrase", "").lower()
                r_bucket = rel_meta.get("relative_bucket", "").lower()
                q_l = query.lower()
                if r_phrase and (f" {r_phrase} " in f" {q_l} " or f" {r_phrase}" in f" {q_l}" or f"{r_phrase} " in f"{q_l} "):
                    score = max(score, 1.0)
                elif r_bucket and (f" {r_bucket.lower()} " in f" {q_l} "):
                    score = max(score, 0.90)

            # C. Dense semantic similarity fallback (only when no explicit temporal/session anchors are present)
            if not parsed_epochs and not target_sessions:
                if ep_vectors is not None and idx < len(ep_vectors) and q_vec is not None:
                    sim = float(np.dot(q_vec, ep_vectors[idx]))
                    if sim > 0.35:
                        semantic_score = min(1.0, max(0.0, (sim - 0.30) / 0.40))
                        score = max(score, 0.85 * semantic_score)

            if score > 0.15:
                seeds[ep_id] = round(score, 4)

        return seeds

    def _decompose_query_intent(self, query: str) -> List[Tuple[str, np.ndarray]]:
        """
        Information-theoretically principled query decomposition:
        1. Sub-clause decomposition via standard compound punctuation delimiters ([;\\n\\?]|\s+and\s+|,\s*).
        2. Dynamic BM25 IDF weighting over self.tensor_engine._turn_idf to extract salient semantic concepts
           without any hardcoded stopwords, phrase lists, or heuristics.
        3. Semantic composition via dense neural embedding of distinctive domain concepts.
        Returns a list of (probe_text, probe_vector) pairs.
        """
        probes: List[Tuple[str, np.ndarray]] = []
        raw_clauses = [c.strip() for c in re.split(r"[;\n\?]|\s+and\s+|,\s*", query) if c.strip()]
        turn_idf = getattr(self.tensor_engine, "_turn_idf", {}) or {}
        max_idf = getattr(self.tensor_engine, "_max_idf", 10.0)

        # Dynamic corpus IDF baseline (mean of observed non-zero IDFs)
        if turn_idf:
            idfs = [v for v in turn_idf.values() if v > 0.0]
            mean_idf = float(np.mean(idfs)) if idfs else 2.0
        else:
            mean_idf = 2.0

        for clause in raw_clauses:
            tokens = [w for w in re.findall(r"\b\w+\b", clause.lower())]
            scored_tokens = sorted(tokens, key=lambda t: turn_idf.get(t, max_idf), reverse=True)
            distinctive_tokens = [t for t in scored_tokens if turn_idf.get(t, max_idf) >= mean_idf]
            if distinctive_tokens and len(distinctive_tokens) < len(tokens):
                sub_probe = " ".join(distinctive_tokens)
                sub_vec = self.embedder.embed_text(sub_probe)
                probes.append((sub_probe, sub_vec))

        return probes

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        t_now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Dual-Stream Retrieval Pipeline:
        1. Query Seed Extraction: extracts entities from query string.
        2. Working Memory Attention Injection: blends volatile session seeds with query seeds.
        3. 4D Spreading Activation: Personalized PageRank diffusion across dynamic tensor.
        4. Tri-Channel Episodic Retrieval: dense semantic + graph + regex.
        5. Dual-Stream Separation: partitions episodes into Session vs. Historical.
        6. Activated Concept Graph: top entity concepts with their active slot values.
        """
        now = t_now if t_now is not None else time.time()
        if hasattr(self.tensor_engine, "sync_relative_temporal_metadata"):
            self.tensor_engine.sync_relative_temporal_metadata(ref_epoch=t_now)

        # Step 1: Structural Embedding Noise Subtraction & Dual-Factor Seed Identification
        raw_q_vec = self.embedder.embed_text(query)
        q_vec = self.embedder.filter_structural_noise(raw_q_vec) if hasattr(self.embedder, "filter_structural_noise") else raw_q_vec

        combined_seeds: Dict[str, float] = {}

        # Factor 1: High-Performance grep-matching of existing known entities directly inside query
        if hasattr(self.registry, "find_entities_in_text"):
            matched_known = self.registry.find_entities_in_text(query)
            for ent, score in matched_known[:5]:
                combined_seeds[ent.canonical_name] = max(combined_seeds.get(ent.canonical_name, 0.0), float(score))

        # Factor 2: Dense Semantic Search across Registered Entities (ALWAYS ACTIVE CONCURRENTLY)
        sim_matches = self.registry.search_entities(query, query_vector=q_vec, top_k=5)
        for ent, score in sim_matches:
            if score > 0.20:
                combined_seeds[ent.canonical_name] = max(combined_seeds.get(ent.canonical_name, 0.0), float(score))

        # Factor 3: Relevant Data Match (Entities with matching slotted knowledge or time-slices)
        matched_slots = []
        if hasattr(self.registry, "query_entities_multi_aspect"):
            matched_slots = self.registry.query_entities_multi_aspect(
                query=query,
                query_vector=q_vec,
                query_epoch=now,
                top_k=max(8, top_k),
                embedder=self.embedder,
            )
            for sm in matched_slots[:6]:
                c_name = sm.get("canonical_name")
                if c_name:
                    combined_seeds[c_name] = max(combined_seeds.get(c_name, 0.0), 0.95)

        # Factor 4: Technical Identifiers (filenames, versions, quoted terms)
        tech_entities = self.nlp.extract_entities(query)
        for te in tech_entities:
            combined_seeds[te] = max(combined_seeds.get(te, 0.0), 1.0)

        # Working memory attention bias
        working_seeds = self.working_memory.get_working_attention_seeds()
        for wm_ent in working_seeds:
            if wm_ent in combined_seeds:
                combined_seeds[wm_ent] = max(combined_seeds[wm_ent], 1.0)
            else:
                combined_seeds[wm_ent] = self.working_bias_weight

        # Factor 5: Dynamic Episode Seed Extraction (Dates, Sessions, Dense Semantics)
        ep_seeds = self._extract_episode_seeds(query=query, query_vector=raw_q_vec)

        # Step 2: 4D Spreading Activation grounded in query, vectors, entity data, and episode seeds
        activation = self.tensor_engine.spreading_activation_4d(
            seed_entities=combined_seeds,
            seed_episodes=ep_seeds,
            query_vector=q_vec,
            query_text=query,
            t_now=now,
        )

        # Step 4: Tri-Channel Episode Retrieval & Session-Stratified Retrieval
        all_episodes = self.tensor_engine.query_episodes(
            query_text=query,
            top_k=max(25, top_k * 5),
            query_vector=q_vec,
            t_now=now,
            seed_entities=combined_seeds,
            seed_episodes=ep_seeds,
        )

        def _get_session_partition_key(ep_item: Dict[str, Any]) -> str:
            ep = ep_item.get("episode", {})
            meta = ep.get("metadata", {})
            if isinstance(meta, dict):
                if "session_id" in meta:
                    return f"session_{meta['session_id']}"
                if "session" in meta:
                    return f"session_{meta['session']}"
            name = ep.get("name", "")
            s_match = re.search(r"\bSession\s*(\d+)\b", name, re.IGNORECASE)
            if s_match:
                return f"session_{s_match.group(1)}"
            start_t = float(ep.get("start_epoch") or 0.0)
            if start_t > 0:
                from .node_models import epoch_to_iso
                return epoch_to_iso(start_t)[:10]
            return ep.get("episode_id", "session_default")

        # Group episodes by session boundary and allocate quota per partition
        from collections import defaultdict
        session_partitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ep_info in all_episodes:
            pkey = _get_session_partition_key(ep_info)
            session_partitions[pkey].append(ep_info)

        # Stratified allocation: take top matches per session partition
        per_partition_quota = max(1, min(2, top_k))
        stratified_episodes: List[Dict[str, Any]] = []
        for pkey, ep_list in session_partitions.items():
            stratified_episodes.extend(ep_list[:per_partition_quota])

        # Backfill with remaining best matches if needed
        seen_ep_ids = {e.get("episode", {}).get("episode_id") for e in stratified_episodes}
        for ep_info in all_episodes:
            eid = ep_info.get("episode", {}).get("episode_id")
            if eid not in seen_ep_ids:
                stratified_episodes.append(ep_info)
                seen_ep_ids.add(eid)

        # Step 5: Dual-Stream Separation (Session vs. Historical)
        session_episodes: List[Dict[str, Any]] = []
        historical_episodes: List[Dict[str, Any]] = []

        for ep_info in stratified_episodes:
            ep = ep_info.get("episode", {})
            start_t = float(ep.get("start_epoch") or 0.0)
            if start_t >= self.session_start_epoch:
                session_episodes.append(ep_info)
            else:
                historical_episodes.append(ep_info)

        # Step 6: Dynamic Subgraph Sizing & Traversal Metrics (Derived from Shannon Entropy)
        traversal_meta = getattr(self.tensor_engine, "last_traversal_meta", {})
        chunk_scale = float(traversal_meta.get("chunk_scale", 1.0))
        dynamic_k_entities = max(top_k, int(round(top_k * chunk_scale)))
        dynamic_k_turns = max(top_k, int(round(top_k * chunk_scale)))

        # Step 7: Activated Entity Concepts with Active Slot Values & Timeframe intervals
        top_entities = self.tensor_engine.get_top_activated_entities(
            activation,
            top_k=dynamic_k_entities,
            min_activation=0.03,
        )
        active_concepts: List[Dict[str, Any]] = []
        for ent, score in top_entities:
            active_concepts.append({
                "entity_id": ent.entity_id,
                "canonical_name": ent.canonical_name,
                "score": round(score, 4),
                "active_slots": ent.get_active_slots(),
                "slots_with_timeframe": ent.get_active_slots_with_timeframe(),
            })

        # Step 8: Granular Dialogue Turn Retrieval with Dual Intent Decomposition
        # Modulated by parent episode activation from 4D graph diffusion
        boost_episodes = {
            ep_info["episode"]["episode_id"]: ep_info.get("final_score", 1.0)
            for ep_info in all_episodes[:10]
            if "episode" in ep_info and "episode_id" in ep_info["episode"]
        }

        # Grounded entities, slots, and episodic frame candidates from knowledge graph
        grounded_entities_list = list(combined_seeds.keys()) + [c["canonical_name"] for c in active_concepts]
        grounded_slots_list = [sm.get("aspect") for sm in matched_slots if sm.get("aspect")]
        candidate_episodes_list = [
            ep_info["episode"]["episode_id"] for ep_info in all_episodes[:25]
            if "episode" in ep_info and "episode_id" in ep_info["episode"]
        ]

        primary_turns = self.tensor_engine.query_conversation_turns(
            query_text=query,
            query_vector=raw_q_vec,
            top_k=dynamic_k_turns,
            boost_episodes=boost_episodes,
            chunk_scale=chunk_scale,
            candidate_episodes=candidate_episodes_list,
            grounded_entities=grounded_entities_list,
            grounded_slots=grounded_slots_list,
        )

        # Multi-Clause & Salient Concept Intent Probing
        intent_probes = self._decompose_query_intent(query)
        if intent_probes:
            all_probe_turns: List[Dict[str, Any]] = list(primary_turns)
            for probe_text, probe_vec in intent_probes:
                sub_turns = self.tensor_engine.query_conversation_turns(
                    query_text=probe_text,
                    query_vector=probe_vec,
                    top_k=dynamic_k_turns,
                    boost_episodes=boost_episodes,
                    chunk_scale=chunk_scale,
                    candidate_episodes=candidate_episodes_list,
                    grounded_entities=grounded_entities_list,
                    grounded_slots=grounded_slots_list,
                )
                all_probe_turns.extend(sub_turns)

            # Deduplicate turns preserving highest relevance score
            merged_turns: List[Dict[str, Any]] = []
            seen_tids: Set[Any] = set()
            for t_rec in sorted(all_probe_turns, key=lambda x: x.get("score", 0.0), reverse=True):
                tid = t_rec.get("turn_id") or t_rec.get("turn_number")
                if tid not in seen_tids:
                    seen_tids.add(tid)
                    merged_turns.append(t_rec)
            exact_dialogue_turns = merged_turns[:dynamic_k_turns]
        else:
            exact_dialogue_turns = primary_turns

        return {
            "working_memory_block": self.working_memory.format_working_context_block(),
            "exact_dialogue_turns": exact_dialogue_turns,
            "dynamic_traversal_meta": traversal_meta,
            "retrieved_episodes": all_episodes[:max(25, top_k * 5)],
            "historical_insights": historical_episodes[:max(20, top_k * 4)],
            "recent_session_insights": session_episodes[:max(20, top_k * 4)],
            "active_graph_concepts": active_concepts,
            "matched_slotted_knowledge": matched_slots,
            "retrieval_seeds": list(combined_seeds.keys()),
            "episode_seeds": list(ep_seeds.keys()),
        }

    # ------------------------------------------------------------------------
    # System Prompt Injection Assembly (Strict Token Cap < 350 Tokens)
    # ------------------------------------------------------------------------

    def assemble_prompt_injection(self, intent: str, max_items: int = 5) -> str:
        """
        Compiles a high-density Markdown context block for agent system prompts.
        Uses exact granular dialogue turns and active slotted concepts (Zero summary dumping).
        Strict Token Cap: Must remain strictly under 350 tokens (< 1400 characters).
        """
        ctx = self.retrieve_context(query=intent, top_k=max_items)
        lines: List[str] = []

        # 1. Working Memory Block
        wm_block = ctx.get("working_memory_block", "")
        if wm_block:
            lines.append(wm_block)

        # 2. Granular Dialogue Evidence (Exact turns, not full summaries)
        turns = ctx.get("exact_dialogue_turns", [])
        if turns:
            lines.append("\n### [Granular Dialogue Evidence]")
            for t in turns[:3]:
                ep_name = t.get("episode_name", "")
                t_num = t.get("turn_number") or t.get("turn_id", "")
                spk = t.get("speaker", "")
                txt = t.get("text", "")
                lines.append(f"- [{ep_name} Turn {t_num}] {spk}: {txt[:120]}")

        # 3. Active Concepts & Verified Standards
        concepts = ctx.get("active_graph_concepts", [])
        if concepts:
            lines.append("\n### [Active Knowledge Concepts]")
            for c in concepts[:3]:
                c_name = c.get("canonical_name", "")
                slots = c.get("active_slots", {})
                slot_strs = [f"{k}={v}" for k, v in list(slots.items())[:2]]
                if slot_strs:
                    lines.append(f"- **{c_name}**: {', '.join(slot_strs)}")

        compiled = "\n".join(lines)

        # Enforce strict budget cap: truncate cleanly if exceeding 1400 characters
        if len(compiled) > 1400:
            compiled = compiled[:1390] + "\n..."

        return compiled

    # ------------------------------------------------------------------------
    # Grounded Query Answering API
    # ------------------------------------------------------------------------

    def query_memory(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Public high-level query endpoint: answers user/agent questions
        grounded in long-term memory graph and episodic history.
        """
        return self.llm_mgr.query_memory(question, top_k=top_k)

    # ------------------------------------------------------------------------
    # Visualization & Graph Export API
    # ------------------------------------------------------------------------

    def export_graph_data(self) -> Dict[str, Any]:
        """
        Exports the complete 4D spatio-temporal memory graph structure
        for interactive visualization, analytics, and inspection.
        """
        self._sync_tensor_entities()
        nodes_out: List[Dict[str, Any]] = []
        edges_out: List[Dict[str, Any]] = []

        # 1. Export Entity Nodes
        for eid, ent in self.registry.entities.items():
            slots_list: List[Dict[str, Any]] = []
            for aspect, kn in ent.slotted_knowledge.items():
                history_slices: List[Dict[str, Any]] = []
                for s in kn.history:
                    history_slices.append({
                        "slice_id": s.slice_id,
                        "value": s.value,
                        "valid_from": s.valid_from,
                        "valid_to": s.valid_to,
                        "mutation_reason": s.mutation_reason,
                        "episode_id": s.episode_id,
                        "taxonomic_state": s.taxonomic_state.tolist() if s.taxonomic_state is not None else None,
                    })

                tau_dict = None
                if kn.taxonomic_state is not None:
                    tau_dict = {
                        "evaluative": float(kn.taxonomic_state[0]) if len(kn.taxonomic_state) > 0 else 0.0,
                        "epistemic": float(kn.taxonomic_state[1]) if len(kn.taxonomic_state) > 1 else 0.0,
                        "deontic": float(kn.taxonomic_state[2]) if len(kn.taxonomic_state) > 2 else 0.0,
                        "temporal": float(kn.taxonomic_state[3]) if len(kn.taxonomic_state) > 3 else 0.0,
                        "affective": float(kn.taxonomic_state[4]) if len(kn.taxonomic_state) > 4 else 0.0,
                    }

                slots_list.append({
                    "aspect": aspect,
                    "active_value": kn.get_active(),
                    "active_since": kn.active_since,
                    "access_count": kn.access_count,
                    "last_activated": kn.last_activated,
                    "current_episode_id": kn.current_episode_id,
                    "taxonomic_state": tau_dict,
                    "history": history_slices,
                })

            nodes_out.append({
                "id": ent.entity_id,
                "label": ent.canonical_name,
                "category": "entity",
                "entity_type": ent.entity_type,
                "importance": ent.importance,
                "aliases": list(ent.aliases),
                "created_at": ent.created_at,
                "last_updated": ent.last_updated,
                "slots": slots_list,
                "connected_episodes": list(ent.connected_episode_ids),
            })

        # 2. Export Episode Nodes
        for ep in self.tensor_engine.episodes:
            nodes_out.append({
                "id": ep.episode_id,
                "label": ep.name,
                "category": "episode",
                "summary": ep.summary,
                "start_epoch": ep.start_epoch,
                "end_epoch": ep.end_epoch,
                "connected_entities": sorted(list(ep.connected_entity_ids)),
                "metadata": ep.metadata,
            })

        # 3. Export Associative Tensor Edges (Signed Hebbian Synapses)
        for (u, v), w in self.tensor_engine.edge_weights.items():
            src_id = self.tensor_engine.id_to_entity.get(u)
            tgt_id = self.tensor_engine.id_to_entity.get(v)
            if src_id and tgt_id:
                count = self.tensor_engine.edge_counts.get((u, v), 1)
                t_last = self.tensor_engine.edge_timestamps.get((u, v), time.time())
                edges_out.append({
                    "id": f"synapse_{src_id}_{tgt_id}",
                    "source": src_id,
                    "target": tgt_id,
                    "type": "associative",
                    "weight": round(float(w), 4),
                    "polarity": "positive" if w >= 0 else "negative",
                    "count": count,
                    "last_timestamp": t_last,
                })

        # 4. Export Episode-Entity Incidence Edges
        for (ep_idx, ent_idx), weight in self.tensor_engine.incidence_entries.items():
            if ep_idx < len(self.tensor_engine.episodes):
                ep = self.tensor_engine.episodes[ep_idx]
                ent_id = self.tensor_engine.id_to_entity.get(ent_idx)
                if ent_id:
                    edges_out.append({
                        "id": f"incidence_{ep.episode_id}_{ent_id}",
                        "source": ep.episode_id,
                        "target": ent_id,
                        "type": "incidence",
                        "weight": round(float(weight), 2),
                    })

        # 5. Export Semantic Entity-to-Entity Relations
        relations = []
        if self.store:
            relations = self.store.load_entity_relations()
        elif self.tensor_engine:
            relations = self.tensor_engine.semantic_relations

        for rel in relations:
            edges_out.append({
                "id": f"rel_{rel['source_id']}_{rel['target_id']}_{rel['relation']}",
                "source": rel["source_id"],
                "target": rel["target_id"],
                "type": "semantic_relation",
                "label": rel["relation"],
                "weight": float(rel.get("weight", 1.0)),
                "episode_id": rel.get("episode_id"),
            })

        return {
            "version": "2.0.0",
            "timestamp": time.time(),
            "nodes": nodes_out,
            "edges": edges_out,
            "stats": {
                "total_entities": len(self.registry.entities),
                "total_episodes": len(self.tensor_engine.episodes),
                "total_edges": len(edges_out),
            },
        }

    def export_graph_json(self, file_path: str) -> str:
        """Serializes the complete memory graph to a JSON file."""
        import json
        import os
        data = self.export_graph_data()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return file_path
