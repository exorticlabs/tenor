"""
4D Spatio-Temporal EEKG (Episodic-Entity Knowledge Graph Memory System v2)
========================================================================

Architecture Overview:
----------------------
This package implements a unified, multi-layered episodic and entity knowledge graph
designed for autonomous agents and long-term memory benchmarks (e.g., BEAM, LongMemEval).
The system replaces static relational graphs with continuous 4D dynamic attention tensors,
signed Hebbian plasticity, topological graph diffusion, dense semantic canonicalization,
and volatile working memory biasing.

Module Topology & Responsibilities:
-----------------------------------
1. taxonomy.py:
   - Continuous bipolar seed-anchored axis projection engine (R^896 manifold).
   - Online Hebbian pole adaptation and autonomous residual axis discovery.
   - Continuous taxonomic state vectors (tau) replacing discrete enums.

2. node_models.py:
   - Core primitives: TimeframeEpisode, TimeSlice, KnowledgeNode, EntityNode, and EntityRegistry.
   - Segregated chronological event & timeframe ordering with multi-entity binding.
   - Dual update pathways: versioned time-slicing vs. destructive in-place overwrite.
   - Recency-first retrieval with on-demand historical slice resolution.
   - Dual-factor (regex match + dense vector similarity) entity resolution and search.

3. embedder.py:
   - Singleton semantic embedding provider leveraging Qwen2.5 (896-dim unit vectors).
   - High-throughput batch matrix dot-product cosine similarity operations.

4. tensor_engine.py:
   - Dynamic spatio-temporal attention graph & episode incidence matrix with sparse CSR representations.
   - Signed Hebbian learning with asymptotic saturation and suppressive acceleration.
   - Speaker attribution and opinion isolation (listener exclusion from tensor).
   - Inverse Hub Frequency (IHF) graph dampening against degree explosion ("hairball immunity").
   - Personalized PageRank spreading activation with dual-factor seed activation and continuous decay.
   - Tri-channel retrieval fusion (dense semantic 0.65 + 4D tensor graph 0.20 + regex/lexical 0.15).
   - Contradiction attenuation (suppression of obsolete facts without deletion).

5. llm_client.py:
   - Lightweight, dependency-free HTTP client built exclusively on standard library urllib.
   - Support for Ollama Cloud and OpenAI-compatible inference backends with retry/jitter logic.

6. nlp_extractor.py:
   - High-speed rule-based entity extraction with linguistic filtering and semantic deduplication.
   - Universal pattern-based mutation and migration detection.
   - Slotted attribute parsing and turn salience scoring.

7. llm_manager.py:
   - LLM-in-the-loop data formatting for memory storage/updating and query search.
   - In-context coreference/pronoun resolution, temporal de-anchoring, and structured mutation formatting.
   - Dual-factor search query formatting (regex patterns + semantic embeddings).
   - Grounded concise answer synthesis.

8. working_memory.py:
   - Session-scoped volatile context engine managing hierarchical task DAGs.
   - Runtime register tracking (active files, variables, active errors).
   - Dynamic working entity attention mask composition for retrieval biasing.

9. orchestrator.py:
   - Top-level facade coordinating Layer 1 (Long-Term), Layer 2 (Episodic),
     Layer 3 (Recency), and Layer 4 (Working Memory).
   - Ingestion dispatch, dual-stream retrieval, and budget-constrained prompt injection.
"""

from .node_models import (
    UpdateMode,
    TimeSlice,
    TimeframeEpisode,
    KnowledgeNode,
    EntityNode,
    EntityRegistry,
    SQLiteGraphStore,
)
from .embedder import QwenSemanticEmbedder
from .taxonomy import TaxonomicAxis, TaxonomicField, TaxonomicState
from .tensor_engine import SpatioTemporalTensor
from .llm_client import MemoryLLMClient
from .nlp_extractor import FastNLPExtractor
from .llm_manager import LLMMemoryManager, MemoryCandidateMatcher, WordFilter
from .working_memory import (
    SubTask,
    TaskStatus,
    TaskPhase,
    ToolExecutionStatus,
    ToolRunRecord,
    TaskProgress,
    TaskRecord,
    BackgroundTaskRecord,
    StructuredScratchpad,
    WorkingMemoryEngine,
    ExecutiveWorkingMemoryEngine,
)
from .turn_extractor import (
    TurnIngestionConfig,
    TurnRecord,
    TurnExtractionResult,
    TurnIngestionReport,
    LLMTurnExtractor,
    TurnIngestionPipeline,
    normalize_turn_input,
)
from .orchestrator import AgentMemoryOrchestrator

# Structural exports for high-level package consumption
__version__ = "2.0.0"
__all__ = [
    "taxonomy",
    "node_models",
    "embedder",
    "tensor_engine",
    "llm_client",
    "nlp_extractor",
    "llm_manager",
    "working_memory",
    "turn_extractor",
    "orchestrator",
    "UpdateMode",
    "TimeSlice",
    "TimeframeEpisode",
    "KnowledgeNode",
    "EntityNode",
    "EntityRegistry",
    "SQLiteGraphStore",
    "QwenSemanticEmbedder",
    "TaxonomicAxis",
    "TaxonomicField",
    "TaxonomicState",
    "SpatioTemporalTensor",
    "MemoryLLMClient",
    "FastNLPExtractor",
    "LLMMemoryManager",
    "MemoryCandidateMatcher",
    "WordFilter",
    "SubTask",
    "TaskStatus",
    "TaskPhase",
    "ToolExecutionStatus",
    "ToolRunRecord",
    "TaskProgress",
    "TaskRecord",
    "BackgroundTaskRecord",
    "StructuredScratchpad",
    "WorkingMemoryEngine",
    "ExecutiveWorkingMemoryEngine",
    "TurnIngestionConfig",
    "TurnRecord",
    "TurnExtractionResult",
    "TurnIngestionReport",
    "LLMTurnExtractor",
    "TurnIngestionPipeline",
    "normalize_turn_input",
    "AgentMemoryOrchestrator",
]
