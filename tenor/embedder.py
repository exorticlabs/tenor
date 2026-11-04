"""
Semantic Embedding Provider & Vector Math (embedder.py)
======================================================

Enterprise-grade semantic embedding provider leveraging the local Qwen2.5 backbone
(896-dimensional dense vectors) with unit L2-normalization, accelerated BLAS vector math,
and a deterministic offline fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

DITTO_PATH = os.environ.get("DITTO_PATH") or os.environ.get("SKILL_NETWORK_PATH", "")
if DITTO_PATH and DITTO_PATH not in sys.path and os.path.exists(DITTO_PATH):
    sys.path.insert(0, DITTO_PATH)

_GLOBAL_QWEN_BACKBONE = None


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Projects a 1D or 2D vector array onto the unit hypersphere."""
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(float(norm), 1e-12)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-12)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two unit vectors via dot product."""
    return float(np.dot(vec1, vec2))


def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes dot-product cosine similarity between a single 1D query vector
    and a 2D matrix of pre-normalized candidate vectors.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(matrix @ query_vec, dtype=np.float32)


def get_qwen_backbone():
    """Singleton loader for QwenEmbeddingBackbone."""
    global _GLOBAL_QWEN_BACKBONE
    if _GLOBAL_QWEN_BACKBONE is None:
        try:
            import torch
            from core.qwen_embedding import QwenEmbeddingBackbone
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _GLOBAL_QWEN_BACKBONE = QwenEmbeddingBackbone(device=device)
        except Exception as e:
            import importlib.util
            qwen_py = os.path.join(DITTO_PATH, "core", "qwen_embedding.py")
            if os.path.exists(qwen_py):
                try:
                    import torch
                    spec = importlib.util.spec_from_file_location("qwen_embedding", qwen_py)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        _GLOBAL_QWEN_BACKBONE = mod.QwenEmbeddingBackbone(device=device)
                except Exception as ex:
                    logger.warning(f"Could not initialize PyTorch Qwen backbone: {ex}")
    return _GLOBAL_QWEN_BACKBONE


class QwenSemanticEmbedder:
    """
    High-performance semantic embedding wrapper for EEKG.
    Outputs unit-normalized float32 numpy arrays for instant cosine similarity.
    Includes a deterministic hash-projection fallback if PyTorch weights are unavailable.
    """
    _instance: Optional[QwenSemanticEmbedder] = None

    def __new__(cls) -> QwenSemanticEmbedder:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, dim: int = 896):
        if getattr(self, "_initialized", False):
            return
        self.dim = dim
        self.backbone = get_qwen_backbone()
        if self.backbone is not None and hasattr(self.backbone, "d_model"):
            self.dim = self.backbone.d_model
        self._initialized = True

    def _pseudo_embed(self, text: str) -> np.ndarray:
        """
        Deterministic, seed-consistent fallback vector generator for testing
        and offline operation when neural weights are ungrounded.
        """
        # Generate pseudo-embedding using SHA256 hashing across dimensional chunks
        tokens = text.lower().strip().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
            return vec

        for t_idx, token in enumerate(tokens):
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            np.random.seed(token_hash % (2**32))
            token_vec = np.random.randn(self.dim).astype(np.float32)
            # Weight earlier tokens slightly higher
            pos_weight = 1.0 / (1.0 + 0.1 * t_idx)
            vec += token_vec * pos_weight

        return l2_normalize(vec)

    def embed_text(self, text_or_texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds one or more strings into normalized [N, 896] or [896] numpy float32 vectors.
        """
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else list(text_or_texts)

        if not texts:
            empty = np.empty((0, self.dim), dtype=np.float32)
            return empty[0] if is_single else empty

        if self.backbone is not None:
            try:
                import torch
                vectors = None
                if hasattr(self.backbone, "embed_text"):
                    vectors = self.backbone.embed_text(texts)
                elif hasattr(self.backbone, "encode"):
                    vectors = self.backbone.encode(texts)

                if vectors is not None:
                    if isinstance(vectors, torch.Tensor):
                        vectors = vectors.detach().cpu().numpy()
                    vectors = np.asarray(vectors, dtype=np.float32)
                    return vectors[0] if is_single else vectors
            except Exception as e:
                logger.warning(f"Error during neural embedding inference: {e}. Falling back to pseudo-embed.")

        # Fallback deterministic pseudo-embedder
        vecs = [self._pseudo_embed(t) for t in texts]
        arr = np.array(vecs, dtype=np.float32)
        return arr[0] if is_single else arr

    def get_structural_centroid(self) -> np.ndarray:
        """
        Computes or returns cached unit-normalized centroid vector of structural/template
        words (what, when, where, to, the, is, are, etc.) that form a generic syntactic cluster.
        """
        if getattr(self, "_structural_centroid", None) is None:
            structural_words = [
                "what", "when", "where", "how", "why", "which", "who", "whom",
                "the", "a", "an", "is", "was", "are", "were", "be", "been",
                "to", "from", "in", "on", "at", "by", "for", "with", "about",
                "into", "through", "during", "before", "after", "above", "below",
                "can", "could", "would", "should", "will", "shall", "may", "might",
                "did", "do", "does", "have", "has", "had", "tell", "me", "you",
                "my", "your", "our", "their", "there", "here", "this", "that"
            ]
            vecs = np.asarray(self.embed_text(structural_words), dtype=np.float32)
            if vecs.ndim == 2 and vecs.shape[0] > 0:
                centroid = np.mean(vecs, axis=0)
                self._structural_centroid = l2_normalize(centroid)
            else:
                self._structural_centroid = np.zeros(self.dim, dtype=np.float32)
        return self._structural_centroid

    def filter_structural_noise(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Subtracts the projection of the query onto the structural/filler word centroid.
        Isolates domain semantics from grammatical question templates.
        """
        q = np.asarray(query_vec, dtype=np.float32)
        c = self.get_structural_centroid()
        dot = float(np.dot(q, c))
        if dot > 0:
            cleaned = q - dot * c
            norm = np.linalg.norm(cleaned)
            if norm > 1e-6:
                return cleaned / norm
        return q
