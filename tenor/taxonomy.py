"""
Emergent Taxonomic Field & Axis Projection Engine (taxonomy.py)
==============================================================

Implements a continuous, geometric taxonomic manifold replacing discrete enums
and static heuristics with dynamic directional axes in R^D (D=896).
Provides:
- TaxonomicAxis: Bipolar axis anchored by positive (p+) and negative (p-) semantic centroids.
- TaxonomicState: Bounded coordinate vector tau in [-1.0, 1.0]^K.
- TaxonomicField: Dynamic manager for K axes, online Hebbian pole adaptation,
  and residual emergent axis discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedder import QwenSemanticEmbedder, cosine_similarity, l2_normalize

logger = logging.getLogger(__name__)


# ============================================================================
# Taxonomic Data Structures
# ============================================================================

@dataclass
class TaxonomicAxis:
    """
    Represents a single bipolar semantic dimension in R^D.
    Direction vector a_k points from negative pole p- to positive pole p+.
    """
    name: str
    pos_centroid: np.ndarray  # p_k^+ in R^D (unit vector)
    neg_centroid: np.ndarray  # p_k^- in R^D (unit vector)
    axis_vector: np.ndarray   # a_k = normalize(p+ - p-)
    n_updates: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def recompute_axis(self) -> None:
        """Re-normalizes centroids and updates unit direction axis."""
        self.pos_centroid = l2_normalize(self.pos_centroid)
        self.neg_centroid = l2_normalize(self.neg_centroid)
        diff = self.pos_centroid - self.neg_centroid
        norm = float(np.linalg.norm(diff))
        if norm > 1e-7:
            self.axis_vector = diff / norm
        else:
            self.axis_vector = np.zeros_like(diff)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_updates": self.n_updates,
            "metadata": self.metadata,
        }


@dataclass
class TaxonomicState:
    """
    Dense continuous coordinate vector tau in [-1.0, 1.0]^K representing
    the multi-axial taxonomic projection of any utterance or knowledge assertion.
    """
    coords: np.ndarray
    axis_names: List[str]

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float32)
        self._name_to_idx = {name: idx for idx, name in enumerate(self.axis_names)}

    @property
    def evaluative(self) -> float:
        """Affinity / Preference (+) vs. Aversion / Prohibition (-)"""
        idx = self._name_to_idx.get("evaluative", 0 if len(self.coords) > 0 else None)
        return float(self.coords[idx]) if idx is not None and idx < len(self.coords) else 0.0

    @property
    def epistemic(self) -> float:
        """Verified Ground-Truth (+) vs. Doubt / Refutation (-)"""
        idx = self._name_to_idx.get("epistemic", 1 if len(self.coords) > 1 else None)
        return float(self.coords[idx]) if idx is not None and idx < len(self.coords) else 0.0

    @property
    def deontic(self) -> float:
        """Mandatory Obligation (+) vs. Strict Taboo (-)"""
        idx = self._name_to_idx.get("deontic", 2 if len(self.coords) > 2 else None)
        return float(self.coords[idx]) if idx is not None and idx < len(self.coords) else 0.0

    @property
    def temporal(self) -> float:
        """Active Present Standard (+) vs. Obsolete / Superseded (-)"""
        idx = self._name_to_idx.get("temporal", 3 if len(self.coords) > 3 else None)
        return float(self.coords[idx]) if idx is not None and idx < len(self.coords) else 0.0

    @property
    def affective(self) -> float:
        """Constructive / Coherent (+) vs. Frustrated / Urgent (-)"""
        idx = self._name_to_idx.get("affective", 4 if len(self.coords) > 4 else None)
        return float(self.coords[idx]) if idx is not None and idx < len(self.coords) else 0.0

    def get(self, name: str, default: float = 0.0) -> float:
        idx = self._name_to_idx.get(name)
        if idx is not None and idx < len(self.coords):
            return float(self.coords[idx])
        return default

    def __getitem__(self, key: Any) -> float:
        if isinstance(key, str):
            idx = self._name_to_idx[key]
            return float(self.coords[idx])
        return float(self.coords[key])

    def to_dict(self) -> Dict[str, float]:
        return {name: float(self.coords[i]) for i, name in enumerate(self.axis_names) if i < len(self.coords)}

    def __repr__(self) -> str:
        items = [f"{name}={float(self.coords[i]):.3f}" for i, name in enumerate(self.axis_names) if i < len(self.coords)]
        return f"TaxonomicState({', '.join(items)})"


# ============================================================================
# Seed Canonical Axes Definitions
# ============================================================================

SEED_AXIS_ANCHORS = {
    "evaluative": {
        "positive": [
            "I really prefer this option",
            "This is my favorite choice",
            "I love this approach and find it excellent",
            "This is an elegant and valuable solution",
            "I strongly like and recommend this",
        ],
        "negative": [
            "I strongly dislike this option",
            "I really hate this approach",
            "This is terrible and should be avoided",
            "This is worthless, broken, and poor",
            "I despise and reject this choice",
        ],
    },
    "epistemic": {
        "positive": [
            "This is verified fact and proven true",
            "This is established ground truth and confirmed reality",
            "This is undeniable, accurate, and factual",
            "Confirmed and verified to work correctly",
        ],
        "negative": [
            "This is completely false and disproven",
            "This is a refuted myth and untrue claim",
            "This is doubtful, debunked, and incorrect",
            "Unverified speculation and false information",
        ],
    },
    "deontic": {
        "positive": [
            "This is strictly required and mandatory",
            "You must follow this rule and enforced standard",
            "This is an imperative protocol obligation",
            "Always follow this required procedure",
            "Enforced security policy and mandatory compliance rule",
            "Strictly required standard and procedure to enforce",
        ],
        "negative": [
            "This is strictly forbidden and prohibited",
            "This is banned, taboo, and never allowed",
            "This is illegal, disallowed, and proscribed",
            "Never do this under any circumstances",
            "Prohibited security violation and banned action",
            "Strictly forbidden practice and disallowed conduct",
        ],
    },
    "temporal": {
        "positive": [
            "This is the current active modern standard in use today",
            "This is the latest contemporary version right now",
            "Active modern present standard",
            "Current release and state of the art",
            "Actively running and deployed in production today",
            "Modern contemporary version currently active",
        ],
        "negative": [
            "This is obsolete, outdated, and superseded",
            "This is deprecated legacy from the past",
            "This is ancient, abandoned, and no longer valid",
            "Replaced by newer standards and discontinued",
            "Retired legacy system and phased out technology",
            "Discontinued and superseded obsolete version",
        ],
    },
    "affective": {
        "positive": [
            "I feel calm, constructive, and harmonious",
            "This is pleasant, peaceful, and collaborative",
            "Working smoothly in a serene environment",
        ],
        "negative": [
            "I feel frustrated, panicked, and irritated",
            "This is urgent, alarming, and distressing",
            "Crisis situation with high anxiety and pressure",
        ],
    },
}


# ============================================================================
# Master Taxonomic Field Manager
# ============================================================================

class TaxonomicField:
    """
    Manages the continuous geometric manifold of taxonomic axes.
    Provides projection, online pole drift (Hebbian EMA), and emergent axis discovery.
    """

    def __init__(
        self,
        embedder: Optional[QwenSemanticEmbedder] = None,
        gain_gamma: float = 6.0,
        hebb_lr: float = 0.03,
        hebb_threshold: float = 0.35,
    ) -> None:
        self.embedder = embedder or QwenSemanticEmbedder()
        self.gain_gamma = gain_gamma
        self.hebb_lr = hebb_lr
        self.hebb_threshold = hebb_threshold

        self.axes: Dict[str, TaxonomicAxis] = {}
        self.axis_names: List[str] = []
        self._residual_buffer: List[np.ndarray] = []

        self._initialize_seed_axes()

    def _initialize_seed_axes(self) -> None:
        """Embeds seed phrase clusters and constructs initial bipolar axes."""
        for name, anchors in SEED_AXIS_ANCHORS.items():
            pos_phrases = anchors["positive"]
            neg_phrases = anchors["negative"]

            pos_vecs = np.asarray(self.embedder.embed_text(pos_phrases), dtype=np.float32)
            neg_vecs = np.asarray(self.embedder.embed_text(neg_phrases), dtype=np.float32)

            pos_centroid = l2_normalize(np.mean(pos_vecs, axis=0))
            neg_centroid = l2_normalize(np.mean(neg_vecs, axis=0))

            diff = pos_centroid - neg_centroid
            norm = float(np.linalg.norm(diff))
            axis_vector = diff / norm if norm > 1e-7 else np.zeros_like(diff)

            axis = TaxonomicAxis(
                name=name,
                pos_centroid=pos_centroid,
                neg_centroid=neg_centroid,
                axis_vector=axis_vector,
            )
            self.axes[name] = axis
            self.axis_names.append(name)

    def project_assertion(self, vector: np.ndarray, gamma: Optional[float] = None) -> TaxonomicState:
        """
        Projects a dense embedding vector v onto all active axes:
        s_k = cos(v, p_k^+) - cos(v, p_k^-)
        tau_k = tanh(gamma * s_k) in [-1.0, 1.0]
        """
        g = gamma if gamma is not None else self.gain_gamma
        v = l2_normalize(np.asarray(vector, dtype=np.float32))

        coords = np.zeros(len(self.axis_names), dtype=np.float32)
        for i, name in enumerate(self.axis_names):
            axis = self.axes[name]
            raw_proj = float(np.dot(v, axis.axis_vector))
            coords[i] = float(np.tanh(g * raw_proj))

        return TaxonomicState(coords=coords, axis_names=list(self.axis_names))

    def update_poles_online(
        self,
        vector: np.ndarray,
        tau: TaxonomicState,
        lr: Optional[float] = None,
        theta_hebb: Optional[float] = None,
    ) -> None:
        """
        Applies continuous Hebbian exponential moving average (EMA) drift
        to anchor poles when an assertion expresses decisive polarity.
        """
        eta = lr if lr is not None else self.hebb_lr
        theta = theta_hebb if theta_hebb is not None else self.hebb_threshold
        v = l2_normalize(np.asarray(vector, dtype=np.float32))

        for name in self.axis_names:
            val = tau.get(name, 0.0)
            if abs(val) < theta:
                continue

            axis = self.axes[name]
            if val > theta:
                # Positive alignment: attract positive pole towards v
                axis.pos_centroid = l2_normalize((1.0 - eta) * axis.pos_centroid + eta * v)
            elif val < -theta:
                # Negative alignment: attract negative pole towards v
                axis.neg_centroid = l2_normalize((1.0 - eta) * axis.neg_centroid + eta * v)

            axis.recompute_axis()
            axis.n_updates += 1

    def compute_residual(self, vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the orthogonal residual of vector v unexplained by existing axes.
        r(v) = v - sum_k <v, a_k> a_k
        E_res = ||r(v)||^2
        """
        v = l2_normalize(np.asarray(vector, dtype=np.float32))
        proj = np.zeros_like(v)
        for name in self.axis_names:
            axis = self.axes[name]
            proj += float(np.dot(v, axis.axis_vector)) * axis.axis_vector

        res = v - proj
        energy = float(np.dot(res, res))
        return res, energy

    def collect_residual(self, vector: np.ndarray, energy_threshold: float = 0.50) -> None:
        """Collects high-energy residuals for emergent axis discovery."""
        res, energy = self.compute_residual(vector)
        if energy >= energy_threshold:
            self._residual_buffer.append(res)
            if len(self._residual_buffer) > 100:
                self._residual_buffer.pop(0)

    def discover_emergent_axis(
        self,
        name: str,
        threshold_count: int = 20,
        max_iterations: int = 50,
    ) -> Optional[TaxonomicAxis]:
        """
        Autonomous discovery: extracts a dominant residual axis via rank-1
        power iteration over collected residuals.
        """
        if len(self._residual_buffer) < threshold_count:
            return None

        R = np.vstack(self._residual_buffer)  # [M, D]
        D = R.shape[1]

        # Power iteration: find leading principal direction
        u = np.random.randn(D).astype(np.float32)
        u = l2_normalize(u)

        for _ in range(max_iterations):
            # u_next = R.T @ (R @ u)
            proj = np.dot(R, u)
            u_next = np.dot(R.T, proj)
            norm = float(np.linalg.norm(u_next))
            if norm < 1e-7:
                break
            u_next = u_next / norm
            if np.linalg.norm(u_next - u) < 1e-5:
                u = u_next
                break
            u = u_next

        # Construct poles from top-decile positive and negative projections
        projections = np.dot(R, u)
        sorted_indices = np.argsort(projections)
        k_decile = max(1, len(sorted_indices) // 10)

        neg_indices = sorted_indices[:k_decile]
        pos_indices = sorted_indices[-k_decile:]

        pos_centroid = l2_normalize(np.mean(R[pos_indices], axis=0))
        neg_centroid = l2_normalize(np.mean(R[neg_indices], axis=0))

        axis = TaxonomicAxis(
            name=name,
            pos_centroid=pos_centroid,
            neg_centroid=neg_centroid,
            axis_vector=u,
            metadata={"emergent": True},
        )
        self.axes[name] = axis
        self.axis_names.append(name)
        self._residual_buffer.clear()
        return axis
