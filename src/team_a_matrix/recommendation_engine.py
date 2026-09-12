"""
QoS-Only Recommendation Engine for Team A.
Ranks candidate web services based exclusively on multi-attribute QoS scores.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from src.common.constants import QoSAttribute
from src.common.qos_normalizer import QoSNormalizer


class QoSRecommendationEngine:
    """Ranks services based on composite QoS utility without economic/cost awareness."""

    def __init__(self, attribute_weights: Optional[Dict[str, float]] = None):
        self.weights = attribute_weights if attribute_weights else {
            QoSAttribute.RESPONSE_TIME.value: 0.25,
            QoSAttribute.AVAILABILITY.value: 0.20,
            QoSAttribute.RELIABILITY.value: 0.20,
            QoSAttribute.THROUGHPUT.value: 0.20,
            QoSAttribute.LATENCY.value: 0.15,
        }

    def compute_composite_qos_matrix(
        self,
        predicted_matrices: Dict[str, np.ndarray],
        normalizers: Dict[str, QoSNormalizer]
    ) -> np.ndarray:
        """
        Combines multi-attribute predictions into a single [0, 1] composite QoS score matrix.
        Higher values strictly indicate superior QoS.
        """
        first_matrix = next(iter(predicted_matrices.values()))
        composite = np.zeros_like(first_matrix, dtype=np.float32)
        total_weight = sum(self.weights.get(attr, 0.0) for attr in predicted_matrices.keys())
        if total_weight == 0:
            total_weight = 1.0

        for attr, pred_mat in predicted_matrices.items():
            weight = self.weights.get(attr, 0.0) / total_weight
            norm = normalizers[attr]
            norm_mat = norm.transform(pred_mat)
            composite += weight * norm_mat

        return np.clip(composite, 0.0, 1.0)

    def recommend_top_k(
        self,
        user_id: int,
        composite_qos_matrix: np.ndarray,
        k: int = 10,
        interacted_mask: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommends top-K services for a given user.
        
        Args:
            user_id: Target user index.
            composite_qos_matrix: 2D array of composite QoS scores.
            k: Top-K cutoff.
            interacted_mask: Optional boolean array marking services to exclude (already invoked).
        """
        user_scores = np.copy(composite_qos_matrix[user_id])
        if interacted_mask is not None:
            user_scores[interacted_mask] = -float("inf")

        # Top-K indices
        top_k_indices = np.argsort(user_scores)[-k:][::-1]
        return [(int(idx), float(user_scores[idx])) for idx in top_k_indices]

    def recommend_all_users(
        self,
        composite_qos_matrix: np.ndarray,
        k: int = 10,
        train_masks: Optional[np.ndarray] = None
    ) -> Dict[int, List[int]]:
        """
        Generates top-K recommendations for all users.
        """
        num_users = composite_qos_matrix.shape[0]
        recommendations = {}
        for u in range(num_users):
            mask = train_masks[u] if train_masks is not None else None
            top_k = self.recommend_top_k(u, composite_qos_matrix, k=k, interacted_mask=mask)
            recommendations[u] = [s_id for s_id, _ in top_k]
        return recommendations
