"""
Cost-Performance Ranking and Recommendation Engine for Team B.
Generates utility-maximizing service recommendations.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class CostPerformanceRankingEngine:
    """Ranks services based on utility scores with comprehensive audit metadata."""

    def __init__(self):
        pass

    def rank_services_for_user(
        self,
        user_id: int,
        utility_matrix: np.ndarray,
        qos_matrix: np.ndarray,
        service_costs: np.ndarray,
        k: int = 10,
        interacted_mask: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float, float, float]]:
        """
        Ranks candidate services for a single user by utility score.
        
        Returns:
            List of tuples: (service_id, utility_score, qos_score, raw_cost)
        """
        scores = np.copy(utility_matrix[user_id])
        if interacted_mask is not None:
            scores[interacted_mask] = -float("inf")

        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for s_idx in top_indices:
            results.append((
                int(s_idx),
                float(scores[s_idx]),
                float(qos_matrix[user_id, s_idx]),
                float(service_costs[s_idx])
            ))
        return results

    def rank_all_users(
        self,
        utility_matrix: np.ndarray,
        k: int = 10,
        train_masks: Optional[np.ndarray] = None
    ) -> Dict[int, List[int]]:
        """
        Generates top-K recommended service IDs for all users.
        """
        num_users = utility_matrix.shape[0]
        recs = {}
        for u in range(num_users):
            scores = np.copy(utility_matrix[u])
            if train_masks is not None and train_masks[u] is not None:
                scores[train_masks[u]] = -float("inf")
            top_k = np.argsort(scores)[-k:][::-1]
            recs[u] = [int(s) for s in top_k]
        return recs
