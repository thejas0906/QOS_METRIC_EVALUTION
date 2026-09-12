"""
Recommendation Ranking Metrics: Precision@K, Recall@K, and NDCG@K.
Evaluates the ranking quality of recommended services for users.
"""

from typing import Dict, List, Set, Optional
import numpy as np


class RecommendationMetricsEvaluator:
    """Computes standard Information Retrieval and Recommendation ranking metrics."""

    @staticmethod
    def compute_precision_at_k(recommended: List[int], ground_truth_relevant: Set[int], k: int) -> float:
        """
        Precision@K = |Recommended@K ∩ Relevant| / K
        """
        if k <= 0:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in ground_truth_relevant)
        return float(hits / k)

    @staticmethod
    def compute_recall_at_k(recommended: List[int], ground_truth_relevant: Set[int], k: int) -> float:
        """
        Recall@K = |Recommended@K ∩ Relevant| / |Relevant|
        """
        if len(ground_truth_relevant) == 0:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in ground_truth_relevant)
        return float(hits / len(ground_truth_relevant))

    @staticmethod
    def compute_dcg_at_k(recommended: List[int], gains: Dict[int, float], k: int) -> float:
        """Computes Discounted Cumulative Gain at rank cutoff K."""
        top_k = recommended[:k]
        dcg = 0.0
        for rank, item in enumerate(top_k):
            rel = gains.get(item, 0.0)
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / np.log2(rank + 2.0)
        return float(dcg)

    @classmethod
    def compute_ndcg_at_k(
        cls,
        recommended: List[int],
        gains: Dict[int, float],
        k: int
    ) -> float:
        """
        NDCG@K = DCG@K / IDCG@K
        """
        if k <= 0 or not gains:
            return 0.0

        dcg = cls.compute_dcg_at_k(recommended, gains, k)

        # Ideal ranking: items sorted descending by gain
        sorted_ideal_items = sorted(gains.keys(), key=lambda x: gains[x], reverse=True)
        idcg = cls.compute_dcg_at_k(sorted_ideal_items, gains, k)

        if idcg <= 0.0:
            return 0.0
        return float(dcg / idcg)

    @classmethod
    def evaluate_recommendations(
        cls,
        ranked_lists: Dict[int, List[int]],
        ground_truth_relevant: Dict[int, Set[int]],
        gains_dict: Optional[Dict[int, Dict[int, float]]] = None,
        k_list: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluates recommendations across all users for multiple cutoff levels K.
        
        Args:
            ranked_lists: Dict mapping user_id -> ordered list of recommended service_ids.
            ground_truth_relevant: Dict mapping user_id -> set of truly relevant service_ids.
            gains_dict: Optional dict mapping user_id -> {service_id: utility/relevance gain}.
            k_list: List of K thresholds (e.g. [5, 10, 20]).
        """
        if k_list is None:
            k_list = [5, 10, 20]

        results = {
            "precision": {k: [] for k in k_list},
            "recall": {k: [] for k in k_list},
            "ndcg": {k: [] for k in k_list}
        }

        user_ids = [u for u in ranked_lists if u in ground_truth_relevant and len(ground_truth_relevant[u]) > 0]

        for u in user_ids:
            rec = ranked_lists[u]
            rel_set = ground_truth_relevant[u]
            u_gains = gains_dict.get(u, {}) if gains_dict else {s: 1.0 for s in rel_set}

            for k in k_list:
                p_k = cls.compute_precision_at_k(rec, rel_set, k)
                r_k = cls.compute_recall_at_k(rec, rel_set, k)
                n_k = cls.compute_ndcg_at_k(rec, u_gains, k)

                results["precision"][k].append(p_k)
                results["recall"][k].append(r_k)
                results["ndcg"][k].append(n_k)

        # Average across users
        mean_results: Dict[str, Dict[int, float]] = {}
        for metric in ["precision", "recall", "ndcg"]:
            mean_results[metric] = {}
            for k in k_list:
                vals = results[metric][k]
                mean_results[metric][k] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        return mean_results
