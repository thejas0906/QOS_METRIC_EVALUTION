"""
End-to-End Evaluator for Team A (Sparse Matrix QoS Recommendation).
Coordinates continuous prediction benchmarking (RMSE, MAE) and
Top-K ranking evaluation (Precision@K, Recall@K, NDCG@K).
"""

from typing import Dict, List, Set, Any
import numpy as np
import scipy.sparse as sp
from src.common.metrics_prediction import PredictionMetricsEvaluator
from src.common.metrics_recommendation import RecommendationMetricsEvaluator
from src.common.qos_normalizer import QoSNormalizer
from src.team_a_matrix.models.base_model import BaseMatrixQoSModel
from src.team_a_matrix.recommendation_engine import QoSRecommendationEngine


class TeamAEvaluator:
    """Evaluates Team A models across Prediction and Recommendation dimensions."""

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values

    def evaluate_single_attribute(
        self,
        model: BaseMatrixQoSModel,
        test_matrix: np.ndarray,
        test_mask: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates prediction RMSE and MAE on masked test entries."""
        pred_matrix = model.predict_matrix()
        return PredictionMetricsEvaluator.compute_all_prediction_metrics(
            y_true=test_matrix,
            y_pred=pred_matrix,
            mask=test_mask
        )

    def evaluate_recommendations(
        self,
        composite_pred: np.ndarray,
        composite_true: np.ndarray,
        test_mask: np.ndarray,
        percentile_threshold: float = 80.0
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluates top-K recommendations against ground-truth top services.
        
        Args:
            composite_pred: Predicted composite QoS score matrix.
            composite_true: True composite QoS score matrix.
            test_mask: Mask of holdout test items.
            percentile_threshold: Percentile cutoff to define 'relevant/optimal' service.
        """
        num_users = composite_pred.shape[0]
        rec_engine = QoSRecommendationEngine()

        ranked_lists: Dict[int, List[int]] = {}
        ground_truth_relevant: Dict[int, Set[int]] = {}
        gains_dict: Dict[int, Dict[int, float]] = {}

        max_k = max(self.k_values)

        for u in range(num_users):
            test_items = np.where(test_mask[u])[0]
            if len(test_items) < 2:
                continue

            # Ranked predictions among test items
            user_scores = composite_pred[u, test_items]
            sorted_order = np.argsort(user_scores)[::-1]
            ranked_lists[u] = [int(test_items[idx]) for idx in sorted_order[:max_k]]

            # Ground truth relevance
            user_true_scores = composite_true[u, test_items]
            thresh = np.percentile(user_true_scores, percentile_threshold)
            relevant_items = set(test_items[user_true_scores >= thresh])
            ground_truth_relevant[u] = relevant_items

            gains_dict[u] = {
                int(item): float(score)
                for item, score in zip(test_items, user_true_scores)
            }

        return RecommendationMetricsEvaluator.evaluate_recommendations(
            ranked_lists=ranked_lists,
            ground_truth_relevant=ground_truth_relevant,
            gains_dict=gains_dict,
            k_list=self.k_values
        )

    def run_full_evaluation(
        self,
        models_dict: Dict[str, BaseMatrixQoSModel],
        true_matrices: Dict[str, np.ndarray],
        normalizers: Dict[str, QoSNormalizer],
        test_masks: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Full benchmark run:
        1. Evaluates RMSE/MAE per QoS attribute.
        2. Computes composite QoS score.
        3. Evaluates Precision@K, Recall@K, NDCG@K.
        """
        prediction_results = {}
        pred_matrices = {}

        for attr, model in models_dict.items():
            if attr in true_matrices and attr in test_masks:
                pred_matrices[attr] = model.predict_matrix()
                res = self.evaluate_single_attribute(
                    model=model,
                    test_matrix=true_matrices[attr],
                    test_mask=test_masks[attr]
                )
                prediction_results[attr] = res

        # Composite QoS ranking evaluation
        rec_engine = QoSRecommendationEngine()
        composite_pred = rec_engine.compute_composite_qos_matrix(pred_matrices, normalizers)
        composite_true = rec_engine.compute_composite_qos_matrix(true_matrices, normalizers)

        # Common test mask (e.g. response time mask)
        common_mask = next(iter(test_masks.values()))
        rec_results = self.evaluate_recommendations(
            composite_pred=composite_pred,
            composite_true=composite_true,
            test_mask=common_mask
        )

        return {
            "prediction": prediction_results,
            "recommendation": rec_results
        }
