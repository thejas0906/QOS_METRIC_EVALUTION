"""
Comprehensive Evaluator for Team B (GNN + Cost-Performance Recommendation).
Coordinates Prediction (RMSE, MAE), Recommendation (Precision@K, Recall@K, NDCG@K),
Economic (Cost Savings, CP Ratio), and Cold-Start (Unseen Users, Unseen Services) benchmarking.
"""

from typing import Dict, List, Set, Any, Optional
import numpy as np
import torch
from src.common.metrics_prediction import PredictionMetricsEvaluator
from src.common.metrics_recommendation import RecommendationMetricsEvaluator
from src.common.qos_normalizer import QoSNormalizer
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.ranking_engine import CostPerformanceRankingEngine
from src.team_b_gnn.metrics_economic import EconomicMetricsEvaluator
from src.team_b_gnn.cold_start_handler import ColdStartInferenceHandler
from src.team_b_gnn.confidence_estimator import GraphConfidenceEstimator


class TeamBEvaluator:
    """End-to-end evaluator for Team B GNN models."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        k_values: List[int] = [5, 10, 20]
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_values = k_values
        self.utility_engine = CostPerformanceUtilityEngine(alpha=alpha, beta=beta, gamma=gamma)
        self.ranking_engine = CostPerformanceRankingEngine()
        self.confidence_estimator = GraphConfidenceEstimator()

    def run_full_evaluation(
        self,
        model: HeteroGraphSAGE,
        decoder: EdgeQoSDecoder,
        user_features: np.ndarray,
        service_features: np.ndarray,
        edge_u_train: np.ndarray,
        edge_s_train: np.ndarray,
        test_edge_dict: Dict[str, np.ndarray],
        ground_truth_qos: Dict[str, np.ndarray],
        normalizers: Dict[str, QoSNormalizer],
        service_costs: np.ndarray,
        cold_users: Optional[np.ndarray] = None,
        cold_services: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark evaluation:
        1. Multi-attribute prediction error on test edges
        2. Utility-based top-K recommendations
        3. Economic metric comparisons against QoS-only baseline
        4. Inductive cold-start assessment
        """
        model.eval()
        decoder.eval()

        with torch.no_grad():
            u_x = torch.from_numpy(user_features).float()
            s_x = torch.from_numpy(service_features).float()
            e_u = torch.from_numpy(edge_u_train).long()
            e_s = torch.from_numpy(edge_s_train).long()

            h_u, h_s = model(u_x, s_x, e_u, e_s)

            # Predict on test edges
            t_u = torch.from_numpy(test_edge_dict["u"]).long()
            t_s = torch.from_numpy(test_edge_dict["s"]).long()

            edge_preds = decoder(h_u[t_u], h_s[t_s]).numpy()  # (E_test, num_attrs)

            # Full matrix predictions for ranking
            num_u, num_s = user_features.shape[0], service_features.shape[0]
            u_rep = h_u.repeat_interleave(num_s, dim=0)
            s_rep = h_s.repeat(num_u, 1)
            full_preds = decoder(u_rep, s_rep).numpy()

        attr_names = sorted(ground_truth_qos.keys())
        full_pred_dict = {}
        prediction_metrics = {}

        for i, attr in enumerate(attr_names):
            p_mat = full_preds[:, i].reshape(num_u, num_s)
            full_pred_dict[attr] = p_mat

            # Test edge error
            t_true = test_edge_dict["qos"][:, i]
            t_pred = edge_preds[:, i]
            prediction_metrics[attr] = {
                "rmse": PredictionMetricsEvaluator.compute_rmse(t_true, t_pred),
                "mae": PredictionMetricsEvaluator.compute_mae(t_true, t_pred),
            }

        # 2. Recommendation & Utility Evaluation
        composite_qos_pred = self.utility_engine.aggregate_composite_qos(full_pred_dict, normalizers)
        composite_qos_true = self.utility_engine.aggregate_composite_qos(ground_truth_qos, normalizers)

        # Estimate prediction confidence from training graph
        self.confidence_estimator.fit_from_edges(
            edge_u=edge_u_train,
            edge_s=edge_s_train,
            num_users=num_u,
            num_services=num_s
        )
        confidence_matrix = self.confidence_estimator.compute_confidence_matrix()

        utility_pred = self.utility_engine.compute_utility(
            composite_qos_matrix=composite_qos_pred,
            service_costs=service_costs,
            confidence_matrix=confidence_matrix
        )
        utility_true = self.utility_engine.compute_utility(
            composite_qos_matrix=composite_qos_true,
            service_costs=service_costs,
            confidence_matrix=confidence_matrix
        )

        # Baseline QoS-only recommendations (equivalent to beta=0.0, gamma=0.0)
        utility_baseline = composite_qos_pred

        # Build candidate test items per user from held-out test edges (Protocol matching Team A)
        test_u = test_edge_dict["u"]
        test_s = test_edge_dict["s"]
        user_test_items: Dict[int, List[int]] = {u: [] for u in range(num_u)}
        for u_idx, s_idx in zip(test_u, test_s):
            user_test_items[int(u_idx)].append(int(s_idx))

        recs_cost_aware: Dict[int, List[int]] = {}
        recs_baseline: Dict[int, List[int]] = {}
        ground_truth_relevant: Dict[int, Set[int]] = {}
        gains_dict: Dict[int, Dict[int, float]] = {}

        max_k = max(self.k_values)

        for u in range(num_u):
            candidates = np.array(user_test_items[u], dtype=int)
            if len(candidates) < max_k:
                continue

            # Ranked predictions among test candidates for user u (Cost-Aware)
            user_scores = utility_pred[u, candidates]
            sorted_order = np.argsort(user_scores)[::-1]
            recs_cost_aware[u] = [int(candidates[idx]) for idx in sorted_order[:max_k]]

            # Ranked predictions for baseline (QoS-Only)
            base_scores = utility_baseline[u, candidates]
            sorted_base_order = np.argsort(base_scores)[::-1]
            recs_baseline[u] = [int(candidates[idx]) for idx in sorted_base_order[:max_k]]

            # Ground truth relevance: Top 20% within user's held-out test candidates
            user_true_scores = utility_true[u, candidates]
            thresh = np.percentile(user_true_scores, 80.0)
            ground_truth_relevant[u] = set(candidates[user_true_scores >= thresh])

            gains_dict[u] = {
                int(s): float(max(0.0, score))
                for s, score in zip(candidates, user_true_scores)
            }

        rec_metrics = RecommendationMetricsEvaluator.evaluate_recommendations(
            ranked_lists=recs_cost_aware,
            ground_truth_relevant=ground_truth_relevant,
            gains_dict=gains_dict,
            k_list=self.k_values
        )

        # 3. Economic & Reliability Metrics
        economic_metrics = EconomicMetricsEvaluator.compute_all_economic_metrics(
            cost_aware_recs=recs_cost_aware,
            baseline_recs=recs_baseline,
            composite_qos_matrix=composite_qos_true,
            service_costs=service_costs,
            utility_matrix=utility_true,
            confidence_matrix=confidence_matrix,
            num_services=num_s,
            k=10
        )

        # 4. Cold-Start Evaluation
        cold_start_metrics = {}
        if cold_users is not None and len(cold_users) > 0:
            cold_start_metrics["cold_users"] = ColdStartInferenceHandler.evaluate_cold_start(
                model=model,
                decoder=decoder,
                user_features=user_features,
                service_features=service_features,
                cold_indices=cold_users,
                ground_truth_qos=ground_truth_qos,
                mode="user",
                k_values=self.k_values
            )

        if cold_services is not None and len(cold_services) > 0:
            cold_start_metrics["cold_services"] = ColdStartInferenceHandler.evaluate_cold_start(
                model=model,
                decoder=decoder,
                user_features=user_features,
                service_features=service_features,
                cold_indices=cold_services,
                ground_truth_qos=ground_truth_qos,
                mode="service",
                k_values=self.k_values
            )

        return {
            "prediction": prediction_metrics,
            "recommendation": rec_metrics,
            "economic": economic_metrics,
            "cold_start": cold_start_metrics
        }
