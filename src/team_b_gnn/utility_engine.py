"""
Cost-Performance Tradeoff Utility Engine for Team B.
Evaluates the configurable utility formulation:
    Utility = alpha * QoS_Score - beta * Cost
Provides multi-attribute QoS aggregation and sensitivity analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from src.common.constants import QoSAttribute
from src.common.qos_normalizer import QoSNormalizer


class CostPerformanceUtilityEngine:
    """Computes Pareto-optimal utility scores balancing QoS quality against invocation cost."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        attribute_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            alpha: Weight assigned to composite QoS score (alpha >= 0).
            beta: Weight assigned to service cost penalty (beta >= 0).
            gamma: Weight assigned to prediction reliability / confidence (gamma >= 0).
            attribute_weights: Importance weights across individual QoS attributes.
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.weights = attribute_weights if attribute_weights else {
            QoSAttribute.RESPONSE_TIME.value: 0.25,
            QoSAttribute.AVAILABILITY.value: 0.20,
            QoSAttribute.RELIABILITY.value: 0.20,
            QoSAttribute.THROUGHPUT.value: 0.20,
            QoSAttribute.LATENCY.value: 0.15,
        }

    def set_tradeoff_parameters(self, alpha: float, beta: float, gamma: float = 0.0) -> None:
        """Updates alpha, beta, and gamma tradeoff weights dynamically."""
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError(f"Tradeoff weights must be non-negative: alpha={alpha}, beta={beta}, gamma={gamma}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def aggregate_composite_qos(
        self,
        predicted_qos_dict: Dict[str, np.ndarray],
        normalizers: Dict[str, QoSNormalizer]
    ) -> np.ndarray:
        """
        Maps multi-attribute QoS predictions into a single [0, 1] composite QoS score.
        Higher value denotes strictly superior service performance.
        """
        first_matrix = next(iter(predicted_qos_dict.values()))
        composite = np.zeros_like(first_matrix, dtype=np.float32)

        total_weight = sum(self.weights.get(attr, 0.0) for attr in predicted_qos_dict.keys())
        if total_weight == 0:
            total_weight = 1.0

        for attr, pred_mat in predicted_qos_dict.items():
            weight = self.weights.get(attr, 0.0) / total_weight
            norm = normalizers[attr]
            norm_mat = norm.transform(pred_mat)
            composite += weight * norm_mat

        return np.clip(composite, 0.0, 1.0)

    def compute_utility(
        self,
        composite_qos_matrix: np.ndarray,
        service_costs: np.ndarray,
        confidence_matrix: Optional[np.ndarray] = None,
        normalize_cost: bool = True
    ) -> np.ndarray:
        """
        Computes the tri-factor utility matrix:
            Utility(u, s) = alpha * QoS_Score(u, s) - beta * Cost(s) + gamma * Confidence(u, s)
            
        Args:
            composite_qos_matrix: 2D array of shape (num_users, num_services) with scores in [0, 1].
            service_costs: 1D array of shape (num_services,).
            confidence_matrix: Optional 2D array of shape (num_users, num_services) with scores in [0, 1].
            normalize_cost: Whether to Min-Max normalize costs into [0, 1] for balanced scale.
            
        Returns:
            Utility matrix of shape (num_users, num_services).
        """
        if normalize_cost:
            c_min = float(np.min(service_costs))
            c_max = float(np.max(service_costs))
            denom = c_max - c_min if c_max > c_min else 1.0
            costs = (service_costs - c_min) / denom
        else:
            costs = service_costs

        # Broadcast costs across all users: (num_users, num_services)
        cost_grid = np.repeat(costs[np.newaxis, :], composite_qos_matrix.shape[0], axis=0)

        utility_matrix = (self.alpha * composite_qos_matrix) - (self.beta * cost_grid)

        if confidence_matrix is not None and self.gamma != 0.0:
            utility_matrix += (self.gamma * confidence_matrix)

        return utility_matrix.astype(np.float32)
