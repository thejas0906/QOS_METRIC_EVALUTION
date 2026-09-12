"""
Economic and Cost-Performance Metrics Evaluator for Team B.
Quantifies monetary efficiency, cost savings, and cost-performance ratios.
"""

from typing import Dict, List, Optional
import numpy as np


class EconomicMetricsEvaluator:
    """Evaluates financial and trade-off metrics of service recommendation lists."""

    @staticmethod
    def compute_average_cost(
        recommendations: Dict[int, List[int]],
        service_costs: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Computes the mean invocation cost of recommended services across all users.
        """
        total_costs = []
        for u, recs in recommendations.items():
            top_k = recs[:k] if k is not None else recs
            if len(top_k) > 0:
                user_avg = float(np.mean(service_costs[top_k]))
                total_costs.append(user_avg)

        return float(np.mean(total_costs)) if len(total_costs) > 0 else 0.0

    @staticmethod
    def compute_cost_savings(
        cost_aware_recs: Dict[int, List[int]],
        baseline_recs: Dict[int, List[int]],
        service_costs: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Calculates percentage cost savings achieved by cost-aware recommendation vs QoS-only baseline:
            Savings (%) = ((Cost_Baseline - Cost_Aware) / Cost_Baseline) * 100
        """
        avg_aware = EconomicMetricsEvaluator.compute_average_cost(cost_aware_recs, service_costs, k)
        avg_base = EconomicMetricsEvaluator.compute_average_cost(baseline_recs, service_costs, k)

        if avg_base <= 0.0:
            return 0.0

        savings = ((avg_base - avg_aware) / avg_base) * 100.0
        return float(savings)

    @staticmethod
    def compute_cost_performance_ratio(
        recommendations: Dict[int, List[int]],
        composite_qos_matrix: np.ndarray,
        service_costs: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Computes the Cost-Performance Ratio:
            CP_Ratio = Mean QoS / Mean Cost
        """
        user_ratios = []
        for u, recs in recommendations.items():
            top_k = recs[:k] if k is not None else recs
            if len(top_k) > 0:
                mean_qos = float(np.mean(composite_qos_matrix[u, top_k]))
                mean_cost = float(np.mean(service_costs[top_k]))
                if mean_cost > 0:
                    user_ratios.append(mean_qos / mean_cost)

        return float(np.mean(user_ratios)) if len(user_ratios) > 0 else 0.0

    @classmethod
    def compute_all_economic_metrics(
        cls,
        cost_aware_recs: Dict[int, List[int]],
        baseline_recs: Dict[int, List[int]],
        composite_qos_matrix: np.ndarray,
        service_costs: np.ndarray,
        utility_matrix: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Generates full economic audit report:
        - Average Cost
        - Baseline QoS-only Cost
        - Cost Savings (%)
        - Cost-Performance Ratio
        - Mean Utility Score
        """
        avg_cost = cls.compute_average_cost(cost_aware_recs, service_costs, k)
        base_cost = cls.compute_average_cost(baseline_recs, service_costs, k)
        savings = cls.compute_cost_savings(cost_aware_recs, baseline_recs, service_costs, k)
        cp_ratio = cls.compute_cost_performance_ratio(cost_aware_recs, composite_qos_matrix, service_costs, k)

        # Average utility achieved
        utilities = []
        for u, recs in cost_aware_recs.items():
            top_k = recs[:k]
            if len(top_k) > 0:
                utilities.append(float(np.mean(utility_matrix[u, top_k])))
        mean_util = float(np.mean(utilities)) if len(utilities) > 0 else 0.0

        return {
            "average_cost": avg_cost,
            "baseline_qos_cost": base_cost,
            "cost_savings_pct": savings,
            "cost_performance_ratio": cp_ratio,
            "mean_utility": mean_util
        }
