"""
Economic and Cost-Performance Metrics Evaluator for Team B.
Quantifies monetary efficiency, cost savings, and cost-performance ratios.
"""

from typing import Dict, List, Optional, Any
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

    @staticmethod
    def compute_average_confidence(
        recommendations: Dict[int, List[int]],
        confidence_matrix: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Computes the mean prediction confidence score of recommended services across users.
        """
        all_confs = []
        for u, recs in recommendations.items():
            top_k = recs[:k] if k is not None else recs
            if len(top_k) > 0:
                all_confs.append(float(np.mean(confidence_matrix[u, top_k])))
        return float(np.mean(all_confs)) if len(all_confs) > 0 else 0.0

    @staticmethod
    def compute_confidence_distribution(
        recommendations: Dict[int, List[int]],
        confidence_matrix: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Computes statistical distribution of confidence scores for recommended services.
        """
        vals = []
        for u, recs in recommendations.items():
            top_k = recs[:k] if k is not None else recs
            for s in top_k:
                vals.append(float(confidence_matrix[u, s]))

        if len(vals) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr))
        }

    @staticmethod
    def compute_recommendation_coverage(
        recommendations: Dict[int, List[int]],
        num_services: int,
        k: Optional[int] = None
    ) -> float:
        """
        Computes the catalog coverage ratio: |unique recommended services| / total services.
        """
        unique_services = set()
        for u, recs in recommendations.items():
            top_k = recs[:k] if k is not None else recs
            unique_services.update(top_k)

        return float(len(unique_services) / max(1, num_services))

    @classmethod
    def compute_all_economic_metrics(
        cls,
        cost_aware_recs: Dict[int, List[int]],
        baseline_recs: Dict[int, List[int]],
        composite_qos_matrix: np.ndarray,
        service_costs: np.ndarray,
        utility_matrix: np.ndarray,
        confidence_matrix: Optional[np.ndarray] = None,
        num_services: Optional[int] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Generates full economic audit report:
        - Average Cost
        - Baseline QoS-only Cost
        - Cost Savings (%)
        - Cost-Performance Ratio
        - Mean Utility Score
        - Average Confidence
        - Recommendation Coverage
        - Confidence Distribution
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

        n_services = num_services if num_services is not None else len(service_costs)
        coverage = cls.compute_recommendation_coverage(cost_aware_recs, n_services, k)

        metrics = {
            "average_cost": avg_cost,
            "baseline_qos_cost": base_cost,
            "cost_savings_pct": savings,
            "cost_performance_ratio": cp_ratio,
            "mean_utility": mean_util,
            "recommendation_coverage": coverage
        }

        if confidence_matrix is not None:
            metrics["average_confidence"] = cls.compute_average_confidence(cost_aware_recs, confidence_matrix, k)
            metrics["confidence_distribution"] = cls.compute_confidence_distribution(cost_aware_recs, confidence_matrix, k)

        return metrics
