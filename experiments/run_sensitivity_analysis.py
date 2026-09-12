"""
Sensitivity Analysis Runner for Cost-Performance Trade-off.
Sweeps alpha (QoS weight) and beta (Cost penalty) to generate the Pareto frontier.
"""

import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.common.logger import ExperimentLogger
from src.common.visualization import ScientificPlotter
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.team_b_gnn.cost_generator import ServiceCostEngine
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.ranking_engine import CostPerformanceRankingEngine
from src.team_b_gnn.metrics_economic import EconomicMetricsEvaluator


def main():
    logger = ExperimentLogger.setup_logger("SensitivityAnalysis")
    logger.info("Running Cost-Performance Sensitivity Sweep over (alpha, beta)...")

    loader = WSDreamLoader()
    qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=42)
    attr_names = sorted(qos_data.keys())

    normalizers = {attr: QoSNormalizer(attribute=attr).fit(qos_data[attr]) for attr in attr_names}
    utility_engine = CostPerformanceUtilityEngine()
    composite_qos = utility_engine.aggregate_composite_qos(qos_data, normalizers)
    service_quality_profiles = np.mean(composite_qos, axis=0)

    # Load or generate service costs
    cost_file = "data/synthetic/service_costs.csv"
    if os.path.exists(cost_file):
        df_costs = ServiceCostEngine.load_cost_table(cost_file)
        service_costs = df_costs["cost"].values.astype(np.float32)
    else:
        service_costs = ServiceCostEngine.generate_correlated_costs(
            service_qos_profiles=service_quality_profiles,
            correlation_strength=0.75,
            base_cost_range=(0.01, 1.00),
            seed=42
        )

    ranking_engine = CostPerformanceRankingEngine()

    # Grid sweep: alpha in [0.0, 1.0], beta = 1.0 - alpha
    alpha_steps = np.linspace(0.0, 1.0, 11)
    results_list = []

    alpha_vals = []
    avg_qos_vals = []
    avg_cost_vals = []
    utility_vals = []

    k = 10
    num_users = composite_qos.shape[0]

    for alpha in alpha_steps:
        beta = 1.0 - alpha
        engine = CostPerformanceUtilityEngine(alpha=alpha, beta=beta)
        utility_matrix = engine.compute_utility(composite_qos, service_costs)

        recs = ranking_engine.rank_all_users(utility_matrix, k=k)

        # Compute average QoS, Cost, and Utility of recommended services
        user_qos = []
        user_costs = []
        user_utils = []

        for u in range(num_users):
            top_k_services = recs[u][:k]
            user_qos.append(float(np.mean(composite_qos[u, top_k_services])))
            user_costs.append(float(np.mean(service_costs[top_k_services])))
            user_utils.append(float(np.mean(utility_matrix[u, top_k_services])))

        mean_q = float(np.mean(user_qos))
        mean_c = float(np.mean(user_costs))
        mean_u = float(np.mean(user_utils))

        alpha_vals.append(float(alpha))
        avg_qos_vals.append(mean_q)
        avg_cost_vals.append(mean_c)
        utility_vals.append(mean_u)

        results_list.append({
            "alpha": round(float(alpha), 2),
            "beta": round(float(beta), 2),
            "mean_qos_score": mean_q,
            "mean_cost": mean_c,
            "mean_utility": mean_u
        })
        logger.info(f"alpha={alpha:.2f}, beta={beta:.2f} -> Avg QoS: {mean_q:.4f} | Avg Cost: ${mean_c:.4f} | Utility: {mean_u:.4f}")

    # Export sensitivity CSV
    os.makedirs("results/metrics", exist_ok=True)
    df_res = pd.DataFrame(results_list)
    out_csv = "results/metrics/sensitivity_alpha_beta.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Sensitivity data saved to {out_csv}")

    # Plot Pareto curve
    os.makedirs("results/figures", exist_ok=True)
    ScientificPlotter.plot_cost_performance_tradeoff(
        alpha_list=alpha_vals,
        qos_scores=avg_qos_vals,
        costs=avg_cost_vals,
        utilities=utility_vals,
        save_path="results/figures/cost_utility_pareto.png"
    )
    logger.info("Saved Pareto frontier curve to results/figures/cost_utility_pareto.png")


if __name__ == "__main__":
    main()
