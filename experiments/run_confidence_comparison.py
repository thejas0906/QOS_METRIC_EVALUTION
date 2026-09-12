"""
Three-Paradigm Recommendation Comparison:
Paradigm A: QoS-Only (alpha=1.0, beta=0.0, gamma=0.0)
Paradigm B: QoS + Cost (alpha=0.7, beta=0.3, gamma=0.0)
Paradigm C: QoS + Cost + Confidence (alpha=0.5, beta=0.3, gamma=0.2)

Evaluates and plots the trade-off across QoS performance, monetary cost,
prediction reliability (confidence), and catalog coverage.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.common.config import ConfigManager
from src.common.logger import ExperimentLogger
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.common.metrics_recommendation import RecommendationMetricsEvaluator
from src.team_b_gnn.graph_builder import BipartiteQoSGraphBuilder
from src.team_b_gnn.cost_generator import ServiceCostEngine
from src.team_b_gnn.confidence_estimator import GraphConfidenceEstimator
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.ranking_engine import CostPerformanceRankingEngine
from src.team_b_gnn.metrics_economic import EconomicMetricsEvaluator
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder


def run_comparison(config_path: str = "configs/team_b/graphsage.yaml", synthetic: bool = False):
    logger = ExperimentLogger.setup_logger("ConfidenceComparison", log_dir="results/logs")
    logger.info("Initializing 3-Paradigm Reliability & Trade-off Experiment")

    torch.manual_seed(42)
    np.random.seed(42)

    config = ConfigManager.load_team_b_config(config_path)

    # 1. Dataset Loading
    loader = WSDreamLoader()
    if synthetic:
        qos_data = loader.generate_synthetic_qos_dataset(num_users=339, num_services=500, seed=42)
    else:
        qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=42)

    attr_names = sorted(qos_data.keys())
    first_mat = qos_data[attr_names[0]]
    num_users, num_services = first_mat.shape

    # 2. Service Costs
    cost_file = "data/synthetic/service_costs.csv"
    if os.path.exists(cost_file):
        df_costs = pd.read_csv(cost_file)
        service_costs = df_costs["cost"].values.astype(np.float32)
        if len(service_costs) != num_services:
            normalizers = {attr: QoSNormalizer(attribute=attr).fit(qos_data[attr]) for attr in attr_names}
            util_eng = CostPerformanceUtilityEngine()
            comp_true = util_eng.aggregate_composite_qos(qos_data, normalizers)
            sq_profiles = np.mean(comp_true, axis=0)
            service_costs = ServiceCostEngine.generate_correlated_costs(sq_profiles, seed=42)
            ServiceCostEngine.save_cost_table(service_costs, cost_file)
    else:
        normalizers = {attr: QoSNormalizer(attribute=attr).fit(qos_data[attr]) for attr in attr_names}
        util_eng = CostPerformanceUtilityEngine()
        comp_true = util_eng.aggregate_composite_qos(qos_data, normalizers)
        sq_profiles = np.mean(comp_true, axis=0)
        service_costs = ServiceCostEngine.generate_correlated_costs(sq_profiles, seed=42)
        ServiceCostEngine.save_cost_table(service_costs, cost_file)

    # 3. Observation Mask & Graph Construction
    obs_mask = (first_mat > 0)
    graph_data = BipartiteQoSGraphBuilder.build_from_matrices(
        qos_matrices=qos_data,
        observation_mask=obs_mask,
        service_costs=service_costs,
        feature_dim=32,
        seed=42
    )

    # 4. Confidence Estimation
    conf_estimator = GraphConfidenceEstimator()
    conf_estimator.fit_from_edges(
        edge_u=graph_data.edge_u,
        edge_s=graph_data.edge_s,
        num_users=num_users,
        num_services=num_services
    )
    conf_matrix = conf_estimator.compute_confidence_matrix()
    conf_stats = conf_estimator.get_confidence_stats(conf_matrix)
    logger.info(f"Graph Confidence Matrix computed: mean={conf_stats['mean']:.4f}, min={conf_stats['min']:.4f}, max={conf_stats['max']:.4f}")

    # 5. Model Predictions
    model = HeteroGraphSAGE(
        in_dim_user=32,
        in_dim_service=32,
        hidden_dim=config.hidden_channels,
        out_dim=config.out_channels,
        num_layers=config.num_layers,
        aggregator=config.aggregator,
        dropout=config.dropout
    )
    decoder = EdgeQoSDecoder(
        u_dim=config.out_channels,
        s_dim=config.out_channels,
        num_qos_attributes=len(attr_names)
    )

    ckpt_path = "results/checkpoints/team_b/best_model.pt"
    if os.path.exists(ckpt_path):
        logger.info(f"Loading trained model checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        decoder.load_state_dict(ckpt["decoder_state"])
    else:
        logger.warning("Checkpoint not found; using initialized GraphSAGE weights for comparative simulation.")

    model.eval()
    decoder.eval()
    with torch.no_grad():
        u_x = torch.from_numpy(graph_data.user_features).float()
        s_x = torch.from_numpy(graph_data.service_features).float()
        e_u = torch.from_numpy(graph_data.edge_u).long()
        e_s = torch.from_numpy(graph_data.edge_s).long()

        h_u, h_s = model(u_x, s_x, e_u, e_s)
        u_rep = h_u.repeat_interleave(num_services, dim=0)
        s_rep = h_s.repeat(num_users, 1)
        full_preds = decoder(u_rep, s_rep).numpy()

    full_pred_dict = {
        attr: full_preds[:, i].reshape(num_users, num_services)
        for i, attr in enumerate(attr_names)
    }

    normalizers = {attr: QoSNormalizer(attribute=attr).fit(qos_data[attr]) for attr in attr_names}
    util_engine = CostPerformanceUtilityEngine()
    composite_qos_pred = util_engine.aggregate_composite_qos(full_pred_dict, normalizers)
    composite_qos_true = util_engine.aggregate_composite_qos(qos_data, normalizers)

    # 6. Evaluate 3 Paradigms
    paradigms = {
        "Paradigm A (QoS-Only)": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        "Paradigm B (QoS + Cost)": {"alpha": 0.7, "beta": 0.3, "gamma": 0.0},
        "Paradigm C (QoS + Cost + Confidence)": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
    }

    ranking_engine = CostPerformanceRankingEngine()
    results_summary = []
    recs_by_paradigm = {}
    metrics_by_paradigm = {}

    # Define ground-truth relevant set using composite QoS true
    k_eval = 10
    top_k_list = [5, 10, 20]

    # Baseline QoS-only recommendations for savings baseline
    recs_baseline_qos = ranking_engine.rank_all_users(composite_qos_pred, k=max(top_k_list))

    for name, params in paradigms.items():
        p_alpha = params["alpha"]
        p_beta = params["beta"]
        p_gamma = params["gamma"]

        util_engine.set_tradeoff_parameters(alpha=p_alpha, beta=p_beta, gamma=p_gamma)
        utility_pred = util_engine.compute_utility(
            composite_qos_matrix=composite_qos_pred,
            service_costs=service_costs,
            confidence_matrix=conf_matrix
        )
        utility_true = util_engine.compute_utility(
            composite_qos_matrix=composite_qos_true,
            service_costs=service_costs,
            confidence_matrix=conf_matrix
        )

        recs = ranking_engine.rank_all_users(utility_pred, k=max(top_k_list))
        recs_by_paradigm[name] = recs

        # Ground truth relevance based on this paradigm's utility objective
        ground_truth_relevant = {}
        gains_dict = {}
        for u in range(num_users):
            thresh = np.percentile(utility_true[u], 80.0)
            ground_truth_relevant[u] = set(np.where(utility_true[u] >= thresh)[0])
            gains_dict[u] = {
                int(s): float(max(0.0, utility_true[u, s]))
                for s in range(num_services)
            }

        rec_metrics = RecommendationMetricsEvaluator.evaluate_recommendations(
            ranked_lists=recs,
            ground_truth_relevant=ground_truth_relevant,
            gains_dict=gains_dict,
            k_list=top_k_list
        )

        econ_metrics = EconomicMetricsEvaluator.compute_all_economic_metrics(
            cost_aware_recs=recs,
            baseline_recs=recs_baseline_qos,
            composite_qos_matrix=composite_qos_true,
            service_costs=service_costs,
            utility_matrix=utility_true,
            confidence_matrix=conf_matrix,
            num_services=num_services,
            k=k_eval
        )

        metrics_by_paradigm[name] = {
            "params": params,
            "recommendation": rec_metrics,
            "economic": econ_metrics,
        }

        row = {
            "Paradigm": name,
            "Alpha (QoS)": p_alpha,
            "Beta (Cost)": p_beta,
            "Gamma (Confidence)": p_gamma,
            "Precision@10": rec_metrics["precision"][10],
            "Recall@10": rec_metrics["recall"][10],
            "NDCG@10": rec_metrics["ndcg"][10],
            "Average Cost ($)": econ_metrics["average_cost"],
            "Cost Savings (%)": econ_metrics["cost_savings_pct"],
            "CP Ratio": econ_metrics["cost_performance_ratio"],
            "Mean Utility": econ_metrics["mean_utility"],
            "Average Confidence": econ_metrics["average_confidence"],
            "Recommendation Coverage (%)": econ_metrics["recommendation_coverage"] * 100.0,
        }
        results_summary.append(row)
        logger.info(
            f"[{name}] Prec@10: {row['Precision@10']:.4f} | NDCG@10: {row['NDCG@10']:.4f} | "
            f"Cost: ${row['Average Cost ($)']:.4f} (Savings: {row['Cost Savings (%)']:.1f}%) | "
            f"Avg Confidence: {row['Average Confidence']:.4f} | Coverage: {row['Recommendation Coverage (%)']:.2f}%"
        )

    # 7. Export CSV & JSON
    os.makedirs("results/metrics", exist_ok=True)
    df_summary = pd.DataFrame(results_summary)
    csv_out = "results/metrics/confidence_comparison.csv"
    df_summary.to_csv(csv_out, index=False)
    logger.info(f"Comparison CSV saved to {csv_out}")

    json_out = "results/metrics/confidence_comparison.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics_by_paradigm, f, indent=2)
    logger.info(f"Comparison JSON saved to {json_out}")

    # 8. Publication-Quality Comparative Visualization
    os.makedirs("results/figures", exist_ok=True)
    fig_out = "results/figures/confidence_paradigm_comparison.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    par_labels = ["A: QoS-Only", "B: QoS+Cost", "C: QoS+Cost+Conf"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    # Subplot 1: Average Confidence (@10)
    ax1 = axes[0, 0]
    confs = [row["Average Confidence"] for row in results_summary]
    bars1 = ax1.bar(par_labels, confs, color=colors, alpha=0.88, edgecolor="black", width=0.55)
    ax1.set_title("Prediction Confidence of Top-10 Services", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Confidence Score [0, 1]", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: Invocation Cost vs Utility
    ax2 = axes[0, 1]
    costs = [row["Average Cost ($)"] for row in results_summary]
    utils = [row["Mean Utility"] for row in results_summary]
    x_indices = np.arange(len(par_labels))
    w = 0.35
    b_cost = ax2.bar(x_indices - w/2, costs, width=w, label="Avg Cost ($/call)", color="#e41a1c", alpha=0.8, edgecolor="black")
    b_util = ax2.bar(x_indices + w/2, utils, width=w, label="Mean Utility Score", color="#377eb8", alpha=0.8, edgecolor="black")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(par_labels)
    ax2.set_title("Economic Efficiency & Achieved Utility", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Metric Value", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    # Subplot 3: Recommendation Quality (NDCG@K)
    ax3 = axes[1, 0]
    for idx, (name, metrics) in enumerate(metrics_by_paradigm.items()):
        rec = metrics["recommendation"]
        ndcg_vals = [rec["ndcg"][k] for k in top_k_list]
        ax3.plot(top_k_list, ndcg_vals, marker="o", linewidth=2.2, label=par_labels[idx], color=colors[idx])
    ax3.set_title("Recommendation Ranking Quality (NDCG@K)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Cutoff K", fontsize=10)
    ax3.set_ylabel("NDCG@K", fontsize=10)
    ax3.set_xticks(top_k_list)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", frameon=True)

    # Subplot 4: Catalog Recommendation Coverage (%)
    ax4 = axes[1, 1]
    coverages = [row["Recommendation Coverage (%)"] for row in results_summary]
    bars4 = ax4.bar(par_labels, coverages, color=colors, alpha=0.88, edgecolor="black", width=0.55)
    ax4.set_title("Catalog Recommendation Coverage (%)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Coverage of Total Services (%)", fontsize=10)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("QoS Recommendation Paradigm Comparison:\nQoS-Only vs. Cost-Aware vs. Confidence-Aware Pareto Utility", fontsize=14, fontweight="bold")
    plt.savefig(fig_out, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparative plot saved to {fig_out}")

    print("\n" + "="*80)
    print("THREE-PARADIGM COMPARISON SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Paradigm Recommendation Comparison.")
    parser.add_argument("--config", type=str, default="configs/team_b/graphsage.yaml", help="Path to YAML config.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation.")
    args = parser.parse_args()

    run_comparison(config_path=args.config, synthetic=args.synthetic)
