"""
Script to generate publication-quality figures for Train and Validation data splits.
Produces:
1. Team A Top-K recommendation curves (Train and Validation)
2. Team B Top-K recommendation curves (Train and Validation)
3. Comparative Top-K curves (Team A vs Team B for Train and Validation)
4. QoS Prediction Accuracy bar plots (Team A vs Team B for Train and Validation)
5. Actual vs Predicted QoS scatter plots (Train and Validation)
6. 3-Way Generalization comparison plots across Train, Val, and Test splits.
"""

import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.common.config import ConfigManager
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.common.visualization import ScientificPlotter
from src.common.evaluate_qos_prediction import QoSPredictionAccuracyEvaluator
from src.team_a_matrix.data_preprocessor import MatrixPreprocessor
from src.team_a_matrix.matrix_builder import QoSMatrixBuilder
from src.team_a_matrix.models.matrix_factorization import BiasedMatrixFactorization
from src.team_b_gnn.cost_generator import ServiceCostEngine
from src.team_b_gnn.data_preprocessor import GraphDataPreprocessor
from src.team_b_gnn.graph_builder import BipartiteQoSGraphBuilder
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder


def main():
    print("=== Generating Train and Validation Figures ===")
    
    out_dirs = [
        "results/split_evaluations/figures",
        "results/figures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Load Split Metrics JSONs
    train_a_path = "results/split_evaluations/train/team_a_train_results.json"
    train_b_path = "results/split_evaluations/train/team_b_train_results.json"
    val_a_path = "results/split_evaluations/val/team_a_val_results.json"
    val_b_path = "results/split_evaluations/val/team_b_val_results.json"
    test_a_path = "results/split_evaluations/test/team_a_test_results.json"
    test_b_path = "results/split_evaluations/test/team_b_test_results.json"

    with open(train_a_path, "r", encoding="utf-8") as f:
        train_a = json.load(f)
    with open(train_b_path, "r", encoding="utf-8") as f:
        train_b = json.load(f)
    with open(val_a_path, "r", encoding="utf-8") as f:
        val_a = json.load(f)
    with open(val_b_path, "r", encoding="utf-8") as f:
        val_b = json.load(f)
    with open(test_a_path, "r", encoding="utf-8") as f:
        test_a = json.load(f)
    with open(test_b_path, "r", encoding="utf-8") as f:
        test_b = json.load(f)

    def convert_rec(rec_dict):
        return {
            m: {int(k): v for k, v in rec_dict[m].items()}
            for m in ["precision", "recall", "ndcg"]
        }

    # -------------------------------------------------------------
    # A. TOP-K RECOMMENDATION PLOTS
    # -------------------------------------------------------------
    print("1. Plotting Top-K recommendation curves...")

    # Train Top-K Plots
    for d in out_dirs:
        ScientificPlotter.plot_topk_metrics(
            {"Team A (BiasedMF)": convert_rec(train_a["recommendation"])},
            save_path=os.path.join(d, "team_a_topk_curves_train.png")
        )
        ScientificPlotter.plot_topk_metrics(
            {"Team B (HeteroGraphSAGE)": convert_rec(train_b["recommendation"])},
            save_path=os.path.join(d, "team_b_topk_curves_train.png")
        )
        ScientificPlotter.plot_topk_metrics(
            {
                "Team A (BiasedMF)": convert_rec(train_a["recommendation"]),
                "Team B (HeteroGraphSAGE)": convert_rec(train_b["recommendation"])
            },
            save_path=os.path.join(d, "comparative_topk_curves_train.png")
        )

        # Validation Top-K Plots
        ScientificPlotter.plot_topk_metrics(
            {"Team A (BiasedMF)": convert_rec(val_a["recommendation"])},
            save_path=os.path.join(d, "team_a_topk_curves_val.png")
        )
        ScientificPlotter.plot_topk_metrics(
            {"Team B (HeteroGraphSAGE)": convert_rec(val_b["recommendation"])},
            save_path=os.path.join(d, "team_b_topk_curves_val.png")
        )
        ScientificPlotter.plot_topk_metrics(
            {
                "Team A (BiasedMF)": convert_rec(val_a["recommendation"]),
                "Team B (HeteroGraphSAGE)": convert_rec(val_b["recommendation"])
            },
            save_path=os.path.join(d, "comparative_topk_curves_val.png")
        )

    # -------------------------------------------------------------
    # B. PREDICTION ACCURACY BAR CHARTS (Team A vs Team B)
    # -------------------------------------------------------------
    print("2. Plotting QoS Prediction Accuracy comparisons...")
    for d in out_dirs:
        # Train
        eval_train = QoSPredictionAccuracyEvaluator(
            team_a_results_path=train_a_path,
            team_b_results_path=train_b_path,
            output_csv_path=os.path.join(d, "qos_prediction_accuracy_train.csv"),
            output_plot_path=os.path.join(d, "qos_prediction_accuracy_train.png")
        )
        eval_train.run_evaluation()

        # Validation
        eval_val = QoSPredictionAccuracyEvaluator(
            team_a_results_path=val_a_path,
            team_b_results_path=val_b_path,
            output_csv_path=os.path.join(d, "qos_prediction_accuracy_val.csv"),
            output_plot_path=os.path.join(d, "qos_prediction_accuracy_val.png")
        )
        eval_val.run_evaluation()

    # -------------------------------------------------------------
    # C. ACTUAL VS PREDICTED SCATTER PLOTS FOR TRAIN & VAL
    # -------------------------------------------------------------
    print("3. Generating Actual vs Predicted scatter plots for Train and Val...")
    loader = WSDreamLoader()
    qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=42)
    attr_names = sorted(qos_data.keys())
    first_attr = attr_names[0]  # availability or response_time
    full_sample_mat = qos_data[first_attr]

    # Team A masks
    observed_mask = MatrixPreprocessor.create_sparsity_mask(full_sample_mat, target_density=0.10, seed=42)
    train_mask, val_mask, test_mask = MatrixPreprocessor.train_val_test_split(full_sample_mat, observed_mask, 0.10, 0.20, seed=42)

    # Fit quick BiasedMF for scatter plotting
    csr_train = QoSMatrixBuilder.build_csr_matrix(full_sample_mat, train_mask)
    csr_val = QoSMatrixBuilder.build_csr_matrix(full_sample_mat, val_mask)
    mf_model = BiasedMatrixFactorization(latent_dim=20, lr=0.005, reg=0.02, epochs=40, batch_size=1024)
    mf_model.fit(csr_train, csr_val)
    mf_pred = mf_model.predict_matrix()

    train_coords = np.argwhere(train_mask)
    val_coords = np.argwhere(val_mask)

    y_train_true_a = full_sample_mat[train_coords[:, 0], train_coords[:, 1]]
    y_train_pred_a = mf_pred[train_coords[:, 0], train_coords[:, 1]]
    y_val_true_a = full_sample_mat[val_coords[:, 0], val_coords[:, 1]]
    y_val_pred_a = mf_pred[val_coords[:, 0], val_coords[:, 1]]

    for d in out_dirs:
        ScientificPlotter.plot_actual_vs_predicted(
            y_true=y_train_true_a,
            y_pred=y_train_pred_a,
            save_path=os.path.join(d, "team_a_actual_vs_pred_train.png"),
            title=f"Team A (BiasedMF) - {first_attr.title()} (Train Data)"
        )
        ScientificPlotter.plot_actual_vs_predicted(
            y_true=y_val_true_a,
            y_pred=y_val_pred_a,
            save_path=os.path.join(d, "team_a_actual_vs_pred_val.png"),
            title=f"Team A (BiasedMF) - {first_attr.title()} (Validation Data)"
        )

    # Team B predictions
    ckpt_path = "results/checkpoints/team_b/best_model.pt"
    if os.path.exists(ckpt_path):
        num_users, num_services = full_sample_mat.shape
        cost_file = "data/synthetic/service_costs.csv"
        service_costs = pd.read_csv(cost_file)["cost"].values.astype(np.float32) if os.path.exists(cost_file) else np.random.uniform(0.1, 1.0, num_services)

        node_splits = GraphDataPreprocessor.create_inductive_node_split(num_users, num_services, 0.10, 0.10, seed=42)
        warm_mask = np.zeros((num_users, num_services), dtype=bool)
        warm_mask[np.ix_(node_splits["warm_users"], node_splits["warm_services"])] = (full_sample_mat > 0)[np.ix_(node_splits["warm_users"], node_splits["warm_services"])]

        graph_data = BipartiteQoSGraphBuilder.build_from_matrices(qos_data, warm_mask, service_costs, 32, seed=42)
        edge_splits = GraphDataPreprocessor.split_interaction_edges(graph_data.edge_u, graph_data.edge_s, graph_data.edge_qos, 0.10, 0.20, seed=42)

        model = HeteroGraphSAGE(32, 32, 64, 32, 2, "mean", 0.1)
        decoder = EdgeQoSDecoder(32, 32, len(attr_names), [64, 32], 0.1)
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        decoder.load_state_dict(ckpt["decoder_state"])
        model.eval()
        decoder.eval()

        with torch.no_grad():
            u_x = torch.from_numpy(graph_data.user_features).float()
            s_x = torch.from_numpy(graph_data.service_features).float()
            e_u = torch.from_numpy(edge_splits["train"]["u"]).long()
            e_s = torch.from_numpy(edge_splits["train"]["s"]).long()
            h_u, h_s = model(u_x, s_x, e_u, e_s)

            # Train predictions
            tr_u = torch.from_numpy(edge_splits["train"]["u"]).long()
            tr_s = torch.from_numpy(edge_splits["train"]["s"]).long()
            pred_train_b = decoder(h_u[tr_u], h_s[tr_s]).numpy()

            # Val predictions
            val_u = torch.from_numpy(edge_splits["val"]["u"]).long()
            val_s = torch.from_numpy(edge_splits["val"]["s"]).long()
            pred_val_b = decoder(h_u[val_u], h_s[val_s]).numpy()

        for d in out_dirs:
            ScientificPlotter.plot_actual_vs_predicted(
                y_true=edge_splits["train"]["qos"][:, 0],
                y_pred=pred_train_b[:, 0],
                save_path=os.path.join(d, "team_b_actual_vs_pred_train.png"),
                title=f"Team B (GraphSAGE) - {first_attr.title()} (Train Data)"
            )
            ScientificPlotter.plot_actual_vs_predicted(
                y_true=edge_splits["val"]["qos"][:, 0],
                y_pred=pred_val_b[:, 0],
                save_path=os.path.join(d, "team_b_actual_vs_pred_val.png"),
                title=f"Team B (GraphSAGE) - {first_attr.title()} (Validation Data)"
            )

    # -------------------------------------------------------------
    # D. 3-WAY GENERALIZATION PLOTS (Train vs Val vs Test)
    # -------------------------------------------------------------
    print("4. Generating 3-Way Generalization comparison plots...")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    # D1: Prediction RMSE Across Splits
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    small_attrs = ["availability", "reliability", "response_time"]
    x = np.arange(len(small_attrs))
    w = 0.13

    for idx, sp, color_a, color_b in [
        (0, "train", "#90CAF9", "#1565C0"),
        (1, "val", "#FFE082", "#FF8F00"),
        (2, "test", "#EF9A9A", "#C62828")
    ]:
        data_a = train_a if sp == "train" else (val_a if sp == "val" else test_a)
        data_b = train_b if sp == "train" else (val_b if sp == "val" else test_b)
        vals_a = [data_a["prediction"][a]["rmse"] for a in small_attrs]
        vals_b = [data_b["prediction"][a]["rmse"] for a in small_attrs]

        axes[0].bar(x + (idx * 2 - 2.5) * w, vals_a, w, label=f"Team A ({sp.title()})", color=color_a)
        axes[0].bar(x + (idx * 2 - 1.5) * w, vals_b, w, label=f"Team B ({sp.title()})", color=color_b)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.replace('_', '\n').title() for a in small_attrs], fontweight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Prediction RMSE: Small-Scale Attributes\n(Train vs Val vs Test)", fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper right")

    # Large attributes
    large_attrs = ["latency", "throughput"]
    x2 = np.arange(len(large_attrs))
    for idx, sp, color_a, color_b in [
        (0, "train", "#90CAF9", "#1565C0"),
        (1, "val", "#FFE082", "#FF8F00"),
        (2, "test", "#EF9A9A", "#C62828")
    ]:
        data_a = train_a if sp == "train" else (val_a if sp == "val" else test_a)
        data_b = train_b if sp == "train" else (val_b if sp == "val" else test_b)
        vals_a = [data_a["prediction"][a]["rmse"] for a in large_attrs]
        vals_b = [data_b["prediction"][a]["rmse"] for a in large_attrs]

        axes[1].bar(x2 + (idx * 2 - 2.5) * w, vals_a, w, label=f"Team A ({sp.title()})", color=color_a)
        axes[1].bar(x2 + (idx * 2 - 1.5) * w, vals_b, w, label=f"Team B ({sp.title()})", color=color_b)

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Latency\n(ms)", "Throughput\n(kbps)"], fontweight="bold")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Prediction RMSE: Large-Scale Attributes\n(Train vs Val vs Test)", fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")

    plt.suptitle("Generalization Comparison: Prediction RMSE Across Train, Val, and Test Splits", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(os.path.join(d, "split_comparison_prediction_rmse.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # D2: Recommendation Quality Across Splits (Precision@K, NDCG@K)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    k_vals = [5, 10, 20]
    metrics = ["precision", "recall", "ndcg"]
    titles = ["Precision@K Generalization", "Recall@K Generalization", "NDCG@K Generalization"]

    for i, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        for sp, style, label_suffix in [
            ("train", "--", "Train"),
            ("val", "-.", "Val"),
            ("test", "-", "Test")
        ]:
            data_a = train_a if sp == "train" else (val_a if sp == "val" else test_a)
            data_b = train_b if sp == "train" else (val_b if sp == "val" else test_b)

            y_a = [data_a["recommendation"][m][str(k)] for k in k_vals]
            y_b = [data_b["recommendation"][m][str(k)] for k in k_vals]

            ax.plot(k_vals, y_a, linestyle=style, marker="o", linewidth=1.8, label=f"Team A ({label_suffix})")
            ax.plot(k_vals, y_b, linestyle=style, marker="s", linewidth=1.8, label=f"Team B ({label_suffix})")

        ax.set_xlabel("Cutoff (K)")
        ax.set_ylabel(title.split()[0])
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle("Top-K Recommendation Generalization Across Splits (Train vs Val vs Test)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for d in out_dirs:
        plt.savefig(os.path.join(d, "split_comparison_topk_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("[SUCCESS] All figures generated successfully in:")
    for d in out_dirs:
        print(f"  -> {d}")


if __name__ == "__main__":
    main()
