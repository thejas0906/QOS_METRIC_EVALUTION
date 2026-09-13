"""
Sparsity Robustness Benchmark Runner.
Evaluates how prediction accuracy degrades under increasing QoS sparsity
for Team A (BiasedMF) and Team B (HeteroGraphSAGE).

Usage:
    python experiments/run_sparsity_benchmark.py
"""

import json
import os
import sys
import time

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim

from src.common.logger import ExperimentLogger
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.common.metrics_prediction import PredictionMetricsEvaluator
from src.common.sparsity_generator import SparsityGenerator
from src.common.sparsity_evaluator import SparsityEvaluator

# Team A imports
from src.team_a_matrix.data_preprocessor import MatrixPreprocessor
from src.team_a_matrix.matrix_builder import QoSMatrixBuilder
from src.team_a_matrix.models.matrix_factorization import BiasedMatrixFactorization

# Team B imports
from src.team_b_gnn.graph_builder import BipartiteQoSGraphBuilder
from src.team_b_gnn.data_preprocessor import GraphDataPreprocessor
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder


# ======================== Configuration ========================

DENSITY_LEVELS = [0.50, 0.30, 0.20, 0.10, 0.05]
SEED = 42

# Team A hyperparameters
TEAM_A_LATENT_DIM = 20
TEAM_A_LR = 0.005
TEAM_A_REG = 0.02
TEAM_A_EPOCHS = 80
TEAM_A_BATCH_SIZE = 1024
TEAM_A_PATIENCE = 5

# Team B hyperparameters
TEAM_B_HIDDEN_DIM = 64
TEAM_B_OUT_DIM = 32
TEAM_B_NUM_LAYERS = 2
TEAM_B_LR = 0.002
TEAM_B_WEIGHT_DECAY = 1e-4
TEAM_B_EPOCHS = 100
TEAM_B_PATIENCE = 10
TEAM_B_FEATURE_DIM = 32
TEAM_B_DROPOUT = 0.1

# Output paths
CSV_OUTPUT = "results/sparsity_benchmark.csv"
RMSE_PLOT = "results/figures/sparsity_rmse_comparison.png"
MAE_PLOT = "results/figures/sparsity_mae_comparison.png"
PER_ATTR_PLOT = "results/figures/sparsity_per_attribute_rmse.png"
JSON_OUTPUT = "results/metrics/sparsity_benchmark.json"


def train_team_a_at_density(qos_data, attr_names, train_mask, val_mask, test_mask, logger):
    """Trains Team A BiasedMF models and returns per-attribute prediction metrics on test set."""
    results = {}
    for attr in attr_names:
        mat = qos_data[attr]
        csr_train = QoSMatrixBuilder.build_csr_matrix(mat, train_mask)
        csr_val = QoSMatrixBuilder.build_csr_matrix(mat, val_mask)

        model = BiasedMatrixFactorization(
            latent_dim=TEAM_A_LATENT_DIM,
            lr=TEAM_A_LR,
            reg=TEAM_A_REG,
            epochs=TEAM_A_EPOCHS,
            batch_size=TEAM_A_BATCH_SIZE,
            early_stopping_patience=TEAM_A_PATIENCE
        )
        model.fit(csr_train, csr_val)

        pred_matrix = model.predict_matrix()
        metrics = PredictionMetricsEvaluator.compute_all_prediction_metrics(
            y_true=mat, y_pred=pred_matrix, mask=test_mask
        )
        results[attr] = metrics
        logger.info(f"    [Team A] {attr}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    return results


def train_team_b_at_density(qos_data, attr_names, observation_mask, test_mask, logger):
    """Trains Team B GraphSAGE model and returns per-attribute prediction metrics on test set."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    first_mat = qos_data[attr_names[0]]
    num_users, num_services = first_mat.shape

    # Minimal cost vector (not used for prediction, but required by graph builder)
    service_costs = np.random.RandomState(SEED).uniform(0.01, 1.0, size=num_services).astype(np.float32)

    # Build graph from observed entries
    graph_data = BipartiteQoSGraphBuilder.build_from_matrices(
        qos_matrices=qos_data,
        observation_mask=observation_mask,
        service_costs=service_costs,
        feature_dim=TEAM_B_FEATURE_DIM,
        seed=SEED
    )

    # Split graph edges into train/val/test
    edge_splits = GraphDataPreprocessor.split_interaction_edges(
        u_indices=graph_data.edge_u,
        s_indices=graph_data.edge_s,
        qos_attributes=graph_data.edge_qos,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=SEED
    )
    train_edges = edge_splits["train"]
    val_edges = edge_splits["val"]
    test_edges_gnn = edge_splits["test"]

    num_attrs = len(attr_names)

    model = HeteroGraphSAGE(
        in_dim_user=TEAM_B_FEATURE_DIM,
        in_dim_service=TEAM_B_FEATURE_DIM,
        hidden_dim=TEAM_B_HIDDEN_DIM,
        out_dim=TEAM_B_OUT_DIM,
        num_layers=TEAM_B_NUM_LAYERS,
        aggregator="mean",
        dropout=TEAM_B_DROPOUT
    )
    decoder = EdgeQoSDecoder(
        u_dim=TEAM_B_OUT_DIM,
        s_dim=TEAM_B_OUT_DIM,
        num_qos_attributes=num_attrs,
        hidden_dims=[64, 32],
        dropout=TEAM_B_DROPOUT
    )

    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=TEAM_B_LR,
        weight_decay=TEAM_B_WEIGHT_DECAY
    )

    # Tensors
    u_x = torch.from_numpy(graph_data.user_features).float()
    s_x = torch.from_numpy(graph_data.service_features).float()
    e_u_train = torch.from_numpy(train_edges["u"]).long()
    e_s_train = torch.from_numpy(train_edges["s"]).long()
    e_qos_train = torch.from_numpy(train_edges["qos"]).float()

    e_u_val = torch.from_numpy(val_edges["u"]).long()
    e_s_val = torch.from_numpy(val_edges["s"]).long()
    e_qos_val = torch.from_numpy(val_edges["qos"]).float()

    # Attribute scale balancing
    qos_scales = torch.tensor([
        max(float(np.std(train_edges["qos"][:, col])), 1.0)
        for col in range(num_attrs)
    ], dtype=torch.float32)

    def balanced_loss(preds, targets):
        return torch.mean(((preds - targets) / qos_scales) ** 2)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, TEAM_B_EPOCHS + 1):
        model.train()
        decoder.train()
        optimizer.zero_grad()

        h_u, h_s = model(u_x, s_x, e_u_train, e_s_train)
        preds_train = decoder(h_u[e_u_train], h_s[e_s_train])
        loss = balanced_loss(preds_train, e_qos_train)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), 5.0)
        optimizer.step()

        model.eval()
        decoder.eval()
        with torch.no_grad():
            h_u_v, h_s_v = model(u_x, s_x, e_u_train, e_s_train)
            preds_val = decoder(h_u_v[e_u_val], h_s_v[e_s_val])
            val_loss = balanced_loss(preds_val, e_qos_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()}
            }
        else:
            patience_counter += 1
            if patience_counter >= TEAM_B_PATIENCE:
                break

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state["model"])
        decoder.load_state_dict(best_state["decoder"])

    # Full matrix prediction
    model.eval()
    decoder.eval()
    with torch.no_grad():
        h_u, h_s = model(u_x, s_x, e_u_train, e_s_train)
        u_rep = h_u.repeat_interleave(num_services, dim=0)
        s_rep = h_s.repeat(num_users, 1)
        full_preds = decoder(u_rep, s_rep).numpy()

    # Evaluate per attribute on the fixed test mask
    results = {}
    for i, attr in enumerate(attr_names):
        pred_mat = full_preds[:, i].reshape(num_users, num_services)
        true_mat = qos_data[attr]
        metrics = PredictionMetricsEvaluator.compute_all_prediction_metrics(
            y_true=true_mat, y_pred=pred_mat, mask=test_mask
        )
        results[attr] = metrics
        logger.info(f"    [Team B] {attr}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    return results


def main():
    logger = ExperimentLogger.setup_logger("SparsityBenchmark", log_dir="results/logs/sparsity")
    logger.info("=" * 60)
    logger.info("SPARSITY ROBUSTNESS BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Target density levels: {DENSITY_LEVELS}")
    logger.info(f"Equivalent missing ratios: {[(1-d)*100 for d in DENSITY_LEVELS]}%")

    # 1. Load dataset
    loader = WSDreamLoader()
    qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=SEED)
    attr_names = sorted(qos_data.keys())
    first_mat = qos_data[attr_names[0]]
    num_users, num_services = first_mat.shape

    logger.info(f"Dataset: {num_users} users x {num_services} services, attributes: {attr_names}")

    # 2. Generate sparsity masks with fixed test set
    logger.info("Generating sparsity masks with fixed test set...")
    density_masks = SparsityGenerator.generate_density_masks(
        qos_matrix=first_mat,
        density_levels=DENSITY_LEVELS,
        test_ratio=0.20,
        val_ratio=0.10,
        seed=SEED
    )

    evaluator = SparsityEvaluator(qos_attributes=attr_names)

    # 3. Run benchmark at each density level
    all_results = {}
    for density in DENSITY_LEVELS:
        missing_pct = (1.0 - density) * 100.0
        masks = density_masks[density]

        logger.info(f"\n{'='*50}")
        logger.info(f"DENSITY = {density:.0%} | MISSING = {missing_pct:.0f}%")
        logger.info(f"  Observed: {masks['num_observed']}, Train: {masks['num_train']}, "
                     f"Val: {masks['num_val']}, Test: {masks['num_test']}")
        logger.info(f"{'='*50}")

        level_results = {"density": density, "missing_pct": missing_pct}

        # ---- Team A ----
        logger.info("  Training Team A (BiasedMF)...")
        t0 = time.time()
        team_a_metrics = train_team_a_at_density(
            qos_data, attr_names,
            masks["train_mask"], masks["val_mask"], masks["test_mask"],
            logger
        )
        t_a = time.time() - t0
        logger.info(f"  Team A completed in {t_a:.1f}s")

        for attr, m in team_a_metrics.items():
            evaluator.add_result(density, "Team A (BiasedMF)", attr, m["rmse"], m["mae"])
        level_results["team_a"] = team_a_metrics

        # ---- Team B ----
        logger.info("  Training Team B (GraphSAGE)...")
        t0 = time.time()
        team_b_metrics = train_team_b_at_density(
            qos_data, attr_names,
            masks["observed_mask"], masks["test_mask"],
            logger
        )
        t_b = time.time() - t0
        logger.info(f"  Team B completed in {t_b:.1f}s")

        for attr, m in team_b_metrics.items():
            evaluator.add_result(density, "Team B (GraphSAGE)", attr, m["rmse"], m["mae"])
        level_results["team_b"] = team_b_metrics

        all_results[f"{density:.2f}"] = level_results

    # 4. Save CSV
    csv_path = evaluator.save_csv(CSV_OUTPUT)
    logger.info(f"\n[OK] Full results CSV saved to: {csv_path}")

    # 5. Generate plots
    evaluator.plot_rmse_comparison(RMSE_PLOT)
    logger.info(f"[OK] RMSE comparison plot saved to: {RMSE_PLOT}")

    evaluator.plot_mae_comparison(MAE_PLOT)
    logger.info(f"[OK] MAE comparison plot saved to: {MAE_PLOT}")

    evaluator.plot_per_attribute_rmse(PER_ATTR_PLOT)
    logger.info(f"[OK] Per-attribute RMSE plot saved to: {PER_ATTR_PLOT}")

    # 6. Degradation analysis
    analysis = evaluator.compute_degradation_analysis()

    logger.info("\n" + "=" * 60)
    logger.info("DEGRADATION ANALYSIS")
    logger.info("=" * 60)

    for team_key in ["Team A (BiasedMF)", "Team B (GraphSAGE)"]:
        if team_key in analysis:
            a = analysis[team_key]
            logger.info(f"\n{team_key}:")
            logger.info(f"  RMSE: {a['rmse_at_densest']:.4f} (at {a['densest_density']:.0%}) -> "
                         f"{a['rmse_at_sparsest']:.4f} (at {a['sparsest_density']:.0%}) "
                         f"| +{a['rmse_increase_pct']:.1f}% increase")
            logger.info(f"  MAE:  {a['mae_at_densest']:.4f} (at {a['densest_density']:.0%}) -> "
                         f"{a['mae_at_sparsest']:.4f} (at {a['sparsest_density']:.0%}) "
                         f"| +{a['mae_increase_pct']:.1f}% increase")

    if "relative_improvement" in analysis:
        logger.info(f"\nRelative Improvement (Team B RMSE vs Team A RMSE):")
        for row in analysis["relative_improvement"]:
            sign = "+" if row["improvement_pct"] > 0 else ""
            logger.info(f"  Missing {row['missing_pct']:.0f}%: "
                         f"Team A RMSE={row['team_a_rmse']:.4f}, Team B RMSE={row['team_b_rmse']:.4f} "
                         f"-> {sign}{row['improvement_pct']:.1f}% improvement")

    # 7. Summary tables
    summary = evaluator.build_summary_table()

    logger.info(f"\n{'='*60}")
    logger.info("TABLE 1: Average RMSE by Sparsity Level")
    logger.info(f"{'='*60}")
    logger.info(f"{'Missing %':<12} {'Team A RMSE':<16} {'Team B RMSE':<16} {'Winner':<12}")
    logger.info("-" * 56)

    for d in sorted(DENSITY_LEVELS, reverse=True):
        mp = (1.0 - d) * 100.0
        a_row = summary[(summary["team"] == "Team A (BiasedMF)") & (summary["density"] == d)]
        b_row = summary[(summary["team"] == "Team B (GraphSAGE)") & (summary["density"] == d)]
        a_rmse = a_row.iloc[0]["avg_rmse"] if len(a_row) else 0.0
        b_rmse = b_row.iloc[0]["avg_rmse"] if len(b_row) else 0.0
        winner = "Team B" if b_rmse < a_rmse else "Team A"
        logger.info(f"{mp:>6.0f}%      {a_rmse:<16.4f} {b_rmse:<16.4f} {winner}")

    logger.info(f"\n{'='*60}")
    logger.info("TABLE 2: Average MAE by Sparsity Level")
    logger.info(f"{'='*60}")
    logger.info(f"{'Missing %':<12} {'Team A MAE':<16} {'Team B MAE':<16} {'Winner':<12}")
    logger.info("-" * 56)

    for d in sorted(DENSITY_LEVELS, reverse=True):
        mp = (1.0 - d) * 100.0
        a_row = summary[(summary["team"] == "Team A (BiasedMF)") & (summary["density"] == d)]
        b_row = summary[(summary["team"] == "Team B (GraphSAGE)") & (summary["density"] == d)]
        a_mae = a_row.iloc[0]["avg_mae"] if len(a_row) else 0.0
        b_mae = b_row.iloc[0]["avg_mae"] if len(b_row) else 0.0
        winner = "Team B" if b_mae < a_mae else "Team A"
        logger.info(f"{mp:>6.0f}%      {a_mae:<16.4f} {b_mae:<16.4f} {winner}")

    # 8. Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(JSON_OUTPUT)), exist_ok=True)
    json_data = {
        "density_levels": DENSITY_LEVELS,
        "results": all_results,
        "degradation_analysis": {
            k: v for k, v in analysis.items()
            if k != "relative_improvement"
        },
        "relative_improvement": analysis.get("relative_improvement", [])
    }
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"\n[OK] Full benchmark JSON saved to: {JSON_OUTPUT}")

    logger.info("\n" + "=" * 60)
    logger.info("SPARSITY ROBUSTNESS BENCHMARK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
