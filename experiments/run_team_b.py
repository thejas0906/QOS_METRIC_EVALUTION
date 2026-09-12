"""
CLI Experiment Runner for Team B: GNN + Cost-Performance QoS Recommendation.
Trains HeteroGraphSAGE with multi-task edge QoS decoder and evaluates Cost-Performance utility.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.common.config import ConfigManager, TeamBConfig
from src.common.logger import ExperimentLogger
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.common.visualization import ScientificPlotter
from src.team_b_gnn.cost_generator import ServiceCostEngine
from src.team_b_gnn.data_preprocessor import GraphDataPreprocessor
from src.team_b_gnn.graph_builder import BipartiteQoSGraphBuilder
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.evaluator_team_b import TeamBEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run Team B GNN + Cost-Performance QoS Recommendation.")
    parser.add_argument("--config", type=str, default="configs/team_b/graphsage.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--alpha", type=float, default=None, help="Utility alpha (QoS weight).")
    parser.add_argument("--beta", type=float, default=None, help="Utility beta (Cost penalty).")
    parser.add_argument("--gamma", type=float, default=None, help="Utility gamma (Confidence weight).")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation using existing checkpoint.")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.lr:
        overrides["lr"] = args.lr
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.beta is not None:
        overrides["beta"] = args.beta
    if args.gamma is not None:
        overrides["gamma"] = args.gamma

    config = ConfigManager.load_team_b_config(args.config, overrides=overrides)
    logger = ExperimentLogger.setup_logger("TeamB_Experiment", log_dir="results/logs/team_b")
    logger.info(f"Initialized Team B Experiment: {config.model_name}, alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}")

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Dataset Loading
    loader = WSDreamLoader()
    if args.synthetic:
        qos_data = loader.generate_synthetic_qos_dataset(num_users=339, num_services=500, seed=42)
    else:
        qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=42)

    attr_names = sorted(qos_data.keys())
    first_mat = qos_data[attr_names[0]]
    num_users, num_services = first_mat.shape

    # 2. Service Cost Generation
    # Mean service quality profile in [0, 1]
    normalizers = {attr: QoSNormalizer(attribute=attr).fit(qos_data[attr]) for attr in attr_names}
    utility_engine = CostPerformanceUtilityEngine(alpha=config.alpha, beta=config.beta)
    composite_true_all = utility_engine.aggregate_composite_qos(qos_data, normalizers)
    service_quality_profiles = np.mean(composite_true_all, axis=0)

    service_costs = ServiceCostEngine.generate_correlated_costs(
        service_qos_profiles=service_quality_profiles,
        correlation_strength=0.75,
        base_cost_range=(config.cost_min, config.cost_max),
        distribution=config.cost_distribution,
        seed=42
    )
    ServiceCostEngine.save_cost_table(service_costs, "data/synthetic/service_costs.csv")
    logger.info(f"Generated service costs: range [${np.min(service_costs):.2f}, ${np.max(service_costs):.2f}], mean=${np.mean(service_costs):.2f}")

    # 3. Inductive Node Partitioning & Edge Splitting
    node_splits = GraphDataPreprocessor.create_inductive_node_split(
        num_users=num_users,
        num_services=num_services,
        cold_user_ratio=config.cold_user_ratio,
        cold_service_ratio=config.cold_service_ratio,
        seed=42
    )
    warm_users = node_splits["warm_users"]
    warm_services = node_splits["warm_services"]
    cold_users = node_splits["cold_users"]
    cold_services = node_splits["cold_services"]

    # Graph observed edges (warm submatrix only for training)
    warm_mask = np.zeros((num_users, num_services), dtype=bool)
    warm_grid = np.ix_(warm_users, warm_services)
    # Density mask
    sample_obs = (first_mat > 0)
    warm_mask[warm_grid] = sample_obs[warm_grid]

    # Build Graph
    graph_data = BipartiteQoSGraphBuilder.build_from_matrices(
        qos_matrices=qos_data,
        observation_mask=warm_mask,
        service_costs=service_costs,
        feature_dim=32,
        seed=42
    )

    edge_splits = GraphDataPreprocessor.split_interaction_edges(
        u_indices=graph_data.edge_u,
        s_indices=graph_data.edge_s,
        qos_attributes=graph_data.edge_qos,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=42
    )
    train_edges = edge_splits["train"]
    val_edges = edge_splits["val"]
    test_edges = edge_splits["test"]

    logger.info(f"Graph nodes: {num_users} users, {num_services} services.")
    logger.info(f"Interaction edges - Train: {len(train_edges['u'])}, Val: {len(val_edges['u'])}, Test: {len(test_edges['u'])}")

    # 4. Model Instantiation
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
        num_qos_attributes=len(attr_names),
        hidden_dims=[64, 32],
        dropout=config.dropout
    )

    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    # Attribute scale balancing so all 5 QoS dimensions contribute equally to gradients
    qos_scales = torch.tensor([
        max(float(np.std(train_edges["qos"][:, col])), 1.0)
        for col in range(len(attr_names))
    ], dtype=torch.float32)

    def balanced_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        normalized_diff = (preds - targets) / qos_scales
        return torch.mean(normalized_diff ** 2)

    # PyTorch Tensors
    u_x_t = torch.from_numpy(graph_data.user_features).float()
    s_x_t = torch.from_numpy(graph_data.service_features).float()
    e_u_train = torch.from_numpy(train_edges["u"]).long()
    e_s_train = torch.from_numpy(train_edges["s"]).long()
    e_qos_train = torch.from_numpy(train_edges["qos"]).float()

    e_u_val = torch.from_numpy(val_edges["u"]).long()
    e_s_val = torch.from_numpy(val_edges["s"]).long()
    e_qos_val = torch.from_numpy(val_edges["qos"]).float()

    # 5. Training Loop
    if not args.eval_only:
        logger.info(f"Starting GraphSAGE training for {config.epochs} epochs...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, config.epochs + 1):
            model.train()
            decoder.train()
            optimizer.zero_grad()

            # Forward pass message passing
            h_u, h_s = model(u_x_t, s_x_t, e_u_train, e_s_train)

            # Predict QoS on training edges
            preds_train = decoder(h_u[e_u_train], h_s[e_s_train])
            train_loss = balanced_loss(preds_train, e_qos_train)

            train_loss.backward()
            # Gradient clipping for stable convergence
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), max_norm=5.0)
            optimizer.step()

            # Validation
            model.eval()
            decoder.eval()
            with torch.no_grad():
                h_u_val, h_s_val = model(u_x_t, s_x_t, e_u_train, e_s_train)
                preds_val = decoder(h_u_val[e_u_val], h_s_val[e_s_val])
                val_loss = balanced_loss(preds_val, e_qos_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save checkpoint
                os.makedirs("results/checkpoints/team_b", exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "epoch": epoch
                }, "results/checkpoints/team_b/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f}")
    else:
        logger.info("Evaluation only mode: skipping training and loading best model checkpoint.")

    # Load best checkpoint
    ckpt = torch.load("results/checkpoints/team_b/best_model.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    # 6. Evaluation
    evaluator = TeamBEvaluator(alpha=config.alpha, beta=config.beta, gamma=config.gamma, k_values=config.top_k_list)
    eval_results = evaluator.run_full_evaluation(
        model=model,
        decoder=decoder,
        user_features=graph_data.user_features,
        service_features=graph_data.service_features,
        edge_u_train=train_edges["u"],
        edge_s_train=train_edges["s"],
        test_edge_dict=test_edges,
        ground_truth_qos=qos_data,
        normalizers=normalizers,
        service_costs=service_costs,
        cold_users=cold_users,
        cold_services=cold_services
    )

    logger.info("=== Team B Multi-Attribute Prediction Results ===")
    for attr, m in eval_results["prediction"].items():
        logger.info(f"[{attr}] RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f}")

    logger.info("=== Team B Top-K Recommendation Results ===")
    rec = eval_results["recommendation"]
    for k in config.top_k_list:
        logger.info(f"Top-{k} -> Precision: {rec['precision'][k]:.4f} | Recall: {rec['recall'][k]:.4f} | NDCG: {rec['ndcg'][k]:.4f}")

    logger.info("=== Team B Economic & Reliability Evaluation ===")
    econ = eval_results["economic"]
    logger.info(f"Average Cost: ${econ['average_cost']:.4f} vs Baseline: ${econ['baseline_qos_cost']:.4f}")
    conf_str = f" | Avg Confidence: {econ['average_confidence']:.4f}" if "average_confidence" in econ else ""
    cov_str = f" | Coverage: {econ['recommendation_coverage']:.2%}" if "recommendation_coverage" in econ else ""
    logger.info(f"Cost Savings: {econ['cost_savings_pct']:.2f}% | CP Ratio: {econ['cost_performance_ratio']:.4f} | Mean Utility: {econ['mean_utility']:.4f}{conf_str}{cov_str}")

    # 7. Save Metrics JSON
    os.makedirs("results/metrics", exist_ok=True)
    out_path = "results/metrics/team_b_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Metrics saved to {out_path}")

    # 8. Publication Figures
    os.makedirs("results/figures", exist_ok=True)
    # Actual vs predicted
    test_u = test_edges["u"]
    test_s = test_edges["s"]
    model.eval()
    decoder.eval()
    with torch.no_grad():
        h_u, h_s = model(u_x_t, s_x_t, e_u_train, e_s_train)
        pred_test_edges = decoder(h_u[torch.from_numpy(test_u).long()], h_s[torch.from_numpy(test_s).long()]).numpy()

    first_attr = attr_names[0]
    ScientificPlotter.plot_actual_vs_predicted(
        y_true=test_edges["qos"][:, 0],
        y_pred=pred_test_edges[:, 0],
        save_path="results/figures/team_b_actual_vs_pred.png",
        title=f"Team B (GraphSAGE) - {first_attr.title()}"
    )

    # Recommendation curves
    ScientificPlotter.plot_topk_metrics(
        metrics_by_model={"HeteroGraphSAGE": rec},
        save_path="results/figures/team_b_topk_curves.png"
    )

    # Cold start bar chart
    if "cold_users" in eval_results["cold_start"]:
        warm_err = {attr: eval_results["prediction"][attr]["rmse"] for attr in attr_names}
        cold_u_err = {attr: eval_results["cold_start"]["cold_users"]["prediction_errors"][attr]["rmse"] for attr in attr_names}
        cold_s_err = {attr: eval_results["cold_start"]["cold_services"]["prediction_errors"][attr]["rmse"] for attr in attr_names}
        ScientificPlotter.plot_cold_start_comparison(
            warm_metrics=warm_err,
            cold_user_metrics=cold_u_err,
            cold_service_metrics=cold_s_err,
            save_path="results/figures/cold_start_robustness.png"
        )
    logger.info("Generated visualization figures in results/figures/")


if __name__ == "__main__":
    main()
