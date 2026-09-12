"""
CLI Experiment Runner for Team A: Sparse Matrix-Based QoS Recommendation.
Executes training and evaluation for Baselines, CF, and Matrix Factorization models.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.common.config import ConfigManager, TeamAConfig
from src.common.logger import ExperimentLogger
from src.common.dataset_loader import WSDreamLoader
from src.common.qos_normalizer import QoSNormalizer
from src.common.visualization import ScientificPlotter
from src.team_a_matrix.data_preprocessor import MatrixPreprocessor
from src.team_a_matrix.matrix_builder import QoSMatrixBuilder
from src.team_a_matrix.models.baselines import (
    GlobalMeanBaseline,
    UserMeanBaseline,
    ServiceMeanBaseline,
    UserItemMeanBaseline
)
from src.team_a_matrix.models.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.team_a_matrix.models.matrix_factorization import (
    ProbabilisticMatrixFactorization,
    BiasedMatrixFactorization
)
from src.team_a_matrix.evaluator_team_a import TeamAEvaluator


def instantiate_model(model_name: str, config: TeamAConfig):
    """Instantiates model by name with configured hyperparameters."""
    name_lower = model_name.lower()
    if name_lower == "globalmean":
        return GlobalMeanBaseline()
    elif name_lower == "usermean":
        return UserMeanBaseline()
    elif name_lower == "servicemean":
        return ServiceMeanBaseline()
    elif name_lower == "useritemmean":
        return UserItemMeanBaseline()
    elif name_lower == "userbasedcf" or name_lower == "usercf":
        return UserBasedCF(k_neighbors=config.k_neighbors)
    elif name_lower == "itembasedcf" or name_lower == "itemcf":
        return ItemBasedCF(k_neighbors=config.k_neighbors)
    elif name_lower == "pmf":
        return ProbabilisticMatrixFactorization(
            latent_dim=config.latent_dim,
            lr=config.lr,
            reg=config.reg,
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience
        )
    elif name_lower == "biasedmf":
        return BiasedMatrixFactorization(
            latent_dim=config.latent_dim,
            lr=config.lr,
            reg=config.reg,
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Team A Sparse Matrix QoS Recommendation Experiments.")
    parser.add_argument("--config", type=str, default="configs/team_a/mf.yaml", help="Path to YAML config.")
    parser.add_argument("--model", type=str, default="BiasedMF", help="Model override.")
    parser.add_argument("--density", type=float, default=None, help="Density override (e.g. 0.10).")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation.")
    args = parser.parse_args()

    # Load and resolve configuration
    overrides = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.density:
        overrides["density"] = args.density
    if args.epochs:
        overrides["epochs"] = args.epochs

    config = ConfigManager.load_team_a_config(args.config, overrides=overrides)
    logger = ExperimentLogger.setup_logger("TeamA_Experiment", log_dir="results/logs/team_a")
    logger.info(f"Initialized Team A Experiment with Model: {config.model_name}, Density: {config.density}")

    # 1. Dataset Loading
    loader = WSDreamLoader()
    if args.synthetic:
        qos_data = loader.generate_synthetic_qos_dataset(num_users=339, num_services=500, seed=42)
    else:
        qos_data = loader.load_or_generate_dataset(num_users=339, num_services=500, seed=42)

    attr_names = sorted(qos_data.keys())
    logger.info(f"Loaded QoS dataset with attributes: {attr_names}")

    # 2. Preprocessing: Masking and Splitting
    sample_attr = "response_time" if "response_time" in qos_data else attr_names[0]
    full_sample_mat = qos_data[sample_attr]

    observed_mask = MatrixPreprocessor.create_sparsity_mask(
        matrix=full_sample_mat,
        target_density=config.density,
        seed=42
    )
    train_mask, val_mask, test_mask = MatrixPreprocessor.train_val_test_split(
        matrix=full_sample_mat,
        observed_mask=observed_mask,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=42
    )

    logger.info(f"Sparsity Mask: {np.sum(observed_mask)} observed ({config.density*100:.1f}%), Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")

    # 3. Model Training per QoS Attribute
    models_dict = {}
    train_matrices = {}
    val_matrices = {}
    normalizers = {}
    test_masks = {}

    for attr in attr_names:
        mat = qos_data[attr]
        # Fit normalizer for utility conversion
        norm = QoSNormalizer(attribute=attr).fit(mat, mask=observed_mask)
        normalizers[attr] = norm
        test_masks[attr] = test_mask

        # Build sparse matrices
        csr_train = QoSMatrixBuilder.build_csr_matrix(mat, train_mask)
        csr_val = QoSMatrixBuilder.build_csr_matrix(mat, val_mask)

        train_matrices[attr] = csr_train
        val_matrices[attr] = csr_val

        model_instance = instantiate_model(config.model_name, config)
        logger.info(f"Fitting {config.model_name} on attribute [{attr}]...")
        model_instance.fit(csr_train, csr_val)
        models_dict[attr] = model_instance

    # 4. Comprehensive Evaluation
    evaluator = TeamAEvaluator(k_values=config.top_k_list)
    results = evaluator.run_full_evaluation(
        models_dict=models_dict,
        true_matrices=qos_data,
        normalizers=normalizers,
        test_masks=test_masks
    )

    logger.info("=== Team A Prediction Metrics ===")
    for attr, m in results["prediction"].items():
        logger.info(f"[{attr}] RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f}")

    logger.info("=== Team A Top-K Recommendation Metrics ===")
    rec = results["recommendation"]
    for k in config.top_k_list:
        logger.info(f"Top-{k} -> Precision: {rec['precision'][k]:.4f} | Recall: {rec['recall'][k]:.4f} | NDCG: {rec['ndcg'][k]:.4f}")

    # 5. Save Artifacts
    os.makedirs("results/metrics", exist_ok=True)
    out_file = "results/metrics/team_a_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {out_file}")

    # 6. Generate Figures
    os.makedirs("results/figures", exist_ok=True)
    # Scatter plot on first attribute
    first_attr = attr_names[0]
    pred_first = models_dict[first_attr].predict_matrix()
    test_coords = np.argwhere(test_mask)
    t_true = qos_data[first_attr][test_coords[:, 0], test_coords[:, 1]]
    p_pred = pred_first[test_coords[:, 0], test_coords[:, 1]]
    ScientificPlotter.plot_actual_vs_predicted(
        y_true=t_true,
        y_pred=p_pred,
        save_path="results/figures/team_a_actual_vs_pred.png",
        title=f"Team A ({config.model_name}) - {first_attr.title()}"
    )

    # Top-K curves
    ScientificPlotter.plot_topk_metrics(
        metrics_by_model={config.model_name: results["recommendation"]},
        save_path="results/figures/team_a_topk_curves.png"
    )
    logger.info("Generated visualization figures in results/figures/")


if __name__ == "__main__":
    main()
