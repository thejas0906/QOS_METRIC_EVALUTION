"""
Academic Visualization Engine.
Generates publication-quality charts (300 DPI PNG and vector PDF) formatted for ACM/IEEE papers.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


class ScientificPlotter:
    """Creates standardized scientific figures for QoS recommendation benchmarks."""

    @staticmethod
    def _apply_styles():
        """Applies clean academic styling without external heavy themes."""
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--"
        })

    @classmethod
    def plot_actual_vs_predicted(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
        title: str = "Actual vs. Predicted QoS",
        sample_size: int = 500
    ) -> None:
        """Plots scatter plot with identity line comparing predicted vs actual QoS values."""
        cls._apply_styles()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Sample for cleaner visualization
        indices = np.random.choice(len(y_true), min(len(y_true), sample_size), replace=False)
        t_sample = y_true[indices]
        p_sample = y_pred[indices]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(t_sample, p_sample, alpha=0.5, color="#1f77b4", edgecolors="none", s=25, label="Predictions")

        min_val = min(float(np.min(t_sample)), float(np.min(p_sample)))
        max_val = max(float(np.max(t_sample)), float(np.max(p_sample)))
        ax.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=1.5, label="Ideal")

        ax.set_xlabel("Ground Truth QoS")
        ax.set_ylabel("Predicted QoS")
        ax.set_title(title)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @classmethod
    def plot_topk_metrics(
        cls,
        metrics_by_model: Dict[str, Dict[str, Dict[int, float]]],
        save_path: str
    ) -> None:
        """
        Plots comparative line charts for Precision@K, Recall@K, NDCG@K across models.
        
        Args:
            metrics_by_model: Dict of {model_name: {"precision": {k: val}, "recall": {...}, "ndcg": {...}}}
        """
        cls._apply_styles()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        metrics = ["precision", "recall", "ndcg"]
        titles = ["Precision@K", "Recall@K", "NDCG@K"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        markers = ["o", "s", "^", "D", "v", "p"]

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            for m_idx, (model_name, model_metrics) in enumerate(metrics_by_model.items()):
                if metric in model_metrics:
                    k_vals = sorted(model_metrics[metric].keys())
                    scores = [model_metrics[metric][k] for k in k_vals]
                    marker = markers[m_idx % len(markers)]
                    ax.plot(k_vals, scores, marker=marker, linewidth=2, label=model_name)

            ax.set_xlabel("Cutoff (K)")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(k_vals)
            ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @classmethod
    def plot_cost_performance_tradeoff(
        cls,
        alpha_list: List[float],
        qos_scores: List[float],
        costs: List[float],
        utilities: List[float],
        save_path: str
    ) -> None:
        """Plots Pareto frontier demonstrating QoS vs Cost trade-off as alpha varies."""
        cls._apply_styles()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(7, 5))

        color1 = "#1f77b4"
        color2 = "#2ca02c"
        color3 = "#ff7f0e"

        ax1.set_xlabel(r"Tradeoff Coefficient $\alpha$ (QoS Weight)")
        ax1.set_ylabel("Composite QoS Score / Utility", color=color1)
        line1 = ax1.plot(alpha_list, qos_scores, color=color1, marker="o", label="QoS Score")
        line2 = ax1.plot(alpha_list, utilities, color=color3, marker="^", linestyle="--", label="Overall Utility")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Service Cost", color=color2)
        line3 = ax2.plot(alpha_list, costs, color=color2, marker="s", linestyle="-.", label="Avg Cost")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center left")

        plt.title(r"Cost-Performance Pareto Frontier ($\alpha \cdot QoS - \beta \cdot Cost$)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @classmethod
    def plot_cold_start_comparison(
        cls,
        warm_metrics: Dict[str, float],
        cold_user_metrics: Dict[str, float],
        cold_service_metrics: Dict[str, float],
        save_path: str
    ) -> None:
        """Comparative grouped bar chart for Warm vs Unseen Users vs Unseen Services."""
        cls._apply_styles()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        metric_keys = list(warm_metrics.keys())
        x = np.arange(len(metric_keys))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, [warm_metrics[m] for m in metric_keys], width, label="Warm (Transductive)", color="#4c72b0")
        ax.bar(x, [cold_user_metrics[m] for m in metric_keys], width, label="Cold Users (Inductive)", color="#dd8452")
        ax.bar(x + width, [cold_service_metrics[m] for m in metric_keys], width, label="Cold Services (Inductive)", color="#55a868")

        ax.set_ylabel("Score")
        ax.set_title("Cold-Start Robustness Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_keys)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
