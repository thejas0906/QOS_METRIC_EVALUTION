"""
QoS Prediction Accuracy Evaluation Module.
Evaluates and compares continuous regression accuracy (MAE, RMSE, NRMSE)
across all 5 QoS attributes between Team A (BiasedMF) and Team B (HeteroGraphSAGE).
Exports results to results/qos_prediction_accuracy.csv and generates comparison plots.
"""

import json
import os
import sys
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.common.metrics_prediction import PredictionMetricsEvaluator


class QoSPredictionAccuracyEvaluator:
    """Evaluates, compares, and visualizes continuous QoS prediction accuracy."""

    def __init__(
        self,
        team_a_results_path: str = "results/metrics/team_a_results.json",
        team_b_results_path: str = "results/metrics/team_b_results.json",
        output_csv_path: str = "results/qos_prediction_accuracy.csv",
        output_plot_path: str = "results/figures/qos_prediction_accuracy_comparison.png"
    ):
        self.team_a_path = team_a_results_path
        self.team_b_path = team_b_results_path
        self.output_csv = output_csv_path
        self.output_plot = output_plot_path

    def load_results(self) -> Dict[str, Dict[str, Any]]:
        """Loads prediction results from JSON benchmark files."""
        if not os.path.exists(self.team_a_path):
            raise FileNotFoundError(f"Team A results not found at: {self.team_a_path}")
        if not os.path.exists(self.team_b_path):
            raise FileNotFoundError(f"Team B results not found at: {self.team_b_path}")

        with open(self.team_a_path, "r", encoding="utf-8") as f:
            team_a_data = json.load(f)
        with open(self.team_b_path, "r", encoding="utf-8") as f:
            team_b_data = json.load(f)

        return {
            "team_a": team_a_data.get("prediction", {}),
            "team_b": team_b_data.get("prediction", {})
        }

    def generate_comparison_dataframe(self) -> pd.DataFrame:
        """Constructs a structured comparison DataFrame."""
        data = self.load_results()
        a_preds = data["team_a"]
        b_preds = data["team_b"]

        attributes = sorted(set(list(a_preds.keys()) + list(b_preds.keys())))
        rows = []

        for attr in attributes:
            a_metrics = a_preds.get(attr, {})
            b_metrics = b_preds.get(attr, {})

            a_rmse = a_metrics.get("rmse", np.nan)
            a_mae = a_metrics.get("mae", np.nan)
            b_rmse = b_metrics.get("rmse", np.nan)
            b_mae = b_metrics.get("mae", np.nan)

            best_rmse = "Team B" if b_rmse <= a_rmse else "Team A"
            best_mae = "Team B" if b_mae <= a_mae else "Team A"

            rows.append({
                "QoS_Attribute": attr,
                "Team_A_BiasedMF_RMSE": a_rmse,
                "Team_B_GraphSAGE_RMSE": b_rmse,
                "Best_RMSE": best_rmse,
                "RMSE_Improvement_Pct": ((a_rmse - b_rmse) / a_rmse * 100.0) if a_rmse > 0 else 0.0,
                "Team_A_BiasedMF_MAE": a_mae,
                "Team_B_GraphSAGE_MAE": b_mae,
                "Best_MAE": best_mae,
                "MAE_Improvement_Pct": ((a_mae - b_mae) / a_mae * 100.0) if a_mae > 0 else 0.0
            })

        df = pd.DataFrame(rows)
        return df

    def save_csv(self, df: Optional[pd.DataFrame] = None) -> str:
        """Saves comparison DataFrame to CSV."""
        if df is None:
            df = self.generate_comparison_dataframe()

        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        return self.output_csv

    def plot_comparison(self, df: Optional[pd.DataFrame] = None) -> str:
        """Generates visual comparison charts for MAE and RMSE."""
        if df is None:
            df = self.generate_comparison_dataframe()

        os.makedirs(os.path.dirname(os.path.abspath(self.output_plot)), exist_ok=True)

        # Plot 1: Bounded attributes (Availability, Reliability, Response Time)
        # Plot 2: Large-scale attributes (Latency in ms, Throughput in kbps)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Subplot 1: Normalized / Ratio scale attributes
        small_attrs = ["availability", "reliability", "response_time"]
        df_small = df[df["QoS_Attribute"].isin(small_attrs)].copy()

        x = np.arange(len(df_small))
        width = 0.20

        axes[0].bar(x - 1.5 * width, df_small["Team_A_BiasedMF_RMSE"], width, label="Team A (BiasedMF) RMSE", color="#90CAF9", edgecolor="#1565C0")
        axes[0].bar(x - 0.5 * width, df_small["Team_B_GraphSAGE_RMSE"], width, label="Team B (GraphSAGE) RMSE", color="#1565C0")
        axes[0].bar(x + 0.5 * width, df_small["Team_A_BiasedMF_MAE"], width, label="Team A (BiasedMF) MAE", color="#A5D6A7", edgecolor="#2E7D32")
        axes[0].bar(x + 1.5 * width, df_small["Team_B_GraphSAGE_MAE"], width, label="Team B (GraphSAGE) MAE", color="#2E7D32")

        axes[0].set_xticks(x)
        axes[0].set_xticklabels([a.replace('_', '\n').title() for a in df_small["QoS_Attribute"]], fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Error (Attribute Units)", fontsize=11, fontweight="bold")
        axes[0].set_title("Prediction Error: Small-Scale Attributes\n(Availability, Reliability, Response Time)", fontsize=11, fontweight="bold")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
        axes[0].legend(loc="upper right", fontsize=8.5)

        # Subplot 2: Large scale attributes (Latency & Throughput)
        large_attrs = ["latency", "throughput"]
        df_large = df[df["QoS_Attribute"].isin(large_attrs)].copy()

        x2 = np.arange(len(df_large))
        axes[1].bar(x2 - 1.5 * width, df_large["Team_A_BiasedMF_RMSE"], width, label="Team A RMSE", color="#90CAF9", edgecolor="#1565C0")
        axes[1].bar(x2 - 0.5 * width, df_large["Team_B_GraphSAGE_RMSE"], width, label="Team B RMSE", color="#1565C0")
        axes[1].bar(x2 + 0.5 * width, df_large["Team_A_BiasedMF_MAE"], width, label="Team A MAE", color="#A5D6A7", edgecolor="#2E7D32")
        axes[1].bar(x2 + 1.5 * width, df_large["Team_B_GraphSAGE_MAE"], width, label="Team B MAE", color="#2E7D32")

        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(["Latency\n(ms)", "Throughput\n(kbps)"], fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Error (Attribute Units)", fontsize=11, fontweight="bold")
        axes[1].set_title("Prediction Error: Large-Scale Attributes\n(Latency & Throughput)", fontsize=11, fontweight="bold")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        axes[1].legend(loc="upper right", fontsize=8.5)

        plt.suptitle("QoS Prediction Accuracy Comparison: Team A (BiasedMF) vs Team B (HeteroGraphSAGE)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_plot, dpi=300, bbox_inches="tight")
        plt.close()

        return self.output_plot

    def run_evaluation(self) -> pd.DataFrame:
        """Executes complete evaluation, writes CSV, and saves plots."""
        df = self.generate_comparison_dataframe()
        csv_file = self.save_csv(df)
        plot_file = self.plot_comparison(df)
        return df


def main():
    evaluator = QoSPredictionAccuracyEvaluator()
    df = evaluator.run_evaluation()
    print("=== QoS Prediction Accuracy Evaluation ===")
    print(df.to_string(index=False))
    print(f"\n[OK] Results saved to: {evaluator.output_csv}")
    print(f"[OK] Comparison plot saved to: {evaluator.output_plot}")


if __name__ == "__main__":
    main()
