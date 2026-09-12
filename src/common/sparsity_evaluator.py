"""
Sparsity Robustness Evaluator.
Aggregates per-attribute RMSE/MAE results across multiple sparsity levels
and produces publication-quality comparison tables and plots.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SparsityEvaluator:
    """Aggregates and visualizes prediction accuracy across sparsity levels."""

    def __init__(self, qos_attributes: List[str]):
        self.qos_attributes = qos_attributes
        self.records: List[Dict] = []

    def add_result(
        self,
        density: float,
        team: str,
        attribute: str,
        rmse: float,
        mae: float
    ) -> None:
        """Records a single evaluation result."""
        self.records.append({
            "density": density,
            "missing_pct": (1.0 - density) * 100.0,
            "team": team,
            "attribute": attribute,
            "rmse": rmse,
            "mae": mae
        })

    def build_dataframe(self) -> pd.DataFrame:
        """Returns the full results as a DataFrame."""
        return pd.DataFrame(self.records)

    def build_summary_table(self) -> pd.DataFrame:
        """Builds an aggregated summary table (average RMSE / MAE across all attributes)."""
        df = self.build_dataframe()
        summary = df.groupby(["density", "missing_pct", "team"]).agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean")
        ).reset_index()
        return summary

    def save_csv(self, output_path: str) -> str:
        """Saves full per-attribute results to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = self.build_dataframe()
        df.to_csv(output_path, index=False)
        return output_path

    def compute_degradation_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Computes degradation statistics:
        - RMSE increase from densest to sparsest level
        - MAE increase from densest to sparsest level
        - Relative improvement of Team B over Team A at each level
        """
        summary = self.build_summary_table()
        teams = summary["team"].unique()
        densities = sorted(summary["density"].unique(), reverse=True)

        analysis = {}
        for team in teams:
            team_data = summary[summary["team"] == team].sort_values("density", ascending=False)
            if len(team_data) < 2:
                continue

            densest = team_data.iloc[0]
            sparsest = team_data.iloc[-1]

            rmse_increase = ((sparsest["avg_rmse"] - densest["avg_rmse"]) / densest["avg_rmse"]) * 100.0
            mae_increase = ((sparsest["avg_mae"] - densest["avg_mae"]) / densest["avg_mae"]) * 100.0

            analysis[team] = {
                "densest_density": densest["density"],
                "sparsest_density": sparsest["density"],
                "rmse_at_densest": densest["avg_rmse"],
                "rmse_at_sparsest": sparsest["avg_rmse"],
                "rmse_increase_pct": rmse_increase,
                "mae_at_densest": densest["avg_mae"],
                "mae_at_sparsest": sparsest["avg_mae"],
                "mae_increase_pct": mae_increase
            }

        # Relative improvement at each density level
        if "Team A (BiasedMF)" in teams and "Team B (GraphSAGE)" in teams:
            improvement = []
            for d in densities:
                a_row = summary[(summary["team"] == "Team A (BiasedMF)") & (summary["density"] == d)]
                b_row = summary[(summary["team"] == "Team B (GraphSAGE)") & (summary["density"] == d)]
                if len(a_row) == 1 and len(b_row) == 1:
                    a_rmse = a_row.iloc[0]["avg_rmse"]
                    b_rmse = b_row.iloc[0]["avg_rmse"]
                    improvement.append({
                        "density": d,
                        "missing_pct": (1.0 - d) * 100.0,
                        "team_a_rmse": a_rmse,
                        "team_b_rmse": b_rmse,
                        "improvement_pct": ((a_rmse - b_rmse) / a_rmse) * 100.0
                    })
            analysis["relative_improvement"] = improvement

        return analysis

    def plot_rmse_comparison(self, save_path: str) -> str:
        """Plots Average RMSE vs Missing Ratio for both teams."""
        summary = self.build_summary_table()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self._apply_styles()

        fig, ax = plt.subplots(figsize=(8, 5.5))

        team_colors = {
            "Team A (BiasedMF)": ("#1565C0", "o", "--"),
            "Team B (GraphSAGE)": ("#D32F2F", "s", "-")
        }

        for team, (color, marker, ls) in team_colors.items():
            data = summary[summary["team"] == team].sort_values("missing_pct")
            ax.plot(
                data["missing_pct"], data["avg_rmse"],
                color=color, marker=marker, linewidth=2.2, markersize=8,
                linestyle=ls, label=team, markeredgecolor="white", markeredgewidth=1.2
            )

        ax.set_xlabel("Missing Ratio (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average RMSE (across 5 QoS attributes)", fontsize=12, fontweight="bold")
        ax.set_title("Sparsity Robustness: RMSE vs Missing Ratio", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_mae_comparison(self, save_path: str) -> str:
        """Plots Average MAE vs Missing Ratio for both teams."""
        summary = self.build_summary_table()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self._apply_styles()

        fig, ax = plt.subplots(figsize=(8, 5.5))

        team_colors = {
            "Team A (BiasedMF)": ("#1565C0", "o", "--"),
            "Team B (GraphSAGE)": ("#D32F2F", "s", "-")
        }

        for team, (color, marker, ls) in team_colors.items():
            data = summary[summary["team"] == team].sort_values("missing_pct")
            ax.plot(
                data["missing_pct"], data["avg_mae"],
                color=color, marker=marker, linewidth=2.2, markersize=8,
                linestyle=ls, label=team, markeredgecolor="white", markeredgewidth=1.2
            )

        ax.set_xlabel("Missing Ratio (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average MAE (across 5 QoS attributes)", fontsize=12, fontweight="bold")
        ax.set_title("Sparsity Robustness: MAE vs Missing Ratio", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_per_attribute_rmse(self, save_path: str) -> str:
        """Plots per-attribute RMSE vs Missing Ratio (one subplot per attribute)."""
        df = self.build_dataframe()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self._apply_styles()

        n_attrs = len(self.qos_attributes)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        team_colors = {
            "Team A (BiasedMF)": ("#1565C0", "o", "--"),
            "Team B (GraphSAGE)": ("#D32F2F", "s", "-")
        }

        for i, attr in enumerate(self.qos_attributes):
            ax = axes[i]
            attr_data = df[df["attribute"] == attr]

            for team, (color, marker, ls) in team_colors.items():
                team_attr = attr_data[attr_data["team"] == team].sort_values("missing_pct")
                ax.plot(
                    team_attr["missing_pct"], team_attr["rmse"],
                    color=color, marker=marker, linewidth=2, markersize=7,
                    linestyle=ls, label=team, markeredgecolor="white", markeredgewidth=1
                )

            ax.set_title(attr.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.set_xlabel("Missing (%)", fontsize=10)
            ax.set_ylabel("RMSE", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            if i == 0:
                ax.legend(fontsize=8.5, loc="upper left")

        # Hide unused subplot
        if n_attrs < len(axes):
            for j in range(n_attrs, len(axes)):
                axes[j].set_visible(False)

        plt.suptitle("Per-Attribute RMSE vs Sparsity Level", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    @staticmethod
    def _apply_styles():
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--"
        })
