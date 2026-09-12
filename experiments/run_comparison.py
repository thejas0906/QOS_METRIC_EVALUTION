"""
Cross-Paradigm Comparative Analysis Runner.
Directly compares Team A (Matrix Factorization) against Team B (HeteroGraphSAGE + Cost-Performance).
Generates publication LaTeX tables and exportable CSV summaries.
"""

import json
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.common.logger import ExperimentLogger
from src.common.visualization import ScientificPlotter


def generate_latex_tables(team_a_data: dict, team_b_data: dict, output_path: str) -> None:
    """Generates clean, IEEE/ACM publication-ready LaTeX tables."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tex_content = []
    tex_content.append("% Auto-generated comparative benchmark tables")
    tex_content.append("\\begin{table}[htbp]")
    tex_content.append("\\centering")
    tex_content.append("\\caption{Prediction Accuracy Comparison (RMSE & MAE)}")
    tex_content.append("\\label{tab:qos_prediction_comparison}")
    tex_content.append("\\begin{tabular}{lcccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{QoS Attribute} & \\multicolumn{2}{c}{\\textbf{Team A (BiasedMF)}} & \\multicolumn{2}{c}{\\textbf{Team B (GraphSAGE)}} \\\\")
    tex_content.append(" & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{MAE} \\\\")
    tex_content.append("\\hline")

    a_pred = team_a_data.get("prediction", {})
    b_pred = team_b_data.get("prediction", {})

    all_attrs = sorted(set(list(a_pred.keys()) + list(b_pred.keys())))
    for attr in all_attrs:
        a_rmse = a_pred.get(attr, {}).get("rmse", 0.0)
        a_mae = a_pred.get(attr, {}).get("mae", 0.0)
        b_rmse = b_pred.get(attr, {}).get("rmse", 0.0)
        b_mae = b_pred.get(attr, {}).get("mae", 0.0)

        # Bold superior result
        s_b_rmse = f"\\textbf{{{b_rmse:.4f}}}" if b_rmse <= a_rmse else f"{b_rmse:.4f}"
        s_a_rmse = f"\\textbf{{{a_rmse:.4f}}}" if a_rmse < b_rmse else f"{a_rmse:.4f}"
        s_b_mae = f"\\textbf{{{b_mae:.4f}}}" if b_mae <= a_mae else f"{b_mae:.4f}"
        s_a_mae = f"\\textbf{{{a_mae:.4f}}}" if a_mae < b_mae else f"{a_mae:.4f}"

        tex_content.append(f"{attr.replace('_', ' ').title()} & {s_a_rmse} & {s_a_mae} & {s_b_rmse} & {s_b_mae} \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table}\n")

    # Table 2: Recommendation Top-K
    tex_content.append("\\begin{table}[htbp]")
    tex_content.append("\\centering")
    tex_content.append("\\caption{Recommendation Quality Comparison (Top-$K$ Ranking)}")
    tex_content.append("\\label{tab:recommendation_comparison}")
    tex_content.append("\\begin{tabular}{lcccccc}")
    tex_content.append("\\hline")
    tex_content.append("\\textbf{Cutoff $K$} & \\multicolumn{3}{c}{\\textbf{Team A (QoS-Only Ranking)}} & \\multicolumn{3}{c}{\\textbf{Team B (Cost-Performance Ranking)}} \\\\")
    tex_content.append(" & \\textbf{Prec@$K$} & \\textbf{Rec@$K$} & \\textbf{NDCG@$K$} & \\textbf{Prec@$K$} & \\textbf{Rec@$K$} & \\textbf{NDCG@$K$} \\\\")
    tex_content.append("\\hline")

    a_rec = team_a_data.get("recommendation", {})
    b_rec = team_b_data.get("recommendation", {})

    k_list = [5, 10, 20]
    for k in k_list:
        k_str = str(k)
        ap = a_rec.get("precision", {}).get(k_str, a_rec.get("precision", {}).get(k, 0.0))
        ar = a_rec.get("recall", {}).get(k_str, a_rec.get("recall", {}).get(k, 0.0))
        an = a_rec.get("ndcg", {}).get(k_str, a_rec.get("ndcg", {}).get(k, 0.0))

        bp = b_rec.get("precision", {}).get(k_str, b_rec.get("precision", {}).get(k, 0.0))
        br = b_rec.get("recall", {}).get(k_str, b_rec.get("recall", {}).get(k, 0.0))
        bn = b_rec.get("ndcg", {}).get(k_str, b_rec.get("ndcg", {}).get(k, 0.0))

        tex_content.append(f"Top-{k} & {ap:.4f} & {ar:.4f} & {an:.4f} & {bp:.4f} & {br:.4f} & {bn:.4f} \\\\")

    tex_content.append("\\hline")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_content))


def main():
    logger = ExperimentLogger.setup_logger("CrossParadigmComparison")
    logger.info("Starting Cross-Paradigm Comparison (Team A vs Team B)...")

    a_path = "results/metrics/team_a_results.json"
    b_path = "results/metrics/team_b_results.json"

    if not os.path.exists(a_path) or not os.path.exists(b_path):
        logger.warning("Results JSON files not found. Ensure both run_team_a.py and run_team_b.py have been executed.")
        return

    with open(a_path, "r", encoding="utf-8") as f:
        team_a_data = json.load(f)
    with open(b_path, "r", encoding="utf-8") as f:
        team_b_data = json.load(f)

    # 1. Export LaTeX tables
    latex_path = "results/metrics/latex_tables.tex"
    generate_latex_tables(team_a_data, team_b_data, latex_path)
    logger.info(f"Generated LaTeX publication tables at {latex_path}")

    # 2. Build summary CSV
    rows = []
    # Prediction rows
    a_pred = team_a_data.get("prediction", {})
    b_pred = team_b_data.get("prediction", {})
    for attr in sorted(set(list(a_pred.keys()) + list(b_pred.keys()))):
        rows.append({
            "Category": "Prediction",
            "Metric": f"RMSE ({attr})",
            "Team A (BiasedMF)": a_pred.get(attr, {}).get("rmse", None),
            "Team B (GraphSAGE)": b_pred.get(attr, {}).get("rmse", None),
        })
        rows.append({
            "Category": "Prediction",
            "Metric": f"MAE ({attr})",
            "Team A (BiasedMF)": a_pred.get(attr, {}).get("mae", None),
            "Team B (GraphSAGE)": b_pred.get(attr, {}).get("mae", None),
        })

    # Recommendation rows
    for k in [5, 10, 20]:
        k_str = str(k)
        rows.append({
            "Category": "Recommendation",
            "Metric": f"Precision@{k}",
            "Team A (BiasedMF)": team_a_data.get("recommendation", {}).get("precision", {}).get(k_str, None),
            "Team B (GraphSAGE)": team_b_data.get("recommendation", {}).get("precision", {}).get(k_str, None),
        })
        rows.append({
            "Category": "Recommendation",
            "Metric": f"Recall@{k}",
            "Team A (BiasedMF)": team_a_data.get("recommendation", {}).get("recall", {}).get(k_str, None),
            "Team B (GraphSAGE)": team_b_data.get("recommendation", {}).get("recall", {}).get(k_str, None),
        })
        rows.append({
            "Category": "Recommendation",
            "Metric": f"NDCG@{k}",
            "Team A (BiasedMF)": team_a_data.get("recommendation", {}).get("ndcg", {}).get(k_str, None),
            "Team B (GraphSAGE)": team_b_data.get("recommendation", {}).get("ndcg", {}).get(k_str, None),
        })

    # Economic rows
    econ = team_b_data.get("economic", {})
    rows.append({
        "Category": "Economic",
        "Metric": "Average Recommended Cost ($)",
        "Team A (BiasedMF)": econ.get("baseline_qos_cost", None),
        "Team B (GraphSAGE)": econ.get("average_cost", None),
    })
    rows.append({
        "Category": "Economic",
        "Metric": "Cost Savings (%)",
        "Team A (BiasedMF)": 0.0,
        "Team B (GraphSAGE)": econ.get("cost_savings_pct", None),
    })
    rows.append({
        "Category": "Economic",
        "Metric": "Cost-Performance Ratio",
        "Team A (BiasedMF)": None,
        "Team B (GraphSAGE)": econ.get("cost_performance_ratio", None),
    })

    df = pd.DataFrame(rows)
    csv_path = "results/metrics/comparative_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated comparative CSV summary at {csv_path}")

    # 3. Combined Comparative Figure
    ScientificPlotter.plot_topk_metrics(
        metrics_by_model={
            "Team A (BiasedMF)": {
                m: {int(k): v for k, v in team_a_data["recommendation"][m].items()}
                for m in ["precision", "recall", "ndcg"]
            },
            "Team B (HeteroGraphSAGE)": {
                m: {int(k): v for k, v in team_b_data["recommendation"][m].items()}
                for m in ["precision", "recall", "ndcg"]
            }
        },
        save_path="results/figures/comparative_topk_curves.png"
    )
    logger.info("Saved comparative figures to results/figures/comparative_topk_curves.png")


if __name__ == "__main__":
    main()
