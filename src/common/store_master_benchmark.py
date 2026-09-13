"""
Master Benchmark Aggregator.
Consolidates all benchmark experiments into:
1. results/MASTER_BENCHMARK_RESULTS.json
2. results/MASTER_BENCHMARK_REPORT.md
"""

import json
import os
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
METRICS_DIR = "results/metrics"

def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def df_to_markdown(df):
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = [
        "| " + " | ".join([str(val) for val in row]) + " |"
        for row in df.values
    ]
    return "\n".join([header_line, sep_line] + data_lines)

def main():
    print("Aggregating all benchmark results...")

    team_a = load_json_safe(os.path.join(METRICS_DIR, "team_a_results.json"))
    team_b = load_json_safe(os.path.join(METRICS_DIR, "team_b_results.json"))
    sparsity_json = load_json_safe(os.path.join(METRICS_DIR, "sparsity_benchmark.json"))
    confidence_json = load_json_safe(os.path.join(METRICS_DIR, "confidence_comparison.json"))
    sensitivity_csv = load_csv_safe(os.path.join(METRICS_DIR, "sensitivity_alpha_beta.csv"))
    qos_acc_csv = load_csv_safe(os.path.join(RESULTS_DIR, "qos_prediction_accuracy.csv"))
    sparsity_csv = load_csv_safe(os.path.join(RESULTS_DIR, "sparsity_benchmark.csv"))
    confidence_csv = load_csv_safe(os.path.join(METRICS_DIR, "confidence_comparison.csv"))
    comparative_csv = load_csv_safe(os.path.join(METRICS_DIR, "comparative_summary.csv"))

    master_dict = {
        "benchmark_metadata": {
            "dataset": "WS-DREAM #1 (339 users x 5825 web services)",
            "models_evaluated": ["Team A (BiasedMF)", "Team B (HeteroGraphSAGE)"],
            "benchmarks_included": [
                "QoS Prediction Accuracy (MAE/RMSE)",
                "Top-K Recommendation Quality (Precision, Recall, NDCG)",
                "Cost-Performance-Confidence Tradeoff (Paradigms A, B, C)",
                "Sparsity Robustness (50% to 95% missing ratio)",
                "Cold-Start Inductive Generalization (Users and Services)",
                "Cost Sensitivity & Pareto Sweep"
            ]
        },
        "qos_prediction_accuracy": qos_acc_csv.to_dict(orient="records") if qos_acc_csv is not None else {},
        "top_k_recommendation": {
            "team_a": team_a.get("recommendation", {}) if team_a else {},
            "team_b": team_b.get("recommendation", {}) if team_b else {}
        },
        "economic_and_confidence": team_b.get("economic", {}) if team_b else {},
        "cold_start": team_b.get("cold_start", {}) if team_b else {},
        "three_paradigm_tradeoff": confidence_json if confidence_json else {},
        "sparsity_robustness": sparsity_json if sparsity_json else {},
        "sensitivity_sweep": sensitivity_csv.to_dict(orient="records") if sensitivity_csv is not None else {}
    }

    # 1. Save master JSON
    master_json_path = os.path.join(RESULTS_DIR, "MASTER_BENCHMARK_RESULTS.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_dict, f, indent=2, default=str)
    print(f"[OK] Master JSON saved to: {master_json_path}")

    # 2. Build Markdown Report
    report = []
    report.append("# MASTER BENCHMARK EVALUATION REPORT")
    report.append("## QoS-Aware Web Service Recommendation System")
    report.append("**Dataset**: WS-DREAM Benchmark (339 Users × 5,825 Web Services)\n")

    report.append("---")
    report.append("## 1. QoS Continuous Prediction Accuracy")
    report.append("Continuous regression performance (RMSE and MAE) across all 5 QoS dimensions:\n")
    if qos_acc_csv is not None:
        report.append(df_to_markdown(qos_acc_csv))
    report.append("\n")

    report.append("---")
    report.append("## 2. Top-K Recommendation Quality Ranking")
    report.append("Evaluation on held-out test interactions comparing Team A (QoS-Only Ranking) vs Team B (Cost-Performance Ranking):\n")
    report.append("| Cutoff (K) | Metric | Team A (BiasedMF) | Team B (GraphSAGE) | Winner |")
    report.append("| :---: | :---: | :---: | :---: | :---: |")
    if team_a and team_b:
        for k in ["5", "10", "20"]:
            p_a = team_a["recommendation"]["precision"][k]
            p_b = team_b["recommendation"]["precision"][k]
            r_a = team_a["recommendation"]["recall"][k]
            r_b = team_b["recommendation"]["recall"][k]
            n_a = team_a["recommendation"]["ndcg"][k]
            n_b = team_b["recommendation"]["ndcg"][k]

            report.append(f"| Top-{k} | Precision@{k} | {p_a:.4f} | {p_b:.4f} | {'Team B' if p_b > p_a else 'Team A'} |")
            report.append(f"| Top-{k} | Recall@{k} | {r_a:.4f} | {r_b:.4f} | {'Team B' if r_b > r_a else 'Team A'} |")
            report.append(f"| Top-{k} | NDCG@{k} | {n_a:.4f} | {n_b:.4f} | {'Team B' if n_b > n_a else 'Team A'} |")
    report.append("\n")

    report.append("---")
    report.append("## 3. Cost-Performance-Confidence Paradigm Comparison")
    report.append("Evaluation across Paradigm A (QoS-Only), Paradigm B (QoS+Cost), and Paradigm C (QoS+Cost+Confidence):\n")
    if confidence_csv is not None:
        report.append(df_to_markdown(confidence_csv))
    report.append("\n")

    report.append("---")
    report.append("## 4. Sparsity Robustness Benchmark")
    report.append("Degradation under extreme missing QoS observations (50% to 95% missing ratio) on a fixed held-out test set:\n")
    if sparsity_json and "results" in sparsity_json:
        report.append("| Missing % | Observed % | Team A RMSE | Team B RMSE | Team A MAE | Team B MAE | Winner |")
        report.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for d_str in ["0.50", "0.30", "0.20", "0.10", "0.05"]:
            if d_str in sparsity_json["results"]:
                res = sparsity_json["results"][d_str]
                mp = res["missing_pct"]
                obs = res["density"] * 100
                a_rmse = np.mean([v["rmse"] for v in res["team_a"].values()])
                b_rmse = np.mean([v["rmse"] for v in res["team_b"].values()])
                a_mae = np.mean([v["mae"] for v in res["team_a"].values()])
                b_mae = np.mean([v["mae"] for v in res["team_b"].values()])
                report.append(f"| {mp:.0f}% | {obs:.0f}% | {a_rmse:.4f} | {b_rmse:.4f} | {a_mae:.4f} | {b_mae:.4f} | {'Team B' if b_rmse < a_rmse else 'Team A'} |")
    report.append("\n")

    report.append("---")
    report.append("## 5. Cold-Start Inductive Robustness")
    report.append("Performance on completely unseen cold users and services without graph retraining:\n")
    if team_b and "cold_start" in team_b:
        cs = team_b["cold_start"]
        u_cs = cs.get("cold_users", {})
        s_cs = cs.get("cold_services", {})
        report.append(f"- **Cold Users Evaluation** (N={u_cs.get('cold_count', 0)}):")
        report.append(f"  - Precision@10: {u_cs.get('precision', {}).get('10', 0.0):.4f}")
        report.append(f"  - Recall@10: {u_cs.get('recall', {}).get('10', 0.0):.4f}")
        report.append(f"  - NDCG@10: {u_cs.get('ndcg', {}).get('10', 0.0):.4f}")
        report.append(f"- **Cold Services Evaluation** (N={s_cs.get('cold_count', 0)}):")
        report.append(f"  - Precision@10: {s_cs.get('precision', {}).get('10', 0.0):.4f}")
        report.append(f"  - Recall@10: {s_cs.get('recall', {}).get('10', 0.0):.4f}")
        report.append(f"  - NDCG@10: {s_cs.get('ndcg', {}).get('10', 0.0):.4f}")
    report.append("\n")

    report.append("---")
    report.append("## 6. Artifact Registry")
    report.append("| Benchmark Category | File / Artifact Path | Description |")
    report.append("| :--- | :--- | :--- |")
    report.append("| **Master Summary** | [results/MASTER_BENCHMARK_RESULTS.json](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/MASTER_BENCHMARK_RESULTS.json) | Central JSON containing all metric records |")
    report.append("| **QoS Prediction Accuracy** | [results/qos_prediction_accuracy.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/qos_prediction_accuracy.csv) | Per-attribute RMSE/MAE comparison |")
    report.append("| **Recommendation Quality** | [results/metrics/comparative_summary.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/comparative_summary.csv) | Top-K precision, recall, and NDCG |")
    report.append("| **Confidence Trade-off** | [results/metrics/confidence_comparison.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/confidence_comparison.csv) | Paradigms A, B, and C comparison |")
    report.append("| **Sparsity Robustness** | [results/sparsity_benchmark.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/sparsity_benchmark.csv) | 50% to 95% missing observation results |")
    report.append("| **LaTeX Tables** | [results/metrics/latex_tables.tex](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/latex_tables.tex) | Ready-to-paste tables for publication |")
    report.append("| **Top-K Ranking Plot** | [results/figures/comparative_topk_curves.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/comparative_topk_curves.png) | High-res curves for Top-K rankings |")
    report.append("| **Confidence Plot** | [results/figures/confidence_paradigm_comparison.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/confidence_paradigm_comparison.png) | Trade-off visualization across paradigms |")
    report.append("| **Sparsity Plot** | [results/figures/sparsity_rmse_comparison.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/sparsity_rmse_comparison.png) | RMSE degradation curves |")

    report_content = "\n".join(report)
    master_report_path = os.path.join(RESULTS_DIR, "MASTER_BENCHMARK_REPORT.md")
    with open(master_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[OK] Master Report saved to: {master_report_path}")

if __name__ == "__main__":
    main()
