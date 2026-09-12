# QoS-Based Service Recommendation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A publication-ready, reproducible research framework for **Service-Oriented Computing (SOC)** comparing traditional **Sparse Matrix QoS Recommendation** against an inductive **Graph Neural Network (GraphSAGE)** architecture with **Cost-Performance Pareto Optimization**.

---

## 🔬 Architectural Paradigms

### Team A: Sparse Matrix-Based QoS Recommendation
- **Representation**: User-Service sparse interaction matrix $\mathbf{R} \in \mathbb{R}^{U \times S}$ under varying sparsity levels ($90\%$, $95\%$, $98\%$).
- **Techniques**:
  - Statistical baselines (Global Mean, User Mean, Service Mean, Additive User-Item Mean)
  - Memory-based Collaborative Filtering (User-Based CF and Item-Based CF with Pearson Correlation Coefficient and significance shrinkage)
  - Matrix Factorization (Probabilistic Matrix Factorization [PMF] and Biased Matrix Factorization [BiasedMF])
- **Evaluation**: RMSE, MAE, Precision@K, Recall@K, NDCG@K.

### Team B: GraphSAGE + Cost-Performance Tradeoff
- **Representation**: Bipartite interaction graph $\mathcal{G} = (\mathcal{V}_U, \mathcal{V}_S, \mathcal{E}_{US})$.
- **Techniques**:
  - Inductive Heterogeneous GraphSAGE message passing over Users and Services
  - Multi-task edge QoS regression decoder head
  - Service invocation cost modeling (Pareto & log-normal pricing distributions)
  - Parameterized Cost-Performance utility engine:
    $$\text{Utility} = \alpha \cdot \text{QoS\_Score} - \beta \cdot \text{Cost}$$
- **Evaluation**: Prediction (RMSE, MAE), Recommendation (Precision@K, Recall@K, NDCG@K), Economic (Average Cost, Cost Savings %, Cost-Performance Ratio), and Inductive Cold-Start (Unseen Users, Unseen Services).

---

## 📁 Repository Structure

```text
QOS_GNN_metric_simulation/
├── configs/
│   ├── common/             # Dataset and logging specifications
│   ├── team_a/             # Matrix Factorization and CF hyperparameters
│   └── team_b/             # GraphSAGE and tradeoff configurations
├── data/
│   ├── raw/ws_dream/       # Benchmark WS-DREAM matrices
│   ├── processed/          # Cached splits and graph data
│   └── synthetic/          # Generated service cost tables
├── experiments/
│   ├── run_team_a.py       # Execution driver for Team A
│   ├── run_team_b.py       # Execution driver for Team B
│   ├── run_comparison.py   # Cross-paradigm LaTeX table & summary generator
│   └── run_sensitivity_analysis.py # Alpha-beta sweep & Pareto curves
├── results/
│   ├── checkpoints/        # Saved model weights
│   ├── figures/            # 300 DPI publication plots
│   ├── logs/               # Detailed execution traces
│   └── metrics/            # JSON results and LaTeX publication tables
├── src/
│   ├── common/             # Normalization, metrics, logging, loader
│   ├── team_a_matrix/      # Sparse matrix models, preprocessors, engines
│   └── team_b_gnn/         # GraphSAGE, decoder, utility, ranking, cold-start
└── tests/                  # Pytest unit and integration test suite
```

---

## 🚀 Quickstart & Execution

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Team A Benchmark
```bash
python experiments/run_team_a.py --model BiasedMF --density 0.10
```

### 3. Run Team B Benchmark
```bash
python experiments/run_team_b.py --epochs 80 --alpha 0.7 --beta 0.3
```

### 4. Cross-Paradigm Comparison & LaTeX Table Generation
```bash
python experiments/run_comparison.py
```
Outputs:
- `results/metrics/comparative_summary.csv`
- `results/metrics/latex_tables.tex` (Paste directly into research paper)
- `results/figures/comparative_topk_curves.png`

### 5. Cost-Performance Sensitivity & Pareto Frontier
```bash
python experiments/run_sensitivity_analysis.py
```
Outputs:
- `results/metrics/sensitivity_alpha_beta.csv`
- `results/figures/cost_utility_pareto.png`

---

## 🧪 Running Unit Tests
```bash
pytest tests/ -v
```
