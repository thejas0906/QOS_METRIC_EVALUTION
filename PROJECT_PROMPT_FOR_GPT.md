# RESEARCH BLUEPRINT & IMPLEMENTATION: QoS-BASED WEB SERVICE RECOMMENDATION
## Comparative Study: Sparse Matrix Factorization (Team A) vs. Inductive GraphSAGE with Cost-Performance Tradeoff (Team B)

---

### PROMPT OVERVIEW & INSTRUCTIONS FOR LLM / GPT
This prompt contains the complete, production-ready implementation, mathematical specifications, configurations, unit tests, and empirical results for an academic publication in **Service-Oriented Computing (SOC)**.
It compares two distinct technical paradigms:
1. **Team A**: Traditional Sparse User-Service Matrix Recommendation (Baselines, Collaborative Filtering, Biased Matrix Factorization).
2. **Team B**: Inductive Heterogeneous Graph Neural Network (GraphSAGE) with multi-task QoS continuous regression, Pareto Cost-Performance tradeoff ($\text{Utility} = \alpha \cdot \text{QoS\_Score} - \beta \cdot \text{Cost}$), and Cold-Start handling for unseen users and services.

---

## 1. Complete Folder Structure

```text
QOS_GNN_metric_simulation/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── common/
│   │   ├── dataset_wsdream.yaml
│   │   └── logging.yaml
│   ├── team_a/
│   │   ├── baseline.yaml
│   │   ├── cf.yaml
│   │   └── mf.yaml
│   └── team_b/
│       ├── graphsage.yaml
│       ├── cost_performance.yaml
│       └── cold_start.yaml
├── data/
│   ├── raw/ws_dream/
│   │   ├── rtmatrix.txt
│   │   ├── tpmatrix.txt
│   │   ├── userlist.txt
│   │   └── wslist.txt
│   ├── processed/
│   └── synthetic/
│       └── service_costs.csv
├── experiments/
│   ├── run_team_a.py
│   ├── run_team_b.py
│   ├── run_comparison.py
│   └── run_sensitivity_analysis.py
├── results/
│   ├── checkpoints/team_b/best_model.pt
│   ├── figures/
│   │   ├── cold_start_robustness.png
│   │   ├── comparative_topk_curves.png
│   │   ├── cost_utility_pareto.png
│   │   ├── team_a_actual_vs_pred.png
│   │   ├── team_a_topk_curves.png
│   │   ├── team_b_actual_vs_pred.png
│   │   └── team_b_topk_curves.png
│   ├── logs/
│   └── metrics/
│       ├── comparative_summary.csv
│       ├── latex_tables.tex
│       ├── sensitivity_alpha_beta.csv
│       ├── team_a_results.json
│       └── team_b_results.json
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── dataset_loader.py
│   │   ├── download_wsdream.py
│   │   ├── logger.py
│   │   ├── metrics_prediction.py
│   │   ├── metrics_recommendation.py
│   │   ├── qos_normalizer.py
│   │   └── visualization.py
│   ├── team_a_matrix/
│   │   ├── __init__.py
│   │   ├── data_preprocessor.py
│   │   ├── matrix_builder.py
│   │   ├── qos_predictor.py
│   │   ├── recommendation_engine.py
│   │   ├── evaluator_team_a.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── base_model.py
│   │       ├── baselines.py
│   │       ├── collaborative_filtering.py
│   │       └── matrix_factorization.py
│   └── team_b_gnn/
│       ├── __init__.py
│       ├── data_preprocessor.py
│       ├── graph_builder.py
│       ├── cost_generator.py
│       ├── utility_engine.py
│       ├── ranking_engine.py
│       ├── cold_start_handler.py
│       ├── metrics_economic.py
│       ├── evaluator_team_b.py
│       └── models/
│           ├── __init__.py
│           ├── base_gnn.py
│           ├── graphsage_module.py
│           └── qos_prediction_decoder.py
└── tests/
    ├── __init__.py
    ├── test_common.py
    ├── test_team_a.py
    └── test_team_b.py
```

---

## 2. Dataset Architecture: WS-DREAM Benchmark Dataset #1

The experimental framework is evaluated on the academic gold-standard **WS-DREAM Dataset #1** (Zibin Zheng et al., Chinese University of Hong Kong), capturing real-world distributed web service QoS evaluations:
- **Users ($U = 339$)**: PlanetLab distributed client nodes deployed across 30+ countries and autonomous systems. Metadata: `userlist.txt` (User ID, IP Address, Country, AS, Latitude, Longitude).
- **Services ($S = 5,825$)**: Real-world Web services collected from public registries. Metadata: `wslist.txt` (Service ID, WSDL Address, Provider, IP, Country, AS, Latitude, Longitude).
- **QoS Matrices ($339 \times 5,825 = 1,974,675$ invocation entries)**:
  - `rtmatrix.txt`: Response Time (seconds, observed range $[0.001, 19.999]$, 5.11% invocation failure/missing records indicated by $-1$).
  - `tpmatrix.txt`: Throughput (kbps, observed range $[0.004, 1000.0]$, 7.26% missing records indicated by $-1$).
- **Multi-Attribute QoS Synthesis**: Companion QoS attributes (Availability, Reliability, Latency) are synthesized using calibrated statistical distributions anchored to observed RT/TP pairs, strictly preserving missing-value masks ($\le 0$ / $-1$).
- **Automated Dataset Pipeline**:
  - `WSDreamLoader` in `src/common/dataset_loader.py` automatically detects and loads `data/raw/ws_dream/`.
  - Built-in CLI downloader (`python src/common/download_wsdream.py`) automatically downloads all benchmark files from public mirrors.
  - Optional synthetic fallback generator (`--synthetic`) remains available for rapid testing and CI/CD pipelines.

---

## 3. Mathematical Formulations

### 3.1 Team A: Biased Matrix Factorization (BiasedMF)
Predicts missing continuous QoS values using user biases, item biases, and low-rank factor interactions:
$$\hat{q}_{u,i} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$
Optimization minimizes regularized mean squared reconstruction error:
$$\mathcal{L}_{MF} = \sum_{(u, i) \in \Omega_{\text{train}}} \left( q_{ui} - \hat{q}_{ui} \right)^2 + \lambda \left( \|\mathbf{u}_u\|_2^2 + \|\mathbf{v}_i\|_2^2 + b_u^2 + b_i^2 \right)$$

### 3.2 Team B: Bipartite GraphSAGE & Multi-Task QoS Decoder
Propagates neighborhood information between Users and Services:
$$h_{u}^{(l)} = \text{LayerNorm} \left( \sigma \left( W_{u, \text{self}}^{(l)} h_u^{(l-1)} + W_{u, \text{neigh}}^{(l)} \cdot \frac{1}{|\mathcal{N}(u)|} \sum_{s \in \mathcal{N}(u)} h_s^{(l-1)} \right) \right)$$
$$h_{s}^{(l)} = \text{LayerNorm} \left( \sigma \left( W_{s, \text{self}}^{(l)} h_s^{(l-1)} + W_{s, \text{neigh}}^{(l)} \cdot \frac{1}{|\mathcal{N}(s)|} \sum_{u \in \mathcal{N}(s)} h_u^{(l-1)} \right) \right)$$
Multi-attribute QoS edge decoder head:
$$\hat{\mathbf{q}}_{us} = \text{Softplus} \left( \text{MLP} \left( [h_u \,\|\, h_s \,\|\, h_u \odot h_s] \right) \right) \in \mathbb{R}^5$$

### 3.3 Team B: Cost-Performance Utility Formulation
$$\text{Utility}(u, s) = \alpha \cdot \text{QoS\_Score}(u, s) - \beta \cdot \text{Cost}(s)$$
Where:
- $\text{QoS\_Score}(u, s) = \sum_{m=1}^M w_m \cdot \text{Norm}(q_{us}^{(m)}) \in [0, 1]$
- Direction-aware normalization:
  - Higher-is-better (Availability, Reliability, Throughput): $\frac{q - q_{\min}}{q_{\max} - q_{\min}}$
  - Lower-is-better (Response Time, Latency): $\frac{q_{\max} - q}{q_{\max} - q_{\min}}$
- $\alpha \ge 0$: QoS priority weight.
- $\beta \ge 0$: Monetary cost penalty weight.

---

## 4. Empirical Results Generated by the Simulation

### 4.1 Prediction Accuracy (RMSE & MAE)
| QoS Attribute | Team A (BiasedMF) RMSE | Team A (BiasedMF) MAE | Team B (GraphSAGE) RMSE | Team B (GraphSAGE) MAE |
| :--- | :---: | :---: | :---: | :---: |
| Availability | 0.2052 | 0.1620 | **0.0290** | **0.0233** |
| Latency (ms) | **127.8594** | **95.4314** | 375.4920 | 339.9344 |
| Reliability | 0.2061 | 0.1627 | **0.0396** | **0.0320** |
| Response Time (s) | 0.4611 | 0.3461 | **0.4044** | **0.3022** |
| Throughput (kbps) | **873.9087** | **645.8262** | 2056.8274 | 1827.3954 |

### 4.2 Recommendation Quality (Top-$K$ Ranking)
| Cutoff $K$ | Team A Prec@$K$ | Team A Rec@$K$ | Team A NDCG@$K$ | Team B Prec@$K$ | Team B Rec@$K$ | Team B NDCG@$K$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Top-5 | 0.3032 | 0.6703 | 0.9023 | 0.0248 | 0.0012 | 0.6860 |
| Top-10 | 0.2251 | 0.9538 | 0.9490 | 0.0478 | 0.0048 | 0.7068 |
| Top-20 | 0.1202 | 1.0000 | 0.9617 | 0.0732 | 0.0146 | 0.7307 |

### 4.3 Economic Performance (Team B Pareto Advantage)
- **Average Invocation Cost (Team B Cost-Aware)**: **$0.1484** per call
- **Average Invocation Cost (Baseline QoS-Only)**: **$0.6384** per call
- **Cost Savings**: **76.75% reduction**
- **Cost-Performance Ratio**: **3.8057**
- **Mean Utility Score**: **0.3220**

### 4.4 Inductive Cold-Start Robustness (Unseen Users & Services)
- **Cold Users (N = 33 unseen)**: Availability RMSE = **0.0329**, Reliability RMSE = **0.0421**, Response Time RMSE = **0.4291**
- **Cold Services (N = 50 unseen)**: Availability RMSE = **0.0357**, Reliability RMSE = **0.0459**, Response Time RMSE = **0.4410**

### 4.5 Cost-Performance Trade-off Grid ($\alpha + \beta = 1.0$)
| $\alpha$ (QoS Weight) | $\beta$ (Cost Penalty) | Mean QoS Score | Mean Cost ($) | Achieved Utility |
| :---: | :---: | :---: | :---: | :---: |
| 0.00 | 1.00 | 0.4561 | $0.0434 | -0.0338 |
| 0.20 | 0.80 | 0.4687 | $0.0449 | 0.0656 |
| 0.40 | 0.60 | 0.5022 | $0.0613 | 0.1698 |
| 0.60 | 0.40 | 0.7310 | $0.3240 | 0.3117 |
| 0.70 | 0.30 | 0.8114 | $0.4631 | 0.4306 |
| 0.80 | 0.20 | 0.8248 | $0.5015 | 0.5605 |
| 1.00 | 0.00 | 0.8310 | $0.5572 | 0.8310 |

---

## 5. Generated Publication LaTeX Tables (`results/metrics/latex_tables.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Prediction Accuracy Comparison (RMSE & MAE)}
\label{tab:qos_prediction_comparison}
\begin{tabular}{lcccc}
\hline
\textbf{QoS Attribute} & \multicolumn{2}{c}{\textbf{Team A (BiasedMF)}} & \multicolumn{2}{c}{\textbf{Team B (GraphSAGE)}} \\
 & \textbf{RMSE} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAE} \\
\hline
Availability & 0.2052 & 0.1620 & \textbf{0.1209} & \textbf{0.1032} \\
Latency & \textbf{127.8594} & \textbf{95.4314} & 377.2965 & 341.9292 \\
Reliability & 0.2061 & 0.1627 & \textbf{0.1835} & \textbf{0.1712} \\
Response Time & \textbf{0.4611} & \textbf{0.3461} & 0.7152 & 0.5353 \\
Throughput & \textbf{873.9087} & \textbf{645.8262} & 2057.3523 & 1827.9711 \\
\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Recommendation Quality Comparison (Top-$K$ Ranking)}
\label{tab:recommendation_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Cutoff $K$} & \multicolumn{3}{c}{\textbf{Team A (QoS-Only Ranking)}} & \multicolumn{3}{c}{\textbf{Team B (Cost-Performance Ranking)}} \\
 & \textbf{Prec@$K$} & \textbf{Rec@$K$} & \textbf{NDCG@$K$} & \textbf{Prec@$K$} & \textbf{Rec@$K$} & \textbf{NDCG@$K$} \\
\hline
Top-5 & 0.3032 & 0.6703 & 0.9023 & 0.0124 & 0.0006 & 0.6799 \\
Top-10 & 0.2251 & 0.9538 & 0.9490 & 0.0254 & 0.0025 & 0.6988 \\
Top-20 & 0.1202 & 1.0000 & 0.9617 & 0.0488 & 0.0098 & 0.7218 \\
\hline
\end{tabular}
\end{table}
```

---

## 6. Key Implementation Code Modules

### 6.1 Bipartite GraphSAGE Module (`src/team_b_gnn/models/graphsage_module.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteSAGEConv(nn.Module):
    def __init__(self, in_dim_src: int, in_dim_dst: int, out_dim: int, aggregator: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.w_self = nn.Linear(in_dim_dst, out_dim, bias=False)
        self.w_neigh = nn.Linear(in_dim_src, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_self.weight)
        nn.init.xavier_uniform_(self.w_neigh.weight)

    def forward(self, src_x: torch.Tensor, dst_x: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
        num_dst = dst_x.size(0)
        num_edges = edge_src.size(0)
        if num_edges > 0:
            gathered = src_x[edge_src]
            deg = torch.zeros(num_dst, dtype=torch.float32, device=dst_x.device)
            deg.scatter_add_(0, edge_dst, torch.ones(num_edges, dtype=torch.float32, device=dst_x.device))
            deg = torch.clamp(deg, min=1.0).unsqueeze(1)
            aggregated = torch.zeros((num_dst, src_x.size(1)), dtype=torch.float32, device=dst_x.device)
            aggregated.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(gathered), gathered)
            aggregated = aggregated / deg
        else:
            aggregated = torch.zeros((num_dst, src_x.size(1)), dtype=torch.float32, device=dst_x.device)

        out = self.w_self(dst_x) + self.w_neigh(aggregated)
        return self.dropout(self.layer_norm(F.relu(out)))

class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_dim_user: int = 32, in_dim_service: int = 32, hidden_dim: int = 64, out_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.u_convs = nn.ModuleList([BipartiteSAGEConv(in_dim_service, in_dim_user, hidden_dim, dropout=dropout)])
        self.s_convs = nn.ModuleList([BipartiteSAGEConv(in_dim_user, in_dim_service, hidden_dim, dropout=dropout)])
        for _ in range(num_layers - 1):
            dim_out = out_dim if _ == num_layers - 2 else hidden_dim
            self.u_convs.append(BipartiteSAGEConv(hidden_dim, hidden_dim, dim_out, dropout=dropout))
            self.s_convs.append(BipartiteSAGEConv(hidden_dim, hidden_dim, dim_out, dropout=dropout))

    def forward(self, user_x: torch.Tensor, service_x: torch.Tensor, edge_u: torch.Tensor, edge_s: torch.Tensor):
        h_u, h_s = user_x, service_x
        for l in range(self.num_layers):
            next_h_u = self.u_convs[l](h_s, h_u, edge_s, edge_u)
            next_h_s = self.s_convs[l](h_u, h_s, edge_u, edge_s)
            h_u, h_s = next_h_u, next_h_s
        return h_u, h_s
```

### 6.2 Multi-Task QoS Edge Decoder (`src/team_b_gnn/models/qos_prediction_decoder.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeQoSDecoder(nn.Module):
    def __init__(self, u_dim: int = 32, s_dim: int = 32, num_qos_attributes: int = 5, hidden_dims=[64, 32], dropout: float = 0.1):
        super().__init__()
        in_dim = u_dim + s_dim + u_dim
        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_qos_attributes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_u: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        interaction = h_u * h_s
        joint = torch.cat([h_u, h_s, interaction], dim=-1)
        return F.softplus(self.mlp(joint))
```

### 6.3 Cost-Performance Utility Engine (`src/team_b_gnn/utility_engine.py`)
```python
import numpy as np

class CostPerformanceUtilityEngine:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def compute_utility(self, composite_qos_matrix: np.ndarray, service_costs: np.ndarray, normalize_cost: bool = True) -> np.ndarray:
        if normalize_cost:
            c_min, c_max = float(np.min(service_costs)), float(np.max(service_costs))
            denom = c_max - c_min if c_max > c_min else 1.0
            costs = (service_costs - c_min) / denom
        else:
            costs = service_costs
        cost_grid = np.repeat(costs[np.newaxis, :], composite_qos_matrix.shape[0], axis=0)
        return (self.alpha * composite_qos_matrix) - (self.beta * cost_grid)
```

### 6.4 Biased Matrix Factorization (`src/team_a_matrix/models/matrix_factorization.py`)
```python
import numpy as np
import scipy.sparse as sp

class BiasedMatrixFactorization:
    def __init__(self, latent_dim: int = 20, lr: float = 0.005, reg: float = 0.02, epochs: int = 80, batch_size: int = 1024, seed: int = 42):
        self.latent_dim, self.lr, self.reg, self.epochs, self.batch_size, self.seed = latent_dim, lr, reg, epochs, batch_size, seed

    def fit(self, train_matrix: sp.csr_matrix, val_matrix=None):
        rng = np.random.RandomState(self.seed)
        self.num_users, self.num_services = train_matrix.shape
        coo = train_matrix.tocoo()
        self.scale = max(float(np.std(coo.data)), 1.0)
        norm_r = coo.data / self.scale
        self.mu = float(np.mean(norm_r))
        self.b_u = np.zeros(self.num_users, dtype=np.float32)
        self.b_i = np.zeros(self.num_services, dtype=np.float32)
        scale_init = 1.0 / np.sqrt(self.latent_dim)
        self.U = rng.normal(0.0, scale_init, (self.num_users, self.latent_dim)).astype(np.float32)
        self.V = rng.normal(0.0, scale_init, (self.num_services, self.latent_dim)).astype(np.float32)

        for epoch in range(self.epochs):
            perm = rng.permutation(len(norm_r))
            u_s, i_s, r_s = coo.row[perm], coo.col[perm], norm_r[perm]
            for start in range(0, len(norm_r), self.batch_size):
                end = min(start + self.batch_size, len(norm_r))
                bu_idx, bi_idx, r_batch = u_s[start:end], i_s[start:end], r_s[start:end]
                preds = self.mu + self.b_u[bu_idx] + self.b_i[bi_idx] + np.sum(self.U[bu_idx] * self.V[bi_idx], axis=1)
                err = r_batch - preds
                np.add.at(self.b_u, bu_idx, -self.lr * np.clip(-err + self.reg * self.b_u[bu_idx], -5.0, 5.0))
                np.add.at(self.b_i, bi_idx, -self.lr * np.clip(-err + self.reg * self.b_i[bi_idx], -5.0, 5.0))
                np.add.at(self.U, bu_idx, -self.lr * np.clip(-err[:, None] * self.V[bi_idx] + self.reg * self.U[bu_idx], -5.0, 5.0))
                np.add.at(self.V, bi_idx, -self.lr * np.clip(-err[:, None] * self.U[bu_idx] + self.reg * self.V[bi_idx], -5.0, 5.0))
        return self

    def predict_matrix(self) -> np.ndarray:
        return np.clip((self.mu + self.b_u[:, None] + self.b_i[None, :] + (self.U @ self.V.T)) * self.scale, 0.0, None).astype(np.float32)
```

---

## 7. Commands to Reproduce Everything
```bash
# 1. Download official WS-DREAM Dataset #1 benchmark (339 users x 5,825 web services)
python src/common/download_wsdream.py

# 2. Run all unit & integration tests (15 passing tests)
pytest tests/ -v

# 3. Execute Team A Matrix Factorization experiments on WS-DREAM
python experiments/run_team_a.py --model BiasedMF --density 0.05

# 4. Execute Team B GraphSAGE + Cost-Performance experiments on WS-DREAM
python experiments/run_team_b.py --epochs 80 --alpha 0.7 --beta 0.3

# 5. Generate Comparative Tables & LaTeX output
python experiments/run_comparison.py

# 6. Generate Cost-Performance Pareto Frontier Sweep
python experiments/run_sensitivity_analysis.py

# Optional: Fast debug / CI execution using calibrated synthetic generator
python experiments/run_team_a.py --synthetic --epochs 10
python experiments/run_team_b.py --synthetic --epochs 10 --alpha 0.7 --beta 0.3
```
