# MASTER BENCHMARK EVALUATION REPORT
## QoS-Aware Web Service Recommendation System
**Dataset**: WS-DREAM Benchmark (339 Users × 5,825 Web Services)

---
## 1. QoS Continuous Prediction Accuracy
Continuous regression performance (RMSE and MAE) across all 5 QoS dimensions:

| QoS_Attribute | Team_A_BiasedMF_RMSE | Team_B_GraphSAGE_RMSE | Best_RMSE | RMSE_Improvement_Pct | Team_A_BiasedMF_MAE | Team_B_GraphSAGE_MAE | Best_MAE | MAE_Improvement_Pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| availability | 0.3267342150211334 | 0.0923648700118064 | Team B | 71.73088529897849 | 0.1882269978523254 | 0.0732439011335372 | Team B | 61.08746249515109 |
| latency | 529.455322265625 | 632.0682983398438 | Team A | -19.38085646870471 | 257.6280212402344 | 263.8559875488281 | Team A | -2.4174258213885294 |
| reliability | 0.3232125341892242 | 0.1089441552758216 | Team B | 66.29333836043607 | 0.1875327229499817 | 0.0886406227946281 | Team B | 52.73325028279455 |
| response_time | 1.7901195287704468 | 1.9300941228866575 | Team A | -7.819287587592168 | 0.8733486533164978 | 0.9637144804000854 | Team A | -10.347050601203534 |
| throughput | 105.3586196899414 | 120.22053527832033 | Team A | -14.106027235470489 | 50.4361572265625 | 46.75131988525391 | Team B | 7.305943878230184 |


---
## 2. Top-K Recommendation Quality Ranking
Evaluation on held-out test interactions comparing Team A (QoS-Only Ranking) vs Team B (Cost-Performance Ranking):

| Cutoff (K) | Metric | Team A (BiasedMF) | Team B (GraphSAGE) | Winner |
| :---: | :---: | :---: | :---: | :---: |
| Top-5 | Precision@5 | 0.2796 | 0.2549 | Team A |
| Top-5 | Recall@5 | 0.1245 | 0.0067 | Team A |
| Top-5 | NDCG@5 | 0.8757 | 0.6518 | Team A |
| Top-10 | Precision@10 | 0.2395 | 0.3493 | Team B |
| Top-10 | Recall@10 | 0.2131 | 0.0182 | Team A |
| Top-10 | NDCG@10 | 0.8920 | 0.6907 | Team A |
| Top-20 | Precision@20 | 0.2077 | 0.4382 | Team B |
| Top-20 | Recall@20 | 0.3689 | 0.0454 | Team A |
| Top-20 | NDCG@20 | 0.9065 | 0.7447 | Team A |


---
## 3. Cost-Performance-Confidence Paradigm Comparison
Evaluation across Paradigm A (QoS-Only), Paradigm B (QoS+Cost), and Paradigm C (QoS+Cost+Confidence):

| Paradigm | Alpha (QoS) | Beta (Cost) | Gamma (Confidence) | Precision@10 | Recall@10 | NDCG@10 | Average Cost ($) | Cost Savings (%) | CP Ratio | Mean Utility | Average Confidence | Recommendation Coverage (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paradigm A (QoS-Only) | 1.0 | 0.0 | 0.0 | 0.1985250737463127 | 0.0017040691999145 | 0.672033095452871 | 0.5042616234416455 | 0.0 | 1.320837966584489 | 0.661296291864727 | 0.949199206885335 | 19.05579399141631 |
| Paradigm B (QoS + Cost) | 0.7 | 0.3 | 0.0 | 0.0702064896755162 | 0.0006026308126653 | 0.4269210555976541 | 0.0826789968506952 | 83.60394822703316 | 4.748156614110042 | 0.2438264584119341 | 0.7006894095984884 | 2.3004291845493565 |
| Paradigm C (QoS + Cost + Confidence) | 0.5 | 0.3 | 0.2 | 0.1899705014749262 | 0.0016306480813298 | 0.6352332386902999 | 0.0705379940745225 | 86.01162753709228 | 5.217141052776901 | 0.3591891374208231 | 0.9840812338488644 | 1.2875536480686696 |


---
## 4. Sparsity Robustness Benchmark
Degradation under extreme missing QoS observations (50% to 95% missing ratio) on a fixed held-out test set:

| Missing % | Observed % | Team A RMSE | Team B RMSE | Team A MAE | Team B MAE | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% | 50% | 69.0580 | 152.6351 | 25.6209 | 61.8916 | Team A |
| 70% | 30% | 72.9730 | 152.6164 | 28.9818 | 61.8709 | Team A |
| 80% | 20% | 78.0633 | 152.6522 | 32.5317 | 61.9312 | Team A |
| 90% | 10% | 91.4850 | 152.6031 | 41.4750 | 61.8541 | Team A |
| 95% | 5% | 107.0939 | 152.5869 | 51.9665 | 61.8363 | Team A |


---
## 5. Cold-Start Inductive Robustness
Performance on completely unseen cold users and services without graph retraining:

- **Cold Users Evaluation** (N=33):
  - Precision@10: 0.0000
  - Recall@10: 0.0000
  - NDCG@10: 0.0000
- **Cold Services Evaluation** (N=582):
  - Precision@10: 0.0000
  - Recall@10: 0.0000
  - NDCG@10: 0.0000


---
## 6. Artifact Registry
| Benchmark Category | File / Artifact Path | Description |
| :--- | :--- | :--- |
| **Master Summary** | [results/MASTER_BENCHMARK_RESULTS.json](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/MASTER_BENCHMARK_RESULTS.json) | Central JSON containing all metric records |
| **QoS Prediction Accuracy** | [results/qos_prediction_accuracy.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/qos_prediction_accuracy.csv) | Per-attribute RMSE/MAE comparison |
| **Recommendation Quality** | [results/metrics/comparative_summary.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/comparative_summary.csv) | Top-K precision, recall, and NDCG |
| **Confidence Trade-off** | [results/metrics/confidence_comparison.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/confidence_comparison.csv) | Paradigms A, B, and C comparison |
| **Sparsity Robustness** | [results/sparsity_benchmark.csv](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/sparsity_benchmark.csv) | 50% to 95% missing observation results |
| **LaTeX Tables** | [results/metrics/latex_tables.tex](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/metrics/latex_tables.tex) | Ready-to-paste tables for publication |
| **Top-K Ranking Plot** | [results/figures/comparative_topk_curves.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/comparative_topk_curves.png) | High-res curves for Top-K rankings |
| **Confidence Plot** | [results/figures/confidence_paradigm_comparison.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/confidence_paradigm_comparison.png) | Trade-off visualization across paradigms |
| **Sparsity Plot** | [results/figures/sparsity_rmse_comparison.png](file:///c:/Users/kickg/OneDrive/Documents/QOS_GNN_metric_simulation/results/figures/sparsity_rmse_comparison.png) | RMSE degradation curves |