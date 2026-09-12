"""
Tests for Team B: Cost Generator, GraphSAGE, QoS Decoder, Utility Engine, Economic Metrics.
"""

import pytest
import numpy as np
import torch
from src.team_b_gnn.cost_generator import ServiceCostEngine
from src.team_b_gnn.graph_builder import BipartiteQoSGraphBuilder
from src.team_b_gnn.data_preprocessor import GraphDataPreprocessor
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.ranking_engine import CostPerformanceRankingEngine
from src.team_b_gnn.metrics_economic import EconomicMetricsEvaluator


def test_service_cost_engine():
    quality_profile = np.array([0.1, 0.5, 0.9])
    costs = ServiceCostEngine.generate_correlated_costs(
        service_qos_profiles=quality_profile,
        correlation_strength=0.9,
        base_cost_range=(0.01, 1.0),
        distribution="pareto",
        seed=42
    )
    assert len(costs) == 3
    assert np.all(costs >= 0.01) and np.all(costs <= 1.0)
    # Higher quality profile should generally yield higher cost under high correlation
    assert costs[-1] >= costs[0]


def test_graphsage_and_decoder_forward():
    num_users = 10
    num_services = 15
    feature_dim = 16

    # Node features
    u_x = torch.randn(num_users, feature_dim)
    s_x = torch.randn(num_services, feature_dim)

    # Directed bipartite interaction edges
    edge_u = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    edge_s = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    model = HeteroGraphSAGE(
        in_dim_user=feature_dim,
        in_dim_service=feature_dim,
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        aggregator="mean"
    )

    h_u, h_s = model(u_x, s_x, edge_u, edge_s)
    assert h_u.shape == (num_users, 16)
    assert h_s.shape == (num_services, 16)

    # Decoder forward
    decoder = EdgeQoSDecoder(u_dim=16, s_dim=16, num_qos_attributes=5, hidden_dims=[32])
    batch_u = h_u[edge_u]
    batch_s = h_s[edge_s]
    preds = decoder(batch_u, batch_s)

    assert preds.shape == (5, 5)
    # QoS predictions must be strictly positive
    assert torch.all(preds >= 0.0)


def test_cost_performance_utility_monotonicity():
    # User 0, Service 0 vs Service 1
    # Service 0: High QoS (0.9), High Cost (0.8)
    # Service 1: Low QoS (0.2), Low Cost (0.1)
    qos_matrix = np.array([[0.9, 0.2]])
    costs = np.array([0.8, 0.1])

    # Case 1: High alpha (QoS-driven, alpha=0.9, beta=0.1)
    engine_qos = CostPerformanceUtilityEngine(alpha=0.9, beta=0.1)
    util_qos = engine_qos.compute_utility(qos_matrix, costs, normalize_cost=False)
    # Utility = 0.9 * 0.9 - 0.1 * 0.8 = 0.81 - 0.08 = 0.73
    # Utility = 0.9 * 0.2 - 0.1 * 0.1 = 0.18 - 0.01 = 0.17
    assert util_qos[0, 0] > util_qos[0, 1]

    # Case 2: High beta (Cost-sensitive, alpha=0.1, beta=0.9)
    engine_cost = CostPerformanceUtilityEngine(alpha=0.1, beta=0.9)
    util_cost = engine_cost.compute_utility(qos_matrix, costs, normalize_cost=False)
    # Utility = 0.1 * 0.9 - 0.9 * 0.8 = 0.09 - 0.72 = -0.63
    # Utility = 0.1 * 0.2 - 0.9 * 0.1 = 0.02 - 0.09 = -0.07
    assert util_cost[0, 1] > util_cost[0, 0]


def test_economic_metrics_savings():
    # Test recommendations
    # Aware recommendations choose cheaper services: [0, 1]
    # Baseline chooses expensive services: [2, 3]
    aware_recs = {0: [0, 1]}
    baseline_recs = {0: [2, 3]}
    costs = np.array([0.10, 0.20, 0.80, 0.90])

    avg_aware = EconomicMetricsEvaluator.compute_average_cost(aware_recs, costs)
    avg_base = EconomicMetricsEvaluator.compute_average_cost(baseline_recs, costs)
    assert avg_aware == pytest.approx(0.15)
    assert avg_base == pytest.approx(0.85)

    savings = EconomicMetricsEvaluator.compute_cost_savings(aware_recs, baseline_recs, costs)
    expected_savings = ((0.85 - 0.15) / 0.85) * 100.0
    assert savings == pytest.approx(expected_savings)
