"""
Unit and Integration Tests for Graph-Based Confidence Estimation and Tri-Factor Utility.
"""

import pytest
import numpy as np
from src.team_b_gnn.confidence_estimator import GraphConfidenceEstimator
from src.team_b_gnn.utility_engine import CostPerformanceUtilityEngine
from src.team_b_gnn.ranking_engine import CostPerformanceRankingEngine
from src.team_b_gnn.metrics_economic import EconomicMetricsEvaluator


def test_confidence_estimator_bounds_and_monotonicity():
    # Construct synthetic observation mask (5 users x 6 services)
    mask = np.zeros((5, 6), dtype=bool)
    # User 0 has 5 interactions (high degree)
    mask[0, :] = True
    # User 1 has only 1 interaction (low degree)
    mask[1, 0] = True
    # User 4 has 0 interactions (isolated)

    # Service 0 has interactions from users 0, 1 (degree 2)
    # Service 5 has interactions only from user 0 (degree 1)

    estimator = GraphConfidenceEstimator()
    estimator.fit_from_mask(mask)
    conf_mat = estimator.compute_confidence_matrix()

    assert conf_mat.shape == (5, 6)
    # Range check
    assert np.all(conf_mat >= 0.0)
    assert np.all(conf_mat <= 1.0)

    # High-degree user (0) should have higher confidence on service 0 than low-degree user (1) or isolated user (4)
    assert conf_mat[0, 0] > conf_mat[1, 0]
    assert conf_mat[1, 0] > conf_mat[4, 0]

    # Isolated user 4 with unobserved service 5 should have very low confidence
    assert conf_mat[4, 5] < 0.15


def test_confidence_estimator_fit_from_edges_consistency():
    num_users, num_services = 4, 4
    edge_u = np.array([0, 0, 1, 2], dtype=np.int64)
    edge_s = np.array([1, 2, 2, 3], dtype=np.int64)

    mask = np.zeros((num_users, num_services), dtype=bool)
    mask[edge_u, edge_s] = True

    est1 = GraphConfidenceEstimator().fit_from_mask(mask)
    est2 = GraphConfidenceEstimator().fit_from_edges(edge_u, edge_s, num_users, num_services)

    mat1 = est1.compute_confidence_matrix()
    mat2 = est2.compute_confidence_matrix()

    np.testing.assert_allclose(mat1, mat2, atol=1e-5)

    stats = est2.get_confidence_stats(mat2)
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    assert 0.0 <= stats["mean"] <= 1.0


def test_tri_factor_utility_engine():
    alpha, beta, gamma = 0.5, 0.3, 0.2
    engine = CostPerformanceUtilityEngine(alpha=alpha, beta=beta, gamma=gamma)

    num_users, num_services = 2, 3
    qos_matrix = np.array([
        [0.95, 0.90, 0.50],
        [0.80, 0.70, 0.60]
    ], dtype=np.float32)

    service_costs = np.array([0.10, 0.50, 0.90], dtype=np.float32)
    conf_matrix = np.array([
        [0.30, 0.95, 0.80],
        [0.50, 0.50, 0.50]
    ], dtype=np.float32)

    # Normalized costs with c_min=0.10, c_max=0.90:
    # Service 0: 0.0, Service 1: 0.5, Service 2: 1.0
    # For User 0, Service 0 (A): QoS=0.95, Cost_norm=0.0, Conf=0.30
    # Utility = 0.5 * 0.95 - 0.3 * 0.0 + 0.2 * 0.30 = 0.475 + 0.06 = 0.535
    # For User 0, Service 1 (B): QoS=0.90, Cost_norm=0.5, Conf=0.95
    # Utility = 0.5 * 0.90 - 0.3 * 0.5 + 0.2 * 0.95 = 0.450 - 0.150 + 0.190 = 0.490

    util_mat = engine.compute_utility(
        composite_qos_matrix=qos_matrix,
        service_costs=service_costs,
        confidence_matrix=conf_matrix,
        normalize_cost=True
    )

    assert util_mat.shape == (2, 3)
    assert util_mat[0, 0] == pytest.approx(0.535, abs=1e-3)
    assert util_mat[0, 1] == pytest.approx(0.490, abs=1e-3)

    # Dynamic parameter setting
    engine.set_tradeoff_parameters(alpha=0.4, beta=0.2, gamma=0.4)
    assert engine.alpha == 0.4
    assert engine.beta == 0.2
    assert engine.gamma == 0.4


def test_ranking_engine_with_confidence():
    ranking_engine = CostPerformanceRankingEngine()
    util_mat = np.array([[0.8, 0.9, 0.6]], dtype=np.float32)
    qos_mat = np.array([[0.7, 0.8, 0.5]], dtype=np.float32)
    costs = np.array([0.2, 0.3, 0.1], dtype=np.float32)
    conf_mat = np.array([[0.9, 0.85, 0.4]], dtype=np.float32)

    results = ranking_engine.rank_services_for_user(
        user_id=0,
        utility_matrix=util_mat,
        qos_matrix=qos_mat,
        service_costs=costs,
        confidence_matrix=conf_mat,
        k=2
    )

    assert len(results) == 2
    # Service 1 has highest utility (0.9), followed by Service 0 (0.8)
    s_idx, util, qos, cost, conf = results[0]
    assert s_idx == 1
    assert util == pytest.approx(0.9)
    assert conf == pytest.approx(0.85)

    s_idx_2, util_2, qos_2, cost_2, conf_2 = results[1]
    assert s_idx_2 == 0
    assert conf_2 == pytest.approx(0.9)


def test_economic_confidence_and_coverage_metrics():
    recs = {
        0: [1, 2, 3],
        1: [2, 3, 4]
    }
    conf_mat = np.ones((2, 5), dtype=np.float32) * 0.75
    conf_mat[0, 1] = 0.90

    avg_conf = EconomicMetricsEvaluator.compute_average_confidence(recs, conf_mat, k=3)
    assert 0.75 <= avg_conf <= 0.90

    dist = EconomicMetricsEvaluator.compute_confidence_distribution(recs, conf_mat, k=3)
    assert dist["max"] == pytest.approx(0.90)
    assert dist["min"] == pytest.approx(0.75)

    # Unique services recommended: {1, 2, 3, 4} out of 5 services -> coverage = 4/5 = 0.8
    coverage = EconomicMetricsEvaluator.compute_recommendation_coverage(recs, num_services=5, k=3)
    assert coverage == pytest.approx(0.80)
