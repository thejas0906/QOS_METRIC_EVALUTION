"""
Tests for Common Module: Normalizer, Metrics, Dataset Loader.
"""

import pytest
import numpy as np
from src.common.constants import QoSAttribute, QoSDirection
from src.common.qos_normalizer import QoSNormalizer
from src.common.metrics_prediction import PredictionMetricsEvaluator
from src.common.metrics_recommendation import RecommendationMetricsEvaluator
from src.common.dataset_loader import WSDreamLoader


def test_qos_normalizer_positive():
    # Positive attribute: throughput (higher is better)
    data = np.array([100.0, 200.0, 500.0, 1000.0])
    norm = QoSNormalizer(attribute=QoSAttribute.THROUGHPUT).fit(data)

    transformed = norm.transform(data)
    assert transformed[0] == pytest.approx(0.0)
    assert transformed[-1] == pytest.approx(1.0)
    assert transformed[1] < transformed[2]

    # Invertibility
    restored = norm.inverse_transform(transformed)
    np.testing.assert_allclose(restored, data, rtol=1e-5)


def test_qos_normalizer_negative():
    # Negative attribute: response_time (lower is better)
    data = np.array([0.1, 0.5, 1.0, 2.0])
    norm = QoSNormalizer(attribute=QoSAttribute.RESPONSE_TIME).fit(data)

    transformed = norm.transform(data)
    assert transformed[0] == pytest.approx(1.0)  # Best quality gets 1.0
    assert transformed[-1] == pytest.approx(0.0) # Worst quality gets 0.0

    # Invertibility
    restored = norm.inverse_transform(transformed)
    np.testing.assert_allclose(restored, data, rtol=1e-5)


def test_prediction_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    # Perfect prediction
    res = PredictionMetricsEvaluator.compute_all_prediction_metrics(y_true, y_pred)
    assert res["rmse"] == pytest.approx(0.0)
    assert res["mae"] == pytest.approx(0.0)

    # Offset of 1.0
    y_pred_off = y_true + 1.0
    res_off = PredictionMetricsEvaluator.compute_all_prediction_metrics(y_true, y_pred_off)
    assert res_off["rmse"] == pytest.approx(1.0)
    assert res_off["mae"] == pytest.approx(1.0)


def test_recommendation_metrics():
    # Recommended top 5 items
    recommended = [10, 20, 30, 40, 50]
    # Ground truth relevant items
    ground_truth = {10, 20, 99}

    p_at_2 = RecommendationMetricsEvaluator.compute_precision_at_k(recommended, ground_truth, k=2)
    assert p_at_2 == pytest.approx(1.0)  # [10, 20] both in ground truth

    p_at_5 = RecommendationMetricsEvaluator.compute_precision_at_k(recommended, ground_truth, k=5)
    assert p_at_5 == pytest.approx(2.0 / 5.0)

    r_at_5 = RecommendationMetricsEvaluator.compute_recall_at_k(recommended, ground_truth, k=5)
    assert r_at_5 == pytest.approx(2.0 / 3.0)

    # Perfect ranking NDCG
    ideal_rec = [10, 20, 99]
    gains = {10: 3.0, 20: 2.0, 99: 1.0}
    ndcg = RecommendationMetricsEvaluator.compute_ndcg_at_k(ideal_rec, gains, k=3)
    assert ndcg == pytest.approx(1.0)


def test_dataset_loader_synthetic():
    dataset = WSDreamLoader.generate_synthetic_qos_dataset(num_users=20, num_services=30, seed=42)
    assert len(dataset) == 5
    for attr in [
        QoSAttribute.RESPONSE_TIME.value,
        QoSAttribute.AVAILABILITY.value,
        QoSAttribute.RELIABILITY.value,
        QoSAttribute.THROUGHPUT.value,
        QoSAttribute.LATENCY.value
    ]:
        assert attr in dataset
        assert dataset[attr].shape == (20, 30)
        assert not np.any(np.isnan(dataset[attr]))
