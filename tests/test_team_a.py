"""
Tests for Team A: Sparse Matrix Preprocessor, Builder, Models, and Evaluator.
"""

import pytest
import numpy as np
from src.team_a_matrix.data_preprocessor import MatrixPreprocessor
from src.team_a_matrix.matrix_builder import QoSMatrixBuilder
from src.team_a_matrix.models.baselines import (
    GlobalMeanBaseline,
    UserMeanBaseline,
    ServiceMeanBaseline,
    UserItemMeanBaseline
)
from src.team_a_matrix.models.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.team_a_matrix.models.matrix_factorization import (
    ProbabilisticMatrixFactorization,
    BiasedMatrixFactorization
)
from src.team_a_matrix.recommendation_engine import QoSRecommendationEngine
from src.common.qos_normalizer import QoSNormalizer


@pytest.fixture
def sample_qos_matrix():
    rng = np.random.RandomState(42)
    # 20 users x 30 services
    return rng.uniform(0.1, 2.0, size=(20, 30)).astype(np.float32)


def test_matrix_preprocessor_and_builder(sample_qos_matrix):
    mask = MatrixPreprocessor.create_sparsity_mask(sample_qos_matrix, target_density=0.20, seed=42)
    assert np.sum(mask) == int(20 * 30 * 0.20)

    train_m, val_m, test_m = MatrixPreprocessor.train_val_test_split(
        sample_qos_matrix, mask, val_ratio=0.10, test_ratio=0.20, seed=42
    )

    # Disjointness check
    assert not np.any(train_m & val_m)
    assert not np.any(train_m & test_m)
    assert not np.any(val_m & test_m)

    csr = QoSMatrixBuilder.build_csr_matrix(sample_qos_matrix, train_m)
    assert csr.nnz == np.sum(train_m)

    u, s, v = QoSMatrixBuilder.extract_triplets(sample_qos_matrix, train_m)
    assert len(u) == csr.nnz


def test_baselines(sample_qos_matrix):
    mask = MatrixPreprocessor.create_sparsity_mask(sample_qos_matrix, target_density=0.30, seed=42)
    train_m, _, _ = MatrixPreprocessor.train_val_test_split(sample_qos_matrix, mask, seed=42)
    csr = QoSMatrixBuilder.build_csr_matrix(sample_qos_matrix, train_m)

    # Global Mean
    gm = GlobalMeanBaseline().fit(csr)
    assert gm.predict(0, 0) == pytest.approx(float(np.mean(csr.data)), rel=1e-3)

    # User Mean
    um = UserMeanBaseline().fit(csr)
    pred_mat = um.predict_matrix()
    assert pred_mat.shape == (20, 30)

    # User Item Mean
    uim = UserItemMeanBaseline().fit(csr)
    assert uim.predict_matrix().shape == (20, 30)


def test_collaborative_filtering(sample_qos_matrix):
    mask = MatrixPreprocessor.create_sparsity_mask(sample_qos_matrix, target_density=0.40, seed=42)
    train_m, _, _ = MatrixPreprocessor.train_val_test_split(sample_qos_matrix, mask, seed=42)
    csr = QoSMatrixBuilder.build_csr_matrix(sample_qos_matrix, train_m)

    ucf = UserBasedCF(k_neighbors=5).fit(csr)
    val = ucf.predict(0, 1)
    assert val >= 0.0

    icf = ItemBasedCF(k_neighbors=5).fit(csr)
    val_i = icf.predict(0, 1)
    assert val_i >= 0.0


def test_matrix_factorization_convergence(sample_qos_matrix):
    mask = MatrixPreprocessor.create_sparsity_mask(sample_qos_matrix, target_density=0.30, seed=42)
    train_m, val_m, _ = MatrixPreprocessor.train_val_test_split(sample_qos_matrix, mask, seed=42)
    csr_train = QoSMatrixBuilder.build_csr_matrix(sample_qos_matrix, train_m)
    csr_val = QoSMatrixBuilder.build_csr_matrix(sample_qos_matrix, val_m)

    # BiasedMF
    bmf = BiasedMatrixFactorization(latent_dim=5, lr=0.01, epochs=15, early_stopping_patience=10, seed=42)
    bmf.fit(csr_train, csr_val)

    # Loss history should decrease
    assert len(bmf.train_loss_history) > 1
    assert bmf.train_loss_history[-1] < bmf.train_loss_history[0]

    # Reconstructed matrix
    pred_mat = bmf.predict_matrix()
    assert pred_mat.shape == (20, 30)
    assert not np.any(np.isnan(pred_mat))


def test_recommendation_engine(sample_qos_matrix):
    norm = QoSNormalizer(attribute="response_time").fit(sample_qos_matrix)
    pred_dict = {"response_time": sample_qos_matrix}
    norm_dict = {"response_time": norm}

    engine = QoSRecommendationEngine(attribute_weights={"response_time": 1.0})
    comp = engine.compute_composite_qos_matrix(pred_dict, norm_dict)
    assert comp.shape == (20, 30)
    assert np.all(comp >= 0.0) and np.all(comp <= 1.0)

    top_k = engine.recommend_top_k(user_id=0, composite_qos_matrix=comp, k=5)
    assert len(top_k) == 5
    # Should be sorted descending
    scores = [s for _, s in top_k]
    assert scores == sorted(scores, reverse=True)
