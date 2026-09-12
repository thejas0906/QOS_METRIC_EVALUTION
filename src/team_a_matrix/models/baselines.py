"""
Statistical Baseline Models for QoS Benchmarking.
Implements Global Mean, User Mean, Service Mean, and Additive User-Item Mean baselines.
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
from src.team_a_matrix.models.base_model import BaseMatrixQoSModel


class GlobalMeanBaseline(BaseMatrixQoSModel):
    """Predicts the dataset-wide mean QoS for all user-service invocations."""

    def __init__(self):
        super().__init__(model_name="GlobalMean")
        self.global_mean: float = 0.0

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "GlobalMeanBaseline":
        self.num_users, self.num_services = train_matrix.shape
        data = train_matrix.data
        self.global_mean = float(np.mean(data)) if len(data) > 0 else 0.0
        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        return self.global_mean

    def predict_matrix(self) -> np.ndarray:
        return np.full((self.num_users, self.num_services), self.global_mean, dtype=np.float32)


class UserMeanBaseline(BaseMatrixQoSModel):
    """Predicts the personalized average QoS experienced by each user."""

    def __init__(self):
        super().__init__(model_name="UserMean")
        self.global_mean: float = 0.0
        self.user_means: np.ndarray = np.array([])

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "UserMeanBaseline":
        self.num_users, self.num_services = train_matrix.shape
        data = train_matrix.data
        self.global_mean = float(np.mean(data)) if len(data) > 0 else 0.0

        user_sums = np.array(train_matrix.sum(axis=1)).flatten()
        user_counts = np.diff(train_matrix.indptr)

        self.user_means = np.where(
            user_counts > 0,
            user_sums / np.maximum(user_counts, 1),
            self.global_mean
        ).astype(np.float32)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if 0 <= user_id < self.num_users:
            return float(self.user_means[user_id])
        return self.global_mean

    def predict_matrix(self) -> np.ndarray:
        return np.repeat(self.user_means[:, np.newaxis], self.num_services, axis=1)


class ServiceMeanBaseline(BaseMatrixQoSModel):
    """Predicts the historical average QoS delivered by each web service."""

    def __init__(self):
        super().__init__(model_name="ServiceMean")
        self.global_mean: float = 0.0
        self.service_means: np.ndarray = np.array([])

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "ServiceMeanBaseline":
        self.num_users, self.num_services = train_matrix.shape
        data = train_matrix.data
        self.global_mean = float(np.mean(data)) if len(data) > 0 else 0.0

        csc = train_matrix.tocsc()
        service_sums = np.array(csc.sum(axis=0)).flatten()
        service_counts = np.diff(csc.indptr)

        self.service_means = np.where(
            service_counts > 0,
            service_sums / np.maximum(service_counts, 1),
            self.global_mean
        ).astype(np.float32)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if 0 <= service_id < self.num_services:
            return float(self.service_means[service_id])
        return self.global_mean

    def predict_matrix(self) -> np.ndarray:
        return np.repeat(self.service_means[np.newaxis, :], self.num_users, axis=0)


class UserItemMeanBaseline(BaseMatrixQoSModel):
    """Additive baseline model: mu + b_u + b_i."""

    def __init__(self, reg_u: float = 5.0, reg_i: float = 5.0):
        super().__init__(model_name="UserItemMean")
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray = np.array([])
        self.item_bias: np.ndarray = np.array([])

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "UserItemMeanBaseline":
        self.num_users, self.num_services = train_matrix.shape
        data = train_matrix.data
        self.global_mean = float(np.mean(data)) if len(data) > 0 else 0.0

        # Item bias: b_i = sum(r_ui - mu) / (count_i + reg_i)
        csc = train_matrix.tocsc()
        item_counts = np.diff(csc.indptr)
        item_sums = np.array((csc - (csc > 0).astype(np.float32) * self.global_mean).sum(axis=0)).flatten()
        self.item_bias = (item_sums / (item_counts + self.reg_i)).astype(np.float32)

        # User bias: b_u = sum(r_ui - mu - b_i) / (count_u + reg_u)
        user_counts = np.diff(train_matrix.indptr)
        user_bias = np.zeros(self.num_users, dtype=np.float32)

        coo = train_matrix.tocoo()
        residuals = coo.data - (self.global_mean + self.item_bias[coo.col])
        for u, res in zip(coo.row, residuals):
            user_bias[u] += res

        self.user_bias = (user_bias / (user_counts + self.reg_u)).astype(np.float32)
        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        bu = self.user_bias[user_id] if 0 <= user_id < self.num_users else 0.0
        bi = self.item_bias[service_id] if 0 <= service_id < self.num_services else 0.0
        return float(self.global_mean + bu + bi)

    def predict_matrix(self) -> np.ndarray:
        return (self.global_mean + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis, :]).astype(np.float32)
