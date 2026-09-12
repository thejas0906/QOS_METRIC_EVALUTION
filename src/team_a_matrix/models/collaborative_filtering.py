"""
Collaborative Filtering Models for Web Service QoS Prediction.
Implements memory-based User-Based CF (U-CF) and Item/Service-Based CF (I-CF)
using Pearson Correlation Coefficient (PCC) with significance weighting.
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
from src.team_a_matrix.models.base_model import BaseMatrixQoSModel


class UserBasedCF(BaseMatrixQoSModel):
    """User-Based Collaborative Filtering using Pearson Correlation Coefficient."""

    def __init__(self, k_neighbors: int = 10, shrinkage: float = 10.0):
        super().__init__(model_name="UserBasedCF")
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.user_means: np.ndarray = np.array([])
        self.similarity_matrix: np.ndarray = np.array([])
        self.train_dense: np.ndarray = np.array([])
        self.train_mask: np.ndarray = np.array([])

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "UserBasedCF":
        self.num_users, self.num_services = train_matrix.shape
        self.train_dense = train_matrix.toarray().astype(np.float32)
        self.train_mask = (self.train_dense > 0)

        # Compute user means over observed entries
        counts = self.train_mask.sum(axis=1)
        sums = self.train_dense.sum(axis=1)
        global_mean = float(np.mean(train_matrix.data)) if len(train_matrix.data) > 0 else 0.0
        self.user_means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean).astype(np.float32)

        # Centered matrix
        centered = np.where(self.train_mask, self.train_dense - self.user_means[:, np.newaxis], 0.0)

        # Compute pairwise PCC similarity with significance weighting
        norms = np.sqrt(np.sum(centered ** 2, axis=1))  # (num_users,)
        dot_products = centered @ centered.T             # (num_users, num_users)
        norm_outer = norms[:, np.newaxis] @ norms[np.newaxis, :] + 1e-8

        raw_sim = dot_products / norm_outer

        # Co-invocation counts
        co_counts = self.train_mask.astype(float) @ self.train_mask.T.astype(float)
        significance = np.minimum(co_counts, self.shrinkage) / self.shrinkage

        self.similarity_matrix = np.clip(raw_sim * significance, -1.0, 1.0)
        np.fill_diagonal(self.similarity_matrix, 0.0)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if not (0 <= user_id < self.num_users and 0 <= service_id < self.num_services):
            return 0.0

        # Find other users who have invoked service_id
        candidate_mask = self.train_mask[:, service_id]
        candidate_mask[user_id] = False
        candidates = np.where(candidate_mask)[0]

        if len(candidates) == 0:
            return float(self.user_means[user_id])

        sims = self.similarity_matrix[user_id, candidates]
        # Keep top-K neighbors with positive similarity
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return float(self.user_means[user_id])

        cand_users = candidates[pos_mask]
        cand_sims = sims[pos_mask]

        if len(cand_users) > self.k_neighbors:
            top_k_idx = np.argsort(cand_sims)[-self.k_neighbors:]
            cand_users = cand_users[top_k_idx]
            cand_sims = cand_sims[top_k_idx]

        residuals = self.train_dense[cand_users, service_id] - self.user_means[cand_users]
        sim_sum = np.sum(cand_sims)

        if sim_sum == 0:
            return float(self.user_means[user_id])

        prediction = self.user_means[user_id] + (np.sum(cand_sims * residuals) / sim_sum)
        return float(max(0.0, prediction))

    def predict_matrix(self) -> np.ndarray:
        pred_matrix = np.zeros((self.num_users, self.num_services), dtype=np.float32)
        for u in range(self.num_users):
            for s in range(self.num_services):
                pred_matrix[u, s] = self.predict(u, s)
        return pred_matrix


class ItemBasedCF(BaseMatrixQoSModel):
    """Item/Service-Based Collaborative Filtering using Pearson Correlation."""

    def __init__(self, k_neighbors: int = 10, shrinkage: float = 10.0):
        super().__init__(model_name="ItemBasedCF")
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.service_means: np.ndarray = np.array([])
        self.similarity_matrix: np.ndarray = np.array([])
        self.train_dense: np.ndarray = np.array([])
        self.train_mask: np.ndarray = np.array([])

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "ItemBasedCF":
        self.num_users, self.num_services = train_matrix.shape
        self.train_dense = train_matrix.toarray().astype(np.float32)
        self.train_mask = (self.train_dense > 0)

        # Service means
        counts = self.train_mask.sum(axis=0)
        sums = self.train_dense.sum(axis=0)
        global_mean = float(np.mean(train_matrix.data)) if len(train_matrix.data) > 0 else 0.0
        self.service_means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean).astype(np.float32)

        # Centered matrix along services
        centered = np.where(self.train_mask, self.train_dense - self.service_means[np.newaxis, :], 0.0)

        # Compute pairwise PCC similarity between services
        norms = np.sqrt(np.sum(centered ** 2, axis=0))  # (num_services,)
        dot_products = centered.T @ centered            # (num_services, num_services)
        norm_outer = norms[:, np.newaxis] @ norms[np.newaxis, :] + 1e-8

        raw_sim = dot_products / norm_outer

        co_counts = self.train_mask.T.astype(float) @ self.train_mask.astype(float)
        significance = np.minimum(co_counts, self.shrinkage) / self.shrinkage

        self.similarity_matrix = np.clip(raw_sim * significance, -1.0, 1.0)
        np.fill_diagonal(self.similarity_matrix, 0.0)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if not (0 <= user_id < self.num_users and 0 <= service_id < self.num_services):
            return 0.0

        # Services that this user has invoked
        candidate_mask = self.train_mask[user_id, :]
        candidate_mask[service_id] = False
        candidates = np.where(candidate_mask)[0]

        if len(candidates) == 0:
            return float(self.service_means[service_id])

        sims = self.similarity_matrix[service_id, candidates]
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return float(self.service_means[service_id])

        cand_services = candidates[pos_mask]
        cand_sims = sims[pos_mask]

        if len(cand_services) > self.k_neighbors:
            top_k_idx = np.argsort(cand_sims)[-self.k_neighbors:]
            cand_services = cand_services[top_k_idx]
            cand_sims = cand_sims[top_k_idx]

        residuals = self.train_dense[user_id, cand_services] - self.service_means[cand_services]
        sim_sum = np.sum(cand_sims)

        if sim_sum == 0:
            return float(self.service_means[service_id])

        prediction = self.service_means[service_id] + (np.sum(cand_sims * residuals) / sim_sum)
        return float(max(0.0, prediction))

    def predict_matrix(self) -> np.ndarray:
        pred_matrix = np.zeros((self.num_users, self.num_services), dtype=np.float32)
        for u in range(self.num_users):
            for s in range(self.num_services):
                pred_matrix[u, s] = self.predict(u, s)
        return pred_matrix
