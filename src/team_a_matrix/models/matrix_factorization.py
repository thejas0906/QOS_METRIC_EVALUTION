"""
Matrix Factorization Models for Continuous QoS Prediction.
Implements Probabilistic Matrix Factorization (PMF) and Biased Matrix Factorization (BiasedMF)
optimized with Mini-Batch Stochastic Gradient Descent and Early Stopping.
"""

from typing import Optional, List
import numpy as np
import scipy.sparse as sp
from src.team_a_matrix.models.base_model import BaseMatrixQoSModel


class ProbabilisticMatrixFactorization(BaseMatrixQoSModel):
    """
    Probabilistic Matrix Factorization (PMF).
    Decomposes the QoS matrix into user latent factors U and service latent factors V.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 5,
        seed: int = 42
    ):
        super().__init__(model_name="PMF")
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        self.U: np.ndarray = np.array([])  # (num_users, latent_dim)
        self.V: np.ndarray = np.array([])  # (num_services, latent_dim)
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "ProbabilisticMatrixFactorization":
        rng = np.random.RandomState(self.seed)
        self.num_users, self.num_services = train_matrix.shape

        # Initialize factors with small variance
        scale = 1.0 / np.sqrt(self.latent_dim)
        self.U = rng.normal(0.0, scale, (self.num_users, self.latent_dim)).astype(np.float32)
        self.V = rng.normal(0.0, scale, (self.num_services, self.latent_dim)).astype(np.float32)

        coo_train = train_matrix.tocoo()
        train_u = coo_train.row
        train_i = coo_train.col
        train_r = coo_train.data
        n_train = len(train_r)

        # Scale factor for numerical stability across diverse QoS scales
        self.scale = max(float(np.std(train_r)), 1.0)
        norm_r = train_r / self.scale

        coo_val = val_matrix.tocoo() if val_matrix is not None else None
        norm_val_r = (coo_val.data / self.scale) if coo_val is not None else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_U = np.copy(self.U)
        best_V = np.copy(self.V)

        for epoch in range(self.epochs):
            perm = rng.permutation(n_train)
            u_shuf = train_u[perm]
            i_shuf = train_i[perm]
            r_shuf = norm_r[perm]

            total_loss = 0.0
            for start_idx in range(0, n_train, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_train)
                b_u = u_shuf[start_idx:end_idx]
                b_i = i_shuf[start_idx:end_idx]
                b_r = r_shuf[start_idx:end_idx]

                # Forward: dot product of U and V
                preds = np.sum(self.U[b_u] * self.V[b_i], axis=1)
                errors = b_r - preds  # (batch,)
                total_loss += float(np.sum(errors ** 2))

                # Gradients with gradient clipping
                u_factors = self.U[b_u]
                v_factors = self.V[b_i]

                grad_u = np.clip(-errors[:, np.newaxis] * v_factors + self.reg * u_factors, -5.0, 5.0)
                grad_v = np.clip(-errors[:, np.newaxis] * u_factors + self.reg * v_factors, -5.0, 5.0)

                # Vectorized scatter update
                np.add.at(self.U, b_u, -self.lr * grad_u)
                np.add.at(self.V, b_i, -self.lr * grad_v)

            train_loss = (total_loss / max(n_train, 1)) * (self.scale ** 2)
            self.train_loss_history.append(train_loss)

            # Validation evaluation
            if coo_val is not None and len(coo_val.data) > 0:
                val_preds = np.sum(self.U[coo_val.row] * self.V[coo_val.col], axis=1)
                val_loss = float(np.mean((norm_val_r - val_preds) ** 2)) * (self.scale ** 2)
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_U = np.copy(self.U)
                    best_V = np.copy(self.V)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.U = best_U
                        self.V = best_V
                        break

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if not (0 <= user_id < self.num_users and 0 <= service_id < self.num_services):
            return 0.0
        val = float(np.dot(self.U[user_id], self.V[service_id])) * self.scale
        return max(0.0, val)

    def predict_matrix(self) -> np.ndarray:
        pred_mat = (self.U @ self.V.T) * self.scale
        return np.clip(pred_mat, 0.0, None).astype(np.float32)


class BiasedMatrixFactorization(BaseMatrixQoSModel):
    """
    Biased Matrix Factorization (BiasedMF).
    q_hat_{u,i} = mu + b_u + b_i + u_u^T v_i
    Captures user-specific biases, service baseline qualities, and latent interactions.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 5,
        seed: int = 42
    ):
        super().__init__(model_name="BiasedMF")
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        self.mu: float = 0.0
        self.b_u: np.ndarray = np.array([])  # (num_users,)
        self.b_i: np.ndarray = np.array([])  # (num_services,)
        self.U: np.ndarray = np.array([])    # (num_users, latent_dim)
        self.V: np.ndarray = np.array([])    # (num_services, latent_dim)
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "BiasedMatrixFactorization":
        rng = np.random.RandomState(self.seed)
        self.num_users, self.num_services = train_matrix.shape

        coo_train = train_matrix.tocoo()
        train_u = coo_train.row
        train_i = coo_train.col
        train_r = coo_train.data
        n_train = len(train_r)

        # Scale factor for numerical stability
        self.scale = max(float(np.std(train_r)), 1.0)
        norm_r = train_r / self.scale

        # Global mean initialization on normalized scale
        self.mu = float(np.mean(norm_r)) if len(norm_r) > 0 else 0.0

        # Biases
        self.b_u = np.zeros(self.num_users, dtype=np.float32)
        self.b_i = np.zeros(self.num_services, dtype=np.float32)

        # Latent factors
        scale_init = 1.0 / np.sqrt(self.latent_dim)
        self.U = rng.normal(0.0, scale_init, (self.num_users, self.latent_dim)).astype(np.float32)
        self.V = rng.normal(0.0, scale_init, (self.num_services, self.latent_dim)).astype(np.float32)

        coo_val = val_matrix.tocoo() if val_matrix is not None else None
        norm_val_r = (coo_val.data / self.scale) if coo_val is not None else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = (np.copy(self.b_u), np.copy(self.b_i), np.copy(self.U), np.copy(self.V))

        for epoch in range(self.epochs):
            perm = rng.permutation(n_train)
            u_shuf = train_u[perm]
            i_shuf = train_i[perm]
            r_shuf = norm_r[perm]

            total_loss = 0.0
            for start_idx in range(0, n_train, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_train)
                b_u_idx = u_shuf[start_idx:end_idx]
                b_i_idx = i_shuf[start_idx:end_idx]
                b_r = r_shuf[start_idx:end_idx]

                # Prediction
                dot_prod = np.sum(self.U[b_u_idx] * self.V[b_i_idx], axis=1)
                preds = self.mu + self.b_u[b_u_idx] + self.b_i[b_i_idx] + dot_prod
                errors = b_r - preds
                total_loss += float(np.sum(errors ** 2))

                # Gradient updates for biases
                grad_bu = np.clip(-errors + self.reg * self.b_u[b_u_idx], -5.0, 5.0)
                grad_bi = np.clip(-errors + self.reg * self.b_i[b_i_idx], -5.0, 5.0)

                np.add.at(self.b_u, b_u_idx, -self.lr * grad_bu)
                np.add.at(self.b_i, b_i_idx, -self.lr * grad_bi)

                # Gradient updates for latent factors
                u_fac = self.U[b_u_idx]
                v_fac = self.V[b_i_idx]

                grad_u = np.clip(-errors[:, np.newaxis] * v_fac + self.reg * u_fac, -5.0, 5.0)
                grad_v = np.clip(-errors[:, np.newaxis] * u_fac + self.reg * v_fac, -5.0, 5.0)

                np.add.at(self.U, b_u_idx, -self.lr * grad_u)
                np.add.at(self.V, b_i_idx, -self.lr * grad_v)

            train_loss = (total_loss / max(n_train, 1)) * (self.scale ** 2)
            self.train_loss_history.append(train_loss)

            # Validation
            if coo_val is not None and len(coo_val.data) > 0:
                val_dot = np.sum(self.U[coo_val.row] * self.V[coo_val.col], axis=1)
                val_preds = self.mu + self.b_u[coo_val.row] + self.b_i[coo_val.col] + val_dot
                val_loss = float(np.mean((norm_val_r - val_preds) ** 2)) * (self.scale ** 2)
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = (np.copy(self.b_u), np.copy(self.b_i), np.copy(self.U), np.copy(self.V))
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.b_u, self.b_i, self.U, self.V = best_state
                        break

        self.is_fitted = True
        return self

    def predict(self, user_id: int, service_id: int) -> float:
        if not (0 <= user_id < self.num_users and 0 <= service_id < self.num_services):
            return self.mu * self.scale
        val = (self.mu + self.b_u[user_id] + self.b_i[service_id] + float(np.dot(self.U[user_id], self.V[service_id]))) * self.scale
        return float(max(0.0, val))

    def predict_matrix(self) -> np.ndarray:
        pred_mat = (self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + (self.U @ self.V.T)) * self.scale
        return np.clip(pred_mat, 0.0, None).astype(np.float32)
