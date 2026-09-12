"""
Abstract Base Class for Matrix-Based QoS Prediction Models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pickle
import os
import numpy as np
import scipy.sparse as sp


class BaseMatrixQoSModel(ABC):
    """Unified interface for Baselines, Collaborative Filtering, and Matrix Factorization models."""

    def __init__(self, model_name: str = "BaseModel"):
        self.model_name = model_name
        self.num_users: int = 0
        self.num_services: int = 0
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, train_matrix: sp.csr_matrix, val_matrix: Optional[sp.csr_matrix] = None) -> "BaseMatrixQoSModel":
        """Fits model parameters to sparse training matrix."""
        pass

    @abstractmethod
    def predict(self, user_id: int, service_id: int) -> float:
        """Predicts QoS value for a single user-service pair."""
        pass

    @abstractmethod
    def predict_matrix(self) -> np.ndarray:
        """Reconstructs full dense predicted matrix of shape (num_users, num_services)."""
        pass

    def save_model(self, file_path: str) -> None:
        """Serializes model parameters to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, file_path: str) -> "BaseMatrixQoSModel":
        """Deserializes model parameters from disk."""
        with open(file_path, "rb") as f:
            return pickle.load(f)
