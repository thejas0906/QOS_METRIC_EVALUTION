"""
Multi-Attribute QoS Prediction Coordinator for Team A.
Orchestrates prediction models across all five QoS dimensions:
Response Time, Availability, Reliability, Throughput, and Latency.
"""

from typing import Dict, Optional, Type
import numpy as np
import scipy.sparse as sp
from src.common.constants import QoSAttribute
from src.team_a_matrix.models.base_model import BaseMatrixQoSModel


class TeamAQoSPredictor:
    """Manages dedicated or unified matrix models for each QoS dimension."""

    def __init__(self):
        self.models: Dict[str, BaseMatrixQoSModel] = {}
        self.is_fitted: bool = False

    def register_model(self, attribute: str, model: BaseMatrixQoSModel) -> None:
        """Assigns an initialized matrix model to a specific QoS attribute."""
        self.models[attribute] = model

    def fit_all(
        self,
        train_matrices: Dict[str, sp.csr_matrix],
        val_matrices: Optional[Dict[str, sp.csr_matrix]] = None
    ) -> "TeamAQoSPredictor":
        """Fits every registered attribute model on its corresponding sparse matrix."""
        for attr, model in self.models.items():
            if attr not in train_matrices:
                continue
            val_mat = val_matrices.get(attr) if val_matrices else None
            model.fit(train_matrices[attr], val_mat)

        self.is_fitted = True
        return self

    def predict_attribute(self, attribute: str, user_id: int, service_id: int) -> float:
        """Predicts single QoS attribute."""
        if attribute not in self.models:
            raise KeyError(f"No model registered for QoS attribute: {attribute}")
        return self.models[attribute].predict(user_id, service_id)

    def predict_all_attributes(self, user_id: int, service_id: int) -> Dict[str, float]:
        """Predicts all registered QoS attributes for a user-service invocation."""
        return {
            attr: model.predict(user_id, service_id)
            for attr, model in self.models.items()
        }

    def predict_matrix_dict(self) -> Dict[str, np.ndarray]:
        """Returns reconstructed dense predicted matrices for all attributes."""
        return {
            attr: model.predict_matrix()
            for attr, model in self.models.items()
        }
