"""
Graph-Based Prediction Reliability and Confidence Estimator for Team B.
Calculates structural prediction confidence from bipartite user-service graph topology:
- User node degrees (invocation history volume)
- Service node degrees (service observation popularity)
- Number of observed interactions & local graph density
- Normalized confidence scores bounded strictly in [0, 1]
"""

from typing import Dict, Optional, Tuple
import numpy as np


class GraphConfidenceEstimator:
    """Estimates prediction reliability based on bipartite interaction graph structure."""

    def __init__(
        self,
        w_pair: float = 0.60,
        w_density: float = 0.25,
        w_obs: float = 0.15,
        eps: float = 1e-6
    ):
        """
        Args:
            w_pair: Weight for pairwise harmonic degree support.
            w_density: Weight for local bipartite density support.
            w_obs: Bonus weight for directly observed historical edges.
            eps: Epsilon for numerical stability.
        """
        total_w = w_pair + w_density + w_obs
        if total_w <= 0:
            total_w = 1.0
        self.w_pair = float(w_pair / total_w)
        self.w_density = float(w_density / total_w)
        self.w_obs = float(w_obs / total_w)
        self.eps = eps

        # State computed during fit
        self.num_users: int = 0
        self.num_services: int = 0
        self.user_degrees: np.ndarray = np.array([], dtype=np.float32)
        self.service_degrees: np.ndarray = np.array([], dtype=np.float32)
        self.observation_mask: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit_from_mask(self, observation_mask: np.ndarray) -> "GraphConfidenceEstimator":
        """
        Fits graph statistics from a boolean or binary 2D observation mask.
        
        Args:
            observation_mask: Array of shape (num_users, num_services) where True/1 indicates observed interaction.
        """
        self.num_users, self.num_services = observation_mask.shape
        self.observation_mask = (observation_mask > 0).astype(bool)

        # Compute degrees
        self.user_degrees = np.sum(self.observation_mask, axis=1).astype(np.float32)
        self.service_degrees = np.sum(self.observation_mask, axis=0).astype(np.float32)
        self.fitted = True
        return self

    def fit_from_edges(
        self,
        edge_u: np.ndarray,
        edge_s: np.ndarray,
        num_users: int,
        num_services: int
    ) -> "GraphConfidenceEstimator":
        """
        Fits graph statistics from coordinate edge index arrays.
        """
        self.num_users = int(num_users)
        self.num_services = int(num_services)

        self.user_degrees = np.zeros(self.num_users, dtype=np.float32)
        self.service_degrees = np.zeros(self.num_services, dtype=np.float32)

        if len(edge_u) > 0 and len(edge_s) > 0:
            np.add.at(self.user_degrees, edge_u, 1.0)
            np.add.at(self.service_degrees, edge_s, 1.0)

            self.observation_mask = np.zeros((self.num_users, self.num_services), dtype=bool)
            self.observation_mask[edge_u, edge_s] = True
        else:
            self.observation_mask = np.zeros((self.num_users, self.num_services), dtype=bool)

        self.fitted = True
        return self

    def compute_confidence_matrix(self) -> np.ndarray:
        """
        Computes normalized confidence scores C(u, s) in [0, 1] for all user-service pairs.
        
        Returns:
            2D numpy array of shape (num_users, num_services).
        """
        if not self.fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit_from_mask or fit_from_edges first.")

        # 1. Logarithmic Degree Centrality
        max_u_deg = float(np.max(self.user_degrees)) if len(self.user_degrees) > 0 else 0.0
        max_s_deg = float(np.max(self.service_degrees)) if len(self.service_degrees) > 0 else 0.0

        denom_u = np.log1p(max_u_deg) if max_u_deg > 0 else 1.0
        denom_s = np.log1p(max_s_deg) if max_s_deg > 0 else 1.0

        u_support = np.log1p(self.user_degrees) / denom_u  # Shape: (num_users,)
        s_support = np.log1p(self.service_degrees) / denom_s  # Shape: (num_services,)

        # 2. Pairwise Harmonic Interaction Support
        # Broadcast to 2D grid (num_users, num_services)
        u_grid = u_support[:, np.newaxis]
        s_grid = s_support[np.newaxis, :]

        pair_support = (2.0 * u_grid * s_grid) / (u_grid + s_grid + self.eps)
        pair_support = np.clip(pair_support, 0.0, 1.0)

        # 3. Local Bipartite Density Support
        max_total = max_u_deg + max_s_deg
        denom_density = np.log1p(max_total) if max_total > 0 else 1.0

        density_matrix = np.log1p(self.user_degrees[:, np.newaxis] + self.service_degrees[np.newaxis, :]) / denom_density
        density_matrix = np.clip(density_matrix, 0.0, 1.0)

        # 4. Direct Observation Bonus
        obs_bonus = self.observation_mask.astype(np.float32) if self.observation_mask is not None else np.zeros((self.num_users, self.num_services), dtype=np.float32)

        # 5. Composite Normalized Confidence
        confidence = (
            self.w_pair * pair_support +
            self.w_density * density_matrix +
            self.w_obs * obs_bonus
        )
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def get_confidence_stats(self, confidence_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Returns statistical summary of confidence distribution.
        """
        mat = confidence_matrix if confidence_matrix is not None else self.compute_confidence_matrix()
        flat = mat.flatten()
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "median": float(np.median(flat)),
            "p25": float(np.percentile(flat, 25.0)),
            "p75": float(np.percentile(flat, 75.0)),
        }
