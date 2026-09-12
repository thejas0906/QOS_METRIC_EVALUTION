"""
Bipartite QoS Graph Construction.
Constructs heterogeneous bipartite graph representations where:
- Nodes: Users and Services
- Edges: Historic User-Service QoS invocations
- Node Features: User profiles / structural embeddings and Service Cost / SLA features
- Edge Features: Multi-attribute QoS values (Response Time, Availability, Reliability, Throughput, Latency)
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BipartiteGraphData:
    """Lightweight and modular graph container compatible with PyTorch tensors."""
    num_users: int
    num_services: int
    user_features: np.ndarray       # Shape (num_users, in_dim_u)
    service_features: np.ndarray    # Shape (num_services, in_dim_s)
    edge_u: np.ndarray              # Shape (num_edges,) - User source index
    edge_s: np.ndarray              # Shape (num_edges,) - Service target index
    edge_qos: np.ndarray            # Shape (num_edges, num_qos_attributes)
    service_costs: np.ndarray       # Shape (num_services,)


class BipartiteQoSGraphBuilder:
    """Constructs bipartite user-service graphs from multi-attribute QoS matrices."""

    @classmethod
    def create_initial_features(
        cls,
        num_users: int,
        num_services: int,
        service_costs: np.ndarray,
        feature_dim: int = 32,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates initial node feature vectors:
        - User features: Initialized with reproducible projection embeddings.
        - Service features: Concatenation of normalized cost, quality priors, and projection embeddings.
        """
        rng = np.random.RandomState(seed)

        # User initial representations
        user_x = rng.normal(0.0, 1.0 / np.sqrt(feature_dim), (num_users, feature_dim)).astype(np.float32)

        # Service initial representations incorporates monetary cost as a first-class feature
        service_base = rng.normal(0.0, 1.0 / np.sqrt(feature_dim - 1), (num_services, feature_dim - 1)).astype(np.float32)
        norm_cost = ((service_costs - np.min(service_costs)) / (np.max(service_costs) - np.min(service_costs) + 1e-6)).astype(np.float32)
        service_x = np.hstack([service_base, norm_cost[:, np.newaxis]]).astype(np.float32)

        return user_x, service_x

    @classmethod
    def build_from_matrices(
        cls,
        qos_matrices: Dict[str, np.ndarray],
        observation_mask: np.ndarray,
        service_costs: np.ndarray,
        feature_dim: int = 32,
        seed: int = 42
    ) -> BipartiteGraphData:
        """
        Builds graph container from multi-attribute QoS matrices and observation mask.
        """
        first_mat = next(iter(qos_matrices.values()))
        num_users, num_services = first_mat.shape

        coords = np.argwhere(observation_mask)
        edge_u = coords[:, 0].astype(np.int64)
        edge_s = coords[:, 1].astype(np.int64)

        # Stack QoS attributes across edges
        attr_names = sorted(qos_matrices.keys())
        stacked_qos = np.column_stack([
            qos_matrices[attr][edge_u, edge_s]
            for attr in attr_names
        ]).astype(np.float32)

        user_x, service_x = cls.create_initial_features(
            num_users=num_users,
            num_services=num_services,
            service_costs=service_costs,
            feature_dim=feature_dim,
            seed=seed
        )

        return BipartiteGraphData(
            num_users=num_users,
            num_services=num_services,
            user_features=user_x,
            service_features=service_x,
            edge_u=edge_u,
            edge_s=edge_s,
            edge_qos=stacked_qos,
            service_costs=service_costs
        )
