"""
Graph Data Preprocessor for Team B.
Constructs inductive node splits (unseen users, unseen services)
and partitions interaction edges into train, validation, and test subsets.
"""

from typing import Dict, Tuple
import numpy as np


class GraphDataPreprocessor:
    """Partitions bipartite graphs for transductive and inductive evaluations."""

    @staticmethod
    def create_inductive_node_split(
        num_users: int,
        num_services: int,
        cold_user_ratio: float = 0.10,
        cold_service_ratio: float = 0.10,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Partitions user and service indices into warm (training graph) and cold (unseen) sets.
        """
        rng = np.random.RandomState(seed)

        # Users split
        shuffled_users = rng.permutation(num_users)
        num_cold_u = max(1, int(num_users * cold_user_ratio))
        cold_users = shuffled_users[:num_cold_u]
        warm_users = shuffled_users[num_cold_u:]

        # Services split
        shuffled_services = rng.permutation(num_services)
        num_cold_s = max(1, int(num_services * cold_service_ratio))
        cold_services = shuffled_services[:num_cold_s]
        warm_services = shuffled_services[num_cold_s:]

        return {
            "warm_users": np.sort(warm_users),
            "cold_users": np.sort(cold_users),
            "warm_services": np.sort(warm_services),
            "cold_services": np.sort(cold_services),
        }

    @staticmethod
    def split_interaction_edges(
        u_indices: np.ndarray,
        s_indices: np.ndarray,
        qos_attributes: np.ndarray,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Splits observed interaction edges into train, val, and test splits.
        
        Args:
            u_indices: 1D array of user indices.
            s_indices: 1D array of service indices.
            qos_attributes: 2D array of shape (E, num_attributes).
        """
        rng = np.random.RandomState(seed)
        total_edges = len(u_indices)

        shuffled = rng.permutation(total_edges)
        num_test = int(total_edges * test_ratio)
        num_val = int(total_edges * val_ratio)
        num_train = total_edges - num_test - num_val

        idx_test = shuffled[:num_test]
        idx_val = shuffled[num_test:num_test + num_val]
        idx_train = shuffled[num_test + num_val:]

        return {
            "train": {
                "u": u_indices[idx_train],
                "s": s_indices[idx_train],
                "qos": qos_attributes[idx_train]
            },
            "val": {
                "u": u_indices[idx_val],
                "s": s_indices[idx_val],
                "qos": qos_attributes[idx_val]
            },
            "test": {
                "u": u_indices[idx_test],
                "s": s_indices[idx_test],
                "qos": qos_attributes[idx_test]
            }
        }
