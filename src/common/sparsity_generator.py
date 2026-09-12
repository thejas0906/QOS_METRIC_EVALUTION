"""
Sparsity Generator for QoS Robustness Benchmarks.
Creates observation masks at controlled density levels from full QoS matrices,
preserving a fixed held-out test set for fair cross-sparsity comparison.
"""

from typing import Dict, Tuple
import numpy as np


class SparsityGenerator:
    """Generates observation masks at multiple density levels for sparsity robustness experiments."""

    @staticmethod
    def generate_density_masks(
        qos_matrix: np.ndarray,
        density_levels: list,
        test_ratio: float = 0.20,
        val_ratio: float = 0.10,
        seed: int = 42
    ) -> Dict[float, Dict[str, np.ndarray]]:
        """
        Generates train/val/test masks at each target density level.

        Design: The test and validation sets are fixed across all density levels
        for fair comparison. Their sizes are determined by the *smallest* density
        level's budget so that even the sparsest setting has training data.

        At each density level, we sample `target = total_valid * density` entries
        from the shuffled pool. The first `num_test` go to test, the next `num_val`
        go to validation, and the rest go to training.

        Args:
            qos_matrix: 2D QoS matrix (num_users x num_services).
            density_levels: List of target observed densities (e.g. [0.50, 0.30, 0.20, 0.10, 0.05]).
            test_ratio: Fraction of each density budget reserved for test.
            val_ratio: Fraction of each density budget reserved for validation.
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping density -> {"observed_mask", "train_mask", "val_mask", "test_mask",
                                     "num_observed", "num_train", "num_val", "num_test"}
        """
        rng = np.random.RandomState(seed)

        # Identify all valid (non-missing) entries
        valid_mask = (qos_matrix > 0) & (~np.isnan(qos_matrix))
        valid_coords = np.argwhere(valid_mask)
        total_valid = len(valid_coords)

        if total_valid == 0:
            raise ValueError("QoS matrix contains no valid entries.")

        # Shuffle all valid entries once (deterministic ordering)
        shuffled_indices = rng.permutation(total_valid)

        # Size the FIXED test and val sets based on the smallest density level
        # so every level has room for training data.
        min_density = min(density_levels)
        min_budget = max(1, int(total_valid * min_density))
        num_test = max(1, int(min_budget * test_ratio))
        num_val = max(1, int(min_budget * val_ratio))

        # Carve out fixed test and val sets from the start of the shuffle
        test_indices = shuffled_indices[:num_test]
        val_indices = shuffled_indices[num_test:num_test + num_val]
        trainable_pool = shuffled_indices[num_test + num_val:]

        # Build fixed test and val masks
        test_mask = np.zeros(qos_matrix.shape, dtype=bool)
        val_mask = np.zeros(qos_matrix.shape, dtype=bool)

        for idx in test_indices:
            r, c = valid_coords[idx]
            test_mask[r, c] = True

        for idx in val_indices:
            r, c = valid_coords[idx]
            val_mask[r, c] = True

        # For each density level, subsample training entries from the trainable pool
        results = {}

        for density in sorted(density_levels, reverse=True):
            # Total observed entries at this density
            target_observed = max(1, int(total_valid * density))

            # Training budget = total observed minus the fixed test + val
            num_train = max(1, target_observed - num_test - num_val)
            num_train = min(num_train, len(trainable_pool))

            # Select first num_train entries from the trainable pool
            train_sample_indices = trainable_pool[:num_train]

            train_mask = np.zeros(qos_matrix.shape, dtype=bool)
            for idx in train_sample_indices:
                r, c = valid_coords[idx]
                train_mask[r, c] = True

            observed_mask = train_mask | val_mask | test_mask

            results[density] = {
                "observed_mask": observed_mask,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "num_observed": int(np.sum(observed_mask)),
                "num_train": int(np.sum(train_mask)),
                "num_val": int(np.sum(val_mask)),
                "num_test": int(np.sum(test_mask)),
                "actual_density": float(np.sum(observed_mask)) / float(qos_matrix.size)
            }

        return results

    @staticmethod
    def generate_graph_edge_masks(
        qos_matrices: Dict[str, np.ndarray],
        observation_mask: np.ndarray,
        test_ratio: float = 0.20,
        val_ratio: float = 0.10,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        From an observation mask, extracts edge arrays suitable for Team B's graph pipeline.

        Returns:
            (edge_u, edge_s, edge_qos, val_mask_bool, test_mask_bool)
            where edge_u/edge_s/edge_qos are the training edges,
            and val/test masks indicate positions used for evaluation.
        """
        coords = np.argwhere(observation_mask)
        edge_u = coords[:, 0].astype(np.int64)
        edge_s = coords[:, 1].astype(np.int64)

        attr_names = sorted(qos_matrices.keys())
        stacked_qos = np.column_stack([
            qos_matrices[attr][edge_u, edge_s]
            for attr in attr_names
        ]).astype(np.float32)

        # Split edges into train/val/test
        rng = np.random.RandomState(seed)
        total_edges = len(edge_u)
        shuffled = rng.permutation(total_edges)

        num_test = int(total_edges * test_ratio)
        num_val = int(total_edges * val_ratio)

        idx_test = shuffled[:num_test]
        idx_val = shuffled[num_test:num_test + num_val]
        idx_train = shuffled[num_test + num_val:]

        return {
            "train": {"u": edge_u[idx_train], "s": edge_s[idx_train], "qos": stacked_qos[idx_train]},
            "val": {"u": edge_u[idx_val], "s": edge_s[idx_val], "qos": stacked_qos[idx_val]},
            "test": {"u": edge_u[idx_test], "s": edge_s[idx_test], "qos": stacked_qos[idx_test]}
        }
