"""
Sparse Matrix Preprocessing and Density Reduction.
Simulates sparse QoS interaction matrices (e.g., 5%, 10%, 20% observed density)
and partitions observed cells into train, validation, and test subsets.
"""

from typing import Tuple
import numpy as np


class MatrixPreprocessor:
    """Handles matrix sparsity masking and train-val-test partitioning."""

    @staticmethod
    def create_sparsity_mask(
        matrix: np.ndarray,
        target_density: float = 0.10,
        seed: int = 42
    ) -> np.ndarray:
        """
        Creates a boolean mask retaining only a target fraction of available entries.
        
        Args:
            matrix: 2D QoS matrix.
            target_density: Fraction of entries to observe (e.g. 0.10 for 90% sparsity).
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)
        # Identify non-missing entries
        valid_coords = np.argwhere((matrix > 0) & (~np.isnan(matrix)))
        total_valid = len(valid_coords)

        if total_valid == 0:
            raise ValueError("Provided matrix contains no valid non-zero entries.")

        num_sample = max(1, int(total_valid * target_density))
        chosen_indices = rng.choice(total_valid, size=num_sample, replace=False)

        mask = np.zeros(matrix.shape, dtype=bool)
        for idx in chosen_indices:
            r, c = valid_coords[idx]
            mask[r, c] = True

        return mask

    @staticmethod
    def train_val_test_split(
        matrix: np.ndarray,
        observed_mask: np.ndarray,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partitions the observed entries into train, validation, and test masks.
        
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        rng = np.random.RandomState(seed)
        coords = np.argwhere(observed_mask)
        total_obs = len(coords)

        shuffled_indices = rng.permutation(total_obs)

        num_test = int(total_obs * test_ratio)
        num_val = int(total_obs * val_ratio)
        num_train = total_obs - num_test - num_val

        test_indices = shuffled_indices[:num_test]
        val_indices = shuffled_indices[num_test:num_test + num_val]
        train_indices = shuffled_indices[num_test + num_val:]

        train_mask = np.zeros(matrix.shape, dtype=bool)
        val_mask = np.zeros(matrix.shape, dtype=bool)
        test_mask = np.zeros(matrix.shape, dtype=bool)

        for idx in train_indices:
            r, c = coords[idx]
            train_mask[r, c] = True

        for idx in val_indices:
            r, c = coords[idx]
            val_mask[r, c] = True

        for idx in test_indices:
            r, c = coords[idx]
            test_mask[r, c] = True

        return train_mask, val_mask, test_mask
