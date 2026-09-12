"""
QoS Matrix Construction.
Constructs SciPy sparse CSR/COO representations and extracts stochastic coordinate triplets.
"""

from typing import Tuple
import numpy as np
import scipy.sparse as sp


class QoSMatrixBuilder:
    """Converts numpy arrays and observation masks into optimized sparse structures."""

    @staticmethod
    def build_csr_matrix(matrix: np.ndarray, mask: np.ndarray) -> sp.csr_matrix:
        """
        Builds a CSR matrix containing only values at masked coordinates.
        """
        masked_data = np.where(mask, matrix, 0.0)
        return sp.csr_matrix(masked_data, dtype=np.float32)

    @staticmethod
    def extract_triplets(matrix: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts coordinate triplets (user_id, service_id, qos_value) for masked entries.
        """
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

        users = coords[:, 0]
        services = coords[:, 1]
        values = matrix[users, services].astype(np.float32)

        return users, services, values
