"""
QoS Normalization and Direction-Aware Utility Transformation.
Ensures heterogeneous attributes (Response Time, Latency vs Availability, Throughput, Reliability)
are mapped onto a uniform [0, 1] utility scale where higher values always signify superior performance.
"""

from typing import Union, Optional
import numpy as np
from src.common.constants import QoSAttribute, QoSDirection, QOS_DIRECTIONS


class QoSNormalizer:
    """Direction-aware Min-Max scaler for continuous QoS metrics."""

    def __init__(self, attribute: Union[str, QoSAttribute], direction: Optional[QoSDirection] = None):
        self.attribute = QoSAttribute(attribute) if isinstance(attribute, str) else attribute
        self.direction = direction if direction is not None else QOS_DIRECTIONS[self.attribute]
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.fitted: bool = False

    def fit(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> "QoSNormalizer":
        """
        Calculates min and max on observed entries.
        
        Args:
            values: Array of QoS values.
            mask: Optional boolean mask of valid observed entries.
        """
        valid_vals = values[mask] if mask is not None else values
        # Exclude non-positive or negative dummy missing values like -1
        valid_vals = valid_vals[valid_vals >= 0]

        if len(valid_vals) == 0:
            self.min_val = 0.0
            self.max_val = 1.0
        else:
            self.min_val = float(np.min(valid_vals))
            self.max_val = float(np.max(valid_vals))
            # Guard against zero range
            if self.max_val == self.min_val:
                self.max_val = self.min_val + 1e-6

        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Transforms QoS values to [0, 1] utility score.
        Positive: (x - min) / (max - min)
        Negative: (max - x) / (max - min)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")

        # Clip values to training range to avoid negative or >1 normalized values
        clipped = np.clip(values, self.min_val, self.max_val)
        denom = self.max_val - self.min_val

        if self.direction == QoSDirection.POSITIVE:
            normalized = (clipped - self.min_val) / denom
        else:
            normalized = (self.max_val - clipped) / denom

        return np.clip(normalized, 0.0, 1.0)

    def inverse_transform(self, normalized_values: np.ndarray) -> np.ndarray:
        """Denormalizes [0, 1] utility scores back to original unit values."""
        if not self.fitted:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")

        denom = self.max_val - self.min_val
        if self.direction == QoSDirection.POSITIVE:
            original = normalized_values * denom + self.min_val
        else:
            original = self.max_val - (normalized_values * denom)

        return original
