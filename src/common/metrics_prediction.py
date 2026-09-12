"""
Prediction Error Metrics for Continuous QoS Regression.
Calculates statistical errors (RMSE, MAE, NRMSE) on holdout entries.
"""

from typing import Dict, Optional
import numpy as np


class PredictionMetricsEvaluator:
    """Evaluates continuous regression fidelity between ground truth and predicted QoS matrices/vectors."""

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Computes Root Mean Squared Error.
        
        Args:
            y_true: Ground truth QoS array.
            y_pred: Predicted QoS array.
            mask: Optional boolean array marking evaluation positions.
        """
        if mask is not None:
            t = y_true[mask]
            p = y_pred[mask]
        else:
            t = y_true.flatten()
            p = y_pred.flatten()

        if len(t) == 0:
            return 0.0

        mse = np.mean((t - p) ** 2)
        return float(np.sqrt(mse))

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Computes Mean Absolute Error.
        
        Args:
            y_true: Ground truth QoS array.
            y_pred: Predicted QoS array.
            mask: Optional boolean array marking evaluation positions.
        """
        if mask is not None:
            t = y_true[mask]
            p = y_pred[mask]
        else:
            t = y_true.flatten()
            p = y_pred.flatten()

        if len(t) == 0:
            return 0.0

        return float(np.mean(np.abs(t - p)))

    @classmethod
    def compute_all_prediction_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Computes RMSE, MAE, and Normalized RMSE (NRMSE).
        """
        rmse = cls.compute_rmse(y_true, y_pred, mask)
        mae = cls.compute_mae(y_true, y_pred, mask)

        t = y_true[mask] if mask is not None else y_true.flatten()
        val_range = float(np.max(t) - np.min(t)) if len(t) > 0 else 1.0
        nrmse = rmse / val_range if val_range > 0 else 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "nrmse": nrmse
        }
