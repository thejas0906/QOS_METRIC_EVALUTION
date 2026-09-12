"""
Inductive Cold-Start Inference Handler for Team B.
Evaluates model generalization to unseen users and unseen services without retraining.
"""

from typing import Dict, List, Set, Any, Tuple
import numpy as np
import torch
from src.common.metrics_prediction import PredictionMetricsEvaluator
from src.common.metrics_recommendation import RecommendationMetricsEvaluator
from src.team_b_gnn.models.graphsage_module import HeteroGraphSAGE
from src.team_b_gnn.models.qos_prediction_decoder import EdgeQoSDecoder


class ColdStartInferenceHandler:
    """Handles inductive embedding extraction and benchmark evaluation for zero-shot cold nodes."""

    @staticmethod
    def infer_cold_embeddings(
        model: HeteroGraphSAGE,
        user_features: np.ndarray,
        service_features: np.ndarray,
        cold_users: np.ndarray,
        cold_services: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes representations for unseen nodes through inductive feature propagation.
        """
        model.eval()
        with torch.no_grad():
            u_x = torch.from_numpy(user_features).float()
            s_x = torch.from_numpy(service_features).float()

            # Pass through model with empty interaction edges to simulate pure inductive prior
            empty_u = torch.empty(0, dtype=torch.int64)
            empty_s = torch.empty(0, dtype=torch.int64)

            h_u, h_s = model(u_x, s_x, empty_u, empty_s)

        return h_u, h_s

    @classmethod
    def evaluate_cold_start(
        cls,
        model: HeteroGraphSAGE,
        decoder: EdgeQoSDecoder,
        user_features: np.ndarray,
        service_features: np.ndarray,
        cold_indices: np.ndarray,
        ground_truth_qos: Dict[str, np.ndarray],
        mode: str = "user",  # "user" or "service"
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluates cold-start prediction and recommendation accuracy.
        
        Args:
            mode: "user" (evaluates unseen users) or "service" (evaluates unseen services).
        """
        model.eval()
        decoder.eval()

        with torch.no_grad():
            u_x = torch.from_numpy(user_features).float()
            s_x = torch.from_numpy(service_features).float()
            empty_u = torch.empty(0, dtype=torch.int64)
            empty_s = torch.empty(0, dtype=torch.int64)

            h_u, h_s = model(u_x, s_x, empty_u, empty_s)

            # Predict across all user-service combinations
            num_u, num_s = user_features.shape[0], service_features.shape[0]
            # Grid combinations
            u_rep = h_u.repeat_interleave(num_s, dim=0)
            s_rep = h_s.repeat(num_u, 1)

            pred_all = decoder(u_rep, s_rep).numpy()  # (num_u * num_s, num_attributes)
            attr_names = sorted(ground_truth_qos.keys())

            pred_dict = {}
            for i, attr in enumerate(attr_names):
                pred_dict[attr] = pred_all[:, i].reshape(num_u, num_s)

        # Build cold mask
        cold_mask = np.zeros((num_u, num_s), dtype=bool)
        if mode == "user":
            cold_mask[cold_indices, :] = True
        else:
            cold_mask[:, cold_indices] = True

        # Prediction errors
        errors = {}
        for attr in attr_names:
            errors[attr] = PredictionMetricsEvaluator.compute_all_prediction_metrics(
                y_true=ground_truth_qos[attr],
                y_pred=pred_dict[attr],
                mask=cold_mask
            )

        return {
            "mode": mode,
            "cold_count": len(cold_indices),
            "prediction_errors": errors
        }
