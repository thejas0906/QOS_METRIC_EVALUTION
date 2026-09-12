"""
Edge-Level QoS Prediction Decoder.
Multi-Layer Perceptron (MLP) mapping concatenated user and service embeddings
into multi-attribute continuous QoS predictions.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeQoSDecoder(nn.Module):
    """Deep non-linear edge regression head for multi-attribute QoS estimation."""

    def __init__(
        self,
        u_dim: int = 32,
        s_dim: int = 32,
        num_qos_attributes: int = 5,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1
    ):
        super().__init__()
        in_dim = u_dim + s_dim + u_dim  # Concatenation + element-wise interaction

        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        # Final regression head for multi-attribute QoS outputs
        layers.append(nn.Linear(curr_dim, num_qos_attributes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_u: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of user and service embeddings.
        
        Args:
            h_u: User embeddings tensor (B, u_dim)
            h_s: Service embeddings tensor (B, s_dim)
            
        Returns:
            Continuous QoS predictions tensor (B, num_qos_attributes)
        """
        # Rich relational representation
        interaction = h_u * h_s
        joint_representation = torch.cat([h_u, h_s, interaction], dim=-1)

        raw_pred = self.mlp(joint_representation)
        # Ensure predictions are positive since QoS metrics are non-negative
        return F.softplus(raw_pred)
