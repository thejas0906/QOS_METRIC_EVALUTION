"""
Abstract Base Class for GNN-Based Recommendation Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn


class BaseGNNRecommendationModel(nn.Module, ABC):
    """Unified interface for bipartite graph representation learners."""

    def __init__(self, model_name: str = "BaseGNN"):
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def forward(
        self,
        user_x: torch.Tensor,
        service_x: torch.Tensor,
        edge_u: torch.Tensor,
        edge_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Message passing over bipartite user-service graph.
        
        Returns:
            Tuple of (user_embeddings, service_embeddings).
        """
        pass
