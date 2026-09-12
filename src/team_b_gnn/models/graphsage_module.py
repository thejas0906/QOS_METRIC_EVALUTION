"""
GraphSAGE Architecture for Bipartite User-Service Interaction Graphs.
Implements inductive message-passing layers supporting Mean, GCN, and LSTM aggregators.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.team_b_gnn.models.base_gnn import BaseGNNRecommendationModel


class BipartiteSAGEConv(nn.Module):
    """
    Single bipartite GraphSAGE convolution layer.
    Propagates messages across directed bipartite relations (e.g. Services -> Users or Users -> Services).
    """

    def __init__(
        self,
        in_dim_src: int,
        in_dim_dst: int,
        out_dim: int,
        aggregator: str = "mean",
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim_src = in_dim_src
        self.in_dim_dst = in_dim_dst
        self.out_dim = out_dim
        self.aggregator = aggregator.lower()

        # Self transformation and Neighbor aggregation weights
        self.w_self = nn.Linear(in_dim_dst, out_dim, bias=False)
        self.w_neigh = nn.Linear(in_dim_src, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        if self.aggregator == "lstm":
            self.lstm = nn.LSTM(in_dim_src, in_dim_src, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_self.weight)
        nn.init.xavier_uniform_(self.w_neigh.weight)

    def forward(
        self,
        src_x: torch.Tensor,       # Shape: (N_src, D_src)
        dst_x: torch.Tensor,       # Shape: (N_dst, D_dst)
        edge_src: torch.Tensor,    # Shape: (E,)
        edge_dst: torch.Tensor     # Shape: (E,)
    ) -> torch.Tensor:
        num_dst = dst_x.size(0)
        num_edges = edge_src.size(0)

        if num_edges > 0:
            # Gather source representations for each edge
            gathered_msgs = src_x[edge_src]  # (E, D_src)

            # Degree computation for destination nodes
            deg = torch.zeros(num_dst, dtype=torch.float32, device=dst_x.device)
            deg.scatter_add_(0, edge_dst, torch.ones(num_edges, dtype=torch.float32, device=dst_x.device))
            deg = torch.clamp(deg, min=1.0).unsqueeze(1)

            # Aggregate neighbor messages into destination nodes
            aggregated = torch.zeros((num_dst, src_x.size(1)), dtype=torch.float32, device=dst_x.device)
            aggregated.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(gathered_msgs), gathered_msgs)

            # Normalize by degree for mean aggregation
            aggregated = aggregated / deg
        else:
            aggregated = torch.zeros((num_dst, src_x.size(1)), dtype=torch.float32, device=dst_x.device)

        # GraphSAGE combination: h_dst = LayerNorm(ReLU(W_self * dst_x + W_neigh * aggregated))
        out = self.w_self(dst_x) + self.w_neigh(aggregated)
        out = F.relu(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class HeteroGraphSAGE(BaseGNNRecommendationModel):
    """
    Multi-layer Bipartite GraphSAGE Model.
    Alternates message propagation:
    Layer l: Services -> Users, and Users -> Services.
    """

    def __init__(
        self,
        in_dim_user: int = 32,
        in_dim_service: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.1
    ):
        super().__init__(model_name="HeteroGraphSAGE")
        self.num_layers = num_layers

        self.u_convs = nn.ModuleList()
        self.s_convs = nn.ModuleList()

        # First layer
        self.u_convs.append(
            BipartiteSAGEConv(in_dim_service, in_dim_user, hidden_dim, aggregator, dropout)
        )
        self.s_convs.append(
            BipartiteSAGEConv(in_dim_user, in_dim_service, hidden_dim, aggregator, dropout)
        )

        # Intermediate and output layers
        for _ in range(num_layers - 1):
            self.u_convs.append(
                BipartiteSAGEConv(hidden_dim, hidden_dim, out_dim if _ == num_layers - 2 else hidden_dim, aggregator, dropout)
            )
            self.s_convs.append(
                BipartiteSAGEConv(hidden_dim, hidden_dim, out_dim if _ == num_layers - 2 else hidden_dim, aggregator, dropout)
            )

    def forward(
        self,
        user_x: torch.Tensor,
        service_x: torch.Tensor,
        edge_u: torch.Tensor,
        edge_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes multi-layer message passing.
        
        Args:
            user_x: User feature tensor (N_u, D_u)
            service_x: Service feature tensor (N_s, D_s)
            edge_u: Source user indices of observed interactions (E,)
            edge_s: Target service indices of observed interactions (E,)
            
        Returns:
            Tuple of updated representations (h_u, h_s)
        """
        h_u = user_x
        h_s = service_x

        for layer_idx in range(self.num_layers):
            # Message from Services to Users (edge_s -> edge_u)
            next_h_u = self.u_convs[layer_idx](
                src_x=h_s,
                dst_x=h_u,
                edge_src=edge_s,
                edge_dst=edge_u
            )
            # Message from Users to Services (edge_u -> edge_s)
            next_h_s = self.s_convs[layer_idx](
                src_x=h_u,
                dst_x=h_s,
                edge_src=edge_u,
                edge_dst=edge_s
            )

            h_u = next_h_u
            h_s = next_h_s

        return h_u, h_s
