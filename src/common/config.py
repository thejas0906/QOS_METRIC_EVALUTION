"""
Configuration Management for QoS Recommendation Experiments.
Provides strongly typed dataclasses and YAML loading/saving mechanisms.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import os
import yaml


@dataclass
class CommonConfig:
    """General project settings."""
    project_name: str = "QOS_GNN_Recommendation"
    seed: int = 42
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    attributes: List[str] = field(default_factory=lambda: [
        "response_time", "availability", "reliability", "throughput", "latency"
    ])
    num_users: int = 339
    num_services: int = 5825


@dataclass
class TeamAConfig:
    """Hyperparameters and settings for Team A (Sparse Matrix-Based QoS Recommendation)."""
    model_name: str = "BiasedMF"  # Options: GlobalMean, UserMean, ServiceMean, UserCF, ItemCF, BiasedMF, PMF
    density: float = 0.10  # 10% observed entries (90% sparse)
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    latent_dim: int = 20
    lr: float = 0.005
    reg: float = 0.02
    epochs: int = 100
    batch_size: int = 1024
    early_stopping_patience: int = 5
    k_neighbors: int = 10
    similarity_metric: str = "pcc"  # pcc or cosine
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    attribute_weights: Dict[str, float] = field(default_factory=lambda: {
        "response_time": 0.25,
        "availability": 0.20,
        "reliability": 0.20,
        "throughput": 0.20,
        "latency": 0.15,
    })


@dataclass
class TeamBConfig:
    """Hyperparameters and settings for Team B (GNN + Cost-Performance Recommendation)."""
    model_name: str = "HeteroGraphSAGE"
    hidden_channels: int = 64
    out_channels: int = 32
    num_layers: int = 2
    aggregator: str = "mean"  # mean, gcn, lstm
    dropout: float = 0.1
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 150
    batch_size: int = 2048
    early_stopping_patience: int = 10
    alpha: float = 0.7  # QoS importance in Utility
    beta: float = 0.3   # Cost penalty in Utility
    cold_user_ratio: float = 0.10
    cold_service_ratio: float = 0.10
    cost_distribution: str = "pareto"  # pareto or lognormal
    cost_min: float = 0.01
    cost_max: float = 1.00
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    attribute_weights: Dict[str, float] = field(default_factory=lambda: {
        "response_time": 0.25,
        "availability": 0.20,
        "reliability": 0.20,
        "throughput": 0.20,
        "latency": 0.15,
    })


class ConfigManager:
    """Loads, updates, validates, and dumps configuration files."""

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Loads raw dictionary from YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def load_team_a_config(cls, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> TeamAConfig:
        """Loads and resolves Team A configuration."""
        data = cls.load_yaml(config_path) if os.path.exists(config_path) else {}
        if overrides:
            data.update(overrides)
        return TeamAConfig(**{k: v for k, v in data.items() if k in TeamAConfig.__dataclass_fields__})

    @classmethod
    def load_team_b_config(cls, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> TeamBConfig:
        """Loads and resolves Team B configuration."""
        data = cls.load_yaml(config_path) if os.path.exists(config_path) else {}
        if overrides:
            data.update(overrides)
        return TeamBConfig(**{k: v for k, v in data.items() if k in TeamBConfig.__dataclass_fields__})

    @staticmethod
    def save_config(config_obj: Any, output_path: str) -> None:
        """Serializes a dataclass instance into YAML."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(config_obj), f, default_flow_style=False, sort_keys=False)
