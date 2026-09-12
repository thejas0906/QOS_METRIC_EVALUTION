"""
Service Invocation Cost Generator and Pricing Loader.
Simulates realistic economic models for web services (e.g., API pricing per 1,000 calls).
Higher QoS services (lower response time, higher availability) realistically correlate with higher unit cost.
"""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd


class ServiceCostEngine:
    """Generates and manages monetary pricing for web services."""

    @classmethod
    def generate_correlated_costs(
        cls,
        service_qos_profiles: np.ndarray,
        correlation_strength: float = 0.75,
        base_cost_range: Tuple[float, float] = (0.01, 1.00),
        distribution: str = "pareto",
        seed: int = 42
    ) -> np.ndarray:
        """
        Generates service invocation costs correlated with overall service quality.
        
        Args:
            service_qos_profiles: 1D array of quality scores per service in [0, 1].
            correlation_strength: Weight between 0 and 1 dictating quality-cost dependency.
            base_cost_range: (min_cost, max_cost) in dollars.
            distribution: "pareto" (heavy-tailed realistic pricing) or "lognormal".
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)
        num_services = len(service_qos_profiles)
        min_c, max_c = base_cost_range

        if distribution == "pareto":
            # Pareto (alpha=2.5) for typical long-tail software pricing
            noise = rng.pareto(a=2.5, size=num_services)
            # Clip outlier extremes
            noise = np.clip(noise / 4.0, 0.0, 1.0)
        else:
            noise = rng.lognormal(mean=0.0, sigma=0.5, size=num_services)
            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-6)

        # Composite pricing signal: quality + market noise
        raw_signal = correlation_strength * service_qos_profiles + (1.0 - correlation_strength) * noise
        # Scale to cost range
        signal_min, signal_max = np.min(raw_signal), np.max(raw_signal)
        denom = signal_max - signal_min if signal_max > signal_min else 1.0
        normalized_costs = (raw_signal - signal_min) / denom
        costs = min_c + normalized_costs * (max_c - min_c)

        return np.round(costs, 4).astype(np.float32)

    @staticmethod
    def load_cost_table(file_path: str) -> pd.DataFrame:
        """Loads costs from CSV file containing ['service_id', 'cost']."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cost table not found at: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def save_cost_table(costs: np.ndarray, file_path: str) -> None:
        """Saves generated costs to CSV for provenance."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        df = pd.DataFrame({
            "service_id": np.arange(len(costs)),
            "cost": costs
        })
        df.to_csv(file_path, index=False)

    @staticmethod
    def normalize_costs(costs: np.ndarray) -> np.ndarray:
        """Min-Max scales costs into [0, 1] for utility calculation."""
        c_min = float(np.min(costs))
        c_max = float(np.max(costs))
        if c_max == c_min:
            return np.zeros_like(costs, dtype=np.float32)
        return ((costs - c_min) / (c_max - c_min)).astype(np.float32)
