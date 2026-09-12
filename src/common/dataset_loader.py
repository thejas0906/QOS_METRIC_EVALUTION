"""
Dataset Loader and Benchmark Synthesizer.
Loads WS-DREAM QoS benchmark datasets (Response Time, Throughput, etc.)
or synthesizes realistic multi-attribute QoS matrices when raw benchmarks are unavailable.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from src.common.constants import QoSAttribute, DEFAULT_NUM_USERS, DEFAULT_NUM_SERVICES


class WSDreamLoader:
    """Loads and synthesizes web service QoS benchmarks."""

    def __init__(self, data_dir: str = "data/raw/ws_dream"):
        self.data_dir = data_dir

    def load_raw_matrix(self, file_path: str) -> np.ndarray:
        """
        Parses standard WS-DREAM space-separated or tab-separated matrix files.
        Values of -1 or <= 0 indicate missing QoS records.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Matrix file not found: {file_path}")

        # WS-DREAM matrices are typically space/tab-separated numbers
        matrix = np.loadtxt(file_path, dtype=np.float32)
        return matrix

    def load_metadata(
        self,
        user_file: str = "userlist.txt",
        service_file: str = "wslist.txt"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads user and service metadata tables."""
        user_path = os.path.join(self.data_dir, user_file)
        service_path = os.path.join(self.data_dir, service_file)

        if os.path.exists(user_path):
            df_users = pd.read_csv(user_path, sep="\t", header=None, names=["user_id", "ip", "country", "as"])
        else:
            df_users = pd.DataFrame({"user_id": np.arange(DEFAULT_NUM_USERS)})

        if os.path.exists(service_path):
            df_services = pd.read_csv(service_path, sep="\t", header=None, names=["service_id", "wsdl_url", "provider", "country"])
        else:
            df_services = pd.DataFrame({"service_id": np.arange(DEFAULT_NUM_SERVICES)})

        return df_users, df_services

    @classmethod
    def generate_synthetic_qos_dataset(
        cls,
        num_users: int = DEFAULT_NUM_USERS,
        num_services: int = 500,  # Scaled for fast execution and benchmark simulations
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Generates realistic multi-attribute QoS matrices:
        - response_time: Log-normal (0.05s to 4.0s)
        - availability: Beta distribution (0.85 to 1.0)
        - reliability: Beta distribution (0.80 to 1.0)
        - throughput: Log-normal (20 kbps to 4000 kbps)
        - latency: Correlated with response time (10ms to 800ms)
        """
        rng = np.random.RandomState(seed)

        # Base latent quality factor per service (high quality = fast, available, reliable)
        service_base_quality = rng.beta(a=3, b=2, size=num_services)  # [0, 1]
        user_network_penalty = rng.gamma(shape=2.0, scale=0.5, size=num_users)  # network latency factor

        # 1. Response Time (seconds, lower is better)
        # Slower services have low quality; high user penalty increases RT
        rt_base = 0.1 + (1.0 - service_base_quality)[np.newaxis, :] * 2.0
        rt_noise = rng.lognormal(mean=0.0, sigma=0.3, size=(num_users, num_services))
        response_time = (rt_base * rt_noise) + (user_network_penalty[:, np.newaxis] * 0.2)
        response_time = np.clip(response_time, 0.02, 10.0).astype(np.float32)

        # 2. Availability (ratio in [0, 1], higher is better)
        avail_base = 0.85 + service_base_quality[np.newaxis, :] * 0.14
        avail_noise = rng.normal(0, 0.02, size=(num_users, num_services))
        availability = np.clip(avail_base + avail_noise, 0.5, 1.0).astype(np.float32)

        # 3. Reliability (ratio in [0, 1], higher is better)
        rel_base = 0.80 + service_base_quality[np.newaxis, :] * 0.19
        rel_noise = rng.normal(0, 0.03, size=(num_users, num_services))
        reliability = np.clip(rel_base + rel_noise, 0.5, 1.0).astype(np.float32)

        # 4. Throughput (kbps, higher is better)
        tp_base = 200.0 + service_base_quality[np.newaxis, :] * 2500.0
        tp_noise = rng.lognormal(mean=0.0, sigma=0.4, size=(num_users, num_services))
        throughput = np.clip(tp_base * tp_noise, 10.0, 10000.0).astype(np.float32)

        # 5. Latency (ms, lower is better)
        lat_base = response_time * 0.3 * 1000.0  # ~30% of total response time in ms
        lat_noise = rng.normal(0, 15.0, size=(num_users, num_services))
        latency = np.clip(lat_base + lat_noise, 5.0, 3000.0).astype(np.float32)

        return {
            QoSAttribute.RESPONSE_TIME.value: response_time,
            QoSAttribute.AVAILABILITY.value: availability,
            QoSAttribute.RELIABILITY.value: reliability,
            QoSAttribute.THROUGHPUT.value: throughput,
            QoSAttribute.LATENCY.value: latency,
        }

    def load_or_generate_dataset(
        self,
        attributes: Optional[list] = None,
        num_users: int = DEFAULT_NUM_USERS,
        num_services: int = 500,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Attempts to load raw WS-DREAM files from data_dir.
        If not found, synthesizes a scientifically calibrated dataset.
        """
        rt_path = os.path.join(self.data_dir, "rtmatrix.txt")
        tp_path = os.path.join(self.data_dir, "tpmatrix.txt")

        if os.path.exists(rt_path) and os.path.exists(tp_path):
            rt = self.load_raw_matrix(rt_path)
            tp = self.load_raw_matrix(tp_path)
            u_count, s_count = rt.shape

            # Synthesize companion attributes based on actual RT and TP
            rng = np.random.RandomState(seed)
            availability = np.clip(0.95 - (rt / np.max(rt)) * 0.15 + rng.normal(0, 0.02, rt.shape), 0.5, 1.0)
            reliability = np.clip(0.92 - (rt / np.max(rt)) * 0.18 + rng.normal(0, 0.02, rt.shape), 0.5, 1.0)
            latency = np.clip(rt * 300.0 + rng.normal(0, 10.0, rt.shape), 5.0, 5000.0)

            return {
                QoSAttribute.RESPONSE_TIME.value: rt,
                QoSAttribute.AVAILABILITY.value: availability.astype(np.float32),
                QoSAttribute.RELIABILITY.value: reliability.astype(np.float32),
                QoSAttribute.THROUGHPUT.value: tp,
                QoSAttribute.LATENCY.value: latency.astype(np.float32),
            }

        # Otherwise synthesize calibrated dataset
        return self.generate_synthetic_qos_dataset(num_users=num_users, num_services=num_services, seed=seed)
