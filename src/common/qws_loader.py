"""
QWS Dataset Loader and Matrix Synthesizer.
Loads authentic QWS Dataset Version 2.0 (2,507 Web Services, Eyhab Al-Masri & Qusay H. Mahmoud)
and builds distributed client-service QoS interaction matrices for recommendation benchmarking.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from src.common.constants import QoSAttribute


class QWSLoader:
    """Loads and processes the authentic QWS 2.0 Web Service Dataset."""

    def __init__(self, data_path: str = "data/raw/qws/qws_dataset_v2.txt"):
        self.data_path = data_path

    def load_raw_services(self) -> pd.DataFrame:
        """
        Parses QWS 2.0 dataset into a pandas DataFrame of 2,507 real web services.
        Parameters:
        1: Response Time (ms)
        2: Availability (%)
        3: Throughput (invocations/sec)
        4: Successability (%)
        5: Reliability (%)
        6: Compliance (%)
        7: Best Practices (%)
        8: Latency (ms)
        9: Documentation (%)
        10: Service Name
        11: WSDL Address
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"QWS dataset file not found at {self.data_path}")

        records = []
        with open(self.data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 11:
                    try:
                        rt_ms = float(parts[0])
                        avail_pct = float(parts[1])
                        tp = float(parts[2])
                        success_pct = float(parts[3])
                        rel_pct = float(parts[4])
                        compliance = float(parts[5])
                        best_practice = float(parts[6])
                        lat_ms = float(parts[7])
                        doc_pct = float(parts[8])
                        name = parts[9].strip()
                        wsdl = parts[10].strip()

                        records.append({
                            "response_time": rt_ms / 1000.0,  # Convert ms to seconds
                            "availability": avail_pct / 100.0,  # Convert % to [0, 1]
                            "throughput": tp,
                            "reliability": rel_pct / 100.0,  # Convert % to [0, 1]
                            "latency": lat_ms,               # ms
                            "successability": success_pct / 100.0,
                            "compliance": compliance / 100.0,
                            "best_practice": best_practice / 100.0,
                            "documentation": doc_pct / 100.0,
                            "service_name": name,
                            "wsdl_url": wsdl
                        })
                    except ValueError:
                        continue

        df = pd.DataFrame(records)
        return df

    def build_user_service_qos_matrices(
        self,
        num_users: int = 200,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Constructs User x Service QoS matrices (num_users x 2507 services)
        anchored directly to the empirical QWS 2.0 service measurements.
        Client distributed environments introduce realistic client-side network latency,
        bandwidth constraints, and measurement variance.
        """
        df = self.load_raw_services()
        num_services = len(df)
        rng = np.random.RandomState(seed)

        # Base service arrays from QWS
        base_rt = df["response_time"].values.astype(np.float32)       # in seconds
        base_avail = df["availability"].values.astype(np.float32)     # [0, 1]
        base_tp = df["throughput"].values.astype(np.float32)          # invokes/sec
        base_rel = df["reliability"].values.astype(np.float32)        # [0, 1]
        base_lat = df["latency"].values.astype(np.float32)            # in ms

        # Client-side network profiles (simulating distributed users across regions)
        user_network_penalty = rng.gamma(shape=2.0, scale=0.15, size=num_users)  # in seconds
        user_bw_factor = rng.uniform(0.7, 1.3, size=num_users)                   # throughput multiplier

        # 1. Response Time (seconds, lower is better)
        rt_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(num_users, num_services))
        mat_rt = (base_rt[np.newaxis, :] * rt_noise) + (user_network_penalty[:, np.newaxis] * 0.5)
        mat_rt = np.clip(mat_rt, 0.02, 10.0).astype(np.float32)

        # 2. Availability (ratio in [0, 1], higher is better)
        avail_noise = rng.normal(0, 0.015, size=(num_users, num_services))
        mat_avail = np.clip(base_avail[np.newaxis, :] + avail_noise, 0.05, 1.0).astype(np.float32)

        # 3. Throughput (invokes/s, higher is better)
        tp_noise = rng.lognormal(mean=0.0, sigma=0.2, size=(num_users, num_services))
        mat_tp = (base_tp[np.newaxis, :] * tp_noise) * user_bw_factor[:, np.newaxis]
        mat_tp = np.clip(mat_tp, 0.05, 200.0).astype(np.float32)

        # 4. Reliability (ratio in [0, 1], higher is better)
        rel_noise = rng.normal(0, 0.02, size=(num_users, num_services))
        mat_rel = np.clip(base_rel[np.newaxis, :] + rel_noise, 0.10, 1.0).astype(np.float32)

        # 5. Latency (ms, lower is better)
        lat_noise = rng.normal(0, 8.0, size=(num_users, num_services))
        mat_lat = (base_lat[np.newaxis, :] + lat_noise) + (user_network_penalty[:, np.newaxis] * 100.0)
        mat_lat = np.clip(mat_lat, 0.5, 5000.0).astype(np.float32)

        return {
            QoSAttribute.RESPONSE_TIME.value: mat_rt,
            QoSAttribute.AVAILABILITY.value: mat_avail,
            QoSAttribute.THROUGHPUT.value: mat_tp,
            QoSAttribute.RELIABILITY.value: mat_rel,
            QoSAttribute.LATENCY.value: mat_lat,
        }
