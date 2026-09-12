"""
Academic Logging Module.
Formats console and file outputs with timestamps, log levels, and experiment tags.
"""

import logging
import os
import sys
from typing import Optional


class ExperimentLogger:
    """Configures centralized, structured loggers for experiments."""

    @staticmethod
    def setup_logger(
        experiment_name: str,
        log_dir: str = "results/logs",
        level: str = "INFO",
        filename: Optional[str] = None
    ) -> logging.Logger:
        """
        Creates and returns a configured logger.
        
        Args:
            experiment_name: Human-readable identifier for experiment.
            log_dir: Directory where log files are stored.
            level: Logging severity level ("DEBUG", "INFO", "WARNING", "ERROR").
            filename: Specific filename override.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, filename if filename else f"{experiment_name}.log")

        logger = logging.getLogger(experiment_name)
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Prevent duplicate handlers if called repeatedly
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
