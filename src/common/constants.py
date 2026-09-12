"""
Constants and Enumerations for QoS-Based Service Recommendation.
"""

from enum import Enum


class QoSAttribute(str, Enum):
    """Supported QoS Attributes in Service-Oriented Computing benchmark."""
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


class QoSDirection(str, Enum):
    """Optimization direction for QoS attributes."""
    POSITIVE = "positive"  # Higher values denote superior quality (e.g., availability, throughput)
    NEGATIVE = "negative"  # Lower values denote superior quality (e.g., response time, latency)


# Default directionality mapping
QOS_DIRECTIONS = {
    QoSAttribute.RESPONSE_TIME: QoSDirection.NEGATIVE,
    QoSAttribute.AVAILABILITY: QoSDirection.POSITIVE,
    QoSAttribute.RELIABILITY: QoSDirection.POSITIVE,
    QoSAttribute.THROUGHPUT: QoSDirection.POSITIVE,
    QoSAttribute.LATENCY: QoSDirection.NEGATIVE,
}


class SplitType(str, Enum):
    """Evaluation data partitioning strategies."""
    RANDOM = "random"
    USER_COLD = "user_cold"
    SERVICE_COLD = "service_cold"


# Canonical benchmark dimensions (e.g., WS-DREAM standard sizes)
DEFAULT_NUM_USERS = 339
DEFAULT_NUM_SERVICES = 5825
