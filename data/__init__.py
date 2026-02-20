"""
Data module for Vessel AI System
"""

from .autoferry_loader import (
    AutoferryDataLoader,
    AutoferryDetection,
    AutoferryGroundTruth,
    load_autoferry_data,
    SCENARIOS,
    SENSOR_TYPES,
    TARGET_NAMES
)

__all__ = [
    'AutoferryDataLoader',
    'AutoferryDetection', 
    'AutoferryGroundTruth',
    'load_autoferry_data',
    'SCENARIOS',
    'SENSOR_TYPES',
    'TARGET_NAMES'
]
