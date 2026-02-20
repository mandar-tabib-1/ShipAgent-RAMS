"""
Data Loaders Package for RAMS Demonstration

Provides unified interfaces for loading maritime datasets:
- UCI Naval Propulsion CBM Dataset
- AutoFerry Sensor Fusion Dataset
"""

from .uci_naval_loader import UCINavalPropulsionLoader
from .navigation_loader import NavigationDataLoader

__all__ = ['UCINavalPropulsionLoader', 'NavigationDataLoader']
