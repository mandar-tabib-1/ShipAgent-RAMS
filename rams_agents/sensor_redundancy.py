"""
Sensor Redundancy Module for Availability Monitoring

Implements multi-sensor voting and fusion for redundancy management.
Supports 2-of-3 voting, weighted fusion, and Bayesian sensor health tracking.

Part of the RAMS Availability Agent enhancement.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime


class SensorHealth(Enum):
    """Sensor health states."""
    HEALTHY = auto()      # Normal operation
    DEGRADED = auto()     # Reduced accuracy/intermittent
    FAILED = auto()       # Non-functional
    UNKNOWN = auto()      # No data available


class VotingStrategy(Enum):
    """Voting strategy for sensor fusion."""
    MAJORITY = "majority"           # 2-of-3 voting
    WEIGHTED = "weighted"           # Reliability-weighted average
    BAYESIAN = "bayesian"           # Bayesian fusion with uncertainty
    BEST_SENSOR = "best_sensor"     # Use most reliable sensor only


@dataclass
class SensorStatus:
    """Status of an individual sensor."""
    sensor_id: int
    sensor_name: str
    health: SensorHealth
    reliability_score: float  # 0-1, from learned errors
    last_update: float = 0.0
    consecutive_failures: int = 0
    error_rate: float = 0.0
    mean_error_m: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sensor_id': self.sensor_id,
            'sensor_name': self.sensor_name,
            'health': self.health.name,
            'reliability_score': self.reliability_score,
            'last_update': self.last_update,
            'consecutive_failures': self.consecutive_failures,
            'error_rate': self.error_rate,
            'mean_error_m': self.mean_error_m
        }


@dataclass
class RedundancyState:
    """Overall sensor redundancy state."""
    active_sensors: int
    total_sensors: int
    redundancy_level: str  # 'FULL', 'PARTIAL', 'MINIMAL', 'FAILED'
    capability_pct: float
    voting_result: Optional[Dict[str, float]] = None
    sensor_statuses: List[SensorStatus] = field(default_factory=list)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_sensors': self.active_sensors,
            'total_sensors': self.total_sensors,
            'redundancy_level': self.redundancy_level,
            'capability_pct': self.capability_pct,
            'voting_result': self.voting_result,
            'sensors': [s.to_dict() for s in self.sensor_statuses],
            'timestamp': self.timestamp
        }


class SensorVotingFusion:
    """
    Multi-sensor voting and fusion for redundancy management.
    
    Implements DNV-style sensor redundancy monitoring:
    - Tracks individual sensor health
    - Performs N-of-M voting for critical measurements
    - Provides degraded operation when sensors fail
    - Calculates overall redundancy capability
    """
    
    # Sensor definitions (matching Autoferry dataset)
    SENSOR_CONFIG = {
        1: {'name': 'LiDAR', 'default_reliability': 0.32, 'critical': True},
        2: {'name': 'Radar', 'default_reliability': 0.33, 'critical': True},
        3: {'name': 'IR_Camera', 'default_reliability': 0.50, 'critical': False},
        4: {'name': 'EO_Camera', 'default_reliability': 0.50, 'critical': False},
    }
    
    # Thresholds
    FAILURE_THRESHOLD = 5       # Consecutive failures to mark as FAILED
    DEGRADED_THRESHOLD = 3      # Consecutive failures to mark as DEGRADED
    STALE_TIMEOUT = 5.0         # Seconds without update = stale
    MIN_SENSORS_FULL = 3        # Minimum for FULL redundancy
    MIN_SENSORS_PARTIAL = 2     # Minimum for PARTIAL redundancy
    
    def __init__(self, 
                 voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED,
                 learned_reliability: Optional[Dict[int, float]] = None):
        """
        Initialize sensor voting fusion.
        
        Args:
            voting_strategy: Strategy for combining sensor data
            learned_reliability: Reliability scores learned from ground truth
        """
        self.voting_strategy = voting_strategy
        
        # Initialize sensor statuses
        self.sensors: Dict[int, SensorStatus] = {}
        for sensor_id, config in self.SENSOR_CONFIG.items():
            reliability = (learned_reliability or {}).get(
                sensor_id, config['default_reliability']
            )
            self.sensors[sensor_id] = SensorStatus(
                sensor_id=sensor_id,
                sensor_name=config['name'],
                health=SensorHealth.UNKNOWN,
                reliability_score=reliability
            )
        
        # History for trend analysis
        self._detection_history: Dict[int, List[float]] = {
            sid: [] for sid in self.SENSOR_CONFIG
        }
    
    def update_from_detections(self, 
                                detections: List[Dict[str, Any]],
                                timestamp: float) -> None:
        """
        Update sensor health based on incoming detections.
        
        Args:
            detections: List of detection dicts with 'sensorId' field
            timestamp: Current timestamp
        """
        # Track which sensors provided data this cycle
        sensors_active = set()
        
        for det in detections:
            sensor_id = det.get('sensorId', det.get('sensor_id', 0))
            if sensor_id in self.sensors:
                sensors_active.add(sensor_id)
                self._detection_history[sensor_id].append(timestamp)
                
                # Reset failure count on successful detection
                sensor = self.sensors[sensor_id]
                sensor.consecutive_failures = 0
                sensor.last_update = timestamp
                sensor.health = SensorHealth.HEALTHY
        
        # Update sensors that didn't provide data
        for sensor_id, sensor in self.sensors.items():
            if sensor_id not in sensors_active:
                # Check for stale data
                time_since_update = timestamp - sensor.last_update
                
                if sensor.last_update == 0:
                    # Never received data
                    sensor.health = SensorHealth.UNKNOWN
                elif time_since_update > self.STALE_TIMEOUT:
                    sensor.consecutive_failures += 1
                    
                    if sensor.consecutive_failures >= self.FAILURE_THRESHOLD:
                        sensor.health = SensorHealth.FAILED
                    elif sensor.consecutive_failures >= self.DEGRADED_THRESHOLD:
                        sensor.health = SensorHealth.DEGRADED
        
        # Trim history
        max_history = 100
        for sensor_id in self._detection_history:
            if len(self._detection_history[sensor_id]) > max_history:
                self._detection_history[sensor_id] = \
                    self._detection_history[sensor_id][-max_history:]
    
    def update_sensor_error(self, 
                            sensor_id: int, 
                            error_m: float) -> None:
        """
        Update sensor error statistics from ground truth comparison.
        
        Args:
            sensor_id: Sensor ID
            error_m: Position error in meters
        """
        if sensor_id in self.sensors:
            sensor = self.sensors[sensor_id]
            alpha = 0.1  # EMA smoothing
            
            if sensor.mean_error_m == 0:
                sensor.mean_error_m = error_m
            else:
                sensor.mean_error_m = (1 - alpha) * sensor.mean_error_m + alpha * error_m
            
            # Update reliability based on error (inverse relationship)
            # Lower error = higher reliability
            max_acceptable_error = 50.0  # meters
            sensor.reliability_score = max(0.1, 1.0 - (sensor.mean_error_m / max_acceptable_error))
    
    def vote_position(self, 
                      sensor_positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Vote on position using configured strategy.
        
        Args:
            sensor_positions: {sensor_id: (north, east)} from each sensor
            
        Returns:
            Fused (north, east) position
        """
        if not sensor_positions:
            return (0.0, 0.0)
        
        if self.voting_strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(sensor_positions)
        elif self.voting_strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(sensor_positions)
        elif self.voting_strategy == VotingStrategy.BAYESIAN:
            return self._bayesian_vote(sensor_positions)
        else:  # BEST_SENSOR
            return self._best_sensor_vote(sensor_positions)
    
    def _majority_vote(self, 
                       sensor_positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        """Simple majority voting (2-of-3 style)."""
        positions = list(sensor_positions.values())
        
        if len(positions) == 1:
            return positions[0]
        
        # Find positions that agree (within tolerance)
        tolerance = 10.0  # meters
        
        # For 3+ sensors, find consensus
        if len(positions) >= 2:
            # Check which positions agree
            for i, p1 in enumerate(positions):
                agreeing = [p1]
                for j, p2 in enumerate(positions):
                    if i != j:
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        if dist < tolerance:
                            agreeing.append(p2)
                
                # If majority agrees, use their average
                if len(agreeing) >= len(positions) // 2 + 1:
                    return (
                        np.mean([p[0] for p in agreeing]),
                        np.mean([p[1] for p in agreeing])
                    )
        
        # No consensus - use simple average
        return (
            np.mean([p[0] for p in positions]),
            np.mean([p[1] for p in positions])
        )
    
    def _weighted_vote(self, 
                       sensor_positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        """Reliability-weighted average."""
        total_weight = 0.0
        weighted_north = 0.0
        weighted_east = 0.0
        
        for sensor_id, (north, east) in sensor_positions.items():
            if sensor_id in self.sensors:
                weight = self.sensors[sensor_id].reliability_score
                
                # Reduce weight for degraded sensors
                if self.sensors[sensor_id].health == SensorHealth.DEGRADED:
                    weight *= 0.5
                elif self.sensors[sensor_id].health == SensorHealth.FAILED:
                    weight = 0.0
                
                weighted_north += north * weight
                weighted_east += east * weight
                total_weight += weight
        
        if total_weight > 0:
            return (weighted_north / total_weight, weighted_east / total_weight)
        else:
            # Fallback to simple average
            positions = list(sensor_positions.values())
            return (
                np.mean([p[0] for p in positions]),
                np.mean([p[1] for p in positions])
            )
    
    def _bayesian_vote(self, 
                       sensor_positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Bayesian fusion using sensor uncertainties.
        
        Uses inverse variance weighting: w_i = 1/σ_i²
        """
        total_precision_n = 0.0
        total_precision_e = 0.0
        weighted_north = 0.0
        weighted_east = 0.0
        
        for sensor_id, (north, east) in sensor_positions.items():
            if sensor_id in self.sensors:
                sensor = self.sensors[sensor_id]
                
                # Convert reliability to variance (lower reliability = higher variance)
                # Variance roughly equals (mean_error)²
                variance = max(1.0, sensor.mean_error_m ** 2) if sensor.mean_error_m > 0 else 100.0
                
                if sensor.health == SensorHealth.FAILED:
                    continue  # Skip failed sensors
                elif sensor.health == SensorHealth.DEGRADED:
                    variance *= 4  # Double the standard deviation
                
                precision = 1.0 / variance
                weighted_north += north * precision
                weighted_east += east * precision
                total_precision_n += precision
                total_precision_e += precision
        
        if total_precision_n > 0:
            return (weighted_north / total_precision_n, weighted_east / total_precision_e)
        else:
            return self._weighted_vote(sensor_positions)
    
    def _best_sensor_vote(self, 
                          sensor_positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        """Use only the most reliable healthy sensor."""
        best_sensor = None
        best_reliability = -1.0
        
        for sensor_id in sensor_positions:
            if sensor_id in self.sensors:
                sensor = self.sensors[sensor_id]
                if (sensor.health in [SensorHealth.HEALTHY, SensorHealth.UNKNOWN] and 
                    sensor.reliability_score > best_reliability):
                    best_reliability = sensor.reliability_score
                    best_sensor = sensor_id
        
        if best_sensor is not None:
            return sensor_positions[best_sensor]
        else:
            # Fallback to weighted
            return self._weighted_vote(sensor_positions)
    
    def get_redundancy_state(self, timestamp: float) -> RedundancyState:
        """
        Get current redundancy state assessment.
        
        Returns:
            RedundancyState with capability and sensor statuses
        """
        # Count healthy sensors
        healthy_count = sum(
            1 for s in self.sensors.values() 
            if s.health in [SensorHealth.HEALTHY, SensorHealth.UNKNOWN]
        )
        
        critical_healthy = sum(
            1 for sid, s in self.sensors.items()
            if s.health == SensorHealth.HEALTHY and 
            self.SENSOR_CONFIG[sid].get('critical', False)
        )
        
        # Determine redundancy level
        total = len(self.sensors)
        if healthy_count >= self.MIN_SENSORS_FULL:
            level = 'FULL'
            capability = 100.0
        elif healthy_count >= self.MIN_SENSORS_PARTIAL:
            level = 'PARTIAL'
            capability = 75.0
        elif healthy_count >= 1:
            level = 'MINIMAL'
            capability = 50.0
        else:
            level = 'FAILED'
            capability = 0.0
        
        # Reduce capability if critical sensors are down
        if critical_healthy == 0 and healthy_count > 0:
            capability *= 0.5  # Major reduction without critical sensors
        
        return RedundancyState(
            active_sensors=healthy_count,
            total_sensors=total,
            redundancy_level=level,
            capability_pct=capability,
            sensor_statuses=list(self.sensors.values()),
            timestamp=timestamp
        )
    
    def simulate_sensor_failure(self, sensor_id: int) -> None:
        """Simulate a sensor failure for testing."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].health = SensorHealth.FAILED
            self.sensors[sensor_id].consecutive_failures = self.FAILURE_THRESHOLD
    
    def reset_sensor(self, sensor_id: int) -> None:
        """Reset a sensor to healthy state."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].health = SensorHealth.HEALTHY
            self.sensors[sensor_id].consecutive_failures = 0


# Convenience function for integration
def create_sensor_fusion_redundancy(
    learned_reliability: Optional[Dict[int, float]] = None
) -> SensorVotingFusion:
    """
    Create a SensorVotingFusion instance with learned reliability scores.
    
    If no learned scores provided, uses defaults from sensor fusion ML training.
    """
    # Default reliability scores from Autoferry ground truth comparison
    default_scores = {
        1: 0.32,  # LiDAR
        2: 0.33,  # Radar
        3: 0.50,  # IR Camera
        4: 0.50,  # EO Camera
    }
    
    return SensorVotingFusion(
        voting_strategy=VotingStrategy.WEIGHTED,
        learned_reliability=learned_reliability or default_scores
    )
