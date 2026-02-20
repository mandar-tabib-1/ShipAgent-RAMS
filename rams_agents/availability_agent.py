"""
Availability Agent - Monitors system availability and detects anomalies

Part of the RAMS multi-agent system demonstrating agentic AI.
Uses anomaly detection to identify degraded operational modes.

Key Responsibilities:
- Monitor system operational state
- Detect anomalies using Isolation Forest and Autoencoder
- Track MTBF/MTTR metrics
- Identify degraded modes and availability impacts
- Multi-sensor redundancy monitoring (DNV-style)
- DP system availability tracking
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .base_agent import (
    RAMSBaseAgent, RAMSCategory, AgentBelief, AgentMessage,
    MessagePriority
)

# Import redundancy monitoring modules
try:
    from .sensor_redundancy import (
        SensorVotingFusion, SensorHealth, RedundancyState,
        create_sensor_fusion_redundancy
    )
    SENSOR_REDUNDANCY_AVAILABLE = True
except ImportError:
    SENSOR_REDUNDANCY_AVAILABLE = False

try:
    from .dp_availability import (
        DPAvailabilityMonitor, DPCapability, DPState,
        create_dp_monitor
    )
    DP_MONITOR_AVAILABLE = True
except ImportError:
    DP_MONITOR_AVAILABLE = False

# Optional ML-based anomaly detection
try:
    from .ml_models import AutoencoderAnomalyDetector, load_trained_models
    ML_ANOMALY_AVAILABLE = True
except ImportError:
    ML_ANOMALY_AVAILABLE = False


class OperationalMode(Enum):
    """System operational modes."""
    FULL_POWER = auto()       # 100% capability
    REDUCED_POWER = auto()    # 75% capability (single turbine, etc.)
    DEGRADED = auto()         # 50% capability (significant limitations)
    EMERGENCY = auto()        # Minimum safe operation
    FAILED = auto()           # Non-operational


@dataclass
class AvailabilityState:
    """Current availability state."""
    mode: OperationalMode
    capability_pct: float  # 0-100%
    anomaly_score: float   # 0-1, higher = more anomalous
    anomalies_detected: List[str] = field(default_factory=list)
    timestamp: float = 0.0


class AvailabilityAgent(RAMSBaseAgent):
    """
    Agent responsible for system availability monitoring.
    
    RAMS Category: A (Availability)
    
    Agentic Behaviors:
    - Perceives: Operational parameters from propulsion system
    - Reasons: Detects anomalies, identifies degraded modes
    - Acts: Reports availability status and mode transitions
    - Communicates: Alerts Supervisor on availability changes
    
    Availability = MTBF / (MTBF + MTTR)
    """
    
    # Anomaly detection thresholds
    ANOMALY_THRESHOLD = 0.6       # Isolation Forest threshold
    DEGRADED_THRESHOLD = 0.7      # Threshold for degraded mode
    EMERGENCY_THRESHOLD = 0.85    # Threshold for emergency mode
    
    # Nominal operational ranges (from UCI dataset characteristics)
    NOMINAL_RANGES = {
        'ship_speed': (3, 27),          # knots
        'gt_shaft_torque': (0, 100),    # kN·m
        'gt_revolutions': (2500, 4500), # rpm
        'hp_turbine_temp': (900, 1200), # °C
        'compressor_outlet_temp': (200, 500),  # °C
        'fuel_flow': (0.3, 2.0),        # kg/s
    }
    
    def __init__(self, 
                 use_sensor_redundancy: bool = True,
                 use_dp_monitoring: bool = True,
                 use_ml_anomaly: bool = True):
        super().__init__(name="AvailabilityAgent", category=RAMSCategory.AVAILABILITY)
        
        # Availability tracking
        self._current_state: Optional[AvailabilityState] = None
        self._state_history: List[AvailabilityState] = []
        
        # MTBF/MTTR tracking
        self._operational_hours = 0.0
        self._downtime_hours = 0.0
        self._failure_count = 0
        self._last_mode = OperationalMode.FULL_POWER
        
        # Anomaly detection model (simplified Isolation Forest-like)
        self._anomaly_model_fitted = False
        self._baseline_stats: Dict[str, Tuple[float, float]] = {}  # mean, std
        
        # ===== ENHANCED: Sensor Redundancy Monitoring =====
        self._sensor_redundancy: Optional['SensorVotingFusion'] = None
        if use_sensor_redundancy and SENSOR_REDUNDANCY_AVAILABLE:
            self._sensor_redundancy = create_sensor_fusion_redundancy()
        self._sensor_redundancy_state: Optional['RedundancyState'] = None
        
        # ===== ENHANCED: DP System Monitoring =====
        self._dp_monitor: Optional['DPAvailabilityMonitor'] = None
        if use_dp_monitoring and DP_MONITOR_AVAILABLE:
            self._dp_monitor = create_dp_monitor()
        self._dp_state: Optional['DPState'] = None
        
        # ===== ENHANCED: ML-based Anomaly Detection =====
        self._ml_anomaly_detector: Optional['AutoencoderAnomalyDetector'] = None
        if use_ml_anomaly and ML_ANOMALY_AVAILABLE:
            try:
                _, self._ml_anomaly_detector = load_trained_models()
            except Exception:
                self._ml_anomaly_detector = None
        
        # Agent goals
        self.goals = [
            "Monitor system operational availability",
            "Detect operational anomalies",
            "Track mode transitions",
            "Monitor sensor redundancy",
            "Track DP system capability"
        ]
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive system operational state.
        
        Expected environment_data keys:
        - 'propulsion': PropulsionDataPoint or dict with operational data
        - 'timestamp': Current simulation time
        """
        propulsion = environment_data.get('propulsion', {})
        timestamp = environment_data.get('timestamp', 0.0)
        
        # Extract operational parameters
        if hasattr(propulsion, 'to_dict'):
            params = propulsion.to_dict()
        elif isinstance(propulsion, dict):
            params = propulsion
        else:
            params = {}
        
        # Update baseline statistics for anomaly detection
        self._update_baseline(params)
        
        # Detect anomalies
        anomalies, anomaly_score = self._detect_anomalies(params)
        
        # Determine operational mode
        mode, capability = self._determine_mode(params, anomaly_score, anomalies)
        
        # Update state
        self._current_state = AvailabilityState(
            mode=mode,
            capability_pct=capability,
            anomaly_score=anomaly_score,
            anomalies_detected=anomalies,
            timestamp=timestamp
        )
        self._state_history.append(self._current_state)
        
        # Track operational/downtime hours
        if len(self._state_history) > 1:
            time_delta = timestamp - self._state_history[-2].timestamp
            if time_delta > 0:
                if mode in [OperationalMode.FULL_POWER, OperationalMode.REDUCED_POWER]:
                    self._operational_hours += time_delta
                elif mode == OperationalMode.FAILED:
                    self._downtime_hours += time_delta
        
        # Detect mode transitions
        if mode != self._last_mode:
            self._handle_mode_transition(self._last_mode, mode)
            self._last_mode = mode
        
        # Keep history manageable
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-500:]
        
        # Form perception belief
        self.update_belief(AgentBelief(
            belief_type="operational_state",
            value={
                'mode': mode.name,
                'capability_pct': capability,
                'anomaly_score': anomaly_score,
                'anomalies': anomalies
            },
            confidence=0.9,
            source="operational_monitoring",
            evidence=[f"Anomaly score: {anomaly_score:.3f}"]
        ))
    
    def reason(self) -> List[AgentBelief]:
        """
        Reason about availability and identify trends.
        """
        new_beliefs = []
        
        if not self._current_state:
            return new_beliefs
        
        # 1. Calculate availability metric
        availability = self._calculate_availability()
        new_beliefs.append(AgentBelief(
            belief_type="availability_metric",
            value={
                'availability': availability,
                'operational_hours': self._operational_hours,
                'downtime_hours': self._downtime_hours,
                'mtbf': self._calculate_mtbf(),
                'failure_count': self._failure_count
            },
            confidence=0.85,
            source="availability_calculation",
            evidence=["MTBF / (MTBF + MTTR) formula"]
        ))
        
        # 2. Analyze availability trend
        if len(self._state_history) > 10:
            trend = self._analyze_availability_trend()
            new_beliefs.append(AgentBelief(
                belief_type="availability_trend",
                value=trend,
                confidence=0.7,
                source="trend_analysis",
                evidence=[f"Based on {len(self._state_history)} observations"]
            ))
        
        # 3. Generate alerts for degraded states
        if self._current_state.mode in [OperationalMode.DEGRADED, OperationalMode.EMERGENCY, OperationalMode.FAILED]:
            self._generate_availability_alert()
        
        return new_beliefs
    
    def act(self) -> Dict[str, Any]:
        """
        Produce availability assessment and recommendations.
        """
        if not self._current_state:
            return {'status': 'NO_DATA'}
        
        availability = self._calculate_availability()
        
        recommendations = []
        if self._current_state.mode == OperationalMode.FAILED:
            recommendations.append("CRITICAL: System non-operational - initiate recovery procedures")
        elif self._current_state.mode == OperationalMode.EMERGENCY:
            recommendations.append("URGENT: Operating in emergency mode - immediate attention required")
            recommendations.append("REDUCE: Minimize operational demands")
        elif self._current_state.mode == OperationalMode.DEGRADED:
            recommendations.append("CAUTION: Operating in degraded mode")
            recommendations.append("PLAN: Schedule maintenance to restore full capability")
        elif self._current_state.mode == OperationalMode.REDUCED_POWER:
            recommendations.append("INFO: Operating at reduced power")
            recommendations.append("MONITOR: Watch for further degradation")
        else:
            recommendations.append("NORMAL: Full operational capability")
        
        return {
            'availability_metric': availability,
            'operational_state': {
                'mode': self._current_state.mode.name,
                'capability_pct': self._current_state.capability_pct,
                'anomaly_score': self._current_state.anomaly_score
            },
            'anomalies_detected': self._current_state.anomalies_detected,
            'statistics': {
                'operational_hours': round(self._operational_hours, 2),
                'downtime_hours': round(self._downtime_hours, 2),
                'mtbf': round(self._calculate_mtbf(), 2),
                'failure_count': self._failure_count
            },
            'recommendations': recommendations
        }
    
    def _update_baseline(self, params: Dict[str, Any]) -> None:
        """Update baseline statistics for anomaly detection."""
        for key in self.NOMINAL_RANGES.keys():
            if key in params:
                value = params[key]
                if key not in self._baseline_stats:
                    self._baseline_stats[key] = (value, 0.1)  # Initial mean, std
                else:
                    # Exponential moving average update
                    alpha = 0.01
                    old_mean, old_std = self._baseline_stats[key]
                    new_mean = (1 - alpha) * old_mean + alpha * value
                    new_std = max(0.1, (1 - alpha) * old_std + alpha * abs(value - new_mean))
                    self._baseline_stats[key] = (new_mean, new_std)
    
    def _detect_anomalies(self, params: Dict[str, Any]) -> Tuple[List[str], float]:
        """
        Detect anomalies in operational parameters.
        
        Uses a simplified z-score based anomaly detection.
        """
        anomalies = []
        total_score = 0
        count = 0
        
        for key, (min_val, max_val) in self.NOMINAL_RANGES.items():
            if key in params:
                value = params[key]
                
                # Check nominal range
                if value < min_val or value > max_val:
                    anomalies.append(f"{key}: {value:.2f} outside range [{min_val}, {max_val}]")
                    total_score += 0.8
                    count += 1
                # Check z-score against baseline
                elif key in self._baseline_stats:
                    mean, std = self._baseline_stats[key]
                    if std > 0:
                        z_score = abs(value - mean) / std
                        if z_score > 3:
                            anomalies.append(f"{key}: z-score {z_score:.1f} (value={value:.2f})")
                            total_score += min(1.0, z_score / 5)
                            count += 1
        
        # Calculate average anomaly score
        anomaly_score = total_score / max(1, len(self.NOMINAL_RANGES))
        
        return anomalies, min(1.0, anomaly_score)
    
    def _determine_mode(self, 
                        params: Dict[str, Any],
                        anomaly_score: float,
                        anomalies: List[str]) -> Tuple[OperationalMode, float]:
        """Determine operational mode based on anomalies."""
        
        # Check for complete failure
        if anomaly_score >= self.EMERGENCY_THRESHOLD and len(anomalies) > 3:
            return OperationalMode.FAILED, 0
        
        # Check for emergency
        if anomaly_score >= self.EMERGENCY_THRESHOLD:
            return OperationalMode.EMERGENCY, 25
        
        # Check for degraded
        if anomaly_score >= self.DEGRADED_THRESHOLD:
            return OperationalMode.DEGRADED, 50
        
        # Check for reduced power
        if anomaly_score >= self.ANOMALY_THRESHOLD or len(anomalies) > 0:
            return OperationalMode.REDUCED_POWER, 75
        
        # Full power operation
        return OperationalMode.FULL_POWER, 100
    
    def _handle_mode_transition(self, old_mode: OperationalMode, new_mode: OperationalMode) -> None:
        """Handle mode transitions and track failures."""
        # Count transitions to failed state
        if new_mode == OperationalMode.FAILED:
            self._failure_count += 1
        
        # Log transition belief
        self.update_belief(AgentBelief(
            belief_type="mode_transition",
            value={
                'from': old_mode.name,
                'to': new_mode.name,
                'timestamp': self._current_state.timestamp if self._current_state else 0
            },
            confidence=1.0,
            source="mode_monitoring",
            evidence=[f"Failure count: {self._failure_count}"]
        ))
    
    def _calculate_availability(self) -> float:
        """
        Calculate system availability.
        
        A = MTBF / (MTBF + MTTR)
        
        For continuous monitoring:
        A = Operational Time / Total Time
        """
        total_time = self._operational_hours + self._downtime_hours
        if total_time <= 0:
            return 1.0  # Assume 100% if no data
        
        return round(self._operational_hours / total_time, 4)
    
    def _calculate_mtbf(self) -> float:
        """Calculate Mean Time Between Failures."""
        if self._failure_count <= 0:
            return self._operational_hours if self._operational_hours > 0 else float('inf')
        return self._operational_hours / self._failure_count
    
    def _analyze_availability_trend(self) -> Dict[str, Any]:
        """Analyze availability trend over recent history."""
        if len(self._state_history) < 10:
            return {'status': 'INSUFFICIENT_DATA'}
        
        recent = self._state_history[-50:]
        
        # Calculate mode distribution
        mode_counts = {}
        for state in recent:
            mode_name = state.mode.name
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
        
        # Average capability
        avg_capability = np.mean([s.capability_pct for s in recent])
        
        # Trend direction
        if len(recent) > 20:
            first_half = np.mean([s.capability_pct for s in recent[:len(recent)//2]])
            second_half = np.mean([s.capability_pct for s in recent[len(recent)//2:]])
            if second_half < first_half - 5:
                trend_status = 'DECLINING'
            elif second_half > first_half + 5:
                trend_status = 'IMPROVING'
            else:
                trend_status = 'STABLE'
        else:
            trend_status = 'INSUFFICIENT_DATA'
        
        return {
            'average_capability': round(avg_capability, 1),
            'mode_distribution': mode_counts,
            'trend': trend_status,
            'sample_size': len(recent)
        }
    
    def _generate_availability_alert(self) -> None:
        """Generate alert for degraded availability."""
        if not self._current_state:
            return
        
        priority = (MessagePriority.CRITICAL if self._current_state.mode in 
                   [OperationalMode.FAILED, OperationalMode.EMERGENCY] 
                   else MessagePriority.HIGH)
        
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'availability_degraded',
                'mode': self._current_state.mode.name,
                'capability_pct': self._current_state.capability_pct,
                'anomalies': self._current_state.anomalies_detected,
                'availability_metric': self._calculate_availability()
            },
            priority=priority
        )
        self.communicate(msg)
    
    def get_rams_contribution(self) -> float:
        """Get this agent's contribution to system availability metric."""
        return self._calculate_availability()
    
    # =========================================================================
    # ENHANCED: Sensor Redundancy Monitoring
    # =========================================================================
    
    def update_sensor_detections(self, 
                                  detections: List[Dict[str, Any]], 
                                  timestamp: float) -> Optional['RedundancyState']:
        """
        Update sensor redundancy state from detection data.
        
        Args:
            detections: List of sensor detections with 'sensorId' field
            timestamp: Current timestamp
            
        Returns:
            RedundancyState if sensor monitoring is enabled
        """
        if self._sensor_redundancy is None:
            return None
        
        self._sensor_redundancy.update_from_detections(detections, timestamp)
        self._sensor_redundancy_state = self._sensor_redundancy.get_redundancy_state(timestamp)
        
        # Generate alert if redundancy is degraded
        if self._sensor_redundancy_state.redundancy_level in ['MINIMAL', 'FAILED']:
            self._generate_sensor_redundancy_alert()
        
        return self._sensor_redundancy_state
    
    def get_sensor_redundancy_state(self) -> Optional[Dict[str, Any]]:
        """Get current sensor redundancy state as dict."""
        if self._sensor_redundancy_state:
            return self._sensor_redundancy_state.to_dict()
        return None
    
    def _generate_sensor_redundancy_alert(self) -> None:
        """Generate alert for sensor redundancy issues."""
        if not self._sensor_redundancy_state:
            return
        
        priority = (MessagePriority.CRITICAL if 
                   self._sensor_redundancy_state.redundancy_level == 'FAILED'
                   else MessagePriority.HIGH)
        
        failed_sensors = [
            s.sensor_name for s in self._sensor_redundancy_state.sensor_statuses
            if hasattr(s, 'health') and s.health.name == 'FAILED'
        ]
        
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'sensor_redundancy_degraded',
                'redundancy_level': self._sensor_redundancy_state.redundancy_level,
                'active_sensors': self._sensor_redundancy_state.active_sensors,
                'total_sensors': self._sensor_redundancy_state.total_sensors,
                'capability_pct': self._sensor_redundancy_state.capability_pct,
                'failed_sensors': failed_sensors
            },
            priority=priority
        )
        self.communicate(msg)
    
    def simulate_sensor_failure(self, sensor_id: int) -> None:
        """Simulate a sensor failure for testing."""
        if self._sensor_redundancy:
            self._sensor_redundancy.simulate_sensor_failure(sensor_id)
    
    def reset_sensor(self, sensor_id: int) -> None:
        """Reset a sensor to healthy state."""
        if self._sensor_redundancy:
            self._sensor_redundancy.reset_sensor(sensor_id)
    
    # =========================================================================
    # ENHANCED: DP System Availability Monitoring
    # =========================================================================
    
    def update_dp_state(self,
                        position_error_m: float,
                        heading_error_deg: float,
                        thruster_status: Optional[Dict[str, Dict]] = None,
                        timestamp: float = 0.0) -> Optional['DPState']:
        """
        Update DP system availability state.
        
        Args:
            position_error_m: Distance from DP setpoint (meters)
            heading_error_deg: Heading error (degrees)
            thruster_status: Optional dict of {thruster_id: {rpm, thrust_pct, ...}}
            timestamp: Current timestamp
            
        Returns:
            DPState if DP monitoring is enabled
        """
        if self._dp_monitor is None:
            return None
        
        # Update thruster status if provided
        if thruster_status:
            for tid, status in thruster_status.items():
                self._dp_monitor.update_thruster_status(
                    thruster_id=tid,
                    rpm=status.get('rpm', 0),
                    thrust_pct=status.get('thrust_pct', 100),
                    azimuth_deg=status.get('azimuth_deg', 0),
                    temperature_c=status.get('temperature_c', 0),
                    current_a=status.get('current_a', 0)
                )
        
        # Update position error
        self._dp_state = self._dp_monitor.update_position_error(
            position_error_m, heading_error_deg, timestamp
        )
        
        # Generate alert if DP capability is degraded
        if self._dp_state.capability in [DPCapability.DEGRADED, DPCapability.FAILED]:
            self._generate_dp_alert()
        
        return self._dp_state
    
    def get_dp_state(self) -> Optional[Dict[str, Any]]:
        """Get current DP state as dict."""
        if self._dp_state:
            return self._dp_state.to_dict()
        return None
    
    def get_dp_thrust_capability(self) -> Dict[str, float]:
        """Get DP thrust allocation capability per DOF."""
        if self._dp_monitor:
            return self._dp_monitor.get_thrust_allocation_capability()
        return {'surge': 100.0, 'sway': 100.0, 'yaw': 100.0}
    
    def _generate_dp_alert(self) -> None:
        """Generate alert for DP capability issues."""
        if not self._dp_state:
            return
        
        priority = (MessagePriority.CRITICAL if 
                   self._dp_state.capability == DPCapability.FAILED
                   else MessagePriority.HIGH)
        
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'dp_capability_degraded',
                'capability': self._dp_state.capability.name,
                'capability_pct': self._dp_state.capability_pct,
                'position_error_m': self._dp_state.position_error_m,
                'heading_error_deg': self._dp_state.heading_error_deg,
                'station_keeping': self._dp_state.station_keeping,
                'active_thrusters': self._dp_state.active_thrusters,
                'alerts': self._dp_state.alerts
            },
            priority=priority
        )
        self.communicate(msg)
    
    def simulate_thruster_failure(self, thruster_id: str) -> None:
        """Simulate a thruster failure for testing."""
        if self._dp_monitor:
            self._dp_monitor.simulate_thruster_failure(thruster_id)
    
    def reset_thruster(self, thruster_id: str) -> None:
        """Reset a thruster to operational state."""
        if self._dp_monitor:
            self._dp_monitor.reset_thruster(thruster_id)
    
    # =========================================================================
    # ENHANCED: ML-based Anomaly Detection
    # =========================================================================
    
    def detect_anomaly_ml(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect anomaly using ML autoencoder model.
        
        Args:
            features: (16,) array of operational features
            
        Returns:
            Anomaly result dict or None if ML not available
        """
        if self._ml_anomaly_detector is None:
            return None
        
        try:
            result = self._ml_anomaly_detector.detect(features)
            return result.to_dict()
        except Exception:
            return None
    
    # =========================================================================
    # Combined Availability Assessment
    # =========================================================================
    
    def get_combined_availability(self) -> Dict[str, Any]:
        """
        Get combined availability assessment from all sources.
        
        Combines:
        - Propulsion system availability
        - Sensor redundancy capability
        - DP system capability
        
        Returns:
            Comprehensive availability dict
        """
        # Base propulsion availability
        propulsion_availability = self._calculate_availability()
        propulsion_capability = self._current_state.capability_pct if self._current_state else 100.0
        
        # Sensor redundancy
        sensor_capability = 100.0
        if self._sensor_redundancy_state:
            sensor_capability = self._sensor_redundancy_state.capability_pct
        
        # DP capability
        dp_capability = 100.0
        if self._dp_state:
            dp_capability = self._dp_state.capability_pct
        
        # Combined capability (weighted or minimum based on criticality)
        # Using minimum as any subsystem degradation affects overall capability
        combined_capability = min(propulsion_capability, sensor_capability, dp_capability)
        
        # Determine overall status
        if combined_capability >= 75:
            status = 'NORMAL'
        elif combined_capability >= 50:
            status = 'DEGRADED'
        elif combined_capability >= 25:
            status = 'CRITICAL'
        else:
            status = 'FAILED'
        
        return {
            'combined_capability_pct': combined_capability,
            'status': status,
            'subsystems': {
                'propulsion': {
                    'availability': propulsion_availability,
                    'capability_pct': propulsion_capability
                },
                'sensors': {
                    'capability_pct': sensor_capability,
                    'redundancy_level': (self._sensor_redundancy_state.redundancy_level 
                                        if self._sensor_redundancy_state else 'UNKNOWN')
                },
                'dp_system': {
                    'capability_pct': dp_capability,
                    'capability': (self._dp_state.capability.name 
                                  if self._dp_state else 'UNKNOWN')
                }
            }
        }
