"""
DP (Dynamic Positioning) Availability Monitor

Monitors dynamic positioning system availability based on thruster status,
position error, and control system health.

Implements DNV DP capability assessment for Vessel:
- 2x Azimuth thrusters (stern)
- 1x Tunnel thruster (bow)

Part of the RAMS Availability Agent enhancement.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class DPClass(Enum):
    """DNV DP Equipment Classes."""
    DP_0 = "DP-0"      # Manual position keeping
    DP_1 = "DP-1"      # Automatic, single failure tolerance
    DP_2 = "DP-2"      # Redundant systems, single fault tolerance
    DP_3 = "DP-3"      # Full redundancy, physically separated


class DPCapability(Enum):
    """DP capability states."""
    FULL = auto()           # All thrusters operational
    REDUCED = auto()        # >= 2 thrusters, position maintainable
    DEGRADED = auto()       # Minimum thrusters, limited capability
    FAILED = auto()         # Cannot maintain position


class ThrusterHealth(Enum):
    """Individual thruster health states."""
    OPERATIONAL = auto()    # Normal operation
    REDUCED = auto()        # Limited thrust available
    FAILED = auto()         # Non-operational
    UNKNOWN = auto()        # Status unknown


@dataclass
class ThrusterStatus:
    """Status of an individual thruster."""
    thruster_id: str
    thruster_type: str  # 'azimuth' or 'tunnel'
    health: ThrusterHealth
    thrust_pct: float  # 0-100% of max thrust available
    rpm: float = 0.0
    azimuth_deg: float = 0.0  # For azimuth thrusters
    temperature_c: float = 0.0
    current_a: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thruster_id': self.thruster_id,
            'thruster_type': self.thruster_type,
            'health': self.health.name,
            'thrust_pct': self.thrust_pct,
            'rpm': self.rpm,
            'azimuth_deg': self.azimuth_deg,
            'temperature_c': self.temperature_c,
            'current_a': self.current_a
        }


@dataclass
class DPState:
    """Current DP system state."""
    capability: DPCapability
    capability_pct: float  # 0-100%
    position_error_m: float
    heading_error_deg: float
    station_keeping: bool
    thrusters: List[ThrusterStatus] = field(default_factory=list)
    active_thrusters: int = 0
    total_thrusters: int = 3
    alerts: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'capability': self.capability.name,
            'capability_pct': self.capability_pct,
            'position_error_m': self.position_error_m,
            'heading_error_deg': self.heading_error_deg,
            'station_keeping': self.station_keeping,
            'thrusters': [t.to_dict() for t in self.thrusters],
            'active_thrusters': self.active_thrusters,
            'total_thrusters': self.total_thrusters,
            'alerts': self.alerts,
            'timestamp': self.timestamp
        }


class DPAvailabilityMonitor:
    """
    Monitors DP system availability for Vessel.
    
    Based on Kongsberg SDP-11 specifications:
    - Monitors thruster health and thrust allocation
    - Tracks position and heading errors
    - Assesses DP capability class
    - Generates alerts for capability loss
    
    Thruster Configuration (Vessel):
    - azimuth_stbd: Stern starboard azimuth thruster
    - azimuth_port: Stern port azimuth thruster
    - tunnel_bow: Bow tunnel thruster
    """
    
    # Vessel thruster configuration (from dp_agent.py)
    THRUSTER_CONFIG = {
        'azimuth_stbd': {
            'type': 'azimuth',
            'position': (-12.0, 3.0),  # (x_aft, y_stbd) from CoG
            'max_thrust_kN': 50.0,
            'critical_for': ['surge', 'sway', 'yaw']
        },
        'azimuth_port': {
            'type': 'azimuth',
            'position': (-12.0, -3.0),
            'max_thrust_kN': 50.0,
            'critical_for': ['surge', 'sway', 'yaw']
        },
        'tunnel_bow': {
            'type': 'tunnel',
            'position': (14.0, 0.0),
            'max_thrust_kN': 30.0,
            'critical_for': ['sway', 'yaw']
        }
    }
    
    # Thresholds for DNV-style capability assessment
    POSITION_ERROR_GOOD = 1.0       # meters - excellent station keeping
    POSITION_ERROR_WARNING = 3.0    # meters - acceptable
    POSITION_ERROR_CRITICAL = 10.0  # meters - position lost
    
    HEADING_ERROR_GOOD = 2.0        # degrees
    HEADING_ERROR_WARNING = 5.0     # degrees
    HEADING_ERROR_CRITICAL = 15.0   # degrees
    
    # Thruster health thresholds
    THRUST_WARNING_PCT = 70.0       # Below this = reduced capability
    THRUST_FAILED_PCT = 10.0        # Below this = failed
    
    # Temperature limits
    TEMP_WARNING_C = 80.0
    TEMP_CRITICAL_C = 100.0
    
    def __init__(self, dp_class: DPClass = DPClass.DP_1):
        """
        Initialize DP availability monitor.
        
        Args:
            dp_class: Target DP equipment class
        """
        self.dp_class = dp_class
        
        # Initialize thruster statuses
        self.thrusters: Dict[str, ThrusterStatus] = {}
        for tid, config in self.THRUSTER_CONFIG.items():
            self.thrusters[tid] = ThrusterStatus(
                thruster_id=tid,
                thruster_type=config['type'],
                health=ThrusterHealth.UNKNOWN,
                thrust_pct=100.0
            )
        
        # State tracking
        self._position_error_history: List[float] = []
        self._heading_error_history: List[float] = []
        self._current_state: Optional[DPState] = None
        
        # Alert history
        self._alert_history: List[Dict[str, Any]] = []
    
    def update_thruster_status(self, 
                               thruster_id: str,
                               rpm: float = 0.0,
                               thrust_pct: float = 100.0,
                               azimuth_deg: float = 0.0,
                               temperature_c: float = 0.0,
                               current_a: float = 0.0) -> ThrusterHealth:
        """
        Update status of a single thruster.
        
        Args:
            thruster_id: Thruster identifier
            rpm: Current RPM
            thrust_pct: Available thrust percentage (0-100)
            azimuth_deg: Azimuth angle for azimuth thrusters
            temperature_c: Motor temperature
            current_a: Motor current
            
        Returns:
            Assessed thruster health
        """
        if thruster_id not in self.thrusters:
            return ThrusterHealth.UNKNOWN
        
        thruster = self.thrusters[thruster_id]
        thruster.rpm = rpm
        thruster.thrust_pct = thrust_pct
        thruster.azimuth_deg = azimuth_deg
        thruster.temperature_c = temperature_c
        thruster.current_a = current_a
        
        # Assess health
        alerts = []
        
        # Check thrust capability
        if thrust_pct < self.THRUST_FAILED_PCT:
            thruster.health = ThrusterHealth.FAILED
            alerts.append(f"{thruster_id}: FAILED - thrust at {thrust_pct:.1f}%")
        elif thrust_pct < self.THRUST_WARNING_PCT:
            thruster.health = ThrusterHealth.REDUCED
            alerts.append(f"{thruster_id}: REDUCED - thrust at {thrust_pct:.1f}%")
        else:
            thruster.health = ThrusterHealth.OPERATIONAL
        
        # Check temperature
        if temperature_c > self.TEMP_CRITICAL_C:
            thruster.health = ThrusterHealth.FAILED
            alerts.append(f"{thruster_id}: OVERTEMP - {temperature_c:.1f}°C")
        elif temperature_c > self.TEMP_WARNING_C:
            if thruster.health == ThrusterHealth.OPERATIONAL:
                thruster.health = ThrusterHealth.REDUCED
            alerts.append(f"{thruster_id}: High temp - {temperature_c:.1f}°C")
        
        # Store alerts
        for alert in alerts:
            self._alert_history.append({
                'timestamp': self._current_state.timestamp if self._current_state else 0,
                'thruster': thruster_id,
                'message': alert
            })
        
        return thruster.health
    
    def update_position_error(self, 
                               position_error_m: float,
                               heading_error_deg: float,
                               timestamp: float) -> DPState:
        """
        Update DP state with position/heading errors.
        
        Args:
            position_error_m: Distance from setpoint in meters
            heading_error_deg: Heading error in degrees
            timestamp: Current timestamp
            
        Returns:
            Updated DPState
        """
        # Store history
        self._position_error_history.append(position_error_m)
        self._heading_error_history.append(heading_error_deg)
        
        # Trim history
        max_history = 100
        if len(self._position_error_history) > max_history:
            self._position_error_history = self._position_error_history[-max_history:]
            self._heading_error_history = self._heading_error_history[-max_history:]
        
        # Assess capability
        return self._assess_capability(position_error_m, heading_error_deg, timestamp)
    
    def _assess_capability(self, 
                           position_error_m: float,
                           heading_error_deg: float,
                           timestamp: float) -> DPState:
        """
        Assess overall DP capability.
        
        Considers:
        - Number of operational thrusters
        - Position/heading error magnitude
        - Ability to generate forces in all DOF
        """
        alerts = []
        
        # Count operational thrusters
        # UNKNOWN is treated as OPERATIONAL (assume nominal until status received)
        operational = sum(
            1 for t in self.thrusters.values() 
            if t.health in [ThrusterHealth.OPERATIONAL, ThrusterHealth.UNKNOWN]
        )
        reduced = sum(
            1 for t in self.thrusters.values()
            if t.health == ThrusterHealth.REDUCED
        )
        
        # Effective operational count (reduced = 0.5)
        effective_thrusters = operational + 0.5 * reduced
        
        # Check if station keeping is possible
        # Need at least surge OR (1 azimuth + tunnel) for sway/yaw
        azimuth_ok = sum(
            1 for tid, t in self.thrusters.items() 
            if 'azimuth' in tid and t.health in [ThrusterHealth.OPERATIONAL, ThrusterHealth.REDUCED, ThrusterHealth.UNKNOWN]
        )
        tunnel_ok = self.thrusters['tunnel_bow'].health in [
            ThrusterHealth.OPERATIONAL, ThrusterHealth.REDUCED, ThrusterHealth.UNKNOWN
        ]
        
        station_keeping = (azimuth_ok >= 1) or (azimuth_ok >= 1 and tunnel_ok)
        
        # Determine capability level
        if effective_thrusters >= 2.5 and position_error_m < self.POSITION_ERROR_WARNING:
            capability = DPCapability.FULL
            capability_pct = 100.0
        elif effective_thrusters >= 2.0 and position_error_m < self.POSITION_ERROR_CRITICAL:
            capability = DPCapability.REDUCED
            capability_pct = 75.0
            alerts.append("DP REDUCED: Limited thruster availability")
        elif effective_thrusters >= 1.0 and station_keeping:
            capability = DPCapability.DEGRADED
            capability_pct = 50.0
            alerts.append("DP DEGRADED: Minimum capability")
        else:
            capability = DPCapability.FAILED
            capability_pct = 0.0
            alerts.append("DP FAILED: Cannot maintain position")
        
        # Adjust for position error
        if position_error_m > self.POSITION_ERROR_CRITICAL:
            capability = DPCapability.FAILED
            capability_pct = 0.0
            alerts.append(f"POSITION LOST: Error {position_error_m:.1f}m")
        elif position_error_m > self.POSITION_ERROR_WARNING:
            if capability == DPCapability.FULL:
                capability = DPCapability.REDUCED
                capability_pct = min(capability_pct, 75.0)
            alerts.append(f"Position warning: Error {position_error_m:.1f}m")
        
        # Adjust for heading error
        if heading_error_deg > self.HEADING_ERROR_CRITICAL:
            if capability in [DPCapability.FULL, DPCapability.REDUCED]:
                capability = DPCapability.DEGRADED
                capability_pct = min(capability_pct, 50.0)
            alerts.append(f"Heading warning: Error {heading_error_deg:.1f}°")
        
        # Create state
        self._current_state = DPState(
            capability=capability,
            capability_pct=capability_pct,
            position_error_m=position_error_m,
            heading_error_deg=heading_error_deg,
            station_keeping=station_keeping,
            thrusters=list(self.thrusters.values()),
            active_thrusters=operational + reduced,
            total_thrusters=len(self.thrusters),
            alerts=alerts,
            timestamp=timestamp
        )
        
        return self._current_state
    
    def get_current_state(self) -> Optional[DPState]:
        """Get current DP state."""
        return self._current_state
    
    def get_capability_percentage(self) -> float:
        """Get current capability as percentage."""
        if self._current_state:
            return self._current_state.capability_pct
        return 100.0  # Assume full if no data
    
    def simulate_thruster_failure(self, thruster_id: str) -> None:
        """Simulate a thruster failure for testing."""
        if thruster_id in self.thrusters:
            self.thrusters[thruster_id].health = ThrusterHealth.FAILED
            self.thrusters[thruster_id].thrust_pct = 0.0
    
    def reset_thruster(self, thruster_id: str) -> None:
        """Reset a thruster to operational state."""
        if thruster_id in self.thrusters:
            self.thrusters[thruster_id].health = ThrusterHealth.OPERATIONAL
            self.thrusters[thruster_id].thrust_pct = 100.0
    
    def get_thrust_allocation_capability(self) -> Dict[str, float]:
        """
        Calculate force/moment generation capability in each DOF.
        
        Returns:
            Dict with 'surge', 'sway', 'yaw' capability percentages
        """
        # Maximum forces from each thruster
        surge_max = 0.0
        sway_max = 0.0
        yaw_max = 0.0
        
        surge_available = 0.0
        sway_available = 0.0
        yaw_available = 0.0
        
        for tid, thruster in self.thrusters.items():
            config = self.THRUSTER_CONFIG[tid]
            max_thrust = config['max_thrust_kN']
            available = max_thrust * (thruster.thrust_pct / 100.0)
            
            # Position for moment arm
            x, y = config['position']
            
            if thruster.thruster_type == 'azimuth':
                # Azimuth can produce force in any direction
                surge_max += max_thrust
                sway_max += max_thrust
                yaw_max += max_thrust * abs(y)  # Moment arm
                
                if thruster.health != ThrusterHealth.FAILED:
                    surge_available += available
                    sway_available += available
                    yaw_available += available * abs(y)
            else:
                # Tunnel thruster: sway only
                sway_max += max_thrust
                yaw_max += max_thrust * abs(x)
                
                if thruster.health != ThrusterHealth.FAILED:
                    sway_available += available
                    yaw_available += available * abs(x)
        
        return {
            'surge': (surge_available / surge_max * 100) if surge_max > 0 else 0,
            'sway': (sway_available / sway_max * 100) if sway_max > 0 else 0,
            'yaw': (yaw_available / yaw_max * 100) if yaw_max > 0 else 0
        }


# Convenience function
def create_dp_monitor(dp_class: DPClass = DPClass.DP_1) -> DPAvailabilityMonitor:
    """Create a DPAvailabilityMonitor for Vessel."""
    return DPAvailabilityMonitor(dp_class=dp_class)
