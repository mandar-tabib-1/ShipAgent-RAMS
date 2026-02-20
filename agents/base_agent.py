"""
Base Agent Module for Vessel AI System
============================================
Foundation class for all Vessel agents.
Provides common functionality for state management, logging, and alerts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import math


class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """System alert structure"""
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'source': self.source,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


@dataclass
class VesselState:
    """Current vessel state in NED frame"""
    timestamp: datetime
    north: float = 0.0  # meters from reference
    east: float = 0.0   # meters from reference
    yaw: float = 0.0    # radians
    surge: float = 0.0  # m/s
    sway: float = 0.0   # m/s
    yaw_rate: float = 0.0  # rad/s
    latitude: float = 63.43  # degrees
    longitude: float = 10.39  # degrees
    sog: float = 0.0  # speed over ground (knots)
    cog: float = 0.0  # course over ground (degrees)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'position': {'north': self.north, 'east': self.east},
            'heading_deg': math.degrees(self.yaw),
            'velocity': {'surge': self.surge, 'sway': self.sway, 'yaw_rate': self.yaw_rate},
            'geo': {'latitude': self.latitude, 'longitude': self.longitude},
            'sog': self.sog,
            'cog': self.cog
        }


@dataclass
class ThrusterState:
    """State of a thruster unit"""
    thruster_id: str
    thruster_type: str = "azimuth"  # "azimuth" or "tunnel"
    rpm: float = 0.0
    azimuth_angle: float = 0.0  # radians (for azimuth thrusters)
    thrust_force: float = 0.0  # N
    power: float = 0.0  # kW
    temperature: float = 25.0  # °C
    
    def to_dict(self) -> Dict:
        return {
            'id': self.thruster_id,
            'type': self.thruster_type,
            'rpm': self.rpm,
            'azimuth_deg': math.degrees(self.azimuth_angle),
            'thrust_kN': self.thrust_force / 1000,
            'power_kW': self.power,
            'temp_C': self.temperature
        }


@dataclass 
class AgentState:
    """Current state of an agent"""
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    progress: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, status: AgentStatus, task: Optional[str] = None, progress: float = 0.0):
        self.status = status
        self.current_task = task
        self.progress = progress
        self.last_activity = datetime.now()
        self.error_message = None
    
    def set_error(self, message: str):
        self.status = AgentStatus.ERROR
        self.error_message = message
        self.last_activity = datetime.now()


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Vessel AI System.
    
    Each agent specializes in a specific task:
    - Navigation agents handle sensor fusion and collision avoidance
    - DP agents handle dynamic positioning control
    - Propulsion agents handle predictive maintenance
    """
    
    def __init__(self, name: str, description: str, domain: str = "general"):
        self.name = name
        self.description = description
        self.domain = domain  # "navigation", "dp", "propulsion", "integration"
        self.state = AgentState()
        self.logger = logging.getLogger(f"Vessel.{name}")
        self.alerts: List[Alert] = []
        self.results_history: List[Dict] = []
        
    @abstractmethod
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by all agents.
        
        Args:
            data: Input data for processing
            context: Optional context from orchestrator or other agents
            
        Returns:
            Dictionary containing results, alerts, and any recommendations
        """
        pass
    
    def update_status(self, status: AgentStatus, task: Optional[str] = None, progress: float = 0.0):
        """Update agent status"""
        self.state.update(status, task, progress)
    
    def create_alert(self, level: AlertLevel, message: str, 
                     details: Dict = None) -> Alert:
        """Create and store an alert"""
        alert = Alert(
            level=level,
            source=self.name,
            message=message,
            details=details or {}
        )
        self.alerts.append(alert)
        
        # Log based on severity
        if level == AlertLevel.EMERGENCY:
            self.logger.critical(f"🚨 {message}")
        elif level == AlertLevel.CRITICAL:
            self.logger.error(f"⚠️ {message}")
        elif level == AlertLevel.WARNING:
            self.logger.warning(f"⚡ {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
            
        return alert
    
    def get_status(self) -> Dict:
        """Return current agent status"""
        return {
            'name': self.name,
            'domain': self.domain,
            'description': self.description,
            'status': self.state.status.value,
            'current_task': self.state.current_task,
            'progress': self.state.progress,
            'last_activity': self.state.last_activity.isoformat(),
            'active_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'metrics': self.state.metrics
        }
    
    def get_alerts(self, unacknowledged_only: bool = False) -> List[Dict]:
        """Get list of alerts"""
        alerts = self.alerts if not unacknowledged_only else [
            a for a in self.alerts if not a.acknowledged
        ]
        return [a.to_dict() for a in alerts]
    
    def log_result(self, result: Dict):
        """Log a processing result to history"""
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        if len(self.results_history) > 100:
            self.results_history = self.results_history[-100:]
    
    def reset(self):
        """Reset agent state"""
        self.state = AgentState()
        self.alerts.clear()
        self.logger.info("Agent reset")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', status='{self.state.status.value}')>"
