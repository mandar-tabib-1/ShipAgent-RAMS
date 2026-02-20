"""
RAMS Agents - Agentic AI for Maritime RAMS Demonstration

This module implements a multi-agent system demonstrating Reliability, Availability,
Maintainability, and Safety (RAMS) principles for naval vessel operations.

Data Sources:
- UCI Naval Propulsion CBM Dataset (CC BY 4.0) - Propulsion health monitoring
- AutoFerry Sensor Fusion Dataset (MIT) - Navigation safety

Enhanced Features:
- Multi-sensor redundancy monitoring (DNV-style)
- DP system availability tracking
- LSTM-based RUL prediction
- Autoencoder-based anomaly detection

This is a methodology demonstration applicable to any naval vessel.
"""

from .base_agent import (
    RAMSCategory,
    AgentMessage,
    MessagePriority,
    AgentBelief,
    RAMSBaseAgent,
)

from .reliability_agent import ReliabilityAgent
from .availability_agent import AvailabilityAgent
from .maintainability_agent import MaintainabilityAgent
from .safety_agent import SafetyAgent
from .supervisor_agent import RAMSSupervisorAgent
from .rul_estimator import RULEstimator, DegradationModel, HealthIndex, RULEstimate

# LLM-powered supervisor
from .llm_supervisor_agent import (
    LLMSupervisorAgent, 
    SupervisorDecision,
    LLMConfig,
    SupervisorConfig,
    VesselContext,
    create_llm_supervisor
)

# Enhanced availability modules
from .sensor_redundancy import (
    SensorVotingFusion, SensorHealth, VotingStrategy, RedundancyState,
    create_sensor_fusion_redundancy
)
from .dp_availability import (
    DPAvailabilityMonitor, DPCapability, DPClass, DPState,
    create_dp_monitor
)

__all__ = [
    # Core RAMS
    'RAMSCategory',
    'AgentMessage',
    'MessagePriority',
    'AgentBelief',
    'RAMSBaseAgent',
    'ReliabilityAgent',
    'AvailabilityAgent',
    'MaintainabilityAgent',
    'SafetyAgent',
    'RAMSSupervisorAgent',
    'RULEstimator',
    'DegradationModel',
    'HealthIndex',
    'RULEstimate',
    # Enhanced Availability
    'SensorVotingFusion',
    'SensorHealth',
    'VotingStrategy',
    'RedundancyState',
    'create_sensor_fusion_redundancy',
    'DPAvailabilityMonitor',
    'DPCapability',
    'DPClass',
    'DPState',
    'create_dp_monitor',
    # LLM Supervisor
    'LLMSupervisorAgent',
    'SupervisorDecision',
    'LLMConfig',
    'SupervisorConfig',
    'VesselContext',
    'create_llm_supervisor',
]

__version__ = '1.2.0'
__author__ = 'RAMS Demonstration Project'
