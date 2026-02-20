"""
Reliability Agent - Predicts component degradation and RUL

Part of the RAMS multi-agent system demonstrating agentic AI.
Uses the UCI Naval Propulsion CBM dataset for reliability analysis.

Key Responsibilities:
- Monitor propulsion system degradation (kMc, kMt coefficients)
- Predict Remaining Useful Life (RUL) using data-driven methods
- Communicate reliability alerts to other agents

RUL Estimation Methodology:
- Semi-supervised approach using Health Index (HI) trajectories
- Exponential decay model fitted to observed degradation
- Confidence intervals from model uncertainty
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base_agent import (
    RAMSBaseAgent, RAMSCategory, AgentBelief, AgentMessage,
    MessagePriority, RAMSMetrics
)
from .rul_estimator import RULEstimator, DegradationModel, RULEstimate


@dataclass
class DegradationState:
    """Current degradation state of components."""
    compressor_decay: float  # kMc: 1.0 (new) → 0.95 (degraded)
    turbine_decay: float     # kMt: 1.0 (new) → 0.975 (degraded)
    timestamp: float
    
    @property
    def compressor_health_pct(self) -> float:
        """Compressor health as percentage (100% = new)."""
        # Map [0.95, 1.0] to [0, 100]
        return max(0, (self.compressor_decay - 0.95) / 0.05 * 100)
    
    @property
    def turbine_health_pct(self) -> float:
        """Turbine health as percentage (100% = new)."""
        # Map [0.975, 1.0] to [0, 100]
        return max(0, (self.turbine_decay - 0.975) / 0.025 * 100)
    
    @property
    def overall_health(self) -> float:
        """Overall propulsion health (0-100)."""
        return (self.compressor_health_pct + self.turbine_health_pct) / 2


class ReliabilityAgent(RAMSBaseAgent):
    """
    Agent responsible for system reliability monitoring.
    
    RAMS Category: R (Reliability)
    
    Agentic Behaviors:
    - Perceives: Operational data from propulsion system
    - Reasons: Predicts degradation using ML, estimates RUL
    - Acts: Generates reliability forecasts and alerts
    - Communicates: Sends degradation warnings to Supervisor and Maintenance agents
    
    Data Source: UCI Naval Propulsion CBM Dataset
    """
    
    # Degradation thresholds
    COMPRESSOR_WARNING = 0.97   # kMc threshold for warning
    COMPRESSOR_CRITICAL = 0.96  # kMc threshold for critical
    TURBINE_WARNING = 0.985    # kMt threshold for warning
    TURBINE_CRITICAL = 0.98    # kMt threshold for critical
    
    # RUL parameters (operating hours)
    MAX_RUL = 5000  # Maximum expected operating hours
    
    def __init__(self):
        super().__init__(name="ReliabilityAgent", category=RAMSCategory.RELIABILITY)
        
        # Degradation tracking
        self._degradation_history: List[DegradationState] = []
        self._current_degradation: Optional[DegradationState] = None
        
        # Data-driven RUL estimator (semi-supervised approach)
        self._rul_estimator = RULEstimator(
            preferred_model=DegradationModel.EXPONENTIAL,
            min_observations=5,
            max_rul_hours=self.MAX_RUL
        )
        self._population_initialized = False
        
        # ML model for degradation prediction (gradient boosting)
        self._model_trained = False
        self._model = None
        self._scaler = None
        
        # Agent goals
        self.goals = [
            "Monitor propulsion system degradation",
            "Predict component RUL using data-driven methods",
            "Alert on reliability degradation"
        ]
    
    def initialize_from_population(self, 
                                    kMc_samples: np.ndarray, 
                                    kMt_samples: np.ndarray) -> Dict[str, Any]:
        """
        Initialize RUL estimator with population data from UCI dataset.
        
        This is the "unsupervised" learning step - we learn degradation
        statistics from the full dataset distribution.
        
        Args:
            kMc_samples: All compressor decay values
            kMt_samples: All turbine decay values
            
        Returns:
            Population statistics learned
        """
        stats = self._rul_estimator.learn_population_statistics(kMc_samples, kMt_samples)
        self._population_initialized = True
        return stats
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive propulsion system state.
        
        Expected environment_data keys:
        - 'propulsion': PropulsionDataPoint or dict with operational data
        - 'timestamp': Current simulation time
        """
        propulsion = environment_data.get('propulsion', {})
        timestamp = environment_data.get('timestamp', 0.0)
        
        # Extract degradation coefficients
        if hasattr(propulsion, 'compressor_decay'):
            kMc = propulsion.compressor_decay
            kMt = propulsion.turbine_decay
        elif isinstance(propulsion, dict):
            kMc = propulsion.get('compressor_decay', 1.0)
            kMt = propulsion.get('turbine_decay', 1.0)
        else:
            kMc = 1.0
            kMt = 1.0
        
        # Update current degradation state
        self._current_degradation = DegradationState(
            compressor_decay=kMc,
            turbine_decay=kMt,
            timestamp=timestamp
        )
        self._degradation_history.append(self._current_degradation)
        
        # Update data-driven RUL estimator with new observation
        self._rul_estimator.update(kMc, kMt, timestamp)
        
        # Keep history manageable
        if len(self._degradation_history) > 1000:
            self._degradation_history = self._degradation_history[-500:]
        
        # Form perception belief
        self.update_belief(AgentBelief(
            belief_type="current_degradation",
            value={
                'compressor_decay': kMc,
                'turbine_decay': kMt,
                'compressor_health_pct': self._current_degradation.compressor_health_pct,
                'turbine_health_pct': self._current_degradation.turbine_health_pct,
                'overall_health': self._current_degradation.overall_health
            },
            confidence=0.95,
            source="propulsion_sensors",
            evidence=["UCI Naval Propulsion Dataset"]
        ))
    
    def reason(self) -> List[AgentBelief]:
        """
        Reason about reliability state and predict future degradation.
        
        Applies:
        1. Degradation trend analysis
        2. RUL estimation
        3. Alert level determination
        """
        new_beliefs = []
        
        if not self._current_degradation:
            return new_beliefs
        
        # 1. Analyze degradation trend
        trend = self._analyze_degradation_trend()
        new_beliefs.append(AgentBelief(
            belief_type="degradation_trend",
            value=trend,
            confidence=0.8 if len(self._degradation_history) > 10 else 0.5,
            source="trend_analysis",
            evidence=[f"Based on {len(self._degradation_history)} observations"]
        ))
        
        # 2. Estimate RUL
        rul_estimate = self._estimate_rul()
        new_beliefs.append(AgentBelief(
            belief_type="rul_estimate",
            value=rul_estimate,
            confidence=0.75,
            source="rul_model",
            evidence=["Linear degradation extrapolation"]
        ))
        
        # 3. Determine alert level
        alert = self._determine_alert_level()
        if alert['level'] != 'NORMAL':
            new_beliefs.append(AgentBelief(
                belief_type="reliability_alert",
                value=alert,
                confidence=0.9,
                source="threshold_check",
                evidence=[f"kMc={self._current_degradation.compressor_decay:.4f}",
                         f"kMt={self._current_degradation.turbine_decay:.4f}"]
            ))
            
            # Communicate alert to supervisor
            self._generate_alert_message(alert)
        
        return new_beliefs
    
    def act(self) -> Dict[str, Any]:
        """
        Produce reliability assessment and recommendations.
        
        Returns comprehensive reliability report.
        """
        if not self._current_degradation:
            return {'status': 'NO_DATA'}
        
        rul = self._estimate_rul()
        trend = self._analyze_degradation_trend()
        alert = self._determine_alert_level()
        
        # Calculate reliability metric R(t)
        # Using exponential reliability model based on degradation
        reliability = self._calculate_reliability()
        
        recommendations = []
        if alert['level'] == 'CRITICAL':
            recommendations.append("IMMEDIATE: Schedule emergency maintenance")
            recommendations.append("REDUCE: Lower operational load to extend RUL")
        elif alert['level'] == 'WARNING':
            recommendations.append("PLAN: Schedule maintenance within estimated RUL")
            recommendations.append("MONITOR: Increase monitoring frequency")
        else:
            recommendations.append("CONTINUE: Normal operations")
        
        return {
            'reliability_metric': reliability,
            'degradation_state': {
                'compressor_decay': self._current_degradation.compressor_decay,
                'turbine_decay': self._current_degradation.turbine_decay,
                'compressor_health_pct': self._current_degradation.compressor_health_pct,
                'turbine_health_pct': self._current_degradation.turbine_health_pct
            },
            'rul_estimate': rul,
            'degradation_trend': trend,
            'alert': alert,
            'recommendations': recommendations
        }
    
    def _analyze_degradation_trend(self) -> Dict[str, Any]:
        """Analyze degradation trend from history."""
        if len(self._degradation_history) < 2:
            return {'compressor_rate': 0, 'turbine_rate': 0, 'status': 'INSUFFICIENT_DATA'}
        
        # Calculate degradation rates (decay per time unit)
        kMc_values = [d.compressor_decay for d in self._degradation_history]
        kMt_values = [d.turbine_decay for d in self._degradation_history]
        times = [d.timestamp for d in self._degradation_history]
        
        if times[-1] - times[0] > 0:
            time_span = times[-1] - times[0]
            compressor_rate = (kMc_values[0] - kMc_values[-1]) / time_span
            turbine_rate = (kMt_values[0] - kMt_values[-1]) / time_span
        else:
            compressor_rate = 0
            turbine_rate = 0
        
        # Classify trend
        if compressor_rate > 0.0001 or turbine_rate > 0.0001:
            status = 'ACCELERATING' if compressor_rate > 0.0005 else 'DEGRADING'
        elif compressor_rate < -0.0001:
            status = 'RECOVERING'  # Unlikely but possible with maintenance
        else:
            status = 'STABLE'
        
        return {
            'compressor_rate': compressor_rate,
            'turbine_rate': turbine_rate,
            'status': status,
            'observation_count': len(self._degradation_history)
        }
    
    def _estimate_rul(self) -> Dict[str, Any]:
        """
        Estimate Remaining Useful Life using data-driven approach.
        
        Methodology (semi-supervised):
        1. Converts kMc/kMt to Health Index [0,1] scale
        2. Fits exponential decay model to observed trajectory
        3. Extrapolates to failure threshold with uncertainty bounds
        
        This is more accurate than simple linear extrapolation because:
        - Learns degradation dynamics from observations
        - Provides confidence intervals
        - Can incorporate population statistics from UCI dataset
        """
        if not self._current_degradation or len(self._degradation_history) < 2:
            return {
                'compressor_rul_hours': self.MAX_RUL,
                'turbine_rul_hours': self.MAX_RUL,
                'system_rul_hours': self.MAX_RUL,
                'limiting_component': 'unknown',
                'confidence': 'low',
                'method': 'insufficient_data'
            }
        
        # Use data-driven RUL estimator
        rul_estimate: RULEstimate = self._rul_estimator.estimate_rul()
        
        # Get component-specific RULs from health trend
        health_trend = self._rul_estimator.get_health_trend()
        
        # Calculate component RULs based on current health and estimated rates
        comp_rate = health_trend['compressor'].get('degradation_rate', 0)
        turb_rate = health_trend['turbine'].get('degradation_rate', 0)
        
        # Current health indices
        if len(self._rul_estimator.compressor_trajectory) > 0:
            comp_hi = self._rul_estimator.compressor_trajectory.health_indices[-1]
            turb_hi = self._rul_estimator.turbine_trajectory.health_indices[-1]
        else:
            comp_hi = 1.0
            turb_hi = 1.0
        
        # Component RULs (time to HI=0)
        compressor_rul = comp_hi / comp_rate if comp_rate > 0 else self.MAX_RUL
        turbine_rul = turb_hi / turb_rate if turb_rate > 0 else self.MAX_RUL
        
        # Clamp to reasonable values
        compressor_rul = max(0, min(self.MAX_RUL, compressor_rul))
        turbine_rul = max(0, min(self.MAX_RUL, turbine_rul))
        
        return {
            'compressor_rul_hours': round(compressor_rul, 1),
            'turbine_rul_hours': round(turbine_rul, 1),
            'system_rul_hours': rul_estimate.rul_hours,
            'limiting_component': rul_estimate.limiting_component,
            'confidence': rul_estimate.confidence_level,
            'confidence_interval': rul_estimate.confidence_interval,
            'model_used': rul_estimate.model_used,
            'health_index': rul_estimate.health_index,
            'method': 'data_driven_exponential'
        }
    
    def _determine_alert_level(self) -> Dict[str, Any]:
        """Determine alert level based on current degradation."""
        if not self._current_degradation:
            return {'level': 'UNKNOWN', 'reason': 'No data'}
        
        kMc = self._current_degradation.compressor_decay
        kMt = self._current_degradation.turbine_decay
        
        # Check critical thresholds
        if kMc < self.COMPRESSOR_CRITICAL:
            return {
                'level': 'CRITICAL',
                'component': 'compressor',
                'reason': f'Compressor decay kMc={kMc:.4f} below critical threshold {self.COMPRESSOR_CRITICAL}'
            }
        if kMt < self.TURBINE_CRITICAL:
            return {
                'level': 'CRITICAL',
                'component': 'turbine',
                'reason': f'Turbine decay kMt={kMt:.4f} below critical threshold {self.TURBINE_CRITICAL}'
            }
        
        # Check warning thresholds
        if kMc < self.COMPRESSOR_WARNING:
            return {
                'level': 'WARNING',
                'component': 'compressor',
                'reason': f'Compressor decay kMc={kMc:.4f} below warning threshold {self.COMPRESSOR_WARNING}'
            }
        if kMt < self.TURBINE_WARNING:
            return {
                'level': 'WARNING',
                'component': 'turbine',
                'reason': f'Turbine decay kMt={kMt:.4f} below warning threshold {self.TURBINE_WARNING}'
            }
        
        return {'level': 'NORMAL', 'reason': 'All components within acceptable limits'}
    
    def _calculate_reliability(self) -> float:
        """
        Calculate reliability metric R(t).
        
        Uses exponential reliability model:
        R(t) = P(kMc > threshold AND kMt > threshold)
        """
        if not self._current_degradation:
            return 1.0
        
        # Map degradation to reliability
        # R = 1 when new, R → 0 as approaching threshold
        kMc = self._current_degradation.compressor_decay
        kMt = self._current_degradation.turbine_decay
        
        # Distance from failure threshold (normalized)
        compressor_margin = (kMc - 0.95) / 0.05  # [0, 1]
        turbine_margin = (kMt - 0.975) / 0.025   # [0, 1]
        
        # Combined reliability (worst case)
        reliability = min(max(0, compressor_margin), max(0, turbine_margin))
        
        return round(reliability, 4)
    
    def _generate_alert_message(self, alert: Dict[str, Any]) -> None:
        """Generate and queue alert message for other agents."""
        priority = MessagePriority.CRITICAL if alert['level'] == 'CRITICAL' else MessagePriority.HIGH
        
        # Send to Supervisor
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'reliability_degradation',
                'level': alert['level'],
                'component': alert.get('component', 'unknown'),
                'reason': alert['reason'],
                'rul_hours': self._estimate_rul()['system_rul_hours'],
                'current_health': self._current_degradation.overall_health if self._current_degradation else 0
            },
            priority=priority
        )
        self.communicate(msg)
        
        # Also notify Maintenance agent
        maintenance_msg = self.create_message(
            recipient='MaintainabilityAgent',
            content={
                'maintenance_trigger': 'reliability_degradation',
                'component': alert.get('component', 'unknown'),
                'severity': alert['level'],
                'recommended_action': 'IMMEDIATE_MAINTENANCE' if alert['level'] == 'CRITICAL' else 'SCHEDULE_MAINTENANCE'
            },
            priority=priority
        )
        self.communicate(maintenance_msg)
    
    def get_rams_contribution(self) -> float:
        """Get this agent's contribution to system reliability metric."""
        return self._calculate_reliability()
