"""
Safety Agent - Monitors navigation safety and COLREGS compliance

Part of the RAMS multi-agent system demonstrating agentic AI.
Uses AutoFerry Sensor Fusion dataset for collision avoidance demonstration.

Key Responsibilities:
- Monitor vessel traffic and collision risks
- Calculate CPA/TCPA for detected targets
- Apply COLREGS rules (13-17)
- Generate safety alerts and avoidance recommendations

RL+PSF Integration:
- Uses pre-trained RL policy for collision avoidance action proposals
- Applies Potential Safety Function (PSF) filter for formal safety guarantees
- Falls back to rule-based COLREGS if RL unavailable

Attribution:
- RL policy based on Acmece/rl-collision-avoidance (arXiv:1709.10082)
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import warnings

from .base_agent import (
    RAMSBaseAgent, RAMSCategory, AgentBelief, AgentMessage,
    MessagePriority
)

# RL+PSF imports (with fallback)
try:
    from .ml_models.rl_collision_policy import RLCollisionPolicy, RLAction
    from .psf_filter import PotentialSafetyFunction, TrackState, PSFResult
    from .rl_observation_adapter import RLObservationAdapter, MultiFrameObservationAdapter
    RL_PSF_AVAILABLE = True
except ImportError as e:
    RL_PSF_AVAILABLE = False
    warnings.warn(f"RL+PSF components not available: {e}. Using rule-based fallback.")


class COLREGSRule(Enum):
    """COLREGS rules for collision avoidance."""
    RULE_13_OVERTAKING = 13    # Overtaking vessel keeps clear
    RULE_14_HEAD_ON = 14       # Both vessels alter course to starboard
    RULE_15_CROSSING = 15      # Give-way vessel keeps clear of stand-on
    RULE_16_GIVE_WAY = 16      # Give-way vessel action
    RULE_17_STAND_ON = 17      # Stand-on vessel action


class RiskLevel(Enum):
    """Collision risk levels."""
    CRITICAL = 1   # CPA < 50m, TCPA < 60s
    HIGH = 2       # CPA < 100m, TCPA < 180s
    MEDIUM = 3     # CPA < 200m, TCPA < 300s
    LOW = 4        # CPA < 500m
    SAFE = 5       # CPA >= 500m


@dataclass
class Track:
    """A tracked target vessel."""
    target_id: int
    position: Tuple[float, float]   # (North, East) meters
    velocity: Tuple[float, float]   # (Vn, Ve) m/s
    cpa: float                      # Closest Point of Approach (meters)
    tcpa: float                     # Time to CPA (seconds)
    risk_level: RiskLevel
    colregs_rule: Optional[COLREGSRule] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_id': self.target_id,
            'position': self.position,
            'velocity': self.velocity,
            'cpa': round(self.cpa, 1),
            'tcpa': round(self.tcpa, 1),
            'risk_level': self.risk_level.name,
            'colregs_rule': self.colregs_rule.name if self.colregs_rule else None
        }


@dataclass
class SafetyState:
    """Current navigation safety state."""
    tracks: List[Track]
    overall_risk: RiskLevel
    active_rules: List[COLREGSRule]
    safety_index: float  # 0-100, higher = safer
    timestamp: float = 0.0


class SafetyAgent(RAMSBaseAgent):
    """
    Agent responsible for navigation safety.
    
    RAMS Category: S (Safety)
    
    Agentic Behaviors:
    - Perceives: Sensor detections from navigation systems
    - Reasons: Calculates CPA/TCPA, applies COLREGS rules
    - Acts: Generates avoidance recommendations
    - Communicates: Alerts Supervisor on collision risks
    
    Data Source: AutoFerry Sensor Fusion Dataset
    """
    
    # Risk thresholds (meters and seconds)
    CPA_CRITICAL = 50
    CPA_HIGH = 100
    CPA_MEDIUM = 200
    CPA_LOW = 500
    
    TCPA_CRITICAL = 60
    TCPA_HIGH = 180
    TCPA_MEDIUM = 300
    
    # Own vessel default parameters
    DEFAULT_SPEED = 5.0  # m/s (≈10 knots)
    DEFAULT_HEADING = 0  # degrees (North)
    
    def __init__(self, use_rl_policy: bool = True):
        super().__init__(name="SafetyAgent", category=RAMSCategory.SAFETY)
        
        # Track management
        self._tracks: Dict[int, Track] = {}
        self._track_history: Dict[int, List[Tuple[float, float, float]]] = {}  # id -> [(t, N, E)]
        
        # Own vessel state
        self._ownship_position = (0.0, 0.0)
        self._ownship_velocity = (self.DEFAULT_SPEED, 0.0)  # Moving north
        self._ownship_heading = self.DEFAULT_HEADING
        
        # Safety state
        self._current_state: Optional[SafetyState] = None
        
        # RL+PSF components
        self._use_rl_policy = use_rl_policy and RL_PSF_AVAILABLE
        self._rl_policy: Optional['RLCollisionPolicy'] = None
        self._psf_filter: Optional['PotentialSafetyFunction'] = None
        self._obs_adapter: Optional['MultiFrameObservationAdapter'] = None
        self._rl_stats: Dict[str, Any] = {'interventions': 0, 'total_calls': 0, 'rl_used': 0}
        
        if self._use_rl_policy:
            self._init_rl_psf()
        
        # Agent goals
        self.goals = [
            "Monitor vessel traffic",
            "Assess collision risks",
            "Apply COLREGS rules",
            "Ensure safe navigation"
        ]
        
        if self._use_rl_policy:
            self.goals.append("Use RL+PSF for optimal avoidance")
    
    def _init_rl_psf(self) -> None:
        """Initialize RL+PSF components."""
        try:
            # RL Policy
            self._rl_policy = RLCollisionPolicy(frames=3, scan_bins=512)
            loaded = self._rl_policy.load()
            
            # PSF Filter  
            self._psf_filter = PotentialSafetyFunction(
                cpa_critical=self.CPA_CRITICAL,
                tcpa_critical=self.TCPA_CRITICAL
            )
            
            # Observation Adapter
            self._obs_adapter = MultiFrameObservationAdapter(
                num_frames=3,
                max_range=500.0
            )
            
            if loaded:
                print(f"[{self.name}] RL+PSF initialized successfully")
            else:
                print(f"[{self.name}] RL+PSF initialized (demo mode - no trained weights)")
                
        except Exception as e:
            warnings.warn(f"Failed to initialize RL+PSF: {e}")
            self._use_rl_policy = False
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive navigation environment.
        
        Expected environment_data keys:
        - 'navigation': Navigation data with detections
        - 'ownship_position': Own vessel position (N, E)
        - 'ownship_velocity': Own vessel velocity (Vn, Ve)
        - 'targets': List of target detections
        - 'timestamp': Current time
        """
        timestamp = environment_data.get('timestamp', 0.0)
        
        # Update own ship state (use normalize helper for robustness)
        raw_pos = environment_data.get('ownship_position', self._ownship_position)
        self._ownship_position = self._normalize_position(raw_pos)
        
        raw_vel = environment_data.get('ownship_velocity', self._ownship_velocity)
        self._ownship_velocity = self._normalize_position(raw_vel)  # Same format
        
        # Calculate heading from velocity
        if self._ownship_velocity[0] != 0 or self._ownship_velocity[1] != 0:
            self._ownship_heading = math.degrees(
                math.atan2(self._ownship_velocity[1], self._ownship_velocity[0])
            )
        
        # Process target detections
        targets = environment_data.get('targets', [])
        nav_data = environment_data.get('navigation', {})
        
        # Combine targets from different sources
        if 'detections' in nav_data:
            for det in nav_data['detections']:
                if hasattr(det, 'get_position') and det.get_position():
                    pos = det.get_position()
                    targets.append({
                        'id': det.sensor_id * 100,  # Synthetic ID
                        'position': pos,
                        'timestamp': det.timestamp
                    })
        
        # Update tracks
        self._update_tracks(targets, timestamp)
        
        # Calculate safety state
        tracks = list(self._tracks.values())
        overall_risk = self._calculate_overall_risk(tracks)
        active_rules = list(set(t.colregs_rule for t in tracks if t.colregs_rule))
        safety_index = self._calculate_safety_index(tracks, overall_risk)
        
        self._current_state = SafetyState(
            tracks=tracks,
            overall_risk=overall_risk,
            active_rules=active_rules,
            safety_index=safety_index,
            timestamp=timestamp
        )
        
        # Form perception belief
        self.update_belief(AgentBelief(
            belief_type="navigation_situation",
            value={
                'track_count': len(tracks),
                'overall_risk': overall_risk.name,
                'safety_index': safety_index,
                'active_rules': [r.name for r in active_rules]
            },
            confidence=0.9,
            source="sensor_fusion",
            evidence=["AutoFerry Sensor Fusion Dataset"]
        ))
    
    def reason(self) -> List[AgentBelief]:
        """
        Reason about collision risks and required actions.
        """
        new_beliefs = []
        
        if not self._current_state:
            return new_beliefs
        
        # 1. Analyze each track for collision risk
        high_risk_tracks = [t for t in self._current_state.tracks 
                           if t.risk_level.value <= RiskLevel.HIGH.value]
        
        if high_risk_tracks:
            new_beliefs.append(AgentBelief(
                belief_type="collision_risks",
                value={
                    'high_risk_count': len(high_risk_tracks),
                    'tracks': [t.to_dict() for t in high_risk_tracks]
                },
                confidence=0.95,
                source="risk_assessment",
                evidence=[f"CPA/TCPA analysis for {len(high_risk_tracks)} targets"]
            ))
            
            # Generate alerts for critical/high risks
            for track in high_risk_tracks:
                if track.risk_level.value <= RiskLevel.HIGH.value:
                    self._generate_collision_alert(track)
        
        # 2. Determine COLREGS actions
        required_actions = self._determine_colregs_actions()
        if required_actions:
            new_beliefs.append(AgentBelief(
                belief_type="colregs_actions",
                value=required_actions,
                confidence=0.9,
                source="colregs_analysis",
                evidence=["IMO COLREGS Rules 13-17"]
            ))
        
        # 3. Update safety metric
        new_beliefs.append(AgentBelief(
            belief_type="safety_metric",
            value={
                'safety_index': self._current_state.safety_index,
                'risk_level': self._current_state.overall_risk.name
            },
            confidence=0.9,
            source="safety_calculation",
            evidence=[f"Based on {len(self._current_state.tracks)} tracked targets"]
        ))
        
        return new_beliefs
    
    def act(self) -> Dict[str, Any]:
        """
        Produce safety assessment and avoidance recommendations.
        """
        if not self._current_state:
            return {'status': 'NO_DATA', 'safety_index': 100}
        
        recommendations = []
        avoidance_maneuvers = []
        
        # Generate recommendations based on risk
        if self._current_state.overall_risk == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL: Immediate collision avoidance required!")
            avoidance_maneuvers.extend(self._generate_avoidance_maneuvers())
        elif self._current_state.overall_risk == RiskLevel.HIGH:
            recommendations.append("WARNING: High collision risk - prepare for avoidance")
            avoidance_maneuvers.extend(self._generate_avoidance_maneuvers())
        elif self._current_state.overall_risk == RiskLevel.MEDIUM:
            recommendations.append("CAUTION: Monitor approaching vessels closely")
        else:
            recommendations.append("SAFE: No immediate collision risks")
        
        # Add COLREGS-specific guidance
        for track in self._current_state.tracks:
            if track.colregs_rule and track.risk_level.value <= RiskLevel.MEDIUM.value:
                guidance = self._get_colregs_guidance(track)
                if guidance:
                    recommendations.append(guidance)
        
        return {
            'safety_index': self._current_state.safety_index,
            'overall_risk': self._current_state.overall_risk.name,
            'track_count': len(self._current_state.tracks),
            'tracks': [t.to_dict() for t in self._current_state.tracks[:5]],  # Top 5 risks
            'active_colregs_rules': [r.name for r in self._current_state.active_rules],
            'avoidance_maneuvers': avoidance_maneuvers,
            'recommendations': recommendations
        }
    
    def _normalize_position(self, pos: Any) -> Tuple[float, float]:
        """Extract (North, East) tuple from various position formats."""
        if pos is None:
            return (0.0, 0.0)
        
        # Handle tuple or list of two floats
        if isinstance(pos, (tuple, list)):
            if len(pos) >= 2:
                # Check if elements are scalars
                n, e = pos[0], pos[1]
                # Handle nested lists like [[N, E]]
                if isinstance(n, (list, tuple)):
                    n = n[0] if len(n) > 0 else 0.0
                if isinstance(e, (list, tuple)):
                    e = e[0] if len(e) > 0 else 0.0
                return (float(n), float(e))
            elif len(pos) == 1 and isinstance(pos[0], (tuple, list)) and len(pos[0]) >= 2:
                # Nested: [(N, E)]
                return (float(pos[0][0]), float(pos[0][1]))
        
        return (0.0, 0.0)
    
    def _update_tracks(self, targets: List[Dict[str, Any]], timestamp: float) -> None:
        """Update track positions and calculate CPA/TCPA."""
        processed_ids = set()
        
        for target in targets:
            target_id = target.get('id', 0)
            position = self._normalize_position(target.get('position', (0, 0)))
            
            processed_ids.add(target_id)
            
            # Update track history
            if target_id not in self._track_history:
                self._track_history[target_id] = []
            
            self._track_history[target_id].append((timestamp, position[0], position[1]))
            
            # Keep history limited
            if len(self._track_history[target_id]) > 50:
                self._track_history[target_id] = self._track_history[target_id][-30:]
            
            # Estimate velocity from history
            velocity = self._estimate_velocity(target_id)
            
            # Calculate CPA/TCPA
            cpa, tcpa = self._calculate_cpa_tcpa(position, velocity)
            
            # Determine risk level
            risk_level = self._assess_risk(cpa, tcpa)
            
            # Determine COLREGS rule if applicable
            colregs_rule = self._determine_colregs_rule(position, velocity) if risk_level.value <= RiskLevel.MEDIUM.value else None
            
            # Create/update track
            self._tracks[target_id] = Track(
                target_id=target_id,
                position=position,
                velocity=velocity,
                cpa=cpa,
                tcpa=tcpa,
                risk_level=risk_level,
                colregs_rule=colregs_rule,
                timestamp=timestamp
            )
        
        # Remove stale tracks
        stale_threshold = 10.0  # seconds
        stale_ids = [tid for tid, track in self._tracks.items()
                    if timestamp - track.timestamp > stale_threshold and tid not in processed_ids]
        for tid in stale_ids:
            del self._tracks[tid]
    
    def _estimate_velocity(self, target_id: int) -> Tuple[float, float]:
        """Estimate target velocity from track history."""
        history = self._track_history.get(target_id, [])
        
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Use last few points for velocity estimation
        recent = history[-5:]
        if len(recent) >= 2:
            dt = recent[-1][0] - recent[0][0]
            if dt > 0:
                dn = recent[-1][1] - recent[0][1]
                de = recent[-1][2] - recent[0][2]
                return (dn / dt, de / dt)
        
        return (0.0, 0.0)
    
    def _calculate_cpa_tcpa(self, 
                           target_pos: Tuple[float, float],
                           target_vel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate Closest Point of Approach and Time to CPA.
        """
        # Relative position (target relative to own ship)
        rel_pos = (
            target_pos[0] - self._ownship_position[0],
            target_pos[1] - self._ownship_position[1]
        )
        
        # Relative velocity
        rel_vel = (
            target_vel[0] - self._ownship_velocity[0],
            target_vel[1] - self._ownship_velocity[1]
        )
        
        # Speed squared
        speed_sq = rel_vel[0]**2 + rel_vel[1]**2
        
        if speed_sq < 0.01:  # Nearly parallel or stationary
            cpa = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            return (cpa, float('inf'))
        
        # Time to CPA
        tcpa = -(rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1]) / speed_sq
        
        if tcpa < 0:
            tcpa = 0  # CPA is now or in the past
        
        # Position at CPA
        cpa_n = rel_pos[0] + rel_vel[0] * tcpa
        cpa_e = rel_pos[1] + rel_vel[1] * tcpa
        cpa = math.sqrt(cpa_n**2 + cpa_e**2)
        
        return (cpa, tcpa)
    
    def _assess_risk(self, cpa: float, tcpa: float) -> RiskLevel:
        """Assess collision risk based on CPA and TCPA."""
        if cpa < self.CPA_CRITICAL and tcpa < self.TCPA_CRITICAL:
            return RiskLevel.CRITICAL
        elif cpa < self.CPA_HIGH and tcpa < self.TCPA_HIGH:
            return RiskLevel.HIGH
        elif cpa < self.CPA_MEDIUM and tcpa < self.TCPA_MEDIUM:
            return RiskLevel.MEDIUM
        elif cpa < self.CPA_LOW:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE
    
    def _determine_colregs_rule(self, 
                                target_pos: Tuple[float, float],
                                target_vel: Tuple[float, float]) -> Optional[COLREGSRule]:
        """
        Determine applicable COLREGS rule based on encounter geometry.
        """
        # Calculate relative bearing to target
        rel_pos = (
            target_pos[0] - self._ownship_position[0],
            target_pos[1] - self._ownship_position[1]
        )
        bearing_to_target = math.degrees(math.atan2(rel_pos[1], rel_pos[0]))
        
        # Calculate target heading
        if abs(target_vel[0]) > 0.1 or abs(target_vel[1]) > 0.1:
            target_heading = math.degrees(math.atan2(target_vel[1], target_vel[0]))
        else:
            return None  # Stationary target
        
        # Relative heading
        relative_heading = (target_heading - self._ownship_heading + 360) % 360
        
        # Bearing from target to own ship
        bearing_from_target = (bearing_to_target + 180) % 360 - target_heading
        
        # Determine encounter type
        # Head-on: vessels approaching each other nearly head-to-head
        if abs(relative_heading - 180) < 15:
            return COLREGSRule.RULE_14_HEAD_ON
        
        # Overtaking: coming from astern (bearing > 112.5° from ahead)
        if 112.5 < bearing_from_target < 247.5:
            return COLREGSRule.RULE_13_OVERTAKING
        
        # Crossing: target on starboard = give way (Rule 15)
        if 0 < bearing_to_target < 112.5:
            return COLREGSRule.RULE_15_CROSSING
        
        # We are stand-on vessel
        return COLREGSRule.RULE_17_STAND_ON
    
    def _calculate_overall_risk(self, tracks: List[Track]) -> RiskLevel:
        """Calculate overall navigation risk from all tracks."""
        if not tracks:
            return RiskLevel.SAFE
        
        # Return worst-case risk
        worst = min(t.risk_level.value for t in tracks)
        return RiskLevel(worst)
    
    def _calculate_safety_index(self, tracks: List[Track], overall_risk: RiskLevel) -> float:
        """
        Calculate safety index (0-100, higher = safer).
        """
        if not tracks:
            return 100.0
        
        # Base safety from risk level
        risk_to_safety = {
            RiskLevel.CRITICAL: 10,
            RiskLevel.HIGH: 30,
            RiskLevel.MEDIUM: 60,
            RiskLevel.LOW: 80,
            RiskLevel.SAFE: 100
        }
        
        base_safety = risk_to_safety[overall_risk]
        
        # Adjust for number of risky targets
        high_risk_count = len([t for t in tracks if t.risk_level.value <= RiskLevel.HIGH.value])
        safety_penalty = min(30, high_risk_count * 10)
        
        return max(0, base_safety - safety_penalty)
    
    def _generate_avoidance_maneuvers(self) -> List[Dict[str, Any]]:
        """
        Generate avoidance maneuver recommendations.
        
        Uses RL+PSF if available, otherwise falls back to rule-based COLREGS.
        """
        maneuvers = []
        
        # Check if RL+PSF should be used
        if self._use_rl_policy and self._rl_policy and self._rl_policy.is_ready:
            rl_maneuver = self._generate_rl_psf_maneuver()
            if rl_maneuver:
                maneuvers.append(rl_maneuver)
                return maneuvers  # RL provides unified avoidance for all threats
        
        # Fallback to rule-based COLREGS
        return self._generate_rule_based_maneuvers()
    
    def _generate_rl_psf_maneuver(self) -> Optional[Dict[str, Any]]:
        """
        Generate avoidance maneuver using RL+PSF.
        
        Returns:
            Single unified avoidance maneuver or None if RL unavailable
        """
        if not self._current_state or not self._current_state.tracks:
            return None
        
        self._rl_stats['total_calls'] += 1
        
        try:
            # Create observation for RL policy
            track_dicts = self._obs_adapter.tracks_from_safety_agent(
                self._current_state.tracks
            )
            
            ownship_state = {
                'position': self._ownship_position,
                'velocity': self._ownship_velocity,
                'heading': math.radians(self._ownship_heading)
            }
            
            # Use original heading as goal (maintain course when safe)
            goal_distance = 500.0  # meters ahead
            goal_position = (
                self._ownship_position[0] + goal_distance * math.cos(math.radians(self._ownship_heading)),
                self._ownship_position[1] + goal_distance * math.sin(math.radians(self._ownship_heading))
            )
            
            obs = self._obs_adapter.create_observation(
                tracks=track_dicts,
                ownship_state=ownship_state,
                goal_position=goal_position
            )
            
            # Get RL action (deterministic for safety)
            rl_action = self._rl_policy.predict(
                scan=obs.scan,
                goal=obs.goal,
                speed=obs.speed,
                deterministic=True
            )
            
            # Apply PSF filter
            psf_tracks = [
                TrackState(
                    target_id=t.target_id,
                    position=t.position,
                    velocity=t.velocity,
                    cpa=t.cpa,
                    tcpa=t.tcpa
                )
                for t in self._current_state.tracks
            ]
            
            psf_result = self._psf_filter.filter_action(
                rl_action=rl_action.raw_action,
                tracks=psf_tracks,
                ownship_state=ownship_state
            )
            
            # Track statistics
            self._rl_stats['rl_used'] += 1
            if psf_result.was_modified:
                self._rl_stats['interventions'] += 1
            
            # Convert to maneuver format
            return self._rl_action_to_maneuver(
                safe_action=psf_result.safe_action,
                original_action=rl_action.raw_action,
                psf_result=psf_result,
                rl_confidence=rl_action.confidence
            )
            
        except Exception as e:
            warnings.warn(f"RL+PSF failed: {e}. Falling back to rules.")
            return None
    
    def _rl_action_to_maneuver(self,
                               safe_action: np.ndarray,
                               original_action: np.ndarray,
                               psf_result: 'PSFResult',
                               rl_confidence: float) -> Dict[str, Any]:
        """
        Convert RL action to COLREGS-compatible maneuver format.
        
        RL action: [speed_ratio (0-1), steering (-1 to 1)]
        Maneuver: {'action': str, 'direction': str, 'magnitude': float, ...}
        """
        speed_ratio = float(safe_action[0])
        steering = float(safe_action[1])
        
        # Determine primary action
        speed_change = abs(1.0 - speed_ratio)
        steering_magnitude = abs(steering)
        
        # Interpret RL action as COLREGS-compatible maneuver
        if speed_change > 0.3 and steering_magnitude < 0.2:
            # Primarily speed change
            action = 'REDUCE_SPEED' if speed_ratio < 0.8 else 'MAINTAIN_SPEED'
            direction = None
            magnitude = round((1.0 - speed_ratio) * 100, 1)  # % reduction
            
        elif steering_magnitude > 0.15:
            # Steering action
            action = 'ALTER_COURSE'
            direction = 'STARBOARD' if steering > 0 else 'PORT'
            # Convert steering [-1,1] to degrees [0,90]
            magnitude = round(abs(steering) * 60, 1)  # degrees
            
        else:
            # Minor adjustments or maintain
            action = 'MAINTAIN_COURSE'
            direction = None
            magnitude = 0
        
        # Build maneuver dict
        maneuver = {
            'action': action,
            'source': 'RL_PSF',
            'reason': 'RL policy + PSF safety filter',
            'rl_confidence': round(rl_confidence, 2),
            'psf_intervention': psf_result.intervention_type.value if psf_result else 'none',
            'barrier_value': round(psf_result.barrier_value, 1) if psf_result else None,
            'speed_ratio': round(speed_ratio, 2),
            'steering': round(steering, 2)
        }
        
        if direction:
            maneuver['direction'] = direction
        if magnitude > 0:
            maneuver['magnitude'] = magnitude
        
        return maneuver
    
    def _generate_rule_based_maneuvers(self) -> List[Dict[str, Any]]:
        """Generate rule-based COLREGS maneuvers (fallback)."""
        maneuvers = []
        
        for track in self._current_state.tracks:
            if track.risk_level.value > RiskLevel.HIGH.value:
                continue
            
            # Determine avoidance direction based on COLREGS
            if track.colregs_rule == COLREGSRule.RULE_14_HEAD_ON:
                maneuvers.append({
                    'target_id': track.target_id,
                    'action': 'ALTER_COURSE',
                    'direction': 'STARBOARD',
                    'magnitude': 30,  # degrees
                    'reason': 'COLREGS Rule 14: Head-on situation',
                    'source': 'COLREGS_RULES'
                })
            elif track.colregs_rule == COLREGSRule.RULE_15_CROSSING:
                maneuvers.append({
                    'target_id': track.target_id,
                    'action': 'ALTER_COURSE',
                    'direction': 'STARBOARD',
                    'magnitude': 45,
                    'reason': 'COLREGS Rule 15: Give-way in crossing',
                    'source': 'COLREGS_RULES'
                })
            elif track.colregs_rule == COLREGSRule.RULE_13_OVERTAKING:
                maneuvers.append({
                    'target_id': track.target_id,
                    'action': 'MAINTAIN_COURSE',
                    'reason': 'COLREGS Rule 13: Being overtaken - maintain course',
                    'source': 'COLREGS_RULES'
                })
            else:
                # Generic avoidance
                maneuvers.append({
                    'target_id': track.target_id,
                    'action': 'REDUCE_SPEED',
                    'reason': 'Increase TCPA',
                    'source': 'COLREGS_RULES'
                })
        
        return maneuvers
    
    def _get_colregs_guidance(self, track: Track) -> Optional[str]:
        """Get COLREGS guidance for a specific track."""
        rule = track.colregs_rule
        if rule == COLREGSRule.RULE_14_HEAD_ON:
            return f"Target {track.target_id}: HEAD-ON - both vessels alter starboard"
        elif rule == COLREGSRule.RULE_15_CROSSING:
            return f"Target {track.target_id}: CROSSING - give-way, alter to starboard"
        elif rule == COLREGSRule.RULE_13_OVERTAKING:
            return f"Target {track.target_id}: OVERTAKING - maintain course/speed"
        elif rule == COLREGSRule.RULE_17_STAND_ON:
            return f"Target {track.target_id}: STAND-ON - maintain course/speed"
        return None
    
    def _determine_colregs_actions(self) -> List[Dict[str, Any]]:
        """Determine all required COLREGS actions."""
        actions = []
        
        for track in self._current_state.tracks:
            if track.colregs_rule and track.risk_level.value <= RiskLevel.MEDIUM.value:
                action = {
                    'target_id': track.target_id,
                    'rule': track.colregs_rule.name,
                    'cpa': track.cpa,
                    'tcpa': track.tcpa,
                    'risk': track.risk_level.name
                }
                actions.append(action)
        
        return actions
    
    def _generate_collision_alert(self, track: Track) -> None:
        """Generate collision alert for supervisor."""
        priority = MessagePriority.CRITICAL if track.risk_level == RiskLevel.CRITICAL else MessagePriority.HIGH
        
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'collision_risk',
                'target_id': track.target_id,
                'cpa': track.cpa,
                'tcpa': track.tcpa,
                'risk_level': track.risk_level.name,
                'colregs_rule': track.colregs_rule.name if track.colregs_rule else None,
                'recommended_action': self._get_colregs_guidance(track)
            },
            priority=priority
        )
        self.communicate(msg)
    
    def get_rams_contribution(self) -> float:
        """Get this agent's contribution to system safety index."""
        if self._current_state:
            return self._current_state.safety_index / 100.0
        return 1.0
    
    def get_rl_psf_stats(self) -> Dict[str, Any]:
        """
        Get RL+PSF usage statistics.
        
        Returns:
            Dict with total_calls, rl_used, interventions, intervention_rate
        """
        stats = self._rl_stats.copy()
        
        if stats['rl_used'] > 0:
            stats['intervention_rate'] = stats['interventions'] / stats['rl_used']
        else:
            stats['intervention_rate'] = 0.0
        
        stats['rl_available'] = self._use_rl_policy and bool(self._rl_policy)
        stats['psf_available'] = bool(self._psf_filter)
        
        if self._psf_filter:
            stats['psf_intervention_rate'] = self._psf_filter.intervention_rate
        
        return stats
    
    @property
    def uses_rl_policy(self) -> bool:
        """Check if RL+PSF is enabled and available."""
        return self._use_rl_policy and self._rl_policy is not None
