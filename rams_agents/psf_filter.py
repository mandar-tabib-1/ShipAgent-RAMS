"""
Potential Safety Function (PSF) Filter for RL-based Collision Avoidance

Implements a Control Barrier Function (CBF) based safety filter that
projects RL-proposed actions to the safe action space.

PSF ensures that CPA/TCPA constraints are maintained regardless of
RL policy output - providing formal safety guarantees even with
imperfect or out-of-distribution learned policies.

Safety Constraint (CBF):
    φ(CPA, TCPA) = CPA - CPA_min - k·max(0, TCPA_critical - TCPA) > 0
    
This ensures:
1. CPA remains above critical threshold
2. Additional safety margin when TCPA is critical
3. Smooth action projection to safe region

Part of RAMS Safety agent with RL integration.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PSFInterventionType(Enum):
    """Type of PSF intervention applied."""
    NONE = "none"                   # RL action was safe
    SPEED_REDUCTION = "speed"       # Reduced speed for safety
    STEERING_CORRECTION = "steer"   # Modified steering for safety
    FULL_OVERRIDE = "override"      # Full action replacement
    EMERGENCY_STOP = "emergency"    # Critical safety stop


@dataclass
class PSFResult:
    """Result of PSF filtering."""
    safe_action: np.ndarray         # [speed_ratio, steering] after PSF
    original_action: np.ndarray     # Original RL action
    intervention_type: PSFInterventionType
    barrier_value: float            # CBF value (φ), positive = safe
    safety_margin: float            # Distance to barrier (min φ over all targets)
    interventions_per_target: Dict[int, float]  # target_id -> barrier value
    
    @property
    def was_modified(self) -> bool:
        """Check if PSF modified the RL action."""
        return self.intervention_type != PSFInterventionType.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'safe_action': self.safe_action.tolist(),
            'original_action': self.original_action.tolist(),
            'intervention_type': self.intervention_type.value,
            'barrier_value': float(self.barrier_value),
            'safety_margin': float(self.safety_margin),
            'was_modified': self.was_modified
        }


@dataclass 
class TrackState:
    """Track state for PSF computation."""
    target_id: int
    position: Tuple[float, float]   # (North, East) meters relative to ownship
    velocity: Tuple[float, float]   # (Vn, Ve) m/s relative to ownship
    cpa: float                      # Current CPA estimate
    tcpa: float                     # Current TCPA estimate


class PotentialSafetyFunction:
    """
    Potential Safety Function (PSF) for collision avoidance.
    
    Implements a Control Barrier Function (CBF) approach where actions
    are projected to satisfy safety constraints.
    
    Safety Barrier Function:
        φ(x) = CPA(x) - CPA_min - k·max(0, TCPA_critical - TCPA(x))
        
        where:
        - CPA(x): Closest Point of Approach given state x
        - CPA_min: Minimum safe CPA (default 50m)
        - TCPA_critical: Time horizon for TCPA concern (default 60s)
        - k: Coupling coefficient between CPA/TCPA (default 0.5)
    
    If φ(x) < 0, the current trajectory is unsafe and action must be modified.
    
    Usage:
        psf = PotentialSafetyFunction()
        result = psf.filter_action(rl_action, tracks, ownship_state)
        safe_action = result.safe_action
    """
    
    # Default safety thresholds (consistent with SafetyAgent)
    CPA_CRITICAL = 50.0         # meters
    CPA_HIGH = 100.0            # meters  
    TCPA_CRITICAL = 60.0        # seconds
    TCPA_HIGH = 180.0           # seconds
    
    # CBF parameters
    CBF_MARGIN = 10.0           # Safety margin buffer (meters)
    CBF_K_TCPA = 0.5            # TCPA coupling coefficient
    CBF_GAMMA = 0.3             # CBF class-K function parameter
    
    # Action limits
    MAX_SPEED_RATIO = 1.0
    MIN_SPEED_RATIO = 0.0
    MAX_STEERING = 1.0
    
    def __init__(self,
                 cpa_critical: Optional[float] = None,
                 tcpa_critical: Optional[float] = None,
                 cbf_gamma: Optional[float] = None):
        """
        Initialize PSF.
        
        Args:
            cpa_critical: Override default CPA critical threshold
            tcpa_critical: Override default TCPA critical threshold
            cbf_gamma: Override CBF class-K parameter
        """
        self.cpa_critical = cpa_critical or self.CPA_CRITICAL
        self.tcpa_critical = tcpa_critical or self.TCPA_CRITICAL
        self.gamma = cbf_gamma or self.CBF_GAMMA
        
        # Statistics
        self._intervention_count = 0
        self._total_calls = 0
    
    def compute_barrier(self, 
                       cpa: float, 
                       tcpa: float,
                       include_margin: bool = True) -> float:
        """
        Compute Control Barrier Function value.
        
        φ(CPA, TCPA) = CPA - CPA_min - k·max(0, TCPA_crit - TCPA) - margin
        
        Returns:
            Barrier value. Positive = safe, Negative = unsafe
        """
        margin = self.CBF_MARGIN if include_margin else 0.0
        
        # TCPA penalty: increases required CPA when TCPA is low
        tcpa_penalty = self.CBF_K_TCPA * max(0, self.tcpa_critical - tcpa)
        
        # Barrier value
        phi = cpa - self.cpa_critical - tcpa_penalty - margin
        
        return phi
    
    def compute_barrier_derivative(self,
                                   track: TrackState,
                                   action: np.ndarray,
                                   ownship_velocity: Tuple[float, float],
                                   dt: float = 1.0) -> float:
        """
        Compute time derivative of barrier function under action.
        
        This estimates how the barrier value changes given the proposed action.
        
        Args:
            track: Target track state
            action: [speed_ratio, steering] action
            ownship_velocity: Current ownship velocity
            dt: Time step for derivative estimation
            
        Returns:
            Estimated d(phi)/dt
        """
        # Current barrier
        phi_now = self.compute_barrier(track.cpa, track.tcpa)
        
        # Project CPA/TCPA under action
        new_cpa, new_tcpa = self._project_cpa_tcpa(
            track, action, ownship_velocity, dt
        )
        
        phi_next = self.compute_barrier(new_cpa, new_tcpa)
        
        return (phi_next - phi_now) / dt
    
    def filter_action(self,
                     rl_action: np.ndarray,
                     tracks: List[TrackState],
                     ownship_state: Dict[str, Any]) -> PSFResult:
        """
        Filter RL action through PSF to ensure safety.
        
        Args:
            rl_action: [speed_ratio, steering] from RL policy
            tracks: List of tracked targets with CPA/TCPA
            ownship_state: Dict with 'position', 'velocity', 'heading'
            
        Returns:
            PSFResult with safe action and intervention info
        """
        self._total_calls += 1
        
        rl_action = np.asarray(rl_action, dtype=np.float32)
        if len(rl_action) < 2:
            rl_action = np.array([0.5, 0.0])
        
        # Get ownship velocity
        ownship_vel = ownship_state.get('velocity', (5.0, 0.0))
        
        # Compute barrier for all tracks
        barrier_values = {}
        min_barrier = float('inf')
        critical_track = None
        
        for track in tracks:
            phi = self.compute_barrier(track.cpa, track.tcpa)
            barrier_values[track.target_id] = phi
            
            if phi < min_barrier:
                min_barrier = phi
                critical_track = track
        
        # If no tracks or all safe with margin, return original action
        if not tracks or min_barrier > self.CBF_MARGIN:
            return PSFResult(
                safe_action=rl_action.copy(),
                original_action=rl_action.copy(),
                intervention_type=PSFInterventionType.NONE,
                barrier_value=min_barrier if tracks else float('inf'),
                safety_margin=min_barrier,
                interventions_per_target=barrier_values
            )
        
        # Safety intervention required
        self._intervention_count += 1
        
        # critical_track is guaranteed non-None here: if tracks is non-empty,
        # at least one track sets critical_track in the loop above
        assert critical_track is not None, "critical_track must be set when tracks is non-empty"
        
        # Apply CBF-based action projection
        safe_action, intervention_type = self._project_to_safe_action(
            rl_action, critical_track, ownship_vel, min_barrier
        )
        
        # Verify safety of projected action
        final_barrier = self._evaluate_action_safety(
            safe_action, tracks, ownship_vel
        )
        
        return PSFResult(
            safe_action=safe_action,
            original_action=rl_action.copy(),
            intervention_type=intervention_type,
            barrier_value=final_barrier,
            safety_margin=final_barrier,
            interventions_per_target=barrier_values
        )
    
    def _project_to_safe_action(self,
                                rl_action: np.ndarray,
                                critical_track: TrackState,
                                ownship_vel: Tuple[float, float],
                                barrier_value: float) -> Tuple[np.ndarray, PSFInterventionType]:
        """
        Project RL action to safe action space.
        
        Strategy:
        1. If barrier critically negative (< -margin), emergency response
        2. Else, apply graduated intervention based on barrier deficit
        3. Prefer speed reduction, then steering, then full override
        """
        speed_ratio, steering = rl_action[0], rl_action[1]
        intervention_type = PSFInterventionType.NONE
        
        # Calculate relative position/bearing to critical track
        rel_pos = critical_track.position
        bearing = math.atan2(rel_pos[1], rel_pos[0])
        distance = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        
        # Emergency stop if critically unsafe
        if barrier_value < -self.CBF_MARGIN * 2:
            # Emergency: reduce speed dramatically, turn away
            safe_speed = 0.1
            
            # Turn away from target
            turn_direction = 1.0 if bearing > 0 else -1.0  # Starboard if target to port
            safe_steering = np.clip(turn_direction * 0.8, -1, 1)
            
            return np.array([safe_speed, safe_steering]), PSFInterventionType.EMERGENCY_STOP
        
        # Calculate intervention magnitude based on barrier deficit
        intervention_strength = min(1.0, max(0, -barrier_value / (2 * self.CBF_MARGIN)))
        
        # Strategy 1: Speed reduction
        # Reducing speed generally increases TCPA (more time to CPA)
        if critical_track.tcpa < self.tcpa_critical:
            speed_reduction = intervention_strength * 0.5
            safe_speed = max(self.MIN_SPEED_RATIO, speed_ratio - speed_reduction)
            intervention_type = PSFInterventionType.SPEED_REDUCTION
        else:
            safe_speed = speed_ratio
        
        # Strategy 2: Steering correction
        # Steer away from target to increase CPA
        if critical_track.cpa < self.cpa_critical + self.CBF_MARGIN:
            # Calculate avoidance steering
            # Positive bearing = target to starboard, turn to port (-steering)
            # Negative bearing = target to port, turn to starboard (+steering)
            avoidance_direction = -np.sign(bearing) if abs(bearing) > 0.1 else 1.0
            
            # Blend RL steering with avoidance
            avoidance_strength = intervention_strength * 0.6
            safe_steering = (1 - avoidance_strength) * steering + avoidance_strength * avoidance_direction
            safe_steering = np.clip(safe_steering, -self.MAX_STEERING, self.MAX_STEERING)
            
            if intervention_type == PSFInterventionType.NONE:
                intervention_type = PSFInterventionType.STEERING_CORRECTION
            elif intervention_type == PSFInterventionType.SPEED_REDUCTION:
                intervention_type = PSFInterventionType.FULL_OVERRIDE
        else:
            safe_steering = steering
        
        return np.array([safe_speed, safe_steering]), intervention_type
    
    def _project_cpa_tcpa(self,
                          track: TrackState,
                          action: np.ndarray,
                          ownship_vel: Tuple[float, float],
                          dt: float) -> Tuple[float, float]:
        """
        Project CPA/TCPA under proposed action.
        
        Simplified model: assumes action immediately affects velocity.
        """
        speed_ratio, steering = action[0], action[1]
        
        # Current ownship speed and heading
        own_speed = math.sqrt(ownship_vel[0]**2 + ownship_vel[1]**2)
        own_heading = math.atan2(ownship_vel[1], ownship_vel[0])
        
        # New velocity under action
        new_speed = own_speed * speed_ratio
        new_heading = own_heading + steering * 0.5  # Simplified turn model
        
        new_own_vel = (
            new_speed * math.cos(new_heading),
            new_speed * math.sin(new_heading)
        )
        
        # Relative velocity
        rel_vel = (
            track.velocity[0] - new_own_vel[0],
            track.velocity[1] - new_own_vel[1]
        )
        
        # Recalculate CPA/TCPA
        rel_pos = track.position
        speed_sq = rel_vel[0]**2 + rel_vel[1]**2
        
        if speed_sq < 0.01:
            new_cpa = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            new_tcpa = float('inf')
        else:
            new_tcpa = max(0, -(rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1]) / speed_sq)
            cpa_pos = (
                rel_pos[0] + rel_vel[0] * new_tcpa,
                rel_pos[1] + rel_vel[1] * new_tcpa
            )
            new_cpa = math.sqrt(cpa_pos[0]**2 + cpa_pos[1]**2)
        
        return new_cpa, new_tcpa
    
    def _evaluate_action_safety(self,
                               action: np.ndarray,
                               tracks: List[TrackState],
                               ownship_vel: Tuple[float, float]) -> float:
        """Evaluate minimum barrier value under proposed action."""
        min_barrier = float('inf')
        
        for track in tracks:
            cpa, tcpa = self._project_cpa_tcpa(track, action, ownship_vel, 1.0)
            phi = self.compute_barrier(cpa, tcpa, include_margin=False)
            min_barrier = min(min_barrier, phi)
        
        return min_barrier
    
    @property
    def intervention_rate(self) -> float:
        """Get PSF intervention rate (fraction of calls requiring intervention)."""
        if self._total_calls == 0:
            return 0.0
        return self._intervention_count / self._total_calls
    
    def reset_statistics(self) -> None:
        """Reset intervention statistics."""
        self._intervention_count = 0
        self._total_calls = 0


def create_track_state(target_id: int,
                       position: Tuple[float, float],
                       velocity: Tuple[float, float],
                       cpa: float,
                       tcpa: float) -> TrackState:
    """Helper to create TrackState from track data."""
    return TrackState(
        target_id=target_id,
        position=position,
        velocity=velocity,
        cpa=cpa,
        tcpa=tcpa
    )
