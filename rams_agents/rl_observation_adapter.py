"""
Observation Adapter for RL-based Collision Avoidance

Converts Kalman filter track predictions to RL policy observation format.

The RL policy (CNNPolicy) expects:
- scan: (frames, 512) - 1D range scan similar to LiDAR
- goal: (2,) - [cos(theta_goal), sin(theta_goal)]
- speed: (2,) - [vx, vy] normalized

This adapter synthesizes a "virtual LiDAR scan" from tracked target positions,
allowing the pre-trained RL policy to work with Kalman-filtered track data.

Part of RAMS Safety agent with RL integration.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RLObservation:
    """Observation for RL policy."""
    scan: np.ndarray            # (scan_bins,) or (frames, scan_bins)
    goal: np.ndarray            # (2,) [cos, sin] of goal direction
    speed: np.ndarray           # (2,) [vx, vy] normalized
    
    # Metadata
    track_count: int
    ownship_speed: float
    goal_distance: float


class RLObservationAdapter:
    """
    Adapts Kalman track data to RL observation format.
    
    Creates a synthetic "LiDAR-like" scan representation from
    tracked target positions. This allows the pre-trained RL policy
    to process discrete track data as if it were continuous sensor input.
    
    Scan Representation:
        - 512 angular bins covering 360 degrees
        - Each bin contains minimum range to any target in that direction
        - Targets are represented as Gaussian blobs in angular space
        - Maximum range (no detection) = 1.0 (normalized)
        
    Usage:
        adapter = RLObservationAdapter()
        obs = adapter.create_observation(
            tracks=kalman_tracks,
            ownship_state={'position': (0,0), 'velocity': (5,0), 'heading': 0},
            goal_position=(100, 50)
        )
        action = rl_policy.predict(obs.scan, obs.goal, obs.speed)
    """
    
    # Configuration
    SCAN_BINS = 512
    MAX_RANGE = 500.0       # meters - max detection range
    TARGET_ANGULAR_WIDTH = 5.0  # degrees - width of target in scan
    
    # Normalization constants
    MAX_SPEED = 10.0        # m/s for speed normalization
    
    def __init__(self, 
                 scan_bins: int = 512,
                 max_range: float = 500.0,
                 target_width_deg: float = 5.0):
        """
        Initialize observation adapter.
        
        Args:
            scan_bins: Number of angular bins in synthetic scan
            max_range: Maximum range for normalization (meters)
            target_width_deg: Angular width of targets in scan (degrees)
        """
        self.scan_bins = scan_bins
        self.max_range = max_range
        self.target_width_deg = target_width_deg
        self.target_width_bins = int(target_width_deg * scan_bins / 360)
        
        # Precompute bin angles
        self.bin_angles = np.linspace(-np.pi, np.pi, scan_bins, endpoint=False)
        self.bin_centers_deg = np.degrees(self.bin_angles)
    
    def create_observation(self,
                          tracks: List[Dict[str, Any]],
                          ownship_state: Dict[str, Any],
                          goal_position: Optional[Tuple[float, float]] = None) -> RLObservation:
        """
        Create RL observation from track data.
        
        Args:
            tracks: List of track dicts with 'position', 'velocity', 'cpa', 'tcpa'
            ownship_state: Dict with 'position', 'velocity', 'heading'
            goal_position: (N, E) goal waypoint. If None, use current heading.
            
        Returns:
            RLObservation ready for policy input
        """
        # Extract ownship state
        own_pos = ownship_state.get('position', (0.0, 0.0))
        own_vel = ownship_state.get('velocity', (5.0, 0.0))
        own_heading = ownship_state.get('heading', 0.0)  # radians
        
        # Create synthetic scan
        scan = self._create_scan(tracks, own_pos, own_heading)
        
        # Create goal vector
        goal, goal_distance = self._create_goal_vector(own_pos, own_heading, goal_position)
        
        # Create normalized speed vector
        speed = self._create_speed_vector(own_vel)
        
        ownship_speed = math.sqrt(own_vel[0]**2 + own_vel[1]**2)
        
        return RLObservation(
            scan=scan,
            goal=goal,
            speed=speed,
            track_count=len(tracks),
            ownship_speed=ownship_speed,
            goal_distance=goal_distance
        )
    
    def _create_scan(self,
                     tracks: List[Dict[str, Any]],
                     ownship_pos: Tuple[float, float],
                     ownship_heading: float) -> np.ndarray:
        """
        Create synthetic LiDAR-like scan from tracks.
        
        Each track is projected to angular space as a Gaussian blob.
        Scan values are normalized range (0=close, 1=max range/no detection).
        """
        # Initialize scan with max range (no obstacles)
        scan = np.ones(self.scan_bins) 
        
        for track in tracks:
            # Get relative position
            track_pos = track.get('position', (0, 0))
            if isinstance(track_pos, (list, tuple)) and len(track_pos) >= 2:
                rel_n = track_pos[0] - ownship_pos[0]
                rel_e = track_pos[1] - ownship_pos[1]
            else:
                continue
            
            # Calculate range and bearing
            distance = math.sqrt(rel_n**2 + rel_e**2)
            if distance < 1.0:  # Skip if too close (probably self)
                continue
            
            # Bearing in world frame
            bearing_world = math.atan2(rel_e, rel_n)
            # Bearing relative to ownship heading
            bearing_rel = bearing_world - ownship_heading
            # Normalize to [-pi, pi]
            bearing_rel = (bearing_rel + math.pi) % (2 * math.pi) - math.pi
            
            # Convert to scan bin
            bin_idx = int((bearing_rel + math.pi) / (2 * math.pi) * self.scan_bins)
            bin_idx = bin_idx % self.scan_bins
            
            # Normalized range
            norm_range = min(1.0, distance / self.max_range)
            
            # Create Gaussian blob around target
            for offset in range(-self.target_width_bins // 2, 
                               self.target_width_bins // 2 + 1):
                idx = (bin_idx + offset) % self.scan_bins
                # Gaussian weight
                weight = np.exp(-0.5 * (offset / (self.target_width_bins / 3))**2)
                # Update scan (minimum of current and new)
                weighted_range = norm_range + (1 - norm_range) * (1 - weight)
                scan[idx] = min(scan[idx], weighted_range)
        
        return scan.astype(np.float32)
    
    def _create_goal_vector(self,
                           ownship_pos: Tuple[float, float],
                           ownship_heading: float,
                           goal_position: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Create goal direction vector [cos(theta), sin(theta)].
        
        Returns:
            (goal_vector, goal_distance)
        """
        if goal_position is None:
            # Default: maintain current heading
            return np.array([np.cos(ownship_heading), np.sin(ownship_heading)], dtype=np.float32), 0.0
        
        # Direction to goal
        rel_n = goal_position[0] - ownship_pos[0]
        rel_e = goal_position[1] - ownship_pos[1]
        
        distance = math.sqrt(rel_n**2 + rel_e**2)
        if distance < 1.0:
            # At goal
            return np.array([np.cos(ownship_heading), np.sin(ownship_heading)], dtype=np.float32), 0.0
        
        # Goal bearing relative to ownship heading
        goal_bearing_world = math.atan2(rel_e, rel_n)
        goal_bearing_rel = goal_bearing_world - ownship_heading
        
        # [cos, sin] of relative goal direction
        goal_vector = np.array([
            np.cos(goal_bearing_rel),
            np.sin(goal_bearing_rel)
        ], dtype=np.float32)
        
        return goal_vector, distance
    
    def _create_speed_vector(self, 
                            ownship_vel: Tuple[float, float]) -> np.ndarray:
        """
        Create normalized speed vector [vx, vy].
        
        Normalized by MAX_SPEED to range approximately [-1, 1].
        """
        vn = ownship_vel[0] / self.MAX_SPEED
        ve = ownship_vel[1] / self.MAX_SPEED
        
        return np.array([vn, ve], dtype=np.float32)
    
    def tracks_from_safety_agent(self, 
                                 safety_tracks: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert SafetyAgent Track objects to dict format.
        
        Args:
            safety_tracks: List of Track dataclass instances
            
        Returns:
            List of track dicts suitable for create_observation()
        """
        tracks = []
        for t in safety_tracks:
            track_dict = {
                'target_id': getattr(t, 'target_id', 0),
                'position': getattr(t, 'position', (0, 0)),
                'velocity': getattr(t, 'velocity', (0, 0)),
                'cpa': getattr(t, 'cpa', 1000),
                'tcpa': getattr(t, 'tcpa', 1000)
            }
            tracks.append(track_dict)
        return tracks


class MultiFrameObservationAdapter:
    """
    Adapter that maintains frame history for temporal RL policies.
    
    Some RL policies use multiple frames of observations to capture
    temporal dynamics. This adapter maintains a buffer and stacks frames.
    """
    
    def __init__(self, 
                 num_frames: int = 3,
                 **kwargs):
        """
        Initialize multi-frame adapter.
        
        Args:
            num_frames: Number of frames to stack
            **kwargs: Passed to RLObservationAdapter
        """
        self.num_frames = num_frames
        self.base_adapter = RLObservationAdapter(**kwargs)
        self._frame_buffer: List[np.ndarray] = []
    
    def create_observation(self,
                          tracks: List[Dict[str, Any]],
                          ownship_state: Dict[str, Any],
                          goal_position: Optional[Tuple[float, float]] = None) -> RLObservation:
        """
        Create multi-frame observation.
        
        Returns observation with stacked scans of shape (num_frames, scan_bins).
        """
        base_obs = self.base_adapter.create_observation(tracks, ownship_state, goal_position)
        
        # Add to frame buffer
        self._frame_buffer.append(base_obs.scan)
        if len(self._frame_buffer) > self.num_frames:
            self._frame_buffer.pop(0)
        
        # Pad if not enough frames
        while len(self._frame_buffer) < self.num_frames:
            self._frame_buffer.insert(0, base_obs.scan.copy())
        
        # Stack frames
        stacked_scan = np.stack(self._frame_buffer, axis=0)
        
        return RLObservation(
            scan=stacked_scan,
            goal=base_obs.goal,
            speed=base_obs.speed,
            track_count=base_obs.track_count,
            ownship_speed=base_obs.ownship_speed,
            goal_distance=base_obs.goal_distance
        )
    
    def reset(self) -> None:
        """Reset frame buffer (e.g., at start of new scenario)."""
        self._frame_buffer.clear()
    
    def tracks_from_safety_agent(self, safety_tracks: List[Any]) -> List[Dict[str, Any]]:
        """Delegate to base adapter."""
        return self.base_adapter.tracks_from_safety_agent(safety_tracks)
