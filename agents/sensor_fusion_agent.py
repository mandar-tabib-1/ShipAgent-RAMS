"""
Sensor Fusion Agent with Kalman Filtering for R/V Gunnerus
==========================================================
Processes multi-sensor tracking data in Autoferry benchmark format.
Implements Kalman filtering for target tracking and track management.

Sensors supported:
- LiDAR (ID: 1)
- Radar (ID: 2)  
- IR Camera (ID: 3)
- EO Camera (ID: 4)

Coordinate frame: NED (North-East-Down) relative to Piren origin
"""

import numpy as np
import math
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_agent import BaseAgent, AgentStatus, AlertLevel


@dataclass
class KalmanTrack:
    """Represents a tracked target with Kalman filter state"""
    track_id: int
    target_id: int
    name: str
    
    # State vector [north, east, vel_north, vel_east]
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Covariance matrix
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 100)
    
    # Track management
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    last_update: float = 0.0
    
    # Track history
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # ML-related fields
    maneuver_probability: float = 0.0
    current_process_noise: float = 0.5
    last_sensor_id: int = 0
    last_measurement_noise: float = 5.0
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'target_id': self.target_id,
            'target_name': self.name,
            'name': self.name,
            'position': [float(self.state[0]), float(self.state[1])],
            'velocity': [float(self.state[2]), float(self.state[3])],
            'speed_ms': float(np.sqrt(self.state[2]**2 + self.state[3]**2)),
            'speed_knots': float(np.sqrt(self.state[2]**2 + self.state[3]**2) * 1.944),
            'status': 'confirmed' if self.confirmed else 'tentative',
            'confirmed': self.confirmed,
            'hits': self.hits,
            'misses': self.misses,
            # ML info
            'maneuver_probability': self.maneuver_probability,
            'process_noise_Q': self.current_process_noise,
            'measurement_noise_R': self.last_measurement_noise,
            'last_sensor_id': self.last_sensor_id,
            'position_uncertainty_m': float(np.sqrt(self.covariance[0,0] + self.covariance[1,1])),
            'history': [(h[1], h[2]) for h in self.position_history]  # [(north, east), ...]
        }


class KalmanFilter:
    """
    Linear Kalman Filter for target tracking in NED frame.
    
    State: [north, east, vel_north, vel_east]
    Measurement: [north, east]
    """
    
    def __init__(self, 
                 process_noise: float = 0.5,
                 measurement_noise: float = 2.0,
                 dt: float = 0.1):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise standard deviation (m/s²)
            measurement_noise: Measurement noise standard deviation (m)
            dt: Default time step (seconds)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise**2
        
    def get_process_noise(self, dt: float) -> np.ndarray:
        """Compute process noise matrix Q for given time step"""
        q = self.process_noise**2
        return np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
    
    def predict(self, state: np.ndarray, covariance: np.ndarray, 
                dt: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter prediction step.
        
        Args:
            state: Current state vector [4,]
            covariance: Current covariance matrix [4,4]
            dt: Time step (uses default if None)
            
        Returns:
            Predicted state and covariance
        """
        if dt is None:
            dt = self.dt
            
        # Update F matrix with actual dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        Q = self.get_process_noise(dt)
        
        # Predict
        state_pred = F @ state
        cov_pred = F @ covariance @ F.T + Q
        
        return state_pred, cov_pred
    
    def update(self, state: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter update step.
        
        Args:
            state: Predicted state vector [4,]
            covariance: Predicted covariance matrix [4,4]
            measurement: Measurement vector [north, east]
            
        Returns:
            Updated state and covariance
        """
        # Innovation
        y = measurement - self.H @ state
        
        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R
        
        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        state_upd = state + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        cov_upd = I_KH @ covariance @ I_KH.T + K @ self.R @ K.T
        
        return state_upd, cov_upd
    
    def gating(self, state: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray, gate_threshold: float = 9.21) -> bool:
        """
        Chi-squared gating for measurement association.
        
        Args:
            state: Current state
            covariance: Current covariance
            measurement: Candidate measurement
            gate_threshold: Chi-squared threshold (9.21 for 99% with 2 DOF)
            
        Returns:
            True if measurement is within gate
        """
        # Innovation
        y = measurement - self.H @ state
        
        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R
        
        # Mahalanobis distance squared
        d2 = y.T @ np.linalg.inv(S) @ y
        
        return d2 < gate_threshold


class SensorFusionAgentKalman(BaseAgent):
    """
    Multi-sensor fusion agent with Kalman filtering.
    
    Processes detections from multiple sensors in Autoferry benchmark format
    and maintains tracked targets using Kalman filtering.
    
    Features:
    - Multi-sensor data association
    - Kalman filter tracking
    - Track initiation and deletion logic
    - Track-to-track fusion
    """
    
    SENSOR_NAMES = {1: 'LiDAR', 2: 'Radar', 3: 'IR_Camera', 4: 'EO_Camera'}
    TARGET_NAMES = {1: 'Havfruen', 2: 'Gunnerus', 3: 'Jetboat'}
    
    # Piren reference point (NTNU)
    PIREN_ORIGIN = [63.4389029083, 10.39908278, 39.923]  # LLA
    
    # Track management parameters
    CONFIRM_HITS = 3  # Hits needed to confirm track
    DELETE_MISSES = 5  # Misses before track deletion
    GATE_THRESHOLD = 16.0  # Chi-squared gate (95% with 2 DOF)
    
    def __init__(self, 
                 process_noise: float = 0.5,
                 measurement_noise: float = 2.0,
                 use_ml: bool = True):
        super().__init__(
            name="SensorFusionAgent",
            description="Multi-sensor tracking with Kalman filtering",
            domain="navigation"
        )
        self.kalman = KalmanFilter(process_noise, measurement_noise)
        self.tracks: Dict[int, KalmanTrack] = {}
        self.next_track_id = 1
        self.current_time = 0.0
        
        # ML Enhancement
        self.use_ml = use_ml
        self.ml_fusion = None
        if use_ml:
            self._load_ml_models()
    
    def _load_ml_models(self):
        """Load trained ML models for adaptive Kalman filtering."""
        import os
        try:
            from sensor_fusion_ml import MLEnhancedSensorFusion
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            if os.path.exists(os.path.join(models_dir, 'maneuver_detector.pkl')):
                self.ml_fusion = MLEnhancedSensorFusion()
                self.ml_fusion.load(models_dir)
                self.logger.info("ML models loaded for adaptive Kalman filtering")
            else:
                self.logger.warning("ML models not found, using fixed parameters")
                self.use_ml = False
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}")
            self.use_ml = False
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process sensor fusion data with Kalman filtering.
        
        Args:
            data: Dict containing 'detections' and optionally 'ground_truth'
            context: Optional processing context
            
        Returns:
            Fusion results with tracked targets
        """
        self.update_status(AgentStatus.PROCESSING, "sensor_fusion", 0.0)
        self.logger.info("Processing sensor fusion data with Kalman filtering")
        
        try:
            detections = data.get('detections', []) if isinstance(data, dict) else []
            ground_truth = data.get('ground_truth', []) if isinstance(data, dict) else []
            
            if not detections:
                self.logger.warning("No detections provided")
                return {'status': 'success', 'tracks': [], 'num_detections': 0}
            
            # Sort detections by time
            detections = sorted(detections, key=lambda d: d.get('time', 0))
            
            # Process detections through time
            self.update_status(AgentStatus.PROCESSING, "kalman_tracking", 0.3)
            processed_times = set()
            
            for det in detections:
                det_time = det.get('time', 0)
                
                # Predict all tracks to this time
                if det_time not in processed_times:
                    self._predict_all_tracks(det_time)
                    processed_times.add(det_time)
                
                # Associate and update
                self._process_detection(det)
            
            # Final track management
            self.update_status(AgentStatus.PROCESSING, "track_management", 0.7)
            self._manage_tracks()
            
            # Generate sensor statistics
            sensor_stats = self._compute_sensor_stats(detections)
            
            # Extract confirmed tracks
            confirmed_tracks = [t.to_dict() for t in self.tracks.values() if t.confirmed]
            all_tracks = [t.to_dict() for t in self.tracks.values()]
            
            # Compute fusion quality metrics
            quality_metrics = self._compute_quality_metrics(ground_truth)
            
            result = {
                'status': 'success',
                'num_detections': len(detections),
                'sensor_statistics': sensor_stats,
                'num_tracks': len(all_tracks),
                'num_confirmed_tracks': len(confirmed_tracks),
                'tracks': all_tracks,
                'confirmed_tracks': confirmed_tracks,
                'quality_metrics': quality_metrics,
                'coordinate_frame': 'Piren NED',
                'kalman_filter': {
                    'process_noise': self.kalman.process_noise,
                    'measurement_noise': self.kalman.measurement_noise
                },
                'ml_enhanced': self.use_ml and self.ml_fusion is not None,
                'ml_status': self._build_ml_status(),
                'alerts': self.get_alerts()
            }
            
            self.log_result(result)
            self.update_status(AgentStatus.COMPLETED, progress=1.0)
            self.logger.info(f"Tracking complete: {len(confirmed_tracks)} confirmed tracks")
            
            return result
            
        except Exception as e:
            self.state.set_error(str(e))
            self.create_alert(AlertLevel.CRITICAL, f"Sensor fusion failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _predict_all_tracks(self, current_time: float):
        """Predict all tracks to current time"""
        for track in self.tracks.values():
            if track.last_update > 0:
                dt = current_time - track.last_update
                if dt > 0:
                    track.state, track.covariance = self.kalman.predict(
                        track.state, track.covariance, dt
                    )
        self.current_time = current_time
    
    def _process_detection(self, detection: Dict):
        """Process a single detection with ML-adaptive parameters"""
        # Extract measurement
        position = detection.get('position', [0, 0, 0])
        measurement = np.array([position[0], position[1]])  # North, East
        det_time = detection.get('time', 0)
        target_id = detection.get('targetID', 0)
        sensor_id = detection.get('sensorID', 0)
        
        # ML: Get sensor-specific measurement noise
        if self.use_ml and self.ml_fusion:
            range_to_target = np.linalg.norm(measurement)
            R = self.ml_fusion.sensor_reliability.get_measurement_noise(sensor_id, range_to_target)
            self.kalman.R = np.eye(2) * R**2
        
        # Try to associate with existing track
        best_track = None
        best_distance = float('inf')
        
        for track in self.tracks.values():
            if self.kalman.gating(track.state, track.covariance, 
                                  measurement, self.GATE_THRESHOLD):
                # Compute Mahalanobis distance for best match
                y = measurement - self.kalman.H @ track.state
                S = self.kalman.H @ track.covariance @ self.kalman.H.T + self.kalman.R
                d2 = y.T @ np.linalg.inv(S) @ y
                
                if d2 < best_distance:
                    best_distance = d2
                    best_track = track
        
        if best_track is not None:
            # ML: Get maneuver-adaptive process noise
            if self.use_ml and self.ml_fusion:
                track_positions = [(h[1], h[2]) for h in best_track.position_history[-6:]]
                Q = self.ml_fusion.maneuver_detector.get_process_noise(track_positions)
                maneuver_prob = self.ml_fusion.maneuver_detector.last_probability
                best_track.maneuver_probability = maneuver_prob
                best_track.current_process_noise = Q
                best_track.last_sensor_id = sensor_id
                best_track.last_measurement_noise = R if self.use_ml and self.ml_fusion else self.kalman.measurement_noise
                # Temporarily update Kalman process noise
                original_Q = self.kalman.process_noise
                self.kalman.process_noise = Q
            
            # Update existing track
            best_track.state, best_track.covariance = self.kalman.update(
                best_track.state, best_track.covariance, measurement
            )
            best_track.hits += 1
            best_track.last_update = det_time
            best_track.position_history.append((det_time, position[0], position[1]))
            
            if not best_track.confirmed and best_track.hits >= self.CONFIRM_HITS:
                best_track.confirmed = True
                self.logger.info(f"Track {best_track.track_id} confirmed as {best_track.name}")
        else:
            # Initiate new track
            self._initiate_track(measurement, det_time, target_id)
    
    def _initiate_track(self, measurement: np.ndarray, time: float, target_id: int):
        """Initiate a new track"""
        track = KalmanTrack(
            track_id=self.next_track_id,
            target_id=target_id,
            name=self.TARGET_NAMES.get(target_id, f'Target_{target_id}'),
            state=np.array([measurement[0], measurement[1], 0.0, 0.0]),
            covariance=np.diag([10.0, 10.0, 5.0, 5.0]),
            hits=1,
            last_update=time
        )
        track.position_history.append((time, measurement[0], measurement[1]))
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.logger.debug(f"Initiated track {track.track_id}")
    
    def _build_ml_status(self) -> Dict:
        """Build comprehensive ML status with diagnostics."""
        ml_loaded = self.ml_fusion is not None
        maneuver_active = ml_loaded and self.ml_fusion.maneuver_detector.is_trained
        reliability_active = ml_loaded and self.ml_fusion.sensor_reliability.is_trained
        
        status = {
            'enabled': self.use_ml,
            'models_loaded': ml_loaded,
            # Boolean flags expected by streamlit
            'maneuver_detector_active': maneuver_active,
            'reliability_estimator_active': reliability_active,
            # Legacy string fields for backward compatibility
            'maneuver_detector': 'active' if maneuver_active else 'inactive',
            'sensor_reliability': 'active' if reliability_active else 'inactive',
            # Diagnostic information
            'diagnostics': {
                'gate_threshold': self.GATE_THRESHOLD,
                'gate_threshold_chi2_dof': 2,
                'confirm_hits': self.CONFIRM_HITS,
                'delete_misses': self.DELETE_MISSES,
            },
            'sensor_stats': {},
        }
        
        # Add detailed ML diagnostic info
        if not ml_loaded:
            status['diagnostics']['ml_load_error'] = 'ML models could not be loaded - check models/ directory for maneuver_detector.pkl and sensor_reliability.pkl'
        else:
            # Add maneuver detector stats
            if hasattr(self.ml_fusion.maneuver_detector, 'training_stats'):
                status['maneuver_detector_stats'] = self.ml_fusion.maneuver_detector.training_stats
            # Add sensor reliability stats
            status['sensor_stats'] = self.ml_fusion.sensor_reliability.get_all_stats()
            if not reliability_active:
                status['diagnostics']['reliability_warning'] = 'Sensor reliability estimator not trained - using default noise values'
        
        return status
    
    def set_gate_threshold(self, threshold: float):
        """Set association gate threshold (χ² critical value).
        
        Common values:
        - 9.21: 99% confidence, 2 DOF (tight)
        - 16.0: 95% confidence, 2 DOF (default)
        - 24.0: 90% confidence, 2 DOF (relaxed for high clutter)
        """
        self.GATE_THRESHOLD = threshold
        self.logger.info(f"Association gate threshold set to {threshold}")
    
    def _manage_tracks(self):
        """Track management - confirm and delete tracks"""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Check for stale tracks
            time_since_update = self.current_time - track.last_update
            
            if time_since_update > 5.0:  # 5 seconds without update
                track.misses += 1
                
            if track.misses >= self.DELETE_MISSES:
                tracks_to_delete.append(track_id)
                self.logger.debug(f"Deleting track {track_id} (too many misses)")
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def _compute_sensor_stats(self, detections: List[Dict]) -> Dict:
        """Compute statistics per sensor"""
        stats = {}
        for sensor_id, name in self.SENSOR_NAMES.items():
            sensor_dets = [d for d in detections if d.get('sensorID') == sensor_id]
            if sensor_dets:
                times = [d.get('time', 0) for d in sensor_dets]
                duration = max(times) - min(times) if len(times) > 1 else 1
                rate = len(sensor_dets) / duration if duration > 0 else 0
                
                stats[name] = {
                    'count': len(sensor_dets),
                    'rate_hz': rate,
                    'time_span': duration
                }
            else:
                stats[name] = {'count': 0, 'rate_hz': 0, 'time_span': 0}
        return stats
    
    def _compute_quality_metrics(self, ground_truth: List[Dict]) -> Dict:
        """Compute tracking quality metrics against ground truth"""
        if not ground_truth or not self.tracks:
            return {'available': False}
        
        # Build ground truth positions by target
        gt_by_target = {}
        for gt in ground_truth:
            tid = gt.get('targetID', 0)
            if tid not in gt_by_target:
                gt_by_target[tid] = []
            gt_by_target[tid].append({
                'time': gt.get('time', 0),
                'position': gt.get('position', [0, 0, 0])
            })
        
        # Compute errors for confirmed tracks
        errors = []
        for track in self.tracks.values():
            if track.confirmed and track.target_id in gt_by_target:
                gt_positions = gt_by_target[track.target_id]
                if gt_positions:
                    # Use last ground truth position
                    last_gt = max(gt_positions, key=lambda x: x['time'])
                    gt_pos = last_gt['position']
                    
                    error_n = track.state[0] - gt_pos[0]
                    error_e = track.state[1] - gt_pos[1]
                    error = np.sqrt(error_n**2 + error_e**2)
                    errors.append(error)
        
        if errors:
            return {
                'available': True,
                'mean_position_error_m': float(np.mean(errors)),
                'max_position_error_m': float(np.max(errors)),
                'rms_error_m': float(np.sqrt(np.mean(np.array(errors)**2)))
            }
        return {'available': False}
    
    def get_track(self, track_id: int) -> Optional[KalmanTrack]:
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def get_confirmed_tracks(self) -> List[KalmanTrack]:
        """Get all confirmed tracks"""
        return [t for t in self.tracks.values() if t.confirmed]
    
    def reset_tracks(self):
        """Reset all tracks"""
        self.tracks.clear()
        self.next_track_id = 1
        self.current_time = 0.0
        self.logger.info("All tracks reset")
