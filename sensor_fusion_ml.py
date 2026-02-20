"""
Machine Learning Enhancements for Sensor Fusion
================================================
1. Maneuver-Adaptive Kalman Filter - Adjusts process noise based on target behavior
2. Sensor Reliability Estimation - Learns measurement noise per sensor

These models are trained on Autoferry ground truth data and can be used
for real-time inference during tracking.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SensorStats:
    """Statistics for a sensor type"""
    sensor_id: int
    sensor_name: str
    mean_error: float
    std_error: float
    detection_count: int
    reliability_score: float  # 0-1, higher is better
    
    def to_dict(self) -> Dict:
        return {
            'sensor_id': self.sensor_id,
            'sensor_name': self.sensor_name,
            'mean_error_m': round(self.mean_error, 2),
            'std_error_m': round(self.std_error, 2),
            'detection_count': self.detection_count,
            'reliability_score': round(self.reliability_score, 3)
        }


class ManeuverDetector:
    """
    ML model to detect if a target is maneuvering based on track history.
    
    Uses velocity changes over a sliding window to predict maneuver probability.
    High maneuver probability → increase Kalman process noise.
    """
    
    WINDOW_SIZE = 5
    MANEUVER_THRESHOLD = 0.3  # m/s² acceleration threshold
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_stats = {}
        self.last_probability = 0.0
        
    def train(self, ground_truth: List[Dict]) -> Dict:
        """
        Train maneuver detector from ground truth trajectories.
        
        Args:
            ground_truth: List of {time, targetID, position} dicts
            
        Returns:
            Training statistics
        """
        # Group by target
        targets = {}
        for gt in sorted(ground_truth, key=lambda x: x['time']):
            tid = gt['targetID']
            if tid not in targets:
                targets[tid] = []
            targets[tid].append(gt)
        
        X, y = [], []
        
        for tid, trajectory in targets.items():
            if len(trajectory) < self.WINDOW_SIZE + 2:
                continue
            
            # Extract positions and times
            positions = np.array([t['position'][:2] for t in trajectory])
            times = np.array([t['time'] for t in trajectory])
            
            # Compute velocities
            velocities = []
            for i in range(1, len(positions)):
                dt = times[i] - times[i-1]
                if dt > 0.01:
                    vel = (positions[i] - positions[i-1]) / dt
                    velocities.append(vel)
                else:
                    velocities.append(np.array([0.0, 0.0]))
            velocities = np.array(velocities)
            
            # Compute accelerations
            accelerations = []
            for i in range(1, len(velocities)):
                dt = times[i+1] - times[i] if i+1 < len(times) else 0.5
                if dt > 0.01:
                    acc = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(np.linalg.norm(acc))
                else:
                    accelerations.append(0.0)
            
            # Create training samples with sliding window
            for i in range(self.WINDOW_SIZE, len(velocities) - 1):
                vel_window = velocities[i-self.WINDOW_SIZE:i]
                speeds = np.linalg.norm(vel_window, axis=1)
                speed_changes = np.diff(speeds)
                
                features = np.concatenate([
                    vel_window.flatten(),
                    speeds,
                    speed_changes
                ])
                
                if i-1 < len(accelerations):
                    label = 1 if accelerations[i-1] > self.MANEUVER_THRESHOLD else 0
                    X.append(features)
                    y.append(label)
        
        if len(X) < 10:
            print("Insufficient training data for maneuver detector")
            return {'status': 'failed', 'error': 'insufficient_data'}
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        accuracy = self.classifier.score(X_scaled, y)
        n_maneuvering = int(np.sum(y))
        n_steady = len(y) - n_maneuvering
        
        self.training_stats = {
            'status': 'success',
            'total_samples': len(y),
            'maneuvering_samples': n_maneuvering,
            'steady_samples': n_steady,
            'maneuvering_ratio': n_maneuvering / len(y),
            'training_accuracy': accuracy
        }
        
        print(f"Maneuver Detector Training:")
        print(f"  Samples: {len(y)} (maneuver: {n_maneuvering}, steady: {n_steady})")
        print(f"  Accuracy: {accuracy:.1%}")
        
        return self.training_stats
    
    def predict(self, track_history: List[Tuple[float, float]]) -> Tuple[int, float]:
        """
        Predict if target is maneuvering.
        
        Args:
            track_history: List of (north, east) positions
            
        Returns:
            (prediction, probability)
        """
        if not self.is_trained or len(track_history) < self.WINDOW_SIZE + 1:
            self.last_probability = 0.0
            return 0, 0.0
        
        positions = np.array(track_history[-(self.WINDOW_SIZE+1):])
        velocities = np.diff(positions, axis=0) / 0.5
        
        speeds = np.linalg.norm(velocities, axis=1)
        speed_changes = np.diff(speeds)
        
        features = np.concatenate([
            velocities.flatten(),
            speeds,
            speed_changes
        ]).reshape(1, -1)
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.classifier.predict(features_scaled)[0]
            probability = self.classifier.predict_proba(features_scaled)[0, 1]
            self.last_probability = float(probability)
            return int(prediction), float(probability)
        except:
            self.last_probability = 0.0
            return 0, 0.0
    
    def get_process_noise(self, track_history: List[Tuple[float, float]],
                          base_noise: float = 0.1,
                          max_noise: float = 2.0) -> float:
        """Get adaptive process noise based on maneuver probability."""
        _, probability = self.predict(track_history)
        noise = base_noise + probability * (max_noise - base_noise)
        return noise
    
    def save(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'training_stats': self.training_stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.training_stats = data.get('training_stats', {})
        self.is_trained = True


class SensorReliabilityEstimator:
    """
    Estimates sensor measurement noise based on ground truth comparison.
    """
    
    SENSOR_NAMES = {1: 'LiDAR', 2: 'Radar', 3: 'IR_Camera', 4: 'EO_Camera'}
    
    def __init__(self):
        self.sensor_stats: Dict[int, SensorStats] = {}
        self.range_model = None
        self.is_trained = False
        self.training_stats = {}
        
    def train(self, detections: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Compute sensor reliability from detections vs ground truth.
        """
        gt_by_time = {}
        for gt in ground_truth:
            t = round(gt['time'], 2)
            if t not in gt_by_time:
                gt_by_time[t] = []
            gt_by_time[t].append(gt)
        
        sensor_errors = {1: [], 2: [], 3: [], 4: []}
        range_data = []
        
        for det in detections:
            sensor_id = det.get('sensorID', 0)
            if sensor_id not in sensor_errors:
                continue
                
            det_pos = np.array(det['position'][:2])
            det_time = round(det.get('time', 0), 2)
            
            # Find ground truth
            if det_time not in gt_by_time:
                for offset in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                    t_check = round(det_time + offset, 2)
                    if t_check in gt_by_time:
                        det_time = t_check
                        break
                else:
                    continue
            
            gt_at_time = gt_by_time[det_time]
            
            min_error = float('inf')
            for gt in gt_at_time:
                gt_pos = np.array(gt['position'][:2])
                error = np.linalg.norm(det_pos - gt_pos)
                if error < min_error:
                    min_error = error
            
            if min_error < 50:
                sensor_errors[sensor_id].append(min_error)
                range_to_target = np.linalg.norm(det_pos)
                range_data.append((range_to_target, min_error, sensor_id))
        
        for sensor_id, errors in sensor_errors.items():
            if len(errors) > 0:
                errors = np.array(errors)
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                reliability = 1.0 / (1.0 + mean_error / 10.0)
                
                self.sensor_stats[sensor_id] = SensorStats(
                    sensor_id=sensor_id,
                    sensor_name=self.SENSOR_NAMES.get(sensor_id, f'Sensor_{sensor_id}'),
                    mean_error=mean_error,
                    std_error=std_error,
                    detection_count=len(errors),
                    reliability_score=reliability
                )
        
        if len(range_data) > 20:
            X = np.array([[r[0], r[2]] for r in range_data])
            y = np.array([r[1] for r in range_data])
            self.range_model = RandomForestRegressor(
                n_estimators=30, max_depth=6, random_state=42
            )
            self.range_model.fit(X, y)
        
        self.is_trained = True
        
        print("\nSensor Reliability:")
        for sid in sorted(self.sensor_stats.keys()):
            s = self.sensor_stats[sid]
            print(f"  {s.sensor_name}: error={s.mean_error:.2f}m, reliability={s.reliability_score:.2f}")
        
        self.training_stats = {
            'status': 'success',
            'sensors_analyzed': len(self.sensor_stats),
            'sensor_stats': {sid: s.to_dict() for sid, s in self.sensor_stats.items()}
        }
        
        return self.training_stats
    
    def get_measurement_noise(self, sensor_id: int, 
                              range_to_target: Optional[float] = None) -> float:
        """Get measurement noise R for a sensor."""
        if not self.is_trained:
            defaults = {1: 2.0, 2: 5.0, 3: 4.0, 4: 3.5}
            return defaults.get(sensor_id, 5.0)
        
        if sensor_id in self.sensor_stats:
            base_noise = self.sensor_stats[sensor_id].std_error
            
            if range_to_target is not None and self.range_model is not None:
                try:
                    predicted = self.range_model.predict([[range_to_target, sensor_id]])[0]
                    base_noise = 0.5 * base_noise + 0.5 * predicted
                except:
                    pass
            
            return max(base_noise, 0.5)
        
        return 5.0
    
    def get_sensor_weight(self, sensor_id: int) -> float:
        """Get fusion weight for a sensor."""
        if sensor_id in self.sensor_stats:
            return self.sensor_stats[sensor_id].reliability_score
        return 0.5
    
    def get_all_stats(self) -> Dict[int, Dict]:
        """Get all sensor statistics."""
        return {sid: s.to_dict() for sid, s in self.sensor_stats.items()}
    
    def save(self, filepath: str):
        """Save model (convert dataclass to dict to avoid pickle issues)."""
        # Convert SensorStats dataclasses to dictionaries
        stats_as_dicts = {}
        for sid, s in self.sensor_stats.items():
            stats_as_dicts[sid] = {
                'sensor_id': s.sensor_id,
                'sensor_name': s.sensor_name,
                'mean_error': s.mean_error,
                'std_error': s.std_error,
                'detection_count': s.detection_count,
                'reliability_score': s.reliability_score
            }
        
        data = {
            'sensor_stats_dicts': stats_as_dicts,
            'range_model': self.range_model,
            'training_stats': self.training_stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load model (reconstruct dataclass from dict)."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct SensorStats from dictionaries
        stats_dicts = data.get('sensor_stats_dicts', data.get('sensor_stats', {}))
        self.sensor_stats = {}
        for sid, s_dict in stats_dicts.items():
            if isinstance(s_dict, dict):
                self.sensor_stats[sid] = SensorStats(
                    sensor_id=s_dict['sensor_id'],
                    sensor_name=s_dict['sensor_name'],
                    mean_error=s_dict['mean_error'],
                    std_error=s_dict['std_error'],
                    detection_count=s_dict['detection_count'],
                    reliability_score=s_dict['reliability_score']
                )
            else:
                # Already a SensorStats object
                self.sensor_stats[sid] = s_dict
        
        self.range_model = data.get('range_model')
        self.training_stats = data.get('training_stats', {})
        self.is_trained = True


class MLEnhancedSensorFusion:
    """
    Combines ML models for enhanced Kalman filtering:
    - Maneuver-adaptive process noise (Q)
    - Sensor-specific measurement noise (R)
    """
    
    def __init__(self):
        self.maneuver_detector = ManeuverDetector()
        self.sensor_reliability = SensorReliabilityEstimator()
        self.models_trained = False
        
    def train(self, detections: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Train all ML models."""
        print("=" * 50)
        print("Training ML-Enhanced Sensor Fusion")
        print("=" * 50)
        
        maneuver_stats = self.maneuver_detector.train(ground_truth)
        sensor_stats = self.sensor_reliability.train(detections, ground_truth)
        
        self.models_trained = True
        
        return {
            'maneuver_detector': maneuver_stats,
            'sensor_reliability': sensor_stats
        }
    
    def save(self, directory: str):
        """Save all models."""
        os.makedirs(directory, exist_ok=True)
        self.maneuver_detector.save(os.path.join(directory, 'maneuver_detector.pkl'))
        self.sensor_reliability.save(os.path.join(directory, 'sensor_reliability.pkl'))
        print(f"Models saved to {directory}/")
        
    def load(self, directory: str):
        """Load all models."""
        self.maneuver_detector.load(os.path.join(directory, 'maneuver_detector.pkl'))
        self.sensor_reliability.load(os.path.join(directory, 'sensor_reliability.pkl'))
        self.models_trained = True
        print(f"Models loaded from {directory}/")
    
    def get_adaptive_params(self, track_history: List, sensor_id: int,
                           range_to_target: float = None) -> Dict:
        """
        Get ML-adapted Kalman parameters for inference.
        
        Returns:
            Dict with Q, R, maneuver_prob, sensor_reliability
        """
        _, maneuver_prob = self.maneuver_detector.predict(track_history)
        Q = self.maneuver_detector.get_process_noise(track_history)
        R = self.sensor_reliability.get_measurement_noise(sensor_id, range_to_target)
        reliability = self.sensor_reliability.get_sensor_weight(sensor_id)
        
        return {
            'process_noise_Q': Q,
            'measurement_noise_R': R,
            'maneuver_probability': maneuver_prob,
            'sensor_reliability': reliability,
            'is_maneuvering': maneuver_prob > 0.5
        }


def train_from_autoferry(train_scenarios: List[str] = None):
    """
    Train ML models from Autoferry data.
    
    Args:
        train_scenarios: Scenarios to use for training (default: env1 scenarios)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data.autoferry_loader import AutoferryDataLoader
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    loader = AutoferryDataLoader(dataset_path)
    
    # Default: train on Environment 1 (scenarios 2-6)
    if train_scenarios is None:
        train_scenarios = ['scenario2', 'scenario3', 'scenario4', 'scenario5', 'scenario6']
    
    all_detections = []
    all_ground_truth = []
    
    print(f"Loading training data from {len(train_scenarios)} scenarios...")
    
    for scenario in train_scenarios:
        if scenario in loader.available_scenarios:
            data = loader.load_for_kalman_filter(scenario)
            all_detections.extend(data['detections'])
            all_ground_truth.extend(data['ground_truth'])
    
    print(f"Training data: {len(all_detections)} detections, {len(all_ground_truth)} ground truth")
    
    ml_fusion = MLEnhancedSensorFusion()
    ml_fusion.train(all_detections, all_ground_truth)
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ml_fusion.save(models_dir)
    
    return ml_fusion


if __name__ == '__main__':
    ml_fusion = train_from_autoferry()
    print("\n" + "=" * 50)
    print("ML Training Complete!")
    print("=" * 50)
