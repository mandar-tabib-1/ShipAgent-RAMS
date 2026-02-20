"""
Autoferry Sensor Fusion Dataset Loader
======================================
Loads and preprocesses the Autoferry benchmark dataset for use with
the Vessel AI System.

Dataset source: https://github.com/Autoferry/sensor_fusion_dataset
Citation: Heterogeneous multi-sensor tracking for an autonomous surface 
         vehicle in a littoral environment (Ocean Engineering, 2022)

Data Format:
- Detections: Sensor measurements in ownship-fixed NED frame
- Ground Truth: Target positions in Piren NED frame
- Sensors: LiDAR (1), Radar (2), IR Camera (3), EO Camera (4)
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Piren NED frame origin (reference point)
PIREN_ORIGIN_LLA = [63.4389029083, 10.39908278, 39.923]

# Sensor type mapping
SENSOR_TYPES = {
    1: {'name': 'LiDAR', 'type': 'position', 'noise_std': 1.0},
    2: {'name': 'Radar', 'type': 'position', 'noise_std': 3.0},
    3: {'name': 'IR_Camera', 'type': 'bearing', 'noise_std': 0.02},
    4: {'name': 'EO_Camera', 'type': 'bearing', 'noise_std': 0.02}
}

# Target names from the dataset
TARGET_NAMES = {
    1: 'Havfruen',
    2: 'Vessel',  # or Jetboat depending on environment
    3: 'Jetboat'
}

# Available scenarios
SCENARIOS = {
    # Environment 1 (May 4, 2021)
    'scenario2': 'Crossing maneuver - Havfruen behind Vessel',
    'scenario3': 'Havfruen maneuvers around Vessel',
    'scenario4': 'Side by side approach, Vessel maneuvers north',
    'scenario5': 'High-speed overtake in sensor shadow',
    'scenario6': 'Both targets travel toward milliAmpere',
    # Environment 2 (May 5, 2021)
    'scenario13': 'Targets approach from each side (docked)',
    'scenario16': 'Targets approach from each side (stationary)',
    'scenario17': 'Both targets approach side by side from east',
    'scenario22': 'Channel crossing with target breaking off'
}


@dataclass
class AutoferryDetection:
    """A single sensor detection"""
    time: float
    sensor_id: int
    sensor_name: str
    measurement_type: str  # 'position' or 'bearing'
    measurement: np.ndarray  # [north, east] for position, angles for bearing
    ownship_position: np.ndarray  # [north, east] in Piren NED
    
    def to_dict(self) -> Dict:
        return {
            'time': self.time,
            'sensorID': self.sensor_id,
            'sensor_name': self.sensor_name,
            'measurement_type': self.measurement_type,
            'measurement': self.measurement.tolist(),
            'ownship_position': self.ownship_position.tolist()
        }


@dataclass
class AutoferryGroundTruth:
    """Ground truth for a single target at a time"""
    time: float
    target_id: int
    target_name: str
    position: np.ndarray  # [north, east, down] in Piren NED
    
    def to_dict(self) -> Dict:
        return {
            'time': self.time,
            'targetID': self.target_id,
            'target_name': self.target_name,
            'position': self.position.tolist()
        }


class AutoferryDataLoader:
    """
    Loader for Autoferry sensor fusion benchmark dataset.
    
    Usage:
        loader = AutoferryDataLoader('path/to/sensor_fusion_dataset')
        detections, ground_truth = loader.load_scenario('scenario2')
        
        # Or get Kalman-filter compatible format:
        data = loader.load_for_kalman_filter('scenario2')
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize loader with path to dataset.
        
        Args:
            dataset_path: Path to the sensor_fusion_dataset folder
        """
        self.dataset_path = dataset_path
        self.available_scenarios = self._find_scenarios()
        
    def _find_scenarios(self) -> List[str]:
        """Find available scenarios in dataset"""
        scenarios = []
        for name in os.listdir(self.dataset_path):
            if name.startswith('scenario') and os.path.isdir(
                os.path.join(self.dataset_path, name)
            ):
                scenarios.append(name)
        return sorted(scenarios)
    
    def load_scenario(self, scenario: str) -> Tuple[List[AutoferryDetection], 
                                                     List[List[AutoferryGroundTruth]]]:
        """
        Load a scenario's detections and ground truth.
        
        Args:
            scenario: Scenario name (e.g., 'scenario2')
            
        Returns:
            Tuple of (detections_list, ground_truth_list)
        """
        scenario_path = os.path.join(self.dataset_path, scenario)
        
        # Load detections
        det_file = os.path.join(scenario_path, f'{scenario}_detections.json')
        with open(det_file, 'r') as f:
            raw_detections = json.load(f)
        
        # Load ground truth
        gt_file = os.path.join(scenario_path, f'{scenario}_groundTruth.json')
        with open(gt_file, 'r') as f:
            raw_ground_truth = json.load(f)
        
        # Parse detections
        detections = []
        for raw_det in raw_detections:
            sensor_id = raw_det['sensorID']
            sensor_info = SENSOR_TYPES.get(sensor_id, {'name': 'Unknown', 'type': 'unknown'})
            
            # Parse measurement based on sensor type
            raw_meas = raw_det['measurement']
            if isinstance(raw_meas, list):
                measurement = np.array(raw_meas)
            else:
                measurement = np.array([raw_meas])
            
            detection = AutoferryDetection(
                time=raw_det['time'],
                sensor_id=sensor_id,
                sensor_name=sensor_info['name'],
                measurement_type=sensor_info['type'],
                measurement=measurement,
                ownship_position=np.array(raw_det['ownshipPosition'])
            )
            detections.append(detection)
        
        # Parse ground truth
        ground_truths = []
        for gt_list in raw_ground_truth:
            gt_frame = []
            for raw_gt in gt_list:
                gt = AutoferryGroundTruth(
                    time=raw_gt['time'],
                    target_id=raw_gt['targetID'],
                    target_name=TARGET_NAMES.get(raw_gt['targetID'], f"Target_{raw_gt['targetID']}"),
                    position=np.array(raw_gt['position'])
                )
                gt_frame.append(gt)
            ground_truths.append(gt_frame)
        
        return detections, ground_truths
    
    def load_for_kalman_filter(self, scenario: str, 
                               use_position_sensors_only: bool = True) -> Dict:
        """
        Load scenario data in format compatible with Kalman filter agent.
        
        Args:
            scenario: Scenario name
            use_position_sensors_only: If True, only use LiDAR and Radar
                                       (sensors that provide position measurements)
        
        Returns:
            Dict with 'detections' and 'ground_truth' in Kalman filter format
        """
        detections, ground_truths = self.load_scenario(scenario)
        
        # Normalize time to start from 0
        if detections:
            t0 = min(d.time for d in detections)
        else:
            t0 = 0
        
        # Convert detections to Kalman filter format
        kalman_detections = []
        for det in detections:
            # Only use position-based sensors for Kalman filter
            if use_position_sensors_only and det.measurement_type != 'position':
                continue
            
            # Skip empty measurements
            if len(det.measurement) == 0:
                continue
            
            # For position sensors, parse measurements based on sensor type
            # LiDAR: [north, east] - single detection
            # Radar: [[north1, north2, ...], [east1, east2, ...]] - multi-target
            if det.measurement_type == 'position':
                meas = det.measurement
                
                # Check if it's a matrix (multi-target radar)
                if len(meas.shape) == 2 and meas.shape[0] == 2:
                    # Radar format: [[norths], [easts]] - transpose to get pairs
                    num_targets = meas.shape[1]
                    for i in range(num_targets):
                        north = meas[0, i]
                        east = meas[1, i]
                        # Convert from ownship-fixed to Piren NED
                        position_piren = np.array([north, east]) + det.ownship_position
                        
                        kalman_detections.append({
                            'time': det.time - t0,
                            'sensorID': det.sensor_id,
                            'targetID': 0,  # Unknown - needs association
                            'position': [float(position_piren[0]), float(position_piren[1]), 0.0]
                        })
                elif len(meas.shape) == 1 and len(meas) >= 2:
                    # LiDAR format: [north, east] - single detection
                    position_piren = meas[:2] + det.ownship_position
                    
                    kalman_detections.append({
                        'time': det.time - t0,
                        'sensorID': det.sensor_id,
                        'targetID': 0,  # Unknown - needs association
                        'position': [float(position_piren[0]), float(position_piren[1]), 0.0]
                    })
        
        # Convert ground truth to Kalman filter format
        kalman_ground_truth = []
        seen_times = set()
        
        for gt_frame in ground_truths:
            for gt in gt_frame:
                # Avoid duplicate entries
                key = (round(gt.time - t0, 3), gt.target_id)
                if key in seen_times:
                    continue
                seen_times.add(key)
                
                kalman_ground_truth.append({
                    'time': gt.time - t0,
                    'targetID': gt.target_id,
                    'position': gt.position.tolist()
                })
        
        # Sort by time
        kalman_detections.sort(key=lambda x: x['time'])
        kalman_ground_truth.sort(key=lambda x: x['time'])
        
        return {
            'detections': kalman_detections,
            'ground_truth': kalman_ground_truth,
            'scenario': scenario,
            'description': SCENARIOS.get(scenario, 'Unknown scenario'),
            'num_targets': len(set(gt['targetID'] for gt in kalman_ground_truth)),
            'duration_seconds': kalman_detections[-1]['time'] if kalman_detections else 0,
            'piren_origin_lla': PIREN_ORIGIN_LLA
        }
    
    def get_scenario_info(self, scenario: str) -> Dict:
        """Get information about a scenario"""
        detections, ground_truths = self.load_scenario(scenario)
        
        # Count detections by sensor
        sensor_counts = {}
        for det in detections:
            sensor_counts[det.sensor_name] = sensor_counts.get(det.sensor_name, 0) + 1
        
        # Get target IDs
        target_ids = set()
        for gt_frame in ground_truths:
            for gt in gt_frame:
                target_ids.add(gt.target_id)
        
        # Time range
        times = [det.time for det in detections]
        
        return {
            'scenario': scenario,
            'description': SCENARIOS.get(scenario, 'Unknown'),
            'num_detections': len(detections),
            'detections_by_sensor': sensor_counts,
            'num_targets': len(target_ids),
            'target_ids': list(target_ids),
            'target_names': [TARGET_NAMES.get(tid, f'Target_{tid}') for tid in target_ids],
            'time_range_seconds': max(times) - min(times) if times else 0,
            'start_time': min(times) if times else 0
        }
    
    def list_scenarios(self) -> List[Dict]:
        """List all available scenarios with info"""
        return [self.get_scenario_info(s) for s in self.available_scenarios]


def load_autoferry_data(scenario: str = 'scenario2') -> Dict:
    """
    Convenience function to load Autoferry data.
    
    Args:
        scenario: Scenario name (default: 'scenario2')
        
    Returns:
        Data dict compatible with Kalman filter agent
    """
    # Find dataset path relative to this file
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Autoferry dataset not found at {dataset_path}. "
            "Please run: git clone https://github.com/Autoferry/sensor_fusion_dataset.git"
        )
    
    loader = AutoferryDataLoader(dataset_path)
    return loader.load_for_kalman_filter(scenario)


if __name__ == '__main__':
    # Test the loader
    import pprint
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    print("=" * 60)
    print("Autoferry Sensor Fusion Dataset Loader Test")
    print("=" * 60)
    
    loader = AutoferryDataLoader(dataset_path)
    
    print(f"\nAvailable scenarios: {loader.available_scenarios}")
    
    print("\n" + "-" * 60)
    for scenario in loader.available_scenarios[:3]:
        info = loader.get_scenario_info(scenario)
        print(f"\n{scenario}: {info['description']}")
        print(f"  Detections: {info['num_detections']}")
        print(f"  Targets: {info['target_names']}")
        print(f"  Duration: {info['time_range_seconds']:.1f}s")
        print(f"  By sensor: {info['detections_by_sensor']}")
    
    print("\n" + "-" * 60)
    print("\nLoading scenario2 for Kalman filter...")
    data = loader.load_for_kalman_filter('scenario2')
    print(f"  Kalman-compatible detections: {len(data['detections'])}")
    print(f"  Ground truth entries: {len(data['ground_truth'])}")
    print(f"  Duration: {data['duration_seconds']:.1f}s")
    print(f"\n  First detection:")
    pprint.pprint(data['detections'][0])
    print(f"\n  First ground truth:")
    pprint.pprint(data['ground_truth'][0])
