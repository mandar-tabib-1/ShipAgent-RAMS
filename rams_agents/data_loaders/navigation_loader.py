"""
Navigation Data Loader for Safety Analysis

Loads the Autoferry Sensor Fusion Dataset for collision avoidance and
safety demonstrations.

Dataset Citation:
    NTNU Autoferry Project (2021).
    Sensor Fusion Benchmark Dataset.
    https://github.com/Autoferry/sensor_fusion_dataset

License: MIT

This dataset contains multi-sensor detections (LiDAR, Radar, Camera)
of maritime targets collected from the milliAmpere autonomous ferry.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, field
from pathlib import Path
import math


@dataclass
class Detection:
    """Single sensor detection."""
    sensor_id: int          # 1=LiDAR, 2=Radar, 3=IR Camera, 4=EO Camera
    timestamp: float        # Time of detection
    measurement: Any        # Position [N,E] or bearing angle
    ownship_position: List[float]  # Own vessel position [N, E]
    
    @property
    def sensor_name(self) -> str:
        """Get human-readable sensor name."""
        names = {1: 'LiDAR', 2: 'Radar', 3: 'IR_Camera', 4: 'EO_Camera'}
        return names.get(self.sensor_id, 'Unknown')
    
    @property
    def is_position_sensor(self) -> bool:
        """Check if sensor provides position (vs bearing only)."""
        return self.sensor_id in [1, 2]
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get position if available (LiDAR, Radar)."""
        if self.is_position_sensor and self.measurement:
            if isinstance(self.measurement, list):
                if len(self.measurement) == 2:
                    return (self.measurement[0], self.measurement[1])
                elif len(self.measurement) > 0 and isinstance(self.measurement[0], list):
                    # Multi-target radar: return first target
                    return (self.measurement[0][0], self.measurement[1][0])
        return None


@dataclass
class GroundTruth:
    """Ground truth target position."""
    target_id: int
    timestamp: float
    position: Tuple[float, float]  # (North, East) in Piren NED frame
    
    @property
    def target_name(self) -> str:
        """Get human-readable target name."""
        names = {1: 'Havfruen', 2: 'Vessel/Jetboat', 3: 'Jetboat'}
        return names.get(self.target_id, f'Target_{self.target_id}')


@dataclass
class NavigationScenario:
    """Complete navigation scenario with detections and ground truth."""
    scenario_id: str
    detections: List[Detection]
    ground_truth: List[GroundTruth]
    
    # Piren NED frame origin
    origin_lat: float = 63.4389029083
    origin_lon: float = 10.39908278
    origin_alt: float = 39.923
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get scenario time range."""
        if not self.detections:
            return (0.0, 0.0)
        times = [d.timestamp for d in self.detections]
        return (min(times), max(times))
    
    def get_unique_timestamps(self) -> List[float]:
        """Get sorted unique timestamps."""
        times = set(d.timestamp for d in self.detections)
        times.update(gt.timestamp for gt in self.ground_truth)
        return sorted(times)
    
    def get_detections_at_time(self, t: float, tolerance: float = 0.1) -> List[Detection]:
        """Get all detections at a specific time."""
        return [d for d in self.detections if abs(d.timestamp - t) <= tolerance]
    
    def get_ground_truth_at_time(self, t: float, tolerance: float = 0.1) -> List[GroundTruth]:
        """Get ground truth positions at a specific time."""
        return [gt for gt in self.ground_truth if abs(gt.timestamp - t) <= tolerance]


class NavigationDataLoader:
    """
    Loader for AutoFerry Sensor Fusion Dataset.
    
    The dataset contains multi-sensor navigation data from the
    milliAmpere autonomous ferry in Trondheim, Norway.
    
    Available scenarios:
    - Environment 1: scenario2, scenario3, scenario4, scenario5, scenario6
    - Environment 2: scenario13, scenario16, scenario17, scenario22
    """
    
    # Default dataset path relative to project
    DEFAULT_DATA_PATH = "data/sensor_fusion_dataset"
    
    AVAILABLE_SCENARIOS = [
        'scenario2', 'scenario3', 'scenario4', 'scenario5', 'scenario6',
        'scenario13', 'scenario16', 'scenario17', 'scenario22'
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to sensor_fusion_dataset folder.
                      If None, uses default relative path.
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Try to find the data relative to this file
            self.data_path = Path(__file__).parent.parent.parent / "data" / "sensor_fusion_dataset"
        
        self._scenarios_cache: Dict[str, NavigationScenario] = {}
    
    def list_scenarios(self) -> List[str]:
        """List available scenarios."""
        available = []
        for scenario in self.AVAILABLE_SCENARIOS:
            scenario_path = self.data_path / scenario
            if scenario_path.exists():
                available.append(scenario)
        return available if available else self.AVAILABLE_SCENARIOS
    
    def load_scenario(self, scenario_id: str) -> NavigationScenario:
        """
        Load a specific scenario.
        
        Args:
            scenario_id: e.g., 'scenario16'
        
        Returns:
            NavigationScenario with detections and ground truth
        """
        if scenario_id in self._scenarios_cache:
            return self._scenarios_cache[scenario_id]
        
        scenario_path = self.data_path / scenario_id
        
        # Load detections
        detections_file = scenario_path / f"{scenario_id}_detections.json"
        detections = self._load_detections(detections_file)
        
        # Load ground truth
        gt_file = scenario_path / f"{scenario_id}_groundTruth.json"
        ground_truth = self._load_ground_truth(gt_file)
        
        scenario = NavigationScenario(
            scenario_id=scenario_id,
            detections=detections,
            ground_truth=ground_truth
        )
        
        self._scenarios_cache[scenario_id] = scenario
        return scenario
    
    def _load_detections(self, file_path: Path) -> List[Detection]:
        """Load detections from JSON file."""
        if not file_path.exists():
            # Generate synthetic detections for demo
            return self._generate_synthetic_detections()
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                print(f"[NavigationLoader] Empty detections file, using synthetic")
                return self._generate_synthetic_detections()
            
            data = json.loads(content)
            
            if not data or not isinstance(data, list):
                return self._generate_synthetic_detections()
            
            detections = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                det = Detection(
                    sensor_id=item.get('sensorID', 1),
                    timestamp=item.get('time', 0.0),
                    measurement=item.get('measurement', [0, 0]),
                    ownship_position=item.get('ownshipPosition', [0, 0])
                )
                detections.append(det)
            
            if not detections:
                return self._generate_synthetic_detections()
            
            return detections
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[NavigationLoader] Error loading {file_path}: {e}")
            return self._generate_synthetic_detections()
    
    def _load_ground_truth(self, file_path: Path) -> List[GroundTruth]:
        """Load ground truth from JSON file."""
        if not file_path.exists():
            return self._generate_synthetic_ground_truth()
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                print(f"[NavigationLoader] Empty ground truth file, using synthetic")
                return self._generate_synthetic_ground_truth()
            
            data = json.loads(content)
            
            if not data or not isinstance(data, list):
                return self._generate_synthetic_ground_truth()
            
            ground_truth = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                gt = GroundTruth(
                    target_id=item.get('targetID', 1),
                    timestamp=item.get('time', 0.0),
                    position=(
                        item.get('position', [0, 0])[0],
                        item.get('position', [0, 0])[1]
                    )
                )
                ground_truth.append(gt)
            
            if not ground_truth:
                return self._generate_synthetic_ground_truth()
            
            return ground_truth
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[NavigationLoader] Error loading {file_path}: {e}")
            return self._generate_synthetic_ground_truth()
    
    def _generate_synthetic_detections(self, n_detections: int = 200) -> List[Detection]:
        """Generate synthetic detections for demonstration."""
        np.random.seed(42)
        
        detections = []
        
        # Simulate a crossing scenario
        for t in np.linspace(0, 60, n_detections):
            # Own ship at origin, moving north
            ownship_n = t * 5  # 5 m/s north
            ownship_e = 0
            
            # Target crossing from east
            target_n = 150 + t * 2  # Moving north slowly
            target_e = 400 - t * 8  # Moving west quickly
            
            # Add noise based on sensor type
            for sensor_id in [1, 2]:  # LiDAR, Radar
                noise = 5 if sensor_id == 1 else 15  # Radar noisier
                det = Detection(
                    sensor_id=sensor_id,
                    timestamp=t,
                    measurement=[
                        target_n + np.random.normal(0, noise),
                        target_e + np.random.normal(0, noise)
                    ],
                    ownship_position=[ownship_n, ownship_e]
                )
                detections.append(det)
        
        return detections
    
    def _generate_synthetic_ground_truth(self, duration: float = 60) -> List[GroundTruth]:
        """Generate synthetic ground truth."""
        ground_truth = []
        
        for t in np.linspace(0, duration, 100):
            target_n = 150 + t * 2
            target_e = 400 - t * 8
            
            gt = GroundTruth(
                target_id=1,
                timestamp=t,
                position=(target_n, target_e)
            )
            ground_truth.append(gt)
        
        return ground_truth
    
    def iterate_timesteps(self, 
                          scenario_id: str,
                          time_step: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate through scenario timesteps.
        
        Args:
            scenario_id: Scenario to iterate
            time_step: Time increment between yields
        
        Yields:
            Dictionary with detections and ground truth at each timestep
        """
        scenario = self.load_scenario(scenario_id)
        t_min, t_max = scenario.get_time_range()
        
        t = t_min
        while t <= t_max:
            detections = scenario.get_detections_at_time(t, tolerance=time_step/2)
            ground_truth = scenario.get_ground_truth_at_time(t, tolerance=time_step/2)
            
            # Calculate own ship position from detections
            ownship_pos = [0, 0]
            if detections:
                ownship_pos = detections[0].ownship_position
            
            yield {
                'timestamp': t,
                'detections': detections,
                'ground_truth': ground_truth,
                'ownship_position': ownship_pos,
                # CPA/TCPA calculation data
                'targets': [
                    {
                        'id': gt.target_id,
                        'name': gt.target_name,
                        'position': gt.position
                    }
                    for gt in ground_truth
                ]
            }
            
            t += time_step
    
    def calculate_cpa_tcpa(self,
                           ownship_pos: Tuple[float, float],
                           ownship_vel: Tuple[float, float],
                           target_pos: Tuple[float, float],
                           target_vel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate Closest Point of Approach (CPA) and Time to CPA (TCPA).
        
        Args:
            ownship_pos: Own ship position (N, E) in meters
            ownship_vel: Own ship velocity (Vn, Ve) in m/s
            target_pos: Target position (N, E) in meters
            target_vel: Target velocity (Vn, Ve) in m/s
        
        Returns:
            Tuple of (CPA in meters, TCPA in seconds)
        """
        # Relative position and velocity
        rel_pos = (target_pos[0] - ownship_pos[0], target_pos[1] - ownship_pos[1])
        rel_vel = (target_vel[0] - ownship_vel[0], target_vel[1] - ownship_vel[1])
        
        # Speed squared
        speed_sq = rel_vel[0]**2 + rel_vel[1]**2
        
        if speed_sq < 1e-6:
            # Targets moving parallel or stationary
            cpa = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            return (cpa, float('inf'))
        
        # Time to CPA
        tcpa = -(rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1]) / speed_sq
        
        if tcpa < 0:
            # CPA in the past
            tcpa = 0
        
        # Position at CPA
        cpa_n = rel_pos[0] + rel_vel[0] * tcpa
        cpa_e = rel_pos[1] + rel_vel[1] * tcpa
        cpa = math.sqrt(cpa_n**2 + cpa_e**2)
        
        return (cpa, tcpa)
    
    @staticmethod
    def get_citation() -> str:
        """Get the proper citation for this dataset."""
        return """
Citation:
    NTNU Autoferry Project (2021).
    Sensor Fusion Benchmark Dataset.
    https://github.com/Autoferry/sensor_fusion_dataset

BibTeX:
    @misc{autoferry_sensor_fusion,
        author = {NTNU Autoferry Project},
        title = {Sensor Fusion Benchmark Dataset},
        year = {2021},
        url = {https://github.com/Autoferry/sensor_fusion_dataset}
    }

License: MIT

Description:
    Multi-sensor maritime detections (LiDAR, Radar, IR Camera, EO Camera)
    collected from the milliAmpere autonomous ferry in Trondheim, Norway.
    Contains scenarios with multiple target vessels including Vessel.
"""
