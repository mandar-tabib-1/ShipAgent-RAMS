# Data Directory - Vessel AI System

This directory contains data loaders and datasets for the Vessel AI System.

## Datasets

### Autoferry Sensor Fusion Dataset

**Source:** https://github.com/Autoferry/sensor_fusion_dataset

**Citation:**
> Brekke, E. F., et al. "Heterogeneous multi-sensor tracking for an autonomous 
> surface vehicle in a littoral environment." Ocean Engineering, 252 (2022): 111168.

**Installation:**
```bash
cd data
git clone https://github.com/Autoferry/sensor_fusion_dataset.git
```

**Description:**
Real-world multi-sensor target tracking data collected on the milliAmpere autonomous 
passenger ferry in Trondheim, Norway. Contains synchronized detections from:

| Sensor ID | Type | Measurement | Range |
|-----------|------|-------------|-------|
| 1 | LiDAR | [north, east] position | ~100m |
| 2 | Radar | [north, east] multi-target | ~500m |
| 3 | IR Camera | bearing angle(s) | ~200m |
| 4 | EO Camera | bearing angle(s) | ~200m |

**Scenarios:**

| Scenario | Environment | Description |
|----------|-------------|-------------|
| scenario2 | May 4, 2021 | Crossing maneuver - Havfruen behind Vessel |
| scenario3 | May 4, 2021 | Havfruen maneuvers around Vessel |
| scenario4 | May 4, 2021 | Side by side approach, Vessel maneuvers north |
| scenario5 | May 4, 2021 | High-speed overtake in sensor shadow |
| scenario6 | May 4, 2021 | Both targets travel toward milliAmpere |
| scenario13 | May 5, 2021 | Targets approach from each side (docked) |
| scenario16 | May 5, 2021 | Targets approach from each side (stationary) |
| scenario17 | May 5, 2021 | Both targets approach side by side from east |
| scenario22 | May 5, 2021 | Channel crossing with target breaking off |

**Targets:**
- **Havfruen** (Target ID: 1): Leisure craft
- **Vessel** (Target ID: 2): Vessel research vessel
- **Jetboat** (Target ID: 3): Fast maneuvering boat

**Coordinate System:**
All positions are in Piren NED (North-East-Down) frame with origin at:
- Latitude: 63.4389029083°
- Longitude: 10.39908278°
- Altitude: 39.923m (above WGS84 ellipsoid)

## Usage

### Load Real Data

```python
from data.autoferry_loader import AutoferryDataLoader

# Initialize loader
loader = AutoferryDataLoader('data/sensor_fusion_dataset')

# List available scenarios
for info in loader.list_scenarios():
    print(f"{info['scenario']}: {info['description']}")

# Load data for Kalman filter
data = loader.load_for_kalman_filter('scenario2')
print(f"Detections: {len(data['detections'])}")
print(f"Ground truth: {len(data['ground_truth'])}")
```

### Run Demo with Real Data

```bash
# Synthetic data (default)
python main_demo.py

# Real Autoferry data
python main_demo.py --real

# Specific scenario
python main_demo.py --real --scenario scenario16
```

### Data Format

**Kalman Filter Format:**
```python
{
    'detections': [
        {'time': 0.0, 'sensorID': 1, 'targetID': 0, 'position': [north, east, 0.0]},
        ...
    ],
    'ground_truth': [
        {'time': 0.0, 'targetID': 1, 'position': [north, east, down]},
        ...
    ],
    'scenario': 'scenario2',
    'description': 'Crossing maneuver - Havfruen behind Vessel',
    'num_targets': 2,
    'duration_seconds': 117.8
}
```

## License

The Autoferry dataset is provided under the terms specified in the original repository.
Data is for research purposes. Please cite the original paper when using this data.
