"""
Vessel Configuration
==========================
Real vessel specifications for NTNU's Research Vessel.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VesselSpecs:
    """Vessel Technical Specifications"""
    name: str = "Vessel"
    mmsi: int = 258342000
    imo: int = 9371361
    call_sign: str = "LNVZ"
    
    # Dimensions (meters)
    length_overall: float = 36.25
    length_pp: float = 33.90
    breadth: float = 9.90
    draught: float = 2.70
    deadweight: float = 72.0  # tonnes
    
    # Classification
    class_notation: str = "+1A1 + Ice C + E0 + R2"


@dataclass
class DieselElectricSystem:
    """Diesel-Electric Propulsion System"""
    num_generators: int = 2
    generator_power_kw: float = 500.0
    total_power_kw: float = 1000.0
    voltage: float = 690  # V AC
    
    # Azimuth Thrusters (PM type, Rolls-Royce)
    num_azimuth_thrusters: int = 2
    azimuth_thruster_power_kw: float = 500
    
    # Tunnel Thruster
    num_tunnel_thrusters: int = 1
    tunnel_thruster_power_kw: float = 200.0
    
    # Performance
    max_speed_knots: float = 10.0


@dataclass
class OperationalLimits:
    """Operational Thresholds for Maintenance"""
    engine_temp_warning_c: float = 85.0
    engine_temp_critical_c: float = 95.0
    vibration_warning_g: float = 0.5
    vibration_critical_g: float = 1.0
    generator_overload_pct: float = 95.0
    thruster_temp_warning_c: float = 70.0


# Sensor Variables (OSP FMU format)
SENSOR_VARIABLES = {
    'vessel_state': [
        'position_north_m', 'position_east_m', 'heading_rad',
        'surge_velocity_ms', 'sway_velocity_ms', 'yaw_rate_rads',
        'roll_deg', 'pitch_deg'
    ],
    'azimuth_thruster': [
        'rpm', 'azimuth_angle_deg', 'thrust_force_n', 
        'torque_nm', 'power_kw', 'motor_temp_c'
    ],
    'tunnel_thruster': [
        'rpm', 'thrust_force_n', 'power_kw'
    ],
    'power_system': [
        'bus_voltage_v', 'bus_frequency_hz', 'total_power_kw',
        'generator_1_load_pct', 'generator_2_load_pct'
    ],
    'dp_controller': [
        'setpoint_north_m', 'setpoint_east_m', 'setpoint_yaw_rad',
        'error_north_m', 'error_east_m', 'error_yaw_rad'
    ],
    'environment': [
        'current_velocity_x_ms', 'current_velocity_y_ms',
        'wind_speed_ms', 'wind_direction_deg'
    ]
}

# Autoferry Dataset Reference
AUTOFERRY_DATA = {
    'source': 'https://github.com/Autoferry/sensor_fusion_dataset',
    'vessel_target_id': 2,
    'recording_date': '2021-05-04',
    'piren_origin_lla': [63.4389029083, 10.39908278, 39.923],
    'sensors': {'lidar': 1, 'radar': 2, 'ir': 3, 'eo': 4}
}


def get_config():
    return {
        'specs': VesselSpecs(),
        'propulsion': DieselElectricSystem(),
        'limits': OperationalLimits(),
        'sensors': SENSOR_VARIABLES,
        'autoferry': AUTOFERRY_DATA
    }
