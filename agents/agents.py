"""
Vessel AI System - Agent Module
=====================================
Complete agent implementations for Vessel predictive maintenance
and navigation system using real vessel specifications.

Agents:
1. SensorFusionAgent - Multi-sensor tracking (Autoferry dataset format)
2. PropulsionAgent - Diesel-electric propulsion monitoring  
3. DPAgent - Dynamic Positioning system monitoring
4. PowerSystemAgent - Power management and generator monitoring
5. MaintenanceAgent - Predictive maintenance planning
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# ============================================================================
# BASE CLASSES
# ============================================================================

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'source': self.source,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class VesselBaseAgent(ABC):
    """Base class for all Vessel agents"""
    
    def __init__(self, name: str, description: str, domain: str):
        self.name = name
        self.description = description
        self.domain = domain
        self.status = AgentStatus.IDLE
        self.progress = 0.0
        self.logger = logging.getLogger(f"Vessel.{name}")
        self.alerts: List[Alert] = []
        
    @abstractmethod
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        pass
    
    def create_alert(self, level: AlertLevel, message: str, details: Dict = None):
        alert = Alert(level=level, source=self.name, message=message, details=details or {})
        self.alerts.append(alert)
        icons = {AlertLevel.EMERGENCY: "🚨", AlertLevel.CRITICAL: "⚠️", 
                 AlertLevel.WARNING: "⚡", AlertLevel.INFO: "ℹ️"}
        self.logger.log(
            logging.CRITICAL if level == AlertLevel.EMERGENCY else
            logging.ERROR if level == AlertLevel.CRITICAL else
            logging.WARNING if level == AlertLevel.WARNING else logging.INFO,
            f"{icons.get(level, '')} {message}"
        )
        return alert
    
    def get_alerts(self) -> List[Dict]:
        return [a.to_dict() for a in self.alerts]


# ============================================================================
# SENSOR FUSION AGENT (Navigation/Tracking)
# ============================================================================

class SensorFusionAgent(VesselBaseAgent):
    """
    Processes multi-sensor tracking data in Autoferry benchmark format.
    Sensors: LiDAR (1), Radar (2), IR Camera (3), EO Camera (4)
    """
    
    SENSOR_NAMES = {1: 'LiDAR', 2: 'Radar', 3: 'IR_Camera', 4: 'EO_Camera'}
    PIREN_ORIGIN = [63.4389029083, 10.39908278, 39.923]  # LLA
    
    def __init__(self):
        super().__init__("SensorFusionAgent", "Multi-sensor tracking data processor", "navigation")
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info("Processing sensor fusion data")
        self.status = AgentStatus.PROCESSING
        
        try:
            detections = data.get('detections', []) if isinstance(data, dict) else []
            ground_truth = data.get('ground_truth', []) if isinstance(data, dict) else []
            
            # Process by sensor type
            sensor_stats = {}
            for sensor_id, name in self.SENSOR_NAMES.items():
                sensor_dets = [d for d in detections if d.get('sensorID') == sensor_id]
                sensor_stats[name] = {
                    'count': len(sensor_dets),
                    'rate_hz': self._compute_rate(sensor_dets)
                }
            
            # Process target tracks from ground truth
            tracks = {}
            target_names = {1: 'Havfruen', 2: 'Vessel', 3: 'Jetboat'}
            
            for gt in ground_truth:
                tid = gt.get('targetID', 0)
                if tid not in tracks:
                    tracks[tid] = {'name': target_names.get(tid, f'Target_{tid}'), 'positions': []}
                tracks[tid]['positions'].append({
                    'time': gt.get('time', 0),
                    'north': gt.get('position', [0,0,0])[0],
                    'east': gt.get('position', [0,0,0])[1]
                })
            
            # Extract Vessel track (target_id=2)
            vessel_track = self._extract_vessel(tracks.get(2, {}))
            
            self.status = AgentStatus.COMPLETED
            self.logger.info(f"Processed {len(detections)} detections, {len(tracks)} targets")
            
            return {
                'status': 'success',
                'num_detections': len(detections),
                'sensor_statistics': sensor_stats,
                'num_targets': len(tracks),
                'vessel_track': vessel_track,
                'coordinate_frame': 'Piren NED',
                'alerts': self.get_alerts()
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.create_alert(AlertLevel.CRITICAL, f"Sensor fusion failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _compute_rate(self, detections: List[Dict]) -> float:
        if len(detections) < 2:
            return 0.0
        times = [d.get('time', 0) for d in detections]
        duration = max(times) - min(times)
        return len(detections) / duration if duration > 0 else 0
    
    def _extract_vessel(self, track: Dict) -> Dict:
        positions = track.get('positions', [])
        if not positions:
            return {'found': False}
        
        positions.sort(key=lambda x: x['time'])
        
        # Compute distance and speed
        total_dist = 0
        speeds = []
        for i in range(1, len(positions)):
            dn = positions[i]['north'] - positions[i-1]['north']
            de = positions[i]['east'] - positions[i-1]['east']
            dt = positions[i]['time'] - positions[i-1]['time']
            dist = math.sqrt(dn**2 + de**2)
            total_dist += dist
            if dt > 0:
                speeds.append(dist / dt)
        
        return {
            'found': True,
            'name': 'Vessel',
            'num_positions': len(positions),
            'total_distance_m': total_dist,
            'avg_speed_ms': np.mean(speeds) if speeds else 0,
            'avg_speed_knots': np.mean(speeds) * 1.944 if speeds else 0
        }


# ============================================================================
# PROPULSION AGENT (Diesel-Electric System)
# ============================================================================

class PropulsionAgent(VesselBaseAgent):
    """
    Monitors Vessel diesel-electric propulsion system:
    - 2x Azimuth thrusters (Rolls-Royce PM, 500kW each)
    - 1x Tunnel thruster (200kW)
    - 2x Diesel generators (500kW each)
    """
    
    # Thresholds based on Vessel specs
    THRESHOLDS = {
        'thruster_temp_warning': 70.0,  # °C
        'thruster_temp_critical': 85.0,
        'motor_overload_pct': 95.0,
        'vibration_warning': 0.5,  # g
        'vibration_critical': 1.0,
        'efficiency_warning': 0.75  # ratio
    }
    
    def __init__(self):
        super().__init__("PropulsionAgent", "Diesel-electric propulsion monitoring", "propulsion")
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info("Processing propulsion system data")
        self.status = AgentStatus.PROCESSING
        
        try:
            df = pd.DataFrame(data) if isinstance(data, (dict, list)) else data
            if not isinstance(df, pd.DataFrame):
                df = self._generate_synthetic_data()
            
            # Analyze azimuth thrusters
            az1_health = self._analyze_thruster(df, 'az1')
            az2_health = self._analyze_thruster(df, 'az2')
            tunnel_health = self._analyze_thruster(df, 'tunnel')
            
            # Overall propulsion health
            thruster_scores = [az1_health['health_score'], az2_health['health_score'], 
                             tunnel_health['health_score']]
            overall_health = np.mean(thruster_scores)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(df)
            
            # Generate faults
            faults = self._identify_faults(az1_health, az2_health, tunnel_health)
            
            self.status = AgentStatus.COMPLETED
            
            return {
                'status': 'success',
                'azimuth_thruster_1': az1_health,
                'azimuth_thruster_2': az2_health,
                'tunnel_thruster': tunnel_health,
                'overall_health': overall_health,
                'health_status': 'good' if overall_health > 80 else 'warning' if overall_health > 60 else 'critical',
                'anomalies_detected': anomalies,
                'faults': faults,
                'alerts': self.get_alerts()
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'status': 'error', 'error': str(e)}
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic propulsion data based on Vessel specs"""
        np.random.seed(42)
        
        # Simulate operating at various loads
        load_pct = np.random.uniform(20, 100, n_samples)
        
        data = {
            # Azimuth Thruster 1
            'az1_rpm': 200 + load_pct * 2 + np.random.normal(0, 5, n_samples),
            'az1_azimuth_deg': np.random.uniform(-180, 180, n_samples),
            'az1_power_kw': load_pct * 5 + np.random.normal(0, 10, n_samples),
            'az1_temp_c': 30 + load_pct * 0.4 + np.random.normal(0, 2, n_samples),
            'az1_vibration_g': 0.1 + load_pct * 0.003 + np.random.normal(0, 0.05, n_samples),
            
            # Azimuth Thruster 2
            'az2_rpm': 200 + load_pct * 2 + np.random.normal(0, 5, n_samples),
            'az2_azimuth_deg': np.random.uniform(-180, 180, n_samples),
            'az2_power_kw': load_pct * 5 + np.random.normal(0, 10, n_samples),
            'az2_temp_c': 30 + load_pct * 0.4 + np.random.normal(0, 2, n_samples),
            'az2_vibration_g': 0.1 + load_pct * 0.003 + np.random.normal(0, 0.05, n_samples),
            
            # Tunnel Thruster
            'tunnel_rpm': np.where(np.random.random(n_samples) > 0.7, 
                                   300 + np.random.normal(0, 20, n_samples), 0),
            'tunnel_power_kw': np.where(np.random.random(n_samples) > 0.7,
                                        100 + np.random.normal(0, 10, n_samples), 0),
            'tunnel_temp_c': 25 + np.random.normal(0, 3, n_samples),
            
            'load_pct': load_pct
        }
        
        # Add some degradation for testing
        data['az1_temp_c'][-200:] += 15  # Simulated overheating
        
        return pd.DataFrame(data)
    
    def _analyze_thruster(self, df: pd.DataFrame, thruster: str) -> Dict:
        """Analyze individual thruster health"""
        prefix = f'{thruster}_'
        
        # Get relevant columns
        temp_col = f'{prefix}temp_c'
        vib_col = f'{prefix}vibration_g'
        power_col = f'{prefix}power_kw'
        
        health_score = 100.0
        issues = []
        
        if temp_col in df.columns:
            max_temp = df[temp_col].max()
            avg_temp = df[temp_col].mean()
            
            if max_temp > self.THRESHOLDS['thruster_temp_critical']:
                health_score -= 30
                issues.append(f'Critical temperature: {max_temp:.1f}°C')
                self.create_alert(AlertLevel.CRITICAL, f"{thruster} critical temperature: {max_temp:.1f}°C")
            elif max_temp > self.THRESHOLDS['thruster_temp_warning']:
                health_score -= 15
                issues.append(f'High temperature: {max_temp:.1f}°C')
                self.create_alert(AlertLevel.WARNING, f"{thruster} high temperature: {max_temp:.1f}°C")
        else:
            avg_temp = max_temp = None
        
        if vib_col in df.columns:
            max_vib = df[vib_col].max()
            if max_vib > self.THRESHOLDS['vibration_critical']:
                health_score -= 25
                issues.append(f'Critical vibration: {max_vib:.2f}g')
            elif max_vib > self.THRESHOLDS['vibration_warning']:
                health_score -= 10
                issues.append(f'High vibration: {max_vib:.2f}g')
        else:
            max_vib = None
        
        if power_col in df.columns:
            avg_power = df[power_col].mean()
            max_power = df[power_col].max()
        else:
            avg_power = max_power = None
        
        return {
            'thruster_id': thruster,
            'health_score': max(0, health_score),
            'avg_temperature_c': float(avg_temp) if avg_temp is not None else None,
            'max_temperature_c': float(max_temp) if max_temp is not None else None,
            'max_vibration_g': float(max_vib) if max_vib is not None else None,
            'avg_power_kw': float(avg_power) if avg_power is not None else None,
            'issues': issues,
            'status': 'good' if health_score > 80 else 'warning' if health_score > 60 else 'critical'
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Simple anomaly detection"""
        anomalies = {'count': 0, 'sensors': []}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > len(df) * 0.05:
                anomalies['count'] += outliers
                anomalies['sensors'].append({'sensor': col, 'outliers': int(outliers)})
        
        return anomalies
    
    def _identify_faults(self, az1: Dict, az2: Dict, tunnel: Dict) -> List[Dict]:
        """Identify faults based on thruster analysis"""
        faults = []
        
        for thruster in [az1, az2, tunnel]:
            if thruster['status'] == 'critical':
                faults.append({
                    'component': thruster['thruster_id'],
                    'severity': 'critical',
                    'issues': thruster['issues'],
                    'recommended_action': 'Immediate inspection required'
                })
            elif thruster['status'] == 'warning':
                faults.append({
                    'component': thruster['thruster_id'],
                    'severity': 'warning',
                    'issues': thruster['issues'],
                    'recommended_action': 'Schedule maintenance within 48 hours'
                })
        
        return faults


# ============================================================================
# DYNAMIC POSITIONING AGENT
# ============================================================================

class DPAgent(VesselBaseAgent):
    """
    Monitors Vessel Dynamic Positioning System (Kongsberg SDP-11).
    Tracks position/heading errors and control performance.
    """
    
    THRESHOLDS = {
        'position_error_warning': 2.0,  # meters
        'position_error_critical': 5.0,
        'heading_error_warning': 5.0,  # degrees
        'heading_error_critical': 10.0
    }
    
    def __init__(self):
        super().__init__("DPAgent", "DP system monitoring", "dp")
        self.setpoint = {'north': 0.0, 'east': 0.0, 'heading': 45.0}
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info("Processing DP system data")
        self.status = AgentStatus.PROCESSING
        
        try:
            # Handle vessel state dict input from orchestrator
            if isinstance(data, dict) and 'north' in data:
                vessel_state = data
                
                # Get setpoint from context
                if context and 'setpoint' in context:
                    self.setpoint = context['setpoint']
                
                # Calculate position error
                error_n = vessel_state.get('north', 0) - self.setpoint.get('north', 0)
                error_e = vessel_state.get('east', 0) - self.setpoint.get('east', 0)
                error_heading = vessel_state.get('heading', 0) - self.setpoint.get('heading', 0)
                
                # Normalize heading error to -180 to 180
                while error_heading > 180:
                    error_heading -= 360
                while error_heading < -180:
                    error_heading += 360
                
                total_error = math.sqrt(error_n**2 + error_e**2)
                
                # Determine status
                if total_error > self.THRESHOLDS['position_error_critical']:
                    pos_status = 'critical'
                    self.create_alert(AlertLevel.CRITICAL, f"Position error critical: {total_error:.1f}m")
                elif total_error > self.THRESHOLDS['position_error_warning']:
                    pos_status = 'warning'
                else:
                    pos_status = 'good'
                
                dp_performance = max(0, 100 - total_error * 10 - abs(error_heading) * 2)
                
                self.status = AgentStatus.COMPLETED
                
                return {
                    'status': 'success',
                    'dp_mode': context.get('mode', 'station_keeping') if context else 'station_keeping',
                    'vessel_state': vessel_state,
                    'setpoint': self.setpoint,
                    'position_error': {
                        'north': error_n,
                        'east': error_e,
                        'total_m': total_error,
                        'heading_deg': error_heading
                    },
                    'position_status': pos_status,
                    'dp_performance_index': dp_performance,
                    'dp_status': 'operational' if dp_performance > 70 else 'degraded',
                    'alerts': self.get_alerts()
                }
            
            # Handle DataFrame input for batch analysis
            df = pd.DataFrame(data) if isinstance(data, (dict, list)) else data
            if not isinstance(df, pd.DataFrame):
                df = self._generate_synthetic_dp_data()
            
            # Analyze position keeping
            position_analysis = self._analyze_position_keeping(df)
            
            # Analyze heading keeping
            heading_analysis = self._analyze_heading_keeping(df)
            
            # Compute DP performance index
            dp_performance = self._compute_dp_performance(position_analysis, heading_analysis)
            
            self.status = AgentStatus.COMPLETED
            
            return {
                'status': 'success',
                'position_keeping': position_analysis,
                'heading_keeping': heading_analysis,
                'dp_performance_index': dp_performance,
                'dp_status': 'operational' if dp_performance > 70 else 'degraded',
                'alerts': self.get_alerts()
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'status': 'error', 'error': str(e)}
    
    def _generate_synthetic_dp_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic DP data"""
        np.random.seed(42)
        t = np.linspace(0, 100, n_samples)
        
        return pd.DataFrame({
            'time': t,
            'position_north_m': np.cumsum(np.random.normal(0, 0.1, n_samples)),
            'position_east_m': np.cumsum(np.random.normal(0, 0.1, n_samples)),
            'heading_deg': 45 + np.cumsum(np.random.normal(0, 0.05, n_samples)),
            'setpoint_north_m': np.zeros(n_samples),
            'setpoint_east_m': np.zeros(n_samples),
            'setpoint_heading_deg': np.full(n_samples, 45.0),
            'wind_speed_ms': 5 + np.random.normal(0, 1, n_samples),
            'current_ms': 0.5 + np.random.normal(0, 0.1, n_samples)
        })
    
    def _analyze_position_keeping(self, df: pd.DataFrame) -> Dict:
        """Analyze position keeping performance"""
        if 'position_north_m' not in df.columns:
            return {'error': 'No position data'}
        
        # Compute errors
        error_n = df['position_north_m'] - df.get('setpoint_north_m', 0)
        error_e = df['position_east_m'] - df.get('setpoint_east_m', 0)
        error_magnitude = np.sqrt(error_n**2 + error_e**2)
        
        max_error = float(error_magnitude.max())
        mean_error = float(error_magnitude.mean())
        std_error = float(error_magnitude.std())
        
        # Check thresholds
        status = 'good'
        if max_error > self.THRESHOLDS['position_error_critical']:
            status = 'critical'
            self.create_alert(AlertLevel.CRITICAL, f"DP position error critical: {max_error:.2f}m")
        elif max_error > self.THRESHOLDS['position_error_warning']:
            status = 'warning'
            self.create_alert(AlertLevel.WARNING, f"DP position error high: {max_error:.2f}m")
        
        return {
            'max_error_m': max_error,
            'mean_error_m': mean_error,
            'std_error_m': std_error,
            'within_tolerance': max_error < self.THRESHOLDS['position_error_warning'],
            'status': status
        }
    
    def _analyze_heading_keeping(self, df: pd.DataFrame) -> Dict:
        """Analyze heading keeping performance"""
        if 'heading_deg' not in df.columns:
            return {'error': 'No heading data'}
        
        heading_error = df['heading_deg'] - df.get('setpoint_heading_deg', df['heading_deg'].mean())
        # Normalize to -180 to 180
        heading_error = (heading_error + 180) % 360 - 180
        
        max_error = float(np.abs(heading_error).max())
        mean_error = float(np.abs(heading_error).mean())
        
        status = 'good'
        if max_error > self.THRESHOLDS['heading_error_critical']:
            status = 'critical'
        elif max_error > self.THRESHOLDS['heading_error_warning']:
            status = 'warning'
        
        return {
            'max_error_deg': max_error,
            'mean_error_deg': mean_error,
            'status': status
        }
    
    def _compute_dp_performance(self, position: Dict, heading: Dict) -> float:
        """Compute overall DP performance index (0-100)"""
        score = 100.0
        
        # Position penalties
        if 'max_error_m' in position:
            if position['max_error_m'] > 5:
                score -= 40
            elif position['max_error_m'] > 2:
                score -= 20
            elif position['max_error_m'] > 1:
                score -= 10
        
        # Heading penalties
        if 'max_error_deg' in heading:
            if heading['max_error_deg'] > 10:
                score -= 30
            elif heading['max_error_deg'] > 5:
                score -= 15
        
        return max(0, score)


# ============================================================================
# POWER SYSTEM AGENT
# ============================================================================

class PowerSystemAgent(VesselBaseAgent):
    """
    Monitors Vessel power system:
    - 2x Diesel generators (500kW each, 690V, 60Hz)
    - Power distribution and load balancing
    """
    
    SPECS = {
        'num_generators': 2,
        'generator_power_kw': 500,
        'nominal_voltage': 690,
        'nominal_frequency': 60
    }
    
    THRESHOLDS = {
        'voltage_tolerance_pct': 5,
        'frequency_tolerance_pct': 2,
        'load_warning_pct': 85,
        'load_critical_pct': 95,
        'imbalance_warning_pct': 15
    }
    
    def __init__(self):
        super().__init__("PowerSystemAgent", "Power system monitoring", "power")
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info("Processing power system data")
        self.status = AgentStatus.PROCESSING
        
        try:
            df = pd.DataFrame(data) if isinstance(data, (dict, list)) else data
            if not isinstance(df, pd.DataFrame):
                df = self._generate_synthetic_power_data()
            
            # Analyze generators
            gen1_analysis = self._analyze_generator(df, 1)
            gen2_analysis = self._analyze_generator(df, 2)
            
            # Analyze bus
            bus_analysis = self._analyze_bus(df)
            
            # Load balancing
            balance_analysis = self._analyze_load_balance(gen1_analysis, gen2_analysis)
            
            # Fuel efficiency
            efficiency = self._estimate_fuel_efficiency(df)
            
            self.status = AgentStatus.COMPLETED
            
            return {
                'status': 'success',
                'generator_1': gen1_analysis,
                'generator_2': gen2_analysis,
                'bus': bus_analysis,
                'load_balance': balance_analysis,
                'fuel_efficiency': efficiency,
                'total_power_kw': gen1_analysis.get('power_kw', 0) + gen2_analysis.get('power_kw', 0),
                'alerts': self.get_alerts()
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'status': 'error', 'error': str(e)}
    
    def _generate_synthetic_power_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic power system data"""
        np.random.seed(42)
        
        # Simulate varying load
        load = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 5, n_samples)
        
        return pd.DataFrame({
            'gen1_power_kw': load * 2.5 + np.random.normal(0, 5, n_samples),
            'gen1_load_pct': load + np.random.normal(0, 2, n_samples),
            'gen2_power_kw': load * 2.5 + np.random.normal(0, 5, n_samples),
            'gen2_load_pct': load + np.random.normal(0, 2, n_samples),
            'bus_voltage_v': 690 + np.random.normal(0, 5, n_samples),
            'bus_frequency_hz': 60 + np.random.normal(0, 0.1, n_samples),
            'fuel_rate_lph': 20 + load * 0.3 + np.random.normal(0, 2, n_samples)
        })
    
    def _analyze_generator(self, df: pd.DataFrame, gen_num: int) -> Dict:
        """Analyze individual generator"""
        prefix = f'gen{gen_num}_'
        
        power_col = f'{prefix}power_kw'
        load_col = f'{prefix}load_pct'
        
        analysis = {'generator': gen_num, 'status': 'good'}
        
        if power_col in df.columns:
            analysis['power_kw'] = float(df[power_col].mean())
            analysis['max_power_kw'] = float(df[power_col].max())
        
        if load_col in df.columns:
            avg_load = df[load_col].mean()
            max_load = df[load_col].max()
            analysis['avg_load_pct'] = float(avg_load)
            analysis['max_load_pct'] = float(max_load)
            
            if max_load > self.THRESHOLDS['load_critical_pct']:
                analysis['status'] = 'critical'
                self.create_alert(AlertLevel.CRITICAL, f"Generator {gen_num} overload: {max_load:.1f}%")
            elif max_load > self.THRESHOLDS['load_warning_pct']:
                analysis['status'] = 'warning'
                self.create_alert(AlertLevel.WARNING, f"Generator {gen_num} high load: {max_load:.1f}%")
        
        return analysis
    
    def _analyze_bus(self, df: pd.DataFrame) -> Dict:
        """Analyze main bus"""
        analysis = {'status': 'good'}
        
        if 'bus_voltage_v' in df.columns:
            avg_v = df['bus_voltage_v'].mean()
            analysis['avg_voltage_v'] = float(avg_v)
            deviation = abs(avg_v - self.SPECS['nominal_voltage']) / self.SPECS['nominal_voltage'] * 100
            analysis['voltage_deviation_pct'] = float(deviation)
            
            if deviation > self.THRESHOLDS['voltage_tolerance_pct']:
                analysis['status'] = 'warning'
                self.create_alert(AlertLevel.WARNING, f"Bus voltage deviation: {deviation:.1f}%")
        
        if 'bus_frequency_hz' in df.columns:
            avg_f = df['bus_frequency_hz'].mean()
            analysis['avg_frequency_hz'] = float(avg_f)
            f_deviation = abs(avg_f - self.SPECS['nominal_frequency']) / self.SPECS['nominal_frequency'] * 100
            analysis['frequency_deviation_pct'] = float(f_deviation)
        
        return analysis
    
    def _analyze_load_balance(self, gen1: Dict, gen2: Dict) -> Dict:
        """Analyze load balance between generators"""
        load1 = gen1.get('avg_load_pct', 50)
        load2 = gen2.get('avg_load_pct', 50)
        
        imbalance = abs(load1 - load2)
        
        return {
            'gen1_load_pct': load1,
            'gen2_load_pct': load2,
            'imbalance_pct': imbalance,
            'balanced': imbalance < self.THRESHOLDS['imbalance_warning_pct']
        }
    
    def _estimate_fuel_efficiency(self, df: pd.DataFrame) -> Dict:
        """Estimate fuel efficiency"""
        if 'fuel_rate_lph' not in df.columns:
            return {'available': False}
        
        total_power = df.get('gen1_power_kw', 0) + df.get('gen2_power_kw', 0)
        if isinstance(total_power, (pd.Series, np.ndarray)):
            total_power = total_power.mean()
        
        avg_fuel = df['fuel_rate_lph'].mean()
        
        # Specific fuel consumption (g/kWh) - approximate
        sfc = (avg_fuel * 0.85 * 1000) / (total_power + 1) if total_power > 0 else 0
        
        return {
            'avg_fuel_rate_lph': float(avg_fuel),
            'specific_fuel_consumption': float(sfc),
            'efficiency_rating': 'good' if sfc < 250 else 'fair' if sfc < 300 else 'poor'
        }


# ============================================================================
# MAINTENANCE AGENT
# ============================================================================

class MaintenanceAgent(VesselBaseAgent):
    """
    Predictive maintenance planning for Vessel.
    Integrates analysis from all other agents.
    """
    
    MAINTENANCE_COSTS = {
        'azimuth_thruster_inspection': 5000,
        'azimuth_thruster_repair': 25000,
        'azimuth_thruster_overhaul': 100000,
        'tunnel_thruster_inspection': 2000,
        'generator_service': 8000,
        'dp_calibration': 3000
    }
    
    def __init__(self):
        super().__init__("MaintenanceAgent", "Predictive maintenance planning", "integration")
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info("Generating maintenance plan")
        self.status = AgentStatus.PROCESSING
        
        try:
            # Extract results from other agents
            propulsion = context.get('propulsion', {}) if context else {}
            dp = context.get('dp', {}) if context else {}
            power = context.get('power', {}) if context else {}
            
            # Generate tasks
            tasks = []
            
            # From propulsion
            for thruster_key in ['azimuth_thruster_1', 'azimuth_thruster_2', 'tunnel_thruster']:
                thruster = propulsion.get(thruster_key, {})
                if thruster.get('status') == 'critical':
                    tasks.append({
                        'id': f"MNT-{len(tasks):03d}",
                        'component': thruster_key,
                        'priority': 'critical',
                        'task': 'Emergency inspection and repair',
                        'estimated_cost': self.MAINTENANCE_COSTS.get(
                            'azimuth_thruster_repair' if 'azimuth' in thruster_key else 'tunnel_thruster_inspection', 5000)
                    })
                elif thruster.get('status') == 'warning':
                    tasks.append({
                        'id': f"MNT-{len(tasks):03d}",
                        'component': thruster_key,
                        'priority': 'high',
                        'task': 'Scheduled inspection',
                        'estimated_cost': self.MAINTENANCE_COSTS.get('azimuth_thruster_inspection', 5000)
                    })
            
            # From DP
            if dp.get('dp_status') == 'degraded':
                tasks.append({
                    'id': f"MNT-{len(tasks):03d}",
                    'component': 'dp_system',
                    'priority': 'high',
                    'task': 'DP calibration and diagnostics',
                    'estimated_cost': self.MAINTENANCE_COSTS['dp_calibration']
                })
            
            # From power
            for gen in ['generator_1', 'generator_2']:
                gen_data = power.get(gen, {})
                if gen_data.get('status') in ['warning', 'critical']:
                    tasks.append({
                        'id': f"MNT-{len(tasks):03d}",
                        'component': gen,
                        'priority': 'high' if gen_data.get('status') == 'critical' else 'medium',
                        'task': 'Generator service',
                        'estimated_cost': self.MAINTENANCE_COSTS['generator_service']
                    })
            
            # Sort by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            tasks.sort(key=lambda x: priority_order.get(x['priority'], 4))
            
            # Cost summary
            total_cost = sum(t['estimated_cost'] for t in tasks)
            
            self.status = AgentStatus.COMPLETED
            
            return {
                'status': 'success',
                'maintenance_tasks': tasks,
                'task_count': len(tasks),
                'by_priority': {
                    'critical': len([t for t in tasks if t['priority'] == 'critical']),
                    'high': len([t for t in tasks if t['priority'] == 'high']),
                    'medium': len([t for t in tasks if t['priority'] == 'medium'])
                },
                'total_estimated_cost': total_cost,
                'alerts': self.get_alerts()
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'status': 'error', 'error': str(e)}
