"""Vessel AI Orchestrator"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from agents.agents import (
    VesselBaseAgent, AgentStatus, AlertLevel, Alert,
    DPAgent, PropulsionAgent
)
from agents.sensor_fusion_agent import SensorFusionAgentKalman
from agents.collision_avoidance_agent import CollisionAvoidanceAgent


@dataclass
class VesselStatus:
    timestamp: datetime
    position_status: str
    propulsion_status: str
    dp_status: str
    sensor_status: str
    overall_risk: str
    active_alerts: int
    tracked_targets: int
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'position_status': self.position_status,
            'propulsion_status': self.propulsion_status,
            'dp_status': self.dp_status,
            'sensor_status': self.sensor_status,
            'overall_risk': self.overall_risk,
            'active_alerts': self.active_alerts,
            'tracked_targets': self.tracked_targets
        }


class VesselOrchestrator:
    """Main orchestrator for Vessel AI System."""
    
    REFERENCE_POSITION = {'latitude': 63.4305, 'longitude': 10.3951}
    
    def __init__(self, vessel_name: str = "Vessel"):
        self.vessel_name = vessel_name
        self.logger = logging.getLogger("Vessel.Orchestrator")
        self.logger.info(f"Initializing {vessel_name} AI System")
        
        # Navigation agents (with Kalman filtering and COLREGS)
        self.sensor_fusion_agent = SensorFusionAgentKalman()
        self.collision_avoidance_agent = CollisionAvoidanceAgent()
        
        # DP and Propulsion agents
        self.dp_agent = DPAgent()
        self.propulsion_agent = PropulsionAgent()
        
        self.agents = {
            'sensor_fusion': self.sensor_fusion_agent,
            'collision_avoidance': self.collision_avoidance_agent,
            'dp_control': self.dp_agent,
            'propulsion_health': self.propulsion_agent
        }
        
        self.vessel_status = None
        self.results = {}
        self.execution_log = []
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def run_full_analysis(self, sensor_data=None, vessel_state=None, 
                         propulsion_data=None, dp_setpoint=None):
        start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info(f"STARTING FULL ANALYSIS - {self.vessel_name}")
        
        try:
            # Phase 1: Sensor Fusion with Kalman Filtering
            self.logger.info("PHASE 1: SENSOR FUSION (Kalman Filter)")
            if sensor_data is not None:
                fusion_result = self.sensor_fusion_agent.process(sensor_data)
            else:
                fusion_result = self._synthetic_sensor_result()
            self.results['sensor_fusion'] = fusion_result
            self._log_execution('sensor_fusion', fusion_result.get('status', 'unknown'))
            
            # Phase 2: Collision Avoidance (COLREGS)
            self.logger.info("PHASE 2: COLLISION AVOIDANCE (COLREGS)")
            if vessel_state is None:
                vessel_state = self._default_vessel_state()
            
            ca_input = {
                'tracks': fusion_result.get('confirmed_tracks', fusion_result.get('tracks', [])),
                'own_vessel': {
                    'position': {'north': vessel_state.get('north', 0), 'east': vessel_state.get('east', 0)},
                    'velocity': {'north': vessel_state.get('surge', 0), 'east': vessel_state.get('sway', 0)},
                    'heading': vessel_state.get('heading', 0),
                    'speed': np.sqrt(vessel_state.get('surge', 0)**2 + vessel_state.get('sway', 0)**2),
                    'course': vessel_state.get('heading', 0)
                }
            }
            ca_result = self.collision_avoidance_agent.process(ca_input)
            self.results['collision_avoidance'] = ca_result
            self._log_execution('collision_avoidance', ca_result.get('status', 'unknown'))
            
            # Phase 3: DP Control
            self.logger.info("PHASE 3: DYNAMIC POSITIONING")
            dp_context = {'mode': 'station_keeping'}
            if dp_setpoint:
                dp_context['setpoint'] = dp_setpoint
            
            dp_result = self.dp_agent.process(vessel_state, context=dp_context)
            self.results['dp_control'] = dp_result
            self._log_execution('dp_control', dp_result.get('status', 'unknown'))
            
            # Phase 4: Propulsion Health
            self.logger.info("PHASE 4: PROPULSION HEALTH")
            if propulsion_data is None:
                propulsion_data = self._synthetic_propulsion_data()
            
            propulsion_result = self.propulsion_agent.process(
                propulsion_data,
                context={'runtime_hours': self._default_runtime()}
            )
            self.results['propulsion_health'] = propulsion_result
            self._log_execution('propulsion_health', propulsion_result.get('status', 'unknown'))
            
            # Synthesis
            self.vessel_status = self._calculate_vessel_status()
            summary = self._generate_summary()
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"ANALYSIS COMPLETE - Duration: {duration:.2f}s")
            
            return {
                'status': 'success',
                'vessel_name': self.vessel_name,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration,
                'vessel_status': self.vessel_status.to_dict(),
                'summary': summary,
                'all_alerts': self._collect_all_alerts(),
                'detailed_results': {
                    'sensor_fusion': self._summarize_result(fusion_result),
                    'collision_avoidance': self._summarize_result(ca_result),
                    'dp_control': self._summarize_result(dp_result),
                    'propulsion_health': self._summarize_result(propulsion_result)
                },
                'execution_log': self.execution_log
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e), 'partial_results': self.results}
    
    def _default_vessel_state(self):
        return {'north': 0.0, 'east': 0.0, 'heading': 45.0, 'surge': 0.1, 
                'sway': 0.05, 'yaw_rate': 0.01, 'latitude': 63.4305, 'longitude': 10.3951}
    
    def _synthetic_sensor_result(self):
        # Generate synthetic tracks for collision avoidance testing
        synthetic_tracks = [
            {
                'track_id': 1, 'target_id': 1, 'name': 'Havfruen', 'confirmed': True,
                'position': {'north': 150.0, 'east': 80.0},
                'velocity': {'north': -1.5, 'east': -0.5},
                'speed_ms': 1.58, 'speed_knots': 3.07
            },
            {
                'track_id': 2, 'target_id': 3, 'name': 'Jetboat', 'confirmed': True,
                'position': {'north': 200.0, 'east': -50.0},
                'velocity': {'north': -2.0, 'east': 1.0},
                'speed_ms': 2.24, 'speed_knots': 4.35
            }
        ]
        return {
            'status': 'success', 
            'num_tracks': len(synthetic_tracks),
            'num_confirmed_tracks': len(synthetic_tracks),
            'tracks': synthetic_tracks,
            'confirmed_tracks': synthetic_tracks,
            'num_detections': 50,
            'kalman_filter': {'process_noise': 0.5, 'measurement_noise': 2.0}
        }
    
    def _synthetic_propulsion_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'gen1_power': np.random.normal(180, 15, 100),
            'gen2_power': np.random.normal(120, 12, 100),
            'gen1_voltage': np.random.normal(400, 1.5, 100),
            'gen1_coolant_temp': np.random.normal(82, 3, 100)
        })
    
    def _default_runtime(self):
        return {'azimuth_stbd': 3500, 'azimuth_port': 3500, 'tunnel_bow': 2000, 'gen1': 8000, 'gen2': 7500}
    
    def _calculate_vessel_status(self):
        fusion = self.results.get('sensor_fusion', {})
        sensor_status = 'normal'
        tracked = fusion.get('active_tracks', 0)
        
        dp = self.results.get('dp_control', {})
        pos_error = dp.get('position_error', {}).get('total_m', 0)
        dp_status = 'critical' if pos_error > 10 else 'warning' if pos_error > 5 else 'normal'
        
        prop = self.results.get('propulsion_health', {})
        critical = prop.get('fault_summary', {}).get('critical', 0)
        prop_status = 'critical' if critical > 0 else 'warning' if prop.get('fault_summary', {}).get('warning', 0) > 0 else 'normal'
        
        statuses = [sensor_status, dp_status, prop_status]
        overall = 'CRITICAL' if 'critical' in statuses else 'ELEVATED' if 'warning' in statuses else 'NORMAL'
        
        return VesselStatus(datetime.now(), dp_status, prop_status, dp_status, sensor_status, overall, 
                             len(self._collect_all_alerts()), tracked)
    
    def _collect_all_alerts(self):
        alerts = []
        for name, agent in self.agents.items():
            for alert in agent.get_alerts():
                if isinstance(alert, dict):
                    alert['agent'] = name
                    alerts.append(alert)
        return alerts
    
    def _generate_summary(self):
        fusion = self.results.get('sensor_fusion', {})
        ca = self.results.get('collision_avoidance', {})
        dp = self.results.get('dp_control', {})
        prop = self.results.get('propulsion_health', {})
        
        return {
            'sensor_fusion': {
                'tracked_targets': fusion.get('num_confirmed_tracks', fusion.get('num_tracks', 0)), 
                'detections_processed': fusion.get('num_detections', 0),
                'kalman_filter': 'active',
                'status': 'operational'
            },
            'collision_avoidance': {
                'overall_risk': ca.get('overall_risk', 'CLEAR'),
                'critical_risks': ca.get('risk_summary', {}).get('critical', 0),
                'high_risks': ca.get('risk_summary', {}).get('high', 0),
                'colregs': 'active',
                'status': 'operational'
            },
            'dynamic_positioning': {
                'mode': dp.get('dp_mode', 'unknown'),
                'position_error_m': dp.get('position_error', {}).get('total_m', 0),
                'heading_error_deg': dp.get('position_error', {}).get('heading_deg', 0),
                'status': 'operational'
            },
            'propulsion': {
                'faults_detected': prop.get('fault_summary', {}).get('total', 0),
                'critical_faults': prop.get('fault_summary', {}).get('critical', 0),
                'recommendations': len(prop.get('maintenance_recommendations', [])),
                          'status': 'operational' if prop.get('fault_summary', {}).get('critical', 0) == 0 else 'degraded'}
        }
    
    def _summarize_result(self, result):
        exclude = ['data', 'raw_data', 'tracks', 'fusion_log']
        return {k: v for k, v in result.items() if k not in exclude}
    
    def _log_execution(self, agent_name, status):
        self.execution_log.append({'timestamp': datetime.now().isoformat(), 'agent': agent_name, 'status': status})
