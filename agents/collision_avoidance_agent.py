"""
Collision Avoidance Agent for Vessel
==========================================
Analyzes tracked targets to detect collision risks and provide
COLREGS-compliant avoidance recommendations.

COLREGS Rules Implemented:
- Rule 13: Overtaking
- Rule 14: Head-on situation
- Rule 15: Crossing situation
- Rule 17: Stand-on vessel responsibilities
"""

import numpy as np
import math
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent, AgentStatus, AlertLevel


@dataclass
class CollisionRisk:
    """Represents a collision risk assessment"""
    target_id: int
    target_name: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    cpa: float  # Closest Point of Approach (meters)
    tcpa: float  # Time to CPA (seconds)
    current_distance: float  # Current distance (meters)
    relative_bearing: float  # Degrees
    relative_speed: float  # m/s
    encounter_type: str  # 'head-on', 'crossing', 'overtaking'
    colreg_rule: str  # Applicable COLREGS rule
    recommended_action: str
    
    def to_dict(self) -> Dict:
        return {
            'target_id': self.target_id,
            'target_name': self.target_name,
            'risk_level': self.risk_level,
            'cpa_m': self.cpa,
            'tcpa_seconds': self.tcpa,
            'tcpa_minutes': self.tcpa / 60,
            'current_distance_m': self.current_distance,
            'relative_bearing_deg': self.relative_bearing,
            'relative_speed_ms': self.relative_speed,
            'encounter_type': self.encounter_type,
            'colreg_rule': self.colreg_rule,
            'recommended_action': self.recommended_action
        }


class CollisionAvoidanceAgent(BaseAgent):
    """
    Collision avoidance agent for Vessel.
    
    Analyzes tracks from the sensor fusion agent to:
    - Calculate CPA/TCPA for all targets
    - Assess collision risk levels
    - Determine encounter types per COLREGS
    - Generate avoidance recommendations
    
    Designed for the Trondheim Fjord operating area.
    """
    
    # Risk thresholds (in meters and seconds)
    # Adjusted for research vessel operations in coastal waters
    CPA_CRITICAL = 50  # meters
    CPA_HIGH = 100  # meters
    CPA_MEDIUM = 200  # meters
    
    TCPA_CRITICAL = 60  # seconds (1 minute)
    TCPA_HIGH = 180  # seconds (3 minutes)
    TCPA_MEDIUM = 300  # seconds (5 minutes)
    
    # Own vessel parameters (Vessel)
    OWN_VESSEL = {
        'name': 'Vessel',
        'mmsi': 258342000,
        'length': 36.25,  # meters
        'beam': 9.60  # meters
    }
    
    def __init__(self):
        super().__init__(
            name="CollisionAvoidanceAgent",
            description="COLREGS-compliant collision risk assessment",
            domain="navigation"
        )
        self.collision_risks: List[CollisionRisk] = []
        self.own_state: Optional[Dict] = None
        
    def process(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process tracked targets for collision risk assessment.
        
        Args:
            data: Dict containing 'tracks' from SensorFusionAgent and 'own_vessel' state
            context: Optional context with own vessel state
            
        Returns:
            Collision risk assessment results
        """
        self.update_status(AgentStatus.PROCESSING, "collision_analysis", 0.0)
        self.logger.info("Starting collision risk assessment")
        
        try:
            # Extract tracks and own ship state
            tracks = data.get('tracks', []) if isinstance(data, dict) else []
            self.own_state = data.get('own_vessel', context.get('own_vessel', None) if context else None)
            
            if self.own_state is None:
                self.own_state = self._default_own_state()
            
            if not tracks:
                self.logger.info("No targets to analyze")
                return {
                    'status': 'success',
                    'collision_risks': [],
                    'risk_summary': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                    'overall_risk': 'CLEAR'
                }
            
            # Calculate CPA/TCPA for each target
            self.update_status(AgentStatus.PROCESSING, "cpa_calculation", 0.3)
            self.collision_risks = []
            
            for track in tracks:
                if not track.get('confirmed', True):
                    continue  # Skip unconfirmed tracks
                    
                risk = self._assess_target_risk(track)
                if risk:
                    self.collision_risks.append(risk)
            
            # Sort by risk level
            risk_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            self.collision_risks.sort(key=lambda r: (risk_order[r.risk_level], r.tcpa))
            
            # Generate alerts for high-risk situations
            self.update_status(AgentStatus.PROCESSING, "alert_generation", 0.6)
            self._generate_alerts()
            
            # Create situation summary
            self.update_status(AgentStatus.PROCESSING, "situation_summary", 0.8)
            situation = self._create_situation_summary()
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            result = {
                'status': 'success',
                'own_vessel': self.own_state,
                'targets_analyzed': len(tracks),
                'collision_risks': [r.to_dict() for r in self.collision_risks],
                'risk_summary': {
                    'critical': sum(1 for r in self.collision_risks if r.risk_level == 'critical'),
                    'high': sum(1 for r in self.collision_risks if r.risk_level == 'high'),
                    'medium': sum(1 for r in self.collision_risks if r.risk_level == 'medium'),
                    'low': sum(1 for r in self.collision_risks if r.risk_level == 'low')
                },
                'overall_risk': situation['overall_risk_level'],
                'situation': situation,
                'recommendations': recommendations,
                'alerts': self.get_alerts()
            }
            
            self.log_result(result)
            self.update_status(AgentStatus.COMPLETED, progress=1.0)
            self.logger.info(f"Collision analysis complete: {len(self.collision_risks)} risks assessed")
            
            return result
            
        except Exception as e:
            self.state.set_error(str(e))
            self.create_alert(AlertLevel.CRITICAL, f"Collision analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _default_own_state(self) -> Dict:
        """Default own vessel state at Trondheim harbor"""
        return {
            'position': {'north': 0.0, 'east': 0.0},
            'velocity': {'north': 2.0, 'east': 0.5},  # ~2 m/s forward
            'heading': 45.0,  # degrees
            'speed': 2.06,  # m/s (~4 knots)
            'course': 45.0
        }
    
    def _assess_target_risk(self, track: Dict) -> Optional[CollisionRisk]:
        """Assess collision risk for a single target"""
        try:
            # Own ship state - handle both dict and list formats
            own_pos_raw = self.own_state.get('position', [0, 0])
            own_vel_raw = self.own_state.get('velocity', [0, 0])
            
            if isinstance(own_pos_raw, dict):
                own_pos = np.array([own_pos_raw.get('north', 0), own_pos_raw.get('east', 0)])
            else:
                own_pos = np.array(own_pos_raw[:2])
            
            if isinstance(own_vel_raw, dict):
                own_vel = np.array([own_vel_raw.get('north', 0), own_vel_raw.get('east', 0)])
            else:
                own_vel = np.array(own_vel_raw[:2])
            
            own_course = self.own_state.get('course', self.own_state.get('heading', 0))
            
            # Target state - handle both dict and list formats
            target_pos_raw = track.get('position', [0, 0])
            target_vel_raw = track.get('velocity', [0, 0])
            
            if isinstance(target_pos_raw, dict):
                target_pos = np.array([target_pos_raw.get('north', 0), target_pos_raw.get('east', 0)])
            else:
                target_pos = np.array(target_pos_raw[:2])
            
            if isinstance(target_vel_raw, dict):
                target_vel = np.array([target_vel_raw.get('north', 0), target_vel_raw.get('east', 0)])
            else:
                target_vel = np.array(target_vel_raw[:2])
            
            # Relative position and velocity
            rel_pos = target_pos - own_pos
            rel_vel = target_vel - own_vel
            
            # Current distance
            current_distance = np.linalg.norm(rel_pos)
            
            # Relative speed
            rel_speed = np.linalg.norm(rel_vel)
            
            # CPA/TCPA calculation
            if rel_speed < 0.001:
                # Nearly stationary relative motion
                tcpa = 0 if current_distance < self.CPA_MEDIUM else float('inf')
                cpa = current_distance
            else:
                # Time to CPA
                tcpa = -np.dot(rel_pos, rel_vel) / (rel_speed**2)
                
                if tcpa < 0:
                    # CPA in the past, vessels diverging
                    tcpa = 0
                    cpa = current_distance
                else:
                    # CPA position
                    cpa_pos = rel_pos + rel_vel * tcpa
                    cpa = np.linalg.norm(cpa_pos)
            
            # Relative bearing
            rel_bearing = math.degrees(math.atan2(rel_pos[1], rel_pos[0]))
            if rel_bearing < 0:
                rel_bearing += 360
            
            # Adjust to own ship heading
            rel_bearing_from_bow = (rel_bearing - own_course) % 360
            
            # Target course
            target_course = math.degrees(math.atan2(target_vel[1], target_vel[0]))
            if target_course < 0:
                target_course += 360
            
            # Determine encounter type and COLREGS rule
            encounter_type, colreg_rule = self._determine_encounter(
                own_course, target_course, rel_bearing_from_bow
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(cpa, tcpa)
            
            # Generate recommendation
            recommendation = self._get_recommendation(
                encounter_type, colreg_rule, risk_level, rel_bearing_from_bow
            )
            
            return CollisionRisk(
                target_id=track.get('track_id', track.get('target_id', 0)),
                target_name=track.get('target_name', track.get('name', f"Target_{track.get('track_id', 0)}")),
                risk_level=risk_level,
                cpa=cpa,
                tcpa=tcpa,
                current_distance=current_distance,
                relative_bearing=rel_bearing_from_bow,
                relative_speed=rel_speed,
                encounter_type=encounter_type,
                colreg_rule=colreg_rule,
                recommended_action=recommendation
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to assess risk for track: {e}")
            return None
    
    def _determine_encounter(self, own_course: float, target_course: float, 
                             rel_bearing: float) -> Tuple[str, str]:
        """Determine encounter type and applicable COLREGS rule"""
        # Course difference
        course_diff = abs(own_course - target_course)
        if course_diff > 180:
            course_diff = 360 - course_diff
        
        # Head-on: vessels approaching on nearly reciprocal courses
        # Rule 14: When two power-driven vessels are meeting on reciprocal 
        # or nearly reciprocal courses so as to involve risk of collision
        if course_diff > 170 and (rel_bearing < 6 or rel_bearing > 354):
            return "head-on", "Rule 14 - Head-on"
        
        # Overtaking: approaching from astern (more than 22.5° abaft the beam)
        # Rule 13: Any vessel overtaking any other shall keep out of the way
        if 112.5 < rel_bearing < 247.5:
            return "overtaking", "Rule 13 - Overtaking"
        
        # Crossing situation
        # Rule 15: When two power-driven vessels are crossing
        if 0 < rel_bearing <= 112.5:
            # Target on starboard side - we are give-way vessel
            return "crossing", "Rule 15 - Crossing (give-way)"
        else:
            # Target on port side - we are stand-on vessel
            return "crossing", "Rule 17 - Crossing (stand-on)"
    
    def _determine_risk_level(self, cpa: float, tcpa: float) -> str:
        """Determine risk level based on CPA and TCPA"""
        if cpa <= self.CPA_CRITICAL and tcpa <= self.TCPA_CRITICAL:
            return 'critical'
        elif cpa <= self.CPA_HIGH and tcpa <= self.TCPA_HIGH:
            return 'high'
        elif cpa <= self.CPA_MEDIUM and tcpa <= self.TCPA_MEDIUM:
            return 'medium'
        return 'low'
    
    def _get_recommendation(self, encounter_type: str, colreg_rule: str,
                           risk_level: str, rel_bearing: float) -> str:
        """Get COLREGS-compliant recommendation"""
        if risk_level == 'low':
            return "Monitor target. No immediate action required."
        
        if encounter_type == "head-on":
            return ("HEAD-ON (Rule 14): Both vessels shall alter course to "
                   "STARBOARD so that each shall pass on the port side of the other.")
        
        elif encounter_type == "crossing":
            if "give-way" in colreg_rule:
                return ("CROSSING - GIVE-WAY (Rule 15): You are the give-way vessel. "
                       "Alter course to STARBOARD to pass astern of target. "
                       "Avoid crossing ahead of the other vessel.")
            else:
                return ("CROSSING - STAND-ON (Rule 17): You are the stand-on vessel. "
                       "Maintain course and speed. Be prepared to take action if "
                       "give-way vessel does not act. If in doubt, take action to avoid.")
        
        elif encounter_type == "overtaking":
            return ("OVERTAKING (Rule 13): You are the overtaking vessel. "
                   "Keep clear of the vessel being overtaken. "
                   "Any alteration of course must keep you clear until past and clear.")
        
        return "Assess situation carefully and take appropriate action to avoid collision."
    
    def _generate_alerts(self):
        """Generate alerts for dangerous situations"""
        for risk in self.collision_risks:
            if risk.risk_level == 'critical':
                self.create_alert(
                    AlertLevel.EMERGENCY,
                    f"CRITICAL: Collision risk with {risk.target_name}",
                    {'cpa_m': risk.cpa, 'tcpa_s': risk.tcpa, 
                     'action': risk.recommended_action}
                )
            elif risk.risk_level == 'high':
                self.create_alert(
                    AlertLevel.CRITICAL,
                    f"HIGH RISK: {risk.target_name} CPA={risk.cpa:.0f}m T={risk.tcpa/60:.1f}min",
                    {'encounter': risk.encounter_type, 'rule': risk.colreg_rule}
                )
    
    def _create_situation_summary(self) -> Dict:
        """Create overall situation awareness summary"""
        critical_count = sum(1 for r in self.collision_risks if r.risk_level == 'critical')
        high_count = sum(1 for r in self.collision_risks if r.risk_level == 'high')
        
        if critical_count > 0:
            overall_risk = 'CRITICAL'
            vigilance = 'IMMEDIATE ACTION REQUIRED'
        elif high_count > 0:
            overall_risk = 'ELEVATED'
            vigilance = 'HIGH VIGILANCE'
        elif len(self.collision_risks) > 0:
            overall_risk = 'MODERATE'
            vigilance = 'NORMAL WATCH'
        else:
            overall_risk = 'CLEAR'
            vigilance = 'ROUTINE'
        
        return {
            'own_position': self.own_state['position'],
            'own_course': self.own_state.get('course', self.own_state.get('heading', 0)),
            'own_speed_ms': self.own_state.get('speed', 0),
            'total_targets': len(self.collision_risks),
            'immediate_threats': critical_count + high_count,
            'overall_risk_level': overall_risk,
            'recommended_vigilance': vigilance
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Get critical and high-risk targets
        urgent_risks = [r for r in self.collision_risks if r.risk_level in ['critical', 'high']]
        
        if not urgent_risks:
            recommendations.append({
                'priority': 'low',
                'action': 'Continue on present course. No immediate collision risks.',
                'targets': []
            })
            return recommendations
        
        # Group by encounter type
        for encounter in ['head-on', 'crossing', 'overtaking']:
            encounter_risks = [r for r in urgent_risks if r.encounter_type == encounter]
            if encounter_risks:
                most_urgent = min(encounter_risks, key=lambda r: r.tcpa)
                recommendations.append({
                    'priority': 'critical' if most_urgent.risk_level == 'critical' else 'high',
                    'encounter_type': encounter,
                    'action': most_urgent.recommended_action,
                    'colreg_rule': most_urgent.colreg_rule,
                    'targets': [
                        {'id': r.target_id, 'name': r.target_name, 
                         'tcpa_min': r.tcpa/60, 'cpa_m': r.cpa}
                        for r in encounter_risks
                    ]
                })
        
        return sorted(recommendations, 
                     key=lambda x: 0 if x['priority'] == 'critical' else 1)
    
    def get_most_urgent_risk(self) -> Optional[CollisionRisk]:
        """Get the most urgent collision risk"""
        if not self.collision_risks:
            return None
        return self.collision_risks[0]  # Already sorted by risk/tcpa
