#!/usr/bin/env python3
"""
Integrated Navigation Display
==============================
Unified GUI showing:
- Sensor Fusion tracks (ML-enhanced Kalman filter)
- Collision risk overlays with CPA/TCPA
- COLREGS recommendations in real-time
- ML status (maneuver detection, sensor reliability)

Usage:
    python integrated_navigation_display.py                    # Synthetic data
    python integrated_navigation_display.py --real             # Autoferry data
    python integrated_navigation_display.py --real --animate   # Animated
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.sensor_fusion_agent import SensorFusionAgentKalman
from agents.collision_avoidance_agent import CollisionAvoidanceAgent
from data.autoferry_loader import AutoferryDataLoader


# Colors
SENSOR_COLORS = {1: '#2ecc71', 2: '#3498db', 3: '#e74c3c', 4: '#f39c12'}
SENSOR_NAMES = {1: 'LiDAR', 2: 'Radar', 3: 'IR Camera', 4: 'EO Camera'}

RISK_COLORS = {
    'CRITICAL': '#c0392b',
    'HIGH': '#e74c3c',
    'MEDIUM': '#f39c12',
    'LOW': '#27ae60',
    'CLEAR': '#95a5a6'
}

COLREGS_ACTIONS = {
    13: "OVERTAKING - Keep clear",
    14: "HEAD-ON - Turn STARBOARD",
    15: "CROSSING - Give way (turn STARBOARD)",
    16: "Action by give-way vessel",
    17: "STAND-ON - Maintain course/speed"
}


class IntegratedNavigationDisplay:
    """Unified navigation display with sensor fusion + collision avoidance."""
    
    def __init__(self, use_real_data: bool = False, scenario: str = 'scenario16'):
        self.use_real_data = use_real_data
        self.scenario = scenario
        
        # Initialize agents
        self.sensor_fusion_agent = SensorFusionAgentKalman(use_ml=True)
        self.collision_agent = CollisionAvoidanceAgent()
        
        # Data storage
        self.data = None
        self.fusion_result = None
        self.collision_result = None
        self.own_ship = {
            'position': [0, 0],
            'velocity': [2.0, 0.5],
            'heading': 0.0,
            'speed': 2.06
        }
        
        # Animation state
        self.current_time_idx = 0
        self.detection_times = []
        self.ani = None
        
    def load_data(self):
        """Load sensor data."""
        if self.use_real_data:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
            loader = AutoferryDataLoader(dataset_path)
            self.data = loader.load_for_kalman_filter(self.scenario)
            
            # Get unique detection times for animation
            times = sorted(set(d['time'] for d in self.data['detections']))
            self.detection_times = times
        else:
            self.data = self._generate_synthetic_data()
            self.detection_times = sorted(set(d['time'] for d in self.data['detections']))
        
        return self.data
    
    def _generate_synthetic_data(self):
        """Generate synthetic multi-target scenario."""
        detections = []
        ground_truth = []
        
        # Target 1: Crossing from starboard (COLREGS Rule 15)
        for i in range(60):
            t = i * 0.5
            north = 200 - 3 * t  # Coming from ahead-starboard
            east = 150 - 2 * t
            
            for sensor_id in [1, 2]:
                noise = np.random.randn(2) * 5
                detections.append({
                    'time': t,
                    'sensorID': sensor_id,
                    'position': [north + noise[0], east + noise[1], 0],
                    'targetID': 1
                })
            
            ground_truth.append({
                'time': t,
                'targetID': 1,
                'position': [north, east, 0]
            })
        
        # Target 2: Head-on (COLREGS Rule 14)
        for i in range(60):
            t = i * 0.5
            north = 300 - 4 * t  # Coming head-on
            east = 10 + np.sin(t * 0.1) * 5  # Slight wobble
            
            for sensor_id in [1, 2]:
                noise = np.random.randn(2) * 5
                detections.append({
                    'time': t,
                    'sensorID': sensor_id,
                    'position': [north + noise[0], east + noise[1], 0],
                    'targetID': 2
                })
            
            ground_truth.append({
                'time': t,
                'targetID': 2,
                'position': [north, east, 0]
            })
        
        # Target 3: Overtaking (COLREGS Rule 13)
        for i in range(60):
            t = i * 0.5
            north = -50 + 5 * t  # Coming from astern, faster
            east = -20 + 0.5 * t
            
            for sensor_id in [1, 2]:
                noise = np.random.randn(2) * 5
                detections.append({
                    'time': t,
                    'sensorID': sensor_id,
                    'position': [north + noise[0], east + noise[1], 0],
                    'targetID': 3
                })
            
            ground_truth.append({
                'time': t,
                'targetID': 3,
                'position': [north, east, 0]
            })
        
        return {
            'detections': detections,
            'ground_truth': ground_truth,
            'scenario': 'synthetic_colregs'
        }
    
    def process(self, up_to_time: float = None):
        """Run sensor fusion and collision avoidance."""
        if up_to_time is not None:
            # Filter detections up to time
            filtered_data = {
                'detections': [d for d in self.data['detections'] if d['time'] <= up_to_time],
                'ground_truth': self.data.get('ground_truth', [])
            }
        else:
            filtered_data = self.data
        
        # Run sensor fusion
        self.fusion_result = self.sensor_fusion_agent.process(filtered_data)
        
        # Prepare collision avoidance input
        collision_data = {
            'tracks': self.fusion_result['tracks'],
            'own_ship': self.own_ship
        }
        
        # Run collision avoidance
        self.collision_result = self.collision_agent.process(collision_data)
        
        return self.fusion_result, self.collision_result
    
    def create_static_display(self):
        """Create the full integrated display."""
        self.load_data()
        self.process()
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('R/V GUNNERUS - INTEGRATED NAVIGATION DISPLAY', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1],
                     hspace=0.3, wspace=0.25)
        
        # Main track view (spans 2 columns)
        ax_track = fig.add_subplot(gs[0, :2])
        self._draw_track_view(ax_track)
        
        # Collision alerts panel
        ax_alerts = fig.add_subplot(gs[0, 2])
        self._draw_collision_alerts(ax_alerts)
        
        # Sensor fusion status
        ax_fusion = fig.add_subplot(gs[1, 0])
        self._draw_fusion_status(ax_fusion)
        
        # ML status
        ax_ml = fig.add_subplot(gs[1, 1])
        self._draw_ml_status(ax_ml)
        
        # COLREGS reference
        ax_colregs = fig.add_subplot(gs[1, 2])
        self._draw_colregs_reference(ax_colregs)
        
        # Track details table
        ax_table = fig.add_subplot(gs[2, :])
        self._draw_track_table(ax_table)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def _draw_track_view(self, ax):
        """Draw the main situational awareness view."""
        ax.set_title('SITUATIONAL AWARENESS - Track View', fontsize=12, fontweight='bold')
        
        # Draw raw detections faintly
        for sensor_id in SENSOR_COLORS:
            sensor_dets = [d for d in self.data['detections'] if d['sensorID'] == sensor_id]
            if sensor_dets:
                easts = [d['position'][1] for d in sensor_dets]
                norths = [d['position'][0] for d in sensor_dets]
                ax.scatter(easts, norths, c=SENSOR_COLORS[sensor_id], 
                          s=3, alpha=0.15, label=SENSOR_NAMES[sensor_id])
        
        # Draw own ship
        own_e, own_n = self.own_ship['position'][1], self.own_ship['position'][0]
        ax.plot(own_e, own_n, 'k^', markersize=15, label='Own Ship (Vessel)')
        
        # Draw own ship velocity vector
        own_vel = self.own_ship['velocity']
        scale = 10
        ax.annotate('', xy=(own_e + own_vel[1]*scale, own_n + own_vel[0]*scale),
                   xytext=(own_e, own_n),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Draw heading sector
        heading_deg = np.degrees(self.own_ship['heading'])
        wedge = Wedge((own_e, own_n), 50, heading_deg - 10, heading_deg + 10, 
                     alpha=0.2, color='blue')
        ax.add_patch(wedge)
        
        # Get collision assessments - handle both 'assessments' and 'collision_risks' keys
        risks = {}
        if self.collision_result:
            raw_assessments = self.collision_result.get('assessments', 
                             self.collision_result.get('collision_risks', []))
            for assessment in raw_assessments:
                track_id = assessment.get('track_id', assessment.get('target_id'))
                # Normalize keys for display
                risks[track_id] = {
                    'track_id': track_id,
                    'target_name': assessment.get('target_name', f"Track {track_id}"),
                    'risk_level': assessment.get('risk_level', 'CLEAR').upper(),
                    'cpa_distance': assessment.get('cpa_distance', assessment.get('cpa_m', assessment.get('cpa', 0))),
                    'tcpa_seconds': assessment.get('tcpa_seconds', assessment.get('tcpa', 0)),
                    'encounter_type': assessment.get('encounter_type', 'unknown'),
                    'colregs_rule': assessment.get('colregs_rule', assessment.get('colreg_rule')),
                    'recommended_action': assessment.get('recommended_action', '')
                }
        
        # Draw tracks with collision risk overlay
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, track in enumerate(self.fusion_result['tracks']):
            if track['status'] != 'confirmed':
                continue
            
            history = track.get('history', [])
            if len(history) < 2:
                continue
            
            track_id = track['track_id']
            risk_info = risks.get(track_id, {})
            risk_level = risk_info.get('risk_level', 'CLEAR')
            
            # Track color based on risk
            track_color = RISK_COLORS.get(risk_level, colors[i % len(colors)])
            
            norths = [h[0] for h in history]
            easts = [h[1] for h in history]
            
            # Draw track trajectory
            ax.plot(easts, norths, '-', color=track_color, linewidth=2.5, alpha=0.8)
            
            # Current position with risk-colored marker
            ax.plot(easts[-1], norths[-1], 'o', color=track_color, 
                   markersize=12, markeredgecolor='white', markeredgewidth=2)
            
            # Velocity arrow
            vel = track.get('velocity', [0, 0])
            if isinstance(vel, dict):
                vel = [vel.get('north', 0), vel.get('east', 0)]
            vel_scale = 5
            ax.annotate('', xy=(easts[-1] + vel[1]*vel_scale, norths[-1] + vel[0]*vel_scale),
                       xytext=(easts[-1], norths[-1]),
                       arrowprops=dict(arrowstyle='->', color=track_color, lw=2))
            
            # Draw CPA line if risk
            if risk_level in ['CRITICAL', 'HIGH', 'MEDIUM']:
                cpa = risk_info.get('cpa_distance', 0)
                tcpa = risk_info.get('tcpa_seconds', 0)
                
                # Predicted position at CPA
                if tcpa > 0:
                    pred_n = norths[-1] + vel[0] * tcpa
                    pred_e = easts[-1] + vel[1] * tcpa
                    own_pred_n = own_n + self.own_ship['velocity'][0] * tcpa
                    own_pred_e = own_e + self.own_ship['velocity'][1] * tcpa
                    
                    # Draw CPA line
                    ax.plot([pred_e, own_pred_e], [pred_n, own_pred_n], 
                           '--', color=track_color, linewidth=1.5, alpha=0.6)
                    ax.plot(pred_e, pred_n, 'x', color=track_color, markersize=8)
                    ax.plot(own_pred_e, own_pred_n, 'x', color='black', markersize=8)
            
            # Track label with risk
            name = track.get('target_name', f"Track {track_id}")
            label_text = f"{name}"
            if risk_level != 'CLEAR':
                label_text += f"\n[{risk_level}]"
            
            ax.annotate(label_text, (easts[-1], norths[-1]), 
                       xytext=(8, 8), textcoords='offset points', fontsize=9,
                       fontweight='bold' if risk_level in ['CRITICAL', 'HIGH'] else 'normal',
                       color=track_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.8, edgecolor=track_color))
        
        # Draw range rings
        for r in [100, 200, 300]:
            circle = Circle((own_e, own_n), r, fill=False, 
                           linestyle='--', color='gray', alpha=0.3)
            ax.add_patch(circle)
            ax.annotate(f'{r}m', (own_e + r, own_n), fontsize=8, color='gray')
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # Set axis limits based on data
        all_norths = [d['position'][0] for d in self.data['detections']]
        all_easts = [d['position'][1] for d in self.data['detections']]
        margin = 50
        ax.set_xlim(min(all_easts) - margin, max(all_easts) + margin)
        ax.set_ylim(min(all_norths) - margin, max(all_norths) + margin)
    
    def _draw_collision_alerts(self, ax):
        """Draw collision alerts panel."""
        ax.set_title('COLLISION ALERTS', fontsize=12, fontweight='bold', color='#c0392b')
        ax.axis('off')
        
        if not self.collision_result:
            ax.text(0.5, 0.5, 'No collision data', ha='center', va='center')
            return
        
        raw_assessments = self.collision_result.get('assessments', 
                         self.collision_result.get('collision_risks', []))
        
        # Normalize assessments
        assessments = []
        for a in raw_assessments:
            assessments.append({
                'track_id': a.get('track_id', a.get('target_id')),
                'target_name': a.get('target_name', f"Track {a.get('track_id', '?')}"),
                'risk_level': a.get('risk_level', 'CLEAR').upper(),
                'cpa_distance': a.get('cpa_distance', a.get('cpa_m', a.get('cpa', 0))),
                'tcpa_seconds': a.get('tcpa_seconds', a.get('tcpa', 0)),
                'encounter_type': a.get('encounter_type', 'unknown'),
                'colregs_rule': a.get('colregs_rule', a.get('colreg_rule')),
                'recommended_action': a.get('recommended_action', '')
            })
        
        if not assessments:
            ax.text(0.5, 0.5, 'ALL CLEAR\nNo collision risks detected', 
                   ha='center', va='center', fontsize=14, color='#27ae60')
            return
        
        # Sort by risk level
        risk_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'CLEAR': 4}
        assessments = sorted(assessments, key=lambda x: risk_order.get(x.get('risk_level', 'CLEAR'), 5))
        
        y = 0.95
        for assessment in assessments[:5]:  # Show top 5
            risk_level = assessment.get('risk_level', 'CLEAR')
            if risk_level == 'CLEAR':
                continue
            
            track_name = assessment.get('target_name', f"Track {assessment.get('track_id', '?')}")
            cpa = assessment.get('cpa_distance', 0)
            tcpa = assessment.get('tcpa_seconds', 0)
            encounter = assessment.get('encounter_type', 'unknown')
            colreg = assessment.get('colregs_rule', None)
            action = assessment.get('recommended_action', '')
            
            color = RISK_COLORS.get(risk_level, 'gray')
            
            # Risk badge
            alert_text = f"{'⚠' if risk_level in ['CRITICAL', 'HIGH'] else '●'} {track_name}"
            ax.text(0.05, y, alert_text, transform=ax.transAxes, fontsize=11,
                   fontweight='bold', color=color)
            y -= 0.08
            
            # Details
            details = f"  CPA: {cpa:.0f}m | TCPA: {tcpa:.0f}s | {encounter.upper()}"
            ax.text(0.05, y, details, transform=ax.transAxes, fontsize=9, color='#34495e')
            y -= 0.06
            
            # COLREGS rule
            if colreg:
                rule_text = f"  Rule {colreg}: {COLREGS_ACTIONS.get(colreg, '')}"
                ax.text(0.05, y, rule_text, transform=ax.transAxes, fontsize=9, 
                       color=color, style='italic')
                y -= 0.06
            
            # Recommended action
            if action:
                ax.text(0.05, y, f"  → {action}", transform=ax.transAxes, 
                       fontsize=9, fontweight='bold', color='#2c3e50')
                y -= 0.08
            
            y -= 0.02  # Spacing between alerts
    
    def _draw_fusion_status(self, ax):
        """Draw sensor fusion status panel."""
        ax.set_title('SENSOR FUSION STATUS', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        if not self.fusion_result:
            return
        
        # Detection counts by sensor
        sensor_counts = {}
        for d in self.data['detections']:
            sid = d['sensorID']
            sensor_counts[sid] = sensor_counts.get(sid, 0) + 1
        
        info = [
            f"Confirmed Tracks: {self.fusion_result['num_confirmed_tracks']}",
            f"Total Detections: {self.fusion_result['num_detections']}",
            "",
            "Detections by Sensor:"
        ]
        
        for sid in sorted(sensor_counts.keys()):
            name = SENSOR_NAMES.get(sid, f'Sensor {sid}')
            count = sensor_counts[sid]
            info.append(f"  {name}: {count}")
        
        info.extend([
            "",
            f"Kalman Filter: {'active' if self.fusion_result.get('kalman_active') else 'inactive'}",
            f"ML Enhanced: {'Yes' if self.fusion_result.get('ml_enhanced') else 'No'}"
        ])
        
        y = 0.92
        for line in info:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=9,
                   fontfamily='monospace')
            y -= 0.08
    
    def _draw_ml_status(self, ax):
        """Draw ML enhancement status panel."""
        ax.set_title('ML ENHANCEMENT STATUS', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        ml_status = self.fusion_result.get('ml_status', {})
        
        info = [
            f"Maneuver Detector: {ml_status.get('maneuver_detector', 'N/A')}",
            f"Sensor Reliability: {ml_status.get('sensor_reliability', 'N/A')}",
            "",
            "Track ML Status:"
        ]
        
        for track in self.fusion_result['tracks'][:5]:
            if track['status'] != 'confirmed':
                continue
            
            name = track.get('target_name', f"T{track['track_id']}")[:10]
            maneuver_prob = track.get('maneuver_probability', 0)
            q = track.get('process_noise_Q', 0.5)
            
            status = "MANEUVER" if maneuver_prob > 0.5 else "steady"
            info.append(f"  {name}: {maneuver_prob:.0%} ({status})")
        
        # Sensor stats
        sensor_stats = ml_status.get('sensor_stats', {})
        if sensor_stats:
            info.extend(["", "Sensor Reliability:"])
            for sid, stats in sorted(sensor_stats.items()):
                name = stats.get('sensor_name', f'Sensor {sid}')[:8]
                score = stats.get('reliability_score', 0.5)
                info.append(f"  {name}: {score:.2f}")
        
        y = 0.92
        for line in info:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=9,
                   fontfamily='monospace')
            y -= 0.07
    
    def _draw_colregs_reference(self, ax):
        """Draw COLREGS quick reference."""
        ax.set_title('COLREGS QUICK REFERENCE', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        rules = [
            ("Rule 13", "OVERTAKING", "Keep clear of vessel ahead"),
            ("Rule 14", "HEAD-ON", "Both alter to STARBOARD"),
            ("Rule 15", "CROSSING", "Give-way to vessel on STARBOARD"),
            ("Rule 16", "GIVE-WAY", "Take early, substantial action"),
            ("Rule 17", "STAND-ON", "Maintain course and speed")
        ]
        
        y = 0.92
        for rule, situation, action in rules:
            ax.text(0.05, y, f"{rule} - {situation}", transform=ax.transAxes, 
                   fontsize=9, fontweight='bold', color='#2c3e50')
            y -= 0.06
            ax.text(0.08, y, action, transform=ax.transAxes, fontsize=8, 
                   color='#7f8c8d', style='italic')
            y -= 0.10
    
    def _draw_track_table(self, ax):
        """Draw track details table."""
        ax.set_title('TRACK DETAILS', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Get risk info - normalize format
        risks = {}
        if self.collision_result:
            raw_assessments = self.collision_result.get('assessments', 
                             self.collision_result.get('collision_risks', []))
            for a in raw_assessments:
                tid = a.get('track_id', a.get('target_id'))
                risks[tid] = {
                    'risk_level': a.get('risk_level', 'CLEAR').upper(),
                    'cpa_distance': a.get('cpa_distance', a.get('cpa_m', a.get('cpa', 0))),
                    'tcpa_seconds': a.get('tcpa_seconds', a.get('tcpa', 0)),
                    'colregs_rule': a.get('colregs_rule', a.get('colreg_rule')),
                    'recommended_action': a.get('recommended_action', '')
                }
        
        # Build table data
        headers = ['Track', 'Position (N,E)', 'Velocity', 'Speed', 'Risk', 'CPA', 'TCPA', 'COLREGS', 'Action']
        rows = []
        
        for track in self.fusion_result['tracks']:
            if track['status'] != 'confirmed':
                continue
            
            track_id = track['track_id']
            name = track.get('target_name', f"Track {track_id}")[:12]
            
            pos = track.get('position', [0, 0])
            if isinstance(pos, dict):
                pos = [pos.get('north', 0), pos.get('east', 0)]
            
            vel = track.get('velocity', [0, 0])
            if isinstance(vel, dict):
                vel = [vel.get('north', 0), vel.get('east', 0)]
            
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            
            risk_info = risks.get(track_id, {})
            risk = risk_info.get('risk_level', 'CLEAR')
            cpa = risk_info.get('cpa_distance', '-')
            tcpa = risk_info.get('tcpa_seconds', '-')
            colreg = risk_info.get('colregs_rule', '-')
            action = risk_info.get('recommended_action', '-')[:25]
            
            rows.append([
                name,
                f"({pos[0]:.0f}, {pos[1]:.0f})",
                f"({vel[0]:.1f}, {vel[1]:.1f})",
                f"{speed:.1f} m/s",
                risk,
                f"{cpa:.0f}m" if isinstance(cpa, (int, float)) else cpa,
                f"{tcpa:.0f}s" if isinstance(tcpa, (int, float)) else tcpa,
                f"R{colreg}" if isinstance(colreg, int) else colreg,
                action if action != '-' else ''
            ])
        
        if not rows:
            ax.text(0.5, 0.5, 'No confirmed tracks', ha='center', va='center')
            return
        
        # Create table
        table = ax.table(cellText=rows, colLabels=headers, loc='center',
                        cellLoc='center', colColours=['#ecf0f1']*len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Color risk cells
        for i, row in enumerate(rows):
            risk = row[4]
            color = RISK_COLORS.get(risk, 'white')
            if risk in ['CRITICAL', 'HIGH']:
                table[(i+1, 4)].set_facecolor(color)
                table[(i+1, 4)].set_text_props(color='white', fontweight='bold')
    
    def run_animated(self, interval: int = 100):
        """Run animated display."""
        self.load_data()
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('R/V GUNNERUS - INTEGRATED NAVIGATION DISPLAY (LIVE)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1],
                     hspace=0.3, wspace=0.25)
        
        self.ax_track = fig.add_subplot(gs[0, :2])
        self.ax_alerts = fig.add_subplot(gs[0, 2])
        self.ax_fusion = fig.add_subplot(gs[1, 0])
        self.ax_ml = fig.add_subplot(gs[1, 1])
        self.ax_colregs = fig.add_subplot(gs[1, 2])
        self.ax_table = fig.add_subplot(gs[2, :])
        
        def animate(frame):
            if frame >= len(self.detection_times):
                return
            
            current_time = self.detection_times[frame]
            self.process(up_to_time=current_time)
            
            for ax in [self.ax_track, self.ax_alerts, self.ax_fusion, 
                      self.ax_ml, self.ax_colregs, self.ax_table]:
                ax.clear()
            
            self._draw_track_view(self.ax_track)
            self._draw_collision_alerts(self.ax_alerts)
            self._draw_fusion_status(self.ax_fusion)
            self._draw_ml_status(self.ax_ml)
            self._draw_colregs_reference(self.ax_colregs)
            self._draw_track_table(self.ax_table)
            
            # Add time indicator
            self.ax_track.set_title(f'SITUATIONAL AWARENESS - Time: {current_time:.1f}s', 
                                   fontsize=12, fontweight='bold')
        
        self.ani = FuncAnimation(fig, animate, frames=len(self.detection_times),
                                interval=interval, repeat=False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def run_static(self):
        """Run static display."""
        fig = self.create_static_display()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Integrated Navigation Display')
    parser.add_argument('--real', action='store_true', 
                       help='Use real Autoferry data')
    parser.add_argument('--scenario', type=str, default='scenario16',
                       help='Scenario to load (default: scenario16)')
    parser.add_argument('--animate', action='store_true',
                       help='Run animated display')
    parser.add_argument('--interval', type=int, default=100,
                       help='Animation interval in ms (default: 100)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  R/V GUNNERUS - INTEGRATED NAVIGATION DISPLAY")
    print("=" * 70)
    print(f"\n  Data source: {'Autoferry (' + args.scenario + ')' if args.real else 'Synthetic COLREGS scenarios'}")
    print(f"  Mode: {'Animated' if args.animate else 'Static'}")
    print(f"  ML Enhancement: Enabled")
    print("\n  Loading data and processing...")
    
    display = IntegratedNavigationDisplay(
        use_real_data=args.real,
        scenario=args.scenario
    )
    
    if args.animate:
        display.run_animated(interval=args.interval)
    else:
        display.run_static()


if __name__ == "__main__":
    main()
