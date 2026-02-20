#!/usr/bin/env python3
"""
Sensor Fusion Visualization GUI
================================
Visualizes multi-sensor tracking results with Kalman filtering.

Usage:
    python visualize_sensor_fusion.py           # Synthetic data
    python visualize_sensor_fusion.py --real    # Real Autoferry data
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.sensor_fusion_agent import SensorFusionAgentKalman


# Sensor colors
SENSOR_COLORS = {
    1: '#2ecc71',  # LiDAR - Green
    2: '#3498db',  # Radar - Blue
    3: '#e74c3c',  # IR Camera - Red
    4: '#f39c12',  # EO Camera - Orange
}

SENSOR_NAMES = {
    1: 'LiDAR',
    2: 'Radar', 
    3: 'IR Camera',
    4: 'EO Camera',
}

TARGET_COLORS = {
    1: '#9b59b6',  # Havfruen - Purple
    2: '#1abc9c',  # Vessel - Teal
    3: '#e67e22',  # Jetboat - Orange
}

TARGET_NAMES = {
    1: 'Havfruen',
    2: 'Vessel',
    3: 'Jetboat',
}


def generate_synthetic_data(n_targets=3, n_timesteps=50):
    """Generate synthetic sensor data for visualization."""
    np.random.seed(42)
    detections = []
    ground_truth = []
    
    targets = [
        {'id': 1, 'start_north': 200.0, 'start_east': 100.0, 'vn': -3.0, 've': -1.0},
        {'id': 2, 'start_north': 300.0, 'start_east': 0.0, 'vn': -2.5, 've': 0.5},
        {'id': 3, 'start_north': 150.0, 'start_east': -80.0, 'vn': -2.0, 've': 2.0}
    ][:n_targets]
    
    sensors = [
        (1, 'LiDAR', 0.85, 2.0),
        (2, 'Radar', 0.80, 5.0),
        (3, 'IR_Camera', 0.70, 4.0),
        (4, 'EO_Camera', 0.75, 3.5)
    ]
    
    for t in range(n_timesteps):
        ts = t * 0.5
        for tgt in targets:
            n = tgt['start_north'] + tgt['vn'] * ts
            e = tgt['start_east'] + tgt['ve'] * ts
            
            ground_truth.append({
                'time': ts,
                'targetID': tgt['id'],
                'position': [n, e, 0.0],
                'velocity': [tgt['vn'], tgt['ve']]
            })
            
            for sensor_id, sensor_name, prob, noise in sensors:
                if np.random.random() < prob:
                    detections.append({
                        'time': ts,
                        'sensorID': sensor_id,
                        'targetID': tgt['id'],
                        'position': [
                            n + np.random.normal(0, noise),
                            e + np.random.normal(0, noise),
                            0.0
                        ]
                    })
    
    return {'detections': detections, 'ground_truth': ground_truth}


def load_real_data(scenario='scenario2'):
    """Load real Autoferry data."""
    from data.autoferry_loader import AutoferryDataLoader
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    if not os.path.exists(dataset_path):
        print("Autoferry dataset not found. Using synthetic data.")
        return None
    
    loader = AutoferryDataLoader(dataset_path)
    return loader.load_for_kalman_filter(scenario)


class SensorFusionVisualizer:
    """Interactive visualization of sensor fusion results."""
    
    def __init__(self, data: dict, title: str = "Sensor Fusion Visualization"):
        self.data = data
        self.detections = data['detections']
        self.ground_truth = data.get('ground_truth', [])
        self.title = title
        
        # Run sensor fusion
        self.agent = SensorFusionAgentKalman()
        self.result = self.agent.process(data)
        self.tracks = self.result.get('tracks', [])
        
        # Organize data by time
        self.times = sorted(set(d['time'] for d in self.detections))
        self.detections_by_time = {}
        for d in self.detections:
            t = d['time']
            if t not in self.detections_by_time:
                self.detections_by_time[t] = []
            self.detections_by_time[t].append(d)
        
    def plot_static(self):
        """Create static visualization of all data."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(self.title, fontsize=14, fontweight='bold')
        
        # Left plot: Raw detections by sensor
        ax1 = axes[0]
        ax1.set_title('Raw Sensor Detections', fontsize=12)
        
        # Plot detections colored by sensor
        for sensor_id in SENSOR_COLORS:
            sensor_dets = [d for d in self.detections if d['sensorID'] == sensor_id]
            if sensor_dets:
                north = [d['position'][0] for d in sensor_dets]
                east = [d['position'][1] for d in sensor_dets]
                ax1.scatter(east, north, c=SENSOR_COLORS[sensor_id], 
                           s=20, alpha=0.6, label=SENSOR_NAMES[sensor_id])
        
        # Plot ground truth if available
        if self.ground_truth:
            for target_id in TARGET_COLORS:
                gt = [g for g in self.ground_truth if g['targetID'] == target_id]
                if gt:
                    north = [g['position'][0] for g in gt]
                    east = [g['position'][1] for g in gt]
                    ax1.plot(east, north, 'k--', linewidth=2, alpha=0.5)
        
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot observer position
        ax1.plot(0, 0, 'k^', markersize=15, label='Observer')
        ax1.annotate('Observer', (0, 0), xytext=(10, -20), fontsize=10)
        
        # Right plot: Kalman-filtered tracks
        ax2 = axes[1]
        ax2.set_title('Kalman-Filtered Tracks', fontsize=12)
        
        # Plot tracks
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, track in enumerate(self.tracks):
            color = colors[i % len(colors)]
            history = track.get('history', [])
            
            if history:
                north = [h[0] for h in history]
                east = [h[1] for h in history]
                
                # Track trajectory
                ax2.plot(east, north, '-', color=color, linewidth=2, alpha=0.8)
                
                # Track start
                ax2.plot(east[0], north[0], 'o', color=color, markersize=8)
                
                # Track end (current position)
                ax2.plot(east[-1], north[-1], 's', color=color, markersize=12)
                
                # Velocity vector
                if 'velocity' in track:
                    vn, ve = track['velocity']
                    scale = 5  # Scale velocity for visibility
                    ax2.arrow(east[-1], north[-1], ve*scale, vn*scale,
                             head_width=3, head_length=2, fc=color, ec=color)
                
                # Label
                label = f"Track {track['track_id']}"
                if 'target_name' in track:
                    label = track['target_name']
                ax2.annotate(label, (east[-1], north[-1]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold')
        
        # Plot observer
        ax2.plot(0, 0, 'k^', markersize=15)
        ax2.annotate('Observer', (0, 0), xytext=(10, -20), fontsize=10)
        
        ax2.set_xlabel('East (m)')
        ax2.set_ylabel('North (m)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add statistics box
        stats_text = (
            f"Detections: {len(self.detections)}\n"
            f"Tracks: {len(self.tracks)}\n"
            f"Duration: {max(self.times):.1f}s"
        )
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_tracks_detail(self):
        """Create detailed track information plot."""
        n_tracks = len(self.tracks)
        if n_tracks == 0:
            print("No tracks to display")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Track Analysis Details', fontsize=14, fontweight='bold')
        
        # Top-left: All tracks overview
        ax1 = axes[0, 0]
        ax1.set_title('Track Trajectories')
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, track in enumerate(self.tracks):
            color = colors[i % len(colors)]
            history = track.get('history', [])
            if history:
                north = [h[0] for h in history]
                east = [h[1] for h in history]
                label = track.get('target_name', f"Track {track['track_id']}")
                ax1.plot(east, north, '-o', color=color, label=label, 
                        markersize=4, linewidth=2)
        
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Top-right: Track status summary
        ax2 = axes[0, 1]
        ax2.set_title('Track Status Summary')
        
        statuses = [t.get('status', 'unknown') for t in self.tracks]
        status_counts = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        
        bars = ax2.bar(status_counts.keys(), status_counts.values(),
                      color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Track Status')
        
        for bar, count in zip(bars, status_counts.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=12)
        
        # Bottom-left: Detections by sensor
        ax3 = axes[1, 0]
        ax3.set_title('Detections by Sensor Type')
        
        sensor_counts = {}
        for d in self.detections:
            sid = d['sensorID']
            name = SENSOR_NAMES.get(sid, f'Sensor {sid}')
            sensor_counts[name] = sensor_counts.get(name, 0) + 1
        
        colors_list = [SENSOR_COLORS.get(i+1, '#95a5a6') for i in range(len(sensor_counts))]
        wedges, texts, autotexts = ax3.pie(
            sensor_counts.values(), 
            labels=sensor_counts.keys(),
            autopct='%1.1f%%',
            colors=colors_list,
            explode=[0.02] * len(sensor_counts)
        )
        
        # Bottom-right: Track information table
        ax4 = axes[1, 1]
        ax4.set_title('Track Details')
        ax4.axis('off')
        
        table_data = []
        headers = ['Track ID', 'Name', 'Status', 'Hits', 'Position (N,E)', 'Velocity']
        
        for track in self.tracks:
            pos = track.get('position', [0, 0])
            vel = track.get('velocity', [0, 0])
            table_data.append([
                track.get('track_id', '-'),
                track.get('target_name', '-'),
                track.get('status', '-'),
                track.get('hits', '-'),
                f"({pos[0]:.1f}, {pos[1]:.1f})",
                f"({vel[0]:.2f}, {vel[1]:.2f})"
            ])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        return fig
    
    def animate(self, interval=200):
        """Create animated visualization of tracking over time."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(f'{self.title} - Animation', fontsize=14, fontweight='bold')
        
        # Calculate plot limits
        all_north = [d['position'][0] for d in self.detections]
        all_east = [d['position'][1] for d in self.detections]
        margin = 50
        ax.set_xlim(min(all_east) - margin, max(all_east) + margin)
        ax.set_ylim(min(all_north) - margin, max(all_north) + margin)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Observer marker
        ax.plot(0, 0, 'k^', markersize=15)
        ax.annotate('Observer', (0, 0), xytext=(10, -20), fontsize=10)
        
        # Initialize plot elements
        scatter = ax.scatter([], [], c=[], s=50, alpha=0.8)
        track_lines = {}
        track_markers = {}
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=SENSOR_COLORS[sid], markersize=10, 
                      label=SENSOR_NAMES[sid])
            for sid in SENSOR_COLORS
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            return scatter, time_text
        
        def update(frame):
            if frame >= len(self.times):
                return scatter, time_text
            
            current_time = self.times[frame]
            
            # Get detections up to current time
            past_dets = [d for d in self.detections if d['time'] <= current_time]
            recent_dets = [d for d in self.detections 
                         if current_time - 2 <= d['time'] <= current_time]
            
            if recent_dets:
                positions = np.array([[d['position'][1], d['position'][0]] 
                                     for d in recent_dets])
                colors = [SENSOR_COLORS.get(d['sensorID'], '#95a5a6') 
                         for d in recent_dets]
                scatter.set_offsets(positions)
                scatter.set_color(colors)
            
            time_text.set_text(f'Time: {current_time:.1f}s\nDetections: {len(past_dets)}')
            
            return scatter, time_text
        
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(self.times), interval=interval,
                            blit=False, repeat=True)
        
        return fig, anim


def main():
    parser = argparse.ArgumentParser(description='Sensor Fusion Visualization')
    parser.add_argument('--real', action='store_true',
                        help='Use real Autoferry data')
    parser.add_argument('--scenario', type=str, default='scenario2',
                        help='Autoferry scenario (default: scenario2)')
    parser.add_argument('--animate', action='store_true',
                        help='Show animated visualization')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SENSOR FUSION VISUALIZATION")
    print("=" * 60)
    
    # Load data
    if args.real:
        print(f"\nLoading Autoferry {args.scenario}...")
        data = load_real_data(args.scenario)
        if data is None:
            print("Falling back to synthetic data...")
            data = generate_synthetic_data()
            title = "Sensor Fusion - Synthetic Data"
        else:
            title = f"Sensor Fusion - Autoferry {args.scenario}"
    else:
        print("\nGenerating synthetic data...")
        data = generate_synthetic_data()
        title = "Sensor Fusion - Synthetic Data"
    
    print(f"  Detections: {len(data['detections'])}")
    print(f"  Ground truth entries: {len(data.get('ground_truth', []))}")
    
    # Create visualizer
    viz = SensorFusionVisualizer(data, title)
    
    print(f"\n  Tracks created: {len(viz.tracks)}")
    for track in viz.tracks:
        print(f"    - {track.get('target_name', 'Track ' + str(track['track_id']))}: "
              f"{track.get('status', 'unknown')}, {track.get('hits', 0)} hits")
    
    # Create plots
    print("\nGenerating visualizations...")
    
    fig1 = viz.plot_static()
    fig2 = viz.plot_tracks_detail()
    
    if args.animate:
        print("Creating animation (this may take a moment)...")
        fig3, anim = viz.animate()
    
    print("\nDisplaying plots. Close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
