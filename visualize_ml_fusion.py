#!/usr/bin/env python3
"""
ML-Enhanced Sensor Fusion Visualization & Inference Demo
=========================================================
Demonstrates real-time inference with:
- Maneuver-adaptive process noise (Q)
- Sensor-specific measurement noise (R)

Usage:
    python visualize_ml_fusion.py              # Test on held-out scenarios
    python visualize_ml_fusion.py --scenario scenario16
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.sensor_fusion_agent import SensorFusionAgentKalman
from data.autoferry_loader import AutoferryDataLoader


# Sensor colors and names
SENSOR_COLORS = {1: '#2ecc71', 2: '#3498db', 3: '#e74c3c', 4: '#f39c12'}
SENSOR_NAMES = {1: 'LiDAR', 2: 'Radar', 3: 'IR Camera', 4: 'EO Camera'}


def load_test_scenario(scenario: str = 'scenario16'):
    """Load a test scenario (Environment 2 - not used for training)."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    loader = AutoferryDataLoader(dataset_path)
    return loader.load_for_kalman_filter(scenario)


def run_inference(data: dict, use_ml: bool = True):
    """Run sensor fusion with or without ML enhancement."""
    agent = SensorFusionAgentKalman(use_ml=use_ml)
    result = agent.process(data)
    return result, agent


def create_comparison_visualization(data: dict, result_ml: dict, result_fixed: dict):
    """Create side-by-side comparison of ML vs Fixed Kalman."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
    
    # ===== Top row: Track visualizations =====
    
    # ML-Enhanced Tracking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('ML-Enhanced Kalman Filter', fontsize=12, fontweight='bold')
    plot_tracks(ax1, result_ml['tracks'], data['detections'], "With ML")
    
    # Fixed Kalman Tracking  
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Fixed Kalman Filter', fontsize=12, fontweight='bold')
    plot_tracks(ax2, result_fixed['tracks'], data['detections'], "Without ML")
    
    # Ground truth overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Ground Truth Comparison', fontsize=12, fontweight='bold')
    plot_ground_truth_comparison(ax3, result_ml['tracks'], data.get('ground_truth', []))
    
    # ===== Middle row: ML Status =====
    
    # Maneuver Detection
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Maneuver Probability per Track', fontsize=11)
    plot_maneuver_status(ax4, result_ml['tracks'])
    
    # Process Noise (Q) adaptation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Adaptive Process Noise (Q)', fontsize=11)
    plot_process_noise(ax5, result_ml['tracks'])
    
    # Sensor Reliability
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('Sensor Reliability Scores', fontsize=11)
    plot_sensor_reliability(ax6, result_ml.get('ml_status', {}))
    
    # ===== Bottom row: Statistics =====
    
    # Track statistics comparison
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title('Track Statistics', fontsize=11)
    plot_track_stats(ax7, result_ml, result_fixed)
    
    # ML info panel
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    plot_info_panel(ax8, result_ml, result_fixed, data)
    
    plt.tight_layout()
    return fig


def plot_tracks(ax, tracks, detections, label):
    """Plot track trajectories."""
    # Plot detections faintly
    for sensor_id in SENSOR_COLORS:
        sensor_dets = [d for d in detections if d['sensorID'] == sensor_id]
        if sensor_dets:
            easts = [d['position'][1] for d in sensor_dets]
            norths = [d['position'][0] for d in sensor_dets]
            ax.scatter(easts, norths, c=SENSOR_COLORS[sensor_id], 
                      s=5, alpha=0.2, label=SENSOR_NAMES[sensor_id])
    
    # Plot tracks
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, track in enumerate(tracks):
        if track['status'] != 'confirmed':
            continue
        history = track.get('history', [])
        if len(history) < 2:
            continue
        
        color = colors[i % len(colors)]
        norths = [h[0] for h in history]
        easts = [h[1] for h in history]
        
        ax.plot(easts, norths, '-', color=color, linewidth=2, alpha=0.8)
        ax.plot(easts[-1], norths[-1], 'o', color=color, markersize=10)
        
        # Velocity arrow
        vel = track.get('velocity', [0, 0])
        if isinstance(vel, dict):
            vel = [vel.get('north', 0), vel.get('east', 0)]
        scale = 3
        ax.arrow(easts[-1], norths[-1], vel[1]*scale, vel[0]*scale,
                head_width=2, head_length=1, fc=color, ec=color, alpha=0.7)
        
        name = track.get('target_name', f"Track {track['track_id']}")
        ax.annotate(name, (easts[-1], norths[-1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Observer
    ax.plot(0, 0, 'k^', markersize=12)
    ax.annotate('Observer', (0, 0), xytext=(5, -15), fontsize=8)
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)


def plot_ground_truth_comparison(ax, tracks, ground_truth):
    """Plot tracks vs ground truth."""
    # Group ground truth by target
    gt_by_target = {}
    for gt in ground_truth:
        tid = gt['targetID']
        if tid not in gt_by_target:
            gt_by_target[tid] = []
        gt_by_target[tid].append(gt['position'])
    
    # Plot ground truth
    colors = ['#9b59b6', '#1abc9c', '#e67e22']
    for i, (tid, positions) in enumerate(gt_by_target.items()):
        norths = [p[0] for p in positions]
        easts = [p[1] for p in positions]
        ax.plot(easts, norths, '--', color=colors[i % len(colors)], 
               linewidth=2, alpha=0.7, label=f'GT Target {tid}')
    
    # Plot tracks
    for track in tracks:
        if track['status'] != 'confirmed':
            continue
        history = track.get('history', [])
        if len(history) < 2:
            continue
        norths = [h[0] for h in history]
        easts = [h[1] for h in history]
        ax.plot(easts, norths, '-', linewidth=2, alpha=0.8, 
               label=track.get('target_name', 'Track'))
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_aspect('equal')


def plot_maneuver_status(ax, tracks):
    """Plot maneuver probability for each track."""
    confirmed = [t for t in tracks if t['status'] == 'confirmed']
    if not confirmed:
        ax.text(0.5, 0.5, 'No confirmed tracks', ha='center', va='center')
        return
    
    names = [t.get('target_name', f"T{t['track_id']}") for t in confirmed]
    probs = [t.get('maneuver_probability', 0) for t in confirmed]
    
    colors = ['#e74c3c' if p > 0.5 else '#2ecc71' for p in probs]
    bars = ax.barh(names, probs, color=colors, alpha=0.8)
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Maneuver Probability')
    
    for bar, prob in zip(bars, probs):
        status = "MANEUVER" if prob > 0.5 else "steady"
        ax.text(min(prob + 0.05, 0.95), bar.get_y() + bar.get_height()/2,
               f'{prob:.0%} ({status})', va='center', fontsize=9)


def plot_process_noise(ax, tracks):
    """Plot adaptive process noise Q for each track."""
    confirmed = [t for t in tracks if t['status'] == 'confirmed']
    if not confirmed:
        ax.text(0.5, 0.5, 'No confirmed tracks', ha='center', va='center')
        return
    
    names = [t.get('target_name', f"T{t['track_id']}") for t in confirmed]
    q_values = [t.get('process_noise_Q', 0.5) for t in confirmed]
    
    # Color by Q value
    norm = plt.Normalize(0.1, 2.0)
    colors = plt.cm.RdYlGn_r(norm(q_values))
    
    bars = ax.barh(names, q_values, color=colors, alpha=0.8)
    
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Steady')
    ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='Maneuver')
    ax.set_xlabel('Process Noise Q')
    ax.legend(loc='lower right', fontsize=7)
    
    for bar, q in zip(bars, q_values):
        ax.text(q + 0.05, bar.get_y() + bar.get_height()/2,
               f'Q={q:.2f}', va='center', fontsize=9)


def plot_sensor_reliability(ax, ml_status):
    """Plot sensor reliability scores."""
    sensor_stats = ml_status.get('sensor_stats', {})
    
    if not sensor_stats:
        ax.text(0.5, 0.5, 'ML not enabled', ha='center', va='center')
        return
    
    names = []
    scores = []
    colors = []
    
    for sid, stats in sorted(sensor_stats.items()):
        names.append(stats['sensor_name'])
        scores.append(stats['reliability_score'])
        colors.append(SENSOR_COLORS.get(int(sid), '#95a5a6'))
    
    bars = ax.barh(names, scores, color=colors, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Reliability Score (1 = best)')
    
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', fontsize=9)


def plot_track_stats(ax, result_ml, result_fixed):
    """Plot comparison statistics."""
    metrics = ['Confirmed Tracks', 'Total Tracks', 'Detections']
    ml_values = [
        result_ml['num_confirmed_tracks'],
        result_ml['num_tracks'],
        result_ml['num_detections'] / 100  # Scale for visibility
    ]
    fixed_values = [
        result_fixed['num_confirmed_tracks'],
        result_fixed['num_tracks'],
        result_fixed['num_detections'] / 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, ml_values, width, label='ML-Enhanced', color='#3498db')
    ax.bar(x + width/2, fixed_values, width, label='Fixed', color='#95a5a6')
    
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()


def plot_info_panel(ax, result_ml, result_fixed, data):
    """Plot information panel with ML status."""
    
    ml_status = result_ml.get('ml_status', {})
    
    info_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     ML-ENHANCED SENSOR FUSION - INFERENCE RESULTS                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  DATA SOURCE: {data.get('scenario', 'Unknown'):15}  │  Duration: {data.get('duration_seconds', 0):.1f}s                      ║
║  Detections:  {result_ml['num_detections']:15}  │  Ground Truth Targets: {data.get('num_targets', '?')}                     ║
║                                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                              ML ENHANCEMENT STATUS                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Maneuver Detector:     {ml_status.get('maneuver_detector', 'N/A'):10}  │  Adapts Q (process noise)            ║
║  Sensor Reliability:    {ml_status.get('sensor_reliability', 'N/A'):10}  │  Adapts R (measurement noise)        ║
║                                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                               TRACKING RESULTS                                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                          ML-Enhanced    │    Fixed Kalman                         ║
║  Confirmed Tracks:      {result_ml['num_confirmed_tracks']:10}    │    {result_fixed['num_confirmed_tracks']:10}                          ║
║  Total Tracks:          {result_ml['num_tracks']:10}    │    {result_fixed['num_tracks']:10}                          ║
║                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

    HOW ML HELPS:
    • Process Noise Q: Low (0.1) for steady targets → smooth tracks
                       High (2.0) for maneuvering → responsive tracking
    • Measurement Noise R: Based on sensor accuracy learned from training data
    """
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=9,
           verticalalignment='top', horizontalalignment='left')


def main():
    parser = argparse.ArgumentParser(description='ML-Enhanced Sensor Fusion Demo')
    parser.add_argument('--scenario', type=str, default='scenario16',
                       choices=['scenario13', 'scenario16', 'scenario17', 'scenario22'],
                       help='Test scenario (default: scenario16)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ML-ENHANCED SENSOR FUSION - INFERENCE DEMO")
    print("=" * 70)
    print(f"\n  Test scenario: {args.scenario}")
    print("  (Environment 2 data - NOT used for training)")
    
    # Load test data
    print("\n  Loading test data...")
    data = load_test_scenario(args.scenario)
    print(f"  Detections: {len(data['detections'])}")
    print(f"  Ground truth: {len(data['ground_truth'])}")
    
    # Run with ML
    print("\n  Running ML-enhanced Kalman filter...")
    result_ml, agent_ml = run_inference(data, use_ml=True)
    print(f"  Confirmed tracks: {result_ml['num_confirmed_tracks']}")
    
    # Run without ML
    print("\n  Running fixed Kalman filter (baseline)...")
    result_fixed, agent_fixed = run_inference(data, use_ml=False)
    print(f"  Confirmed tracks: {result_fixed['num_confirmed_tracks']}")
    
    # Show ML track details
    print("\n  ML-Enhanced Track Details:")
    for track in result_ml['tracks']:
        if track['status'] == 'confirmed':
            maneuver = track.get('maneuver_probability', 0)
            q = track.get('process_noise_Q', 0.5)
            status = "MANEUVERING" if maneuver > 0.5 else "steady"
            print(f"    {track['target_name']:12} | Maneuver: {maneuver:5.1%} ({status:11}) | Q={q:.2f}")
    
    # Create visualization
    print("\n  Creating visualization...")
    fig = create_comparison_visualization(data, result_ml, result_fixed)
    
    print("\n  Displaying plots. Close window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
