#!/usr/bin/env python3
"""Vessel AI System - Main Demo

Usage:
    python main_demo.py              # Run with synthetic data
    python main_demo.py --real       # Run with real Autoferry data (scenario2)
    python main_demo.py --scenario scenario16   # Run with specific scenario
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from orchestrator import VesselOrchestrator


def load_autoferry_real_data(scenario: str = 'scenario2'):
    """Load real data from Autoferry benchmark dataset."""
    from data.autoferry_loader import AutoferryDataLoader
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    dataset_path = os.path.join(data_dir, 'sensor_fusion_dataset')
    
    if not os.path.exists(dataset_path):
        print(f"  ERROR: Autoferry dataset not found at {dataset_path}")
        print("  Please run: git clone https://github.com/Autoferry/sensor_fusion_dataset.git")
        return None
    
    loader = AutoferryDataLoader(dataset_path)
    print(f"Loading real Autoferry data: {scenario}")
    
    info = loader.get_scenario_info(scenario)
    print(f"  Scenario: {info['description']}")
    print(f"  Targets: {', '.join(info['target_names'])}")
    print(f"  Duration: {info['time_range_seconds']:.1f}s")
    print(f"  Sensors: {info['detections_by_sensor']}")
    
    data = loader.load_for_kalman_filter(scenario)
    print(f"  Kalman-compatible detections: {len(data['detections'])}")
    return data


def generate_autoferry_format_data(n_targets=3, n_timesteps=20):
    """Generate synthetic sensor data in Autoferry format (compatible with Kalman filter)."""
    print("Generating Autoferry-format sensor data...")
    np.random.seed(42)
    detections, ground_truth = [], []
    
    # Target definitions matching Autoferry benchmark
    target_names = {1: 'Havfruen', 2: 'Vessel', 3: 'Jetboat'}
    targets = [
        {'id': 1, 'start_north': 150.0, 'start_east': 80.0, 'vn': -1.5, 've': -0.5},
        {'id': 2, 'start_north': 300.0, 'start_east': 0.0, 'vn': -2.0, 've': 0.0},
        {'id': 3, 'start_north': 200.0, 'start_east': -50.0, 'vn': -2.0, 've': 1.0}
    ][:n_targets]
    
    # Sensor configurations: (sensorID, name, detection_prob, noise_std)
    sensors = [
        (1, 'LiDAR', 0.9, 1.0),
        (2, 'Radar', 0.85, 3.0),
        (3, 'IR_Camera', 0.75, 4.0),
        (4, 'EO_Camera', 0.8, 3.0)
    ]
    
    for t in range(n_timesteps):
        ts = t * 0.5  # 2 Hz sampling
        for tgt in targets:
            # Ground truth position
            n = tgt['start_north'] + tgt['vn'] * ts
            e = tgt['start_east'] + tgt['ve'] * ts
            
            ground_truth.append({
                'time': ts, 
                'targetID': tgt['id'], 
                'position': [n, e, 0.0]  # NED format
            })
            
            # Generate detections from each sensor
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
    
    print(f"  Generated {len(detections)} detections from {len(sensors)} sensors")
    return {'detections': detections, 'ground_truth': ground_truth}


def generate_propulsion_data(n=200):
    """Generate propulsion system data"""
    print("Generating propulsion data...")
    np.random.seed(42)
    data = pd.DataFrame({
        'azimuth_stbd_rpm': np.random.normal(600, 30, n),
        'azimuth_port_rpm': np.random.normal(580, 30, n),
        'tunnel_rpm': np.random.normal(400, 40, n),
        'gen1_power': np.random.normal(180, 15, n),
        'gen2_power': np.random.normal(120, 12, n),
        'gen1_voltage': np.random.normal(400, 1.5, n),
        'gen2_voltage': np.random.normal(400, 1.5, n),
        'gen1_coolant_temp': np.random.normal(82, 3, n),
        'gen2_coolant_temp': np.random.normal(80, 3, n),
        'gen1_oil_pressure': np.random.normal(4.2, 0.2, n)
    })
    data.loc[50:70, 'gen1_voltage'] += np.random.normal(0, 3, 21)
    print(f"  Generated {n} samples")
    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vessel AI System Demo')
    parser.add_argument('--real', action='store_true', 
                        help='Use real Autoferry dataset instead of synthetic')
    parser.add_argument('--scenario', type=str, default='scenario2',
                        help='Autoferry scenario to load (default: scenario2)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  R/V GUNNERUS AI SYSTEM - MULTI-AGENT DEMO")
    print("="*70)
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              R/V GUNNERUS - NTNU RESEARCH VESSEL              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  MMSI: 258342000  │  IMO: 9371361  │  Home: Trondheim         ║
    ║  LOA: 36.25m  │  Beam: 9.6m  │  2x Azimuth + 1x Tunnel        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Agents: Sensor Fusion (Kalman) │ Collision Avoidance (COLREGS)
            Dynamic Positioning    │ Propulsion Health
    """)
    
    print("\n" + "="*70)
    print("  DATA LOADING")
    print("="*70)
    
    if args.real:
        sensor_data = load_autoferry_real_data(args.scenario)
        if sensor_data is None:
            print("  Falling back to synthetic data...")
            sensor_data = generate_autoferry_format_data()
        data_source = f"Autoferry {args.scenario}"
    else:
        sensor_data = generate_autoferry_format_data()
        data_source = "Synthetic"
    
    print(f"\n  Data source: {data_source}")
    
    vessel_state = {'north': 5.0, 'east': -3.0, 'heading': 47.0, 'surge': 0.15,
                    'sway': -0.08, 'yaw_rate': 0.02, 'latitude': 63.4305, 'longitude': 10.3951}
    propulsion_data = generate_propulsion_data()
    
    print("\n" + "="*70)
    print("  RUNNING ANALYSIS")
    print("="*70)
    orchestrator = VesselOrchestrator(vessel_name="Vessel")
    
    results = orchestrator.run_full_analysis(
        sensor_data=sensor_data,
        vessel_state=vessel_state,
        propulsion_data=propulsion_data,
        dp_setpoint={'north': 0.0, 'east': 0.0, 'heading': 45.0}
    )
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    
    if results.get('status') == 'success':
        status = results.get('vessel_status', {})
        print(f"\n  Vessel Status:")
        print(f"    Position:    {status.get('position_status', 'N/A')}")
        print(f"    Propulsion:  {status.get('propulsion_status', 'N/A')}")
        print(f"    Overall:     {status.get('overall_risk', 'N/A')}")
        print(f"    Targets:     {status.get('tracked_targets', 0)}")
        
        summary = results.get('summary', {})
        
        # Sensor Fusion with Kalman Filtering
        fusion = summary.get('sensor_fusion', {})
        print(f"\n  Sensor Fusion (Kalman Filter):")
        print(f"    Tracked targets:  {fusion.get('tracked_targets', 0)}")
        print(f"    Detections:       {fusion.get('detections_processed', 0)}")
        print(f"    Kalman filter:    {fusion.get('kalman_filter', 'N/A')}")
        
        # Collision Avoidance with COLREGS
        ca = summary.get('collision_avoidance', {})
        print(f"\n  Collision Avoidance (COLREGS):")
        print(f"    Overall risk:   {ca.get('overall_risk', 'CLEAR')}")
        print(f"    Critical risks: {ca.get('critical_risks', 0)}")
        print(f"    High risks:     {ca.get('high_risks', 0)}")
        print(f"    COLREGS:        {ca.get('colregs', 'N/A')}")
        
        # Show collision risk details if any
        ca_detail = results.get('detailed_results', {}).get('collision_avoidance', {})
        risks = ca_detail.get('collision_risks', [])
        if risks:
            print(f"\n  Collision Risks Detected:")
            for risk in risks[:3]:  # Show top 3
                print(f"    • {risk['target_name']:12} [{risk['risk_level'].upper():8}] "
                      f"CPA={risk['cpa_m']:.0f}m TCPA={risk['tcpa_minutes']:.1f}min")
                print(f"      {risk['colreg_rule']}")
        
        dp = summary.get('dynamic_positioning', {})
        print(f"\n  DP Control:")
        print(f"    Position Error: {dp.get('position_error_m', 0):.2f} m")
        print(f"    Heading Error:  {dp.get('heading_error_deg', 0):.2f}°")
        
        dp_detail = results.get('detailed_results', {}).get('dp_control', {})
        if 'thruster_commands' in dp_detail:
            print(f"\n  Thruster Commands:")
            for cmd in dp_detail['thruster_commands']:
                print(f"    {cmd['id']:15}: RPM={cmd['rpm']:.0f}, Az={cmd['azimuth_deg']:.1f}°")
        
        prop = summary.get('propulsion', {})
        print(f"\n  Propulsion Health:")
        print(f"    Faults: {prop.get('faults_detected', 0)}")
        print(f"    Status: {prop.get('status', 'N/A')}")
        
        print(f"\n  Execution Log:")
        for entry in results.get('execution_log', []):
            print(f"    {entry['agent']:25} : {entry['status']}")
        
        print(f"\n  Duration: {results.get('duration_seconds', 0):.2f}s")
    else:
        print(f"\n  ❌ Analysis failed: {results.get('error')}")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print("""
    ✓ Multi-sensor fusion with KALMAN FILTERING
    ✓ Collision avoidance with COLREGS compliance (Rules 13-17)
    ✓ Dynamic Positioning control with thrust allocation
    ✓ Propulsion health monitoring with anomaly detection
    
    Open-source data sources:
    • github.com/Autoferry/sensor_fusion_dataset
    • open-simulation-platform.github.io (Vessel FMUs)
    • MarineTraffic AIS (MMSI: 258342000)
    """)
    
    return results

if __name__ == "__main__":
    main()
