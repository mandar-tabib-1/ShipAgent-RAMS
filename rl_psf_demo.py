#!/usr/bin/env python3
"""
RL+PSF Collision Avoidance Demo

Demonstrates the integration of:
1. Pre-trained RL policy (PPO) for collision avoidance
2. Potential Safety Function (PSF) for formal safety guarantees
3. SafetyAgent with COLREGS compliance

This demo shows how the RL policy proposes avoidance maneuvers and
the PSF filter ensures they maintain safety constraints (CPA > 50m).

Attribution:
- RL policy based on Acmece/rl-collision-avoidance (arXiv:1709.10082)
- See rams_agents/CREDITS.md for full attribution

Usage:
    python rl_psf_demo.py                    # Synthetic collision scenarios
    python rl_psf_demo.py --real             # Use AutoFerry dataset
    python rl_psf_demo.py --visualize        # Show real-time visualization
"""

import argparse
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time


def create_synthetic_collision_scenario(scenario_type: str = "head_on") -> List[Dict]:
    """
    Create synthetic collision scenarios for demo.
    
    Args:
        scenario_type: 'head_on', 'crossing', 'overtaking', or 'multi_target'
        
    Returns:
        List of timestep dicts with target positions
    """
    timesteps = []
    
    if scenario_type == "head_on":
        # Head-on: Target approaching from ahead
        for t in range(100):
            target_n = 500 - t * 6  # Approaching at 6 m/s
            target_e = 5 + np.sin(t * 0.1) * 10  # Slight lateral motion
            timesteps.append({
                'timestamp': t,
                'ownship_position': (0, 0),
                'ownship_velocity': (5, 0),  # Moving north at 5 m/s
                'targets': [{
                    'id': 1001,
                    'position': (target_n, target_e),
                    'velocity': (-6, 0)
                }]
            })
    
    elif scenario_type == "crossing":
        # Crossing: Target from starboard
        for t in range(100):
            target_n = 200 + t * 2
            target_e = 400 - t * 5  # Approaching from starboard
            timesteps.append({
                'timestamp': t,
                'ownship_position': (0, 0),
                'ownship_velocity': (5, 0),
                'targets': [{
                    'id': 2001,
                    'position': (target_n, target_e),
                    'velocity': (2, -5)
                }]
            })
    
    elif scenario_type == "overtaking":
        # Being overtaken from astern
        for t in range(100):
            # Faster vessel approaching from behind
            target_n = -300 + t * 8  # Faster at 8 m/s
            target_e = 20
            timesteps.append({
                'timestamp': t,
                'ownship_position': (t * 5, 0),  # Own ship moving
                'ownship_velocity': (5, 0),
                'targets': [{
                    'id': 3001,
                    'position': (target_n, target_e),
                    'velocity': (8, 0)
                }]
            })
    
    elif scenario_type == "multi_target":
        # Multiple targets - complex scenario
        for t in range(100):
            timesteps.append({
                'timestamp': t,
                'ownship_position': (0, 0),
                'ownship_velocity': (5, 0),
                'targets': [
                    {  # Head-on
                        'id': 4001,
                        'position': (400 - t * 5, 10),
                        'velocity': (-5, 0)
                    },
                    {  # Crossing from port
                        'id': 4002,
                        'position': (150, -300 + t * 4),
                        'velocity': (0, 4)
                    },
                    {  # Distant target
                        'id': 4003,
                        'position': (800, 200 - t),
                        'velocity': (-2, -1)
                    }
                ]
            })
    
    return timesteps


def run_rl_psf_demo(scenario_type: str = "head_on", 
                    use_real_data: bool = False,
                    visualize: bool = False,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Run RL+PSF collision avoidance demo.
    
    Args:
        scenario_type: Type of collision scenario
        use_real_data: Use AutoFerry dataset if available
        visualize: Show matplotlib visualization
        verbose: Print detailed output
        
    Returns:
        Dict with demo results and statistics
    """
    # Import SafetyAgent
    try:
        from rams_agents.safety_agent import SafetyAgent, RiskLevel
        from rams_agents.psf_filter import PSFInterventionType
    except ImportError as e:
        print(f"Error importing RAMS agents: {e}")
        print("Make sure you're running from the gunnerus_ai_system directory")
        return {'error': str(e)}
    
    print("\n" + "="*70)
    print("RL+PSF COLLISION AVOIDANCE DEMO")
    print("="*70)
    print("\nComponents:")
    print("  - RL Policy: Pre-trained PPO (arXiv:1709.10082)")
    print("  - PSF Filter: Control Barrier Function (CPA > 50m)")
    print("  - Integration: SafetyAgent with COLREGS compliance")
    print("="*70 + "\n")
    
    # Initialize SafetyAgent with RL+PSF
    agent = SafetyAgent(use_rl_policy=True)
    
    print(f"[SafetyAgent] RL policy enabled: {agent.uses_rl_policy}")
    
    # Generate or load scenario data
    if use_real_data:
        # Try to load AutoFerry data
        try:
            from rams_agents.data_loaders.navigation_loader import NavigationDataLoader
            loader = NavigationDataLoader()
            timesteps = loader.load_scenario("scenario16")[:100]
            print(f"Loaded {len(timesteps)} timesteps from AutoFerry scenario16")
        except Exception as e:
            print(f"Could not load AutoFerry data: {e}")
            print("Falling back to synthetic data")
            timesteps = create_synthetic_collision_scenario(scenario_type)
    else:
        print(f"\nScenario: {scenario_type.upper()}")
        timesteps = create_synthetic_collision_scenario(scenario_type)
    
    # Run simulation
    results = {
        'scenario': scenario_type,
        'timesteps': len(timesteps),
        'maneuvers': [],
        'risk_history': [],
        'psf_interventions': []
    }
    
    print(f"\nRunning {len(timesteps)} timesteps...")
    print("-" * 50)
    
    for i, ts_data in enumerate(timesteps):
        # Run agent cycle
        result = agent.run_cycle(ts_data)
        
        # Record results
        risk = result.get('overall_risk', 'UNKNOWN')
        safety_idx = result.get('safety_index', 100)
        maneuvers = result.get('avoidance_maneuvers', [])
        
        results['risk_history'].append({
            'timestep': i,
            'risk': risk,
            'safety_index': safety_idx
        })
        
        # Print interesting events
        if verbose and (i % 20 == 0 or risk in ['CRITICAL', 'HIGH'] or maneuvers):
            tracks = result.get('tracks', [])
            print(f"\n[t={i:3d}] Risk: {risk:8s} | Safety Index: {safety_idx:.0f}")
            
            if tracks:
                for t in tracks[:2]:  # Show top 2 tracks
                    print(f"         Track {t.get('target_id', '?')}: "
                          f"CPA={t.get('cpa', 0):.0f}m, "
                          f"TCPA={t.get('tcpa', 0):.0f}s, "
                          f"Risk={t.get('risk_level', '?')}")
            
            if maneuvers:
                for m in maneuvers:
                    source = m.get('source', 'UNKNOWN')
                    action = m.get('action', 'UNKNOWN')
                    direction = m.get('direction', '')
                    magnitude = m.get('magnitude', 0)
                    
                    results['maneuvers'].append({
                        'timestep': i,
                        'action': action,
                        'source': source,
                        'direction': direction,
                        'magnitude': magnitude
                    })
                    
                    if source == 'RL_PSF':
                        psf_intervention = m.get('psf_intervention', 'none')
                        barrier = m.get('barrier_value', 0)
                        conf = m.get('rl_confidence', 0)
                        
                        results['psf_interventions'].append({
                            'timestep': i,
                            'intervention': psf_intervention,
                            'barrier_value': barrier
                        })
                        
                        print(f"         [RL+PSF] {action} {direction} {magnitude:.0f}° "
                              f"(PSF: {psf_intervention}, φ={barrier:.0f}m, conf={conf:.2f})")
                    else:
                        print(f"         [COLREGS] {action} {direction} {magnitude:.0f}°")
    
    # Final statistics
    stats = agent.get_rl_psf_stats()
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETE - STATISTICS")
    print("=" * 50)
    
    print(f"\nRL+PSF Usage:")
    print(f"  - Total calls: {stats.get('total_calls', 0)}")
    print(f"  - RL policy used: {stats.get('rl_used', 0)} times")
    print(f"  - PSF interventions: {stats.get('interventions', 0)}")
    print(f"  - Intervention rate: {stats.get('intervention_rate', 0):.1%}")
    
    # Count risk events
    critical_count = sum(1 for r in results['risk_history'] if r['risk'] == 'CRITICAL')
    high_count = sum(1 for r in results['risk_history'] if r['risk'] == 'HIGH')
    
    print(f"\nRisk Events:")
    print(f"  - Critical risk timesteps: {critical_count}")
    print(f"  - High risk timesteps: {high_count}")
    print(f"  - Avoidance maneuvers issued: {len(results['maneuvers'])}")
    
    # RL vs COLREGS breakdown
    rl_maneuvers = sum(1 for m in results['maneuvers'] if m.get('source') == 'RL_PSF')
    colregs_maneuvers = sum(1 for m in results['maneuvers'] if m.get('source') == 'COLREGS_RULES')
    
    print(f"\nManeuver Sources:")
    print(f"  - RL+PSF: {rl_maneuvers}")
    print(f"  - Rule-based COLREGS: {colregs_maneuvers}")
    
    # PSF intervention breakdown
    if results['psf_interventions']:
        intervention_types = {}
        for p in results['psf_interventions']:
            t = p['intervention']
            intervention_types[t] = intervention_types.get(t, 0) + 1
        
        print(f"\nPSF Intervention Types:")
        for t, count in intervention_types.items():
            print(f"  - {t}: {count}")
    
    print("\n" + "="*70)
    print("See rams_agents/CREDITS.md for RL policy attribution")
    print("="*70 + "\n")
    
    results['stats'] = stats
    
    # Visualization
    if visualize:
        try:
            visualize_results(results)
        except ImportError:
            print("Matplotlib not available for visualization")
    
    return results


def visualize_results(results: Dict[str, Any]) -> None:
    """Create visualization of demo results."""
    import matplotlib.pyplot as plt
    
    # Risk history
    timesteps = [r['timestep'] for r in results['risk_history']]
    safety_indices = [r['safety_index'] for r in results['risk_history']]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Safety Index over time
    ax1 = axes[0]
    ax1.plot(timesteps, safety_indices, 'b-', linewidth=2, label='Safety Index')
    ax1.axhline(y=30, color='r', linestyle='--', label='High Risk Threshold')
    ax1.axhline(y=60, color='orange', linestyle='--', label='Medium Risk Threshold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Safety Index')
    ax1.set_title('Safety Index Over Time (RL+PSF Demo)')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Mark maneuver events
    for m in results['maneuvers']:
        ts = m['timestep']
        color = 'green' if m.get('source') == 'RL_PSF' else 'orange'
        ax1.axvline(x=ts, color=color, alpha=0.5, linewidth=1)
    
    # Plot 2: PSF Barrier Values
    ax2 = axes[1]
    if results['psf_interventions']:
        psf_ts = [p['timestep'] for p in results['psf_interventions']]
        barriers = [p['barrier_value'] for p in results['psf_interventions']]
        
        ax2.scatter(psf_ts, barriers, c=['red' if b < 0 else 'green' for b in barriers],
                   s=50, label='PSF Barrier φ')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=2)
        ax2.axhline(y=10, color='orange', linestyle='--', label='Safety Margin')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Barrier Value φ (meters)')
        ax2.set_title('PSF Barrier Values (φ > 0 = Safe)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No PSF interventions recorded', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('rl_psf_demo_results.png', dpi=150)
    print("\nVisualization saved to: rl_psf_demo_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='RL+PSF Collision Avoidance Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rl_psf_demo.py                    # Head-on collision scenario
  python rl_psf_demo.py --scenario crossing # Crossing situation
  python rl_psf_demo.py --real             # Use AutoFerry dataset
  python rl_psf_demo.py --visualize        # Show matplotlib plots

Attribution:
  RL policy based on Acmece/rl-collision-avoidance (arXiv:1709.10082)
  See rams_agents/CREDITS.md for full attribution
        """
    )
    
    parser.add_argument('--scenario', type=str, default='head_on',
                       choices=['head_on', 'crossing', 'overtaking', 'multi_target'],
                       help='Collision scenario type (default: head_on)')
    
    parser.add_argument('--real', action='store_true',
                       help='Use real AutoFerry dataset')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show matplotlib visualization')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    results = run_rl_psf_demo(
        scenario_type=args.scenario,
        use_real_data=args.real,
        visualize=args.visualize,
        verbose=not args.quiet
    )
    
    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    exit(main())
