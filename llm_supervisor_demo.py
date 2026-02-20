"""
LLM Supervisor Agent Demo

Demonstrates the LLM-powered supervisor that orchestrates RAMS agents
and provides expert maritime reasoning.

Requirements:
- Set NTNU_API_KEY environment variable
- Install: pip install openai pyyaml

Usage:
    python llm_supervisor_demo.py
    python llm_supervisor_demo.py --test-fallback  # Test without LLM
"""

import os
import sys
import time
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="LLM Supervisor Agent Demo")
    parser.add_argument('--test-fallback', action='store_true',
                        help='Test fallback mode without LLM')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM SUPERVISOR AGENT DEMO")
    print("Vessel RAMS Framework")
    print("=" * 70)
    
    # Check for API key
    api_key = os.environ.get('NTNU_API_KEY', '')
    if not api_key and not args.test_fallback:
        print("\n⚠️  NTNU_API_KEY not set. Running in fallback mode.")
        print("   Set with: $env:NTNU_API_KEY = 'your-key'")
        args.test_fallback = True
    
    # Import RAMS agents
    print("\n[1] Loading RAMS agents...")
    
    try:
        from rams_agents import (
            SafetyAgent,
            ReliabilityAgent,
            AvailabilityAgent,
            MaintainabilityAgent,
            LLMSupervisorAgent,
            create_llm_supervisor
        )
        print("    ✓ All agents loaded successfully")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return 1
    
    # Initialize agents
    print("\n[2] Initializing agents...")
    
    safety_agent = SafetyAgent(use_rl_policy=True)
    print(f"    ✓ SafetyAgent (RL+PSF: {safety_agent.uses_rl_policy})")
    
    reliability_agent = ReliabilityAgent()
    print("    ✓ ReliabilityAgent")
    
    availability_agent = AvailabilityAgent()
    print("    ✓ AvailabilityAgent")
    
    maintenance_agent = MaintainabilityAgent()
    print("    ✓ MaintainabilityAgent")
    
    # Initialize LLM Supervisor
    if args.test_fallback:
        # Force fallback by not providing API key
        from rams_agents.llm_supervisor_agent import LLMConfig, LLMSupervisorAgent
        llm_config = LLMConfig(api_key='', api_base='', model='none')
        supervisor = LLMSupervisorAgent(llm_config=llm_config)
        print("    ✓ LLMSupervisorAgent (FALLBACK MODE)")
    else:
        supervisor = create_llm_supervisor()
        if supervisor.is_ready:
            print(f"    ✓ LLMSupervisorAgent (LLM: {supervisor.llm_config.model})")
        else:
            print("    ⚠ LLMSupervisorAgent (fallback - no connection)")
    
    # Create synthetic scenario data
    print("\n[3] Running RAMS scenario: Head-on collision situation...")
    print("-" * 70)
    
    # Simulate a scenario with multiple RAMS concerns
    scenarios = [
        {
            'name': 'Normal Operations',
            'safety': {
                'risk_level': 'LOW',
                'safety_index': 95,
                'active_tracks': 2,
                'rl_psf_active': True,
                'maneuvers': []
            },
            'reliability': {
                'overall_health': 'GOOD',
                'components': {
                    'main_engine': {'rul_hours': 2500, 'health': 0.92},
                    'generator_1': {'rul_hours': 1800, 'health': 0.88}
                },
                'critical_alerts': [],
                'warnings': []
            },
            'availability': {
                'mode': 'FULL_POWER',
                'capability_pct': 100,
                'anomaly_score': 0.1,
                'sensors': {'radar': 'HEALTHY', 'lidar': 'HEALTHY', 'camera': 'HEALTHY'},
                'anomalies': []
            },
            'maintenance': {
                'pending_tasks_count': 1,
                'maintainability_metric': 0.95,
                'cost_forecast': {'pending_cost': 2000, 'pending_downtime_hours': 4},
                'recommendations': ['LOW: Scheduled inspection in 200 hours']
            }
        },
        {
            'name': 'Collision Risk Scenario',
            'safety': {
                'risk_level': 'HIGH',
                'safety_index': 35,
                'active_tracks': 3,
                'rl_psf_active': True,
                'maneuvers': [
                    {'action': 'ALTER_COURSE', 'direction': 'STARBOARD', 'magnitude': 30,
                     'reason': 'COLREGS Rule 14 - Head-on situation'},
                    {'action': 'REDUCE_SPEED', 'magnitude': 20, 'reason': 'PSF intervention'}
                ]
            },
            'reliability': {
                'overall_health': 'GOOD',
                'components': {
                    'main_engine': {'rul_hours': 2500, 'health': 0.92},
                    'generator_1': {'rul_hours': 1800, 'health': 0.88}
                },
                'critical_alerts': [],
                'warnings': []
            },
            'availability': {
                'mode': 'FULL_POWER',
                'capability_pct': 100,
                'anomaly_score': 0.15,
                'sensors': {'radar': 'HEALTHY', 'lidar': 'HEALTHY', 'camera': 'HEALTHY'},
                'anomalies': []
            },
            'maintenance': {
                'pending_tasks_count': 1,
                'maintainability_metric': 0.95,
                'cost_forecast': {'pending_cost': 2000, 'pending_downtime_hours': 4},
                'recommendations': ['LOW: Scheduled inspection in 200 hours']
            }
        },
        {
            'name': 'Multi-Concern Scenario',
            'safety': {
                'risk_level': 'CRITICAL',
                'safety_index': 15,
                'active_tracks': 4,
                'rl_psf_active': True,
                'maneuvers': [
                    {'action': 'ALTER_COURSE', 'direction': 'STARBOARD', 'magnitude': 45,
                     'reason': 'COLREGS Rule 15 - Crossing from starboard'},
                    {'action': 'REDUCE_SPEED', 'magnitude': 50, 'reason': 'Emergency PSF'}
                ]
            },
            'reliability': {
                'overall_health': 'DEGRADED',
                'components': {
                    'main_engine': {'rul_hours': 150, 'health': 0.45},
                    'generator_1': {'rul_hours': 80, 'health': 0.35}
                },
                'critical_alerts': ['Generator 1 RUL below 100 hours - schedule maintenance'],
                'warnings': ['Main engine showing degradation signs']
            },
            'availability': {
                'mode': 'DEGRADED',
                'capability_pct': 65,
                'anomaly_score': 0.72,
                'sensors': {'radar': 'HEALTHY', 'lidar': 'DEGRADED', 'camera': 'FAILED'},
                'anomalies': ['Camera sensor failed', 'LiDAR showing intermittent readings']
            },
            'maintenance': {
                'pending_tasks_count': 5,
                'maintainability_metric': 0.65,
                'cost_forecast': {'pending_cost': 45000, 'pending_downtime_hours': 72},
                'recommendations': [
                    'CRITICAL: Generator 1 requires immediate inspection',
                    'HIGH: Camera sensor replacement needed',
                    'MEDIUM: LiDAR calibration overdue'
                ]
            }
        }
    ]
    
    # Run through scenarios
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"{'='*70}")
        
        # Update supervisor with RAMS data
        supervisor.update_rams_state(
            safety_report=scenario['safety'],
            reliability_report=scenario['reliability'],
            availability_report=scenario['availability'],
            maintenance_report=scenario['maintenance']
        )
        
        # Get LLM reasoning
        print(f"\n[Reasoning @ {datetime.now().strftime('%H:%M:%S')}]")
        start = time.time()
        result = supervisor.act()
        elapsed = time.time() - start
        
        if result.get('status') == 'complete':
            print(f"  Latency: {result['metadata']['latency_ms']:.0f}ms")
            print(f"  Tokens: {result['metadata']['tokens_used']}")
            print(f"  LLM Used: {result['metadata']['llm_used']}")
            
            print(f"\n📊 RISK ASSESSMENT:")
            print(f"   {result['risk_assessment']}")
            
            print(f"\n🎯 PRIORITY RANKING:")
            for j, priority in enumerate(result['priority_ranking'][:5], 1):
                print(f"   {j}. {priority}")
            
            print(f"\n⚡ RECOMMENDED ACTIONS:")
            for action in result['recommended_actions'][:5]:
                print(f"   • {action}")
            
            if result['alerts']:
                print(f"\n🚨 ALERTS:")
                for alert in result['alerts']:
                    print(f"   ⚠️  {alert}")
            
            print(f"\n📝 SUMMARY:")
            print(f"   {result['summary']}")
        else:
            print(f"  Status: {result.get('status')}")
            print(f"  Reason: {result.get('reason')}")
        
        # Small delay between scenarios
        if i < len(scenarios) - 1:
            time.sleep(1)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("SESSION STATISTICS")
    print(f"{'='*70}")
    stats = supervisor.get_statistics()
    print(f"  Total Decisions: {stats['total_decisions']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Avg Latency: {stats['avg_latency_ms']:.0f}ms")
    print(f"  LLM Ready: {stats['llm_ready']}")
    print(f"  Model: {stats['model']}")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
