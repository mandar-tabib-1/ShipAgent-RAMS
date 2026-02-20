"""
RAMS Multi-Agent System Demonstration

Demonstrates agentic AI for Reliability, Availability, Maintainability, and Safety
in maritime vessel operations.

Data Sources:
    - UCI Naval Propulsion CBM Dataset (CC BY 4.0): Propulsion health monitoring
    - AutoFerry Sensor Fusion Dataset (MIT): Navigation safety

Usage:
    python rams_demo.py                          # Full demonstration
    python rams_demo.py --scenario collision     # Collision + degradation scenario
    python rams_demo.py --scenario maintenance   # Maintenance tradeoff scenario
    python rams_demo.py --steps 50               # Run for 50 timesteps

This is a methodology demonstration applicable to any naval vessel.
"""

import argparse
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Import RAMS agent framework
from rams_agents.base_agent import RAMSMetrics, RAMSCategory
from rams_agents.reliability_agent import ReliabilityAgent
from rams_agents.availability_agent import AvailabilityAgent
from rams_agents.maintainability_agent import MaintainabilityAgent
from rams_agents.safety_agent import SafetyAgent
from rams_agents.supervisor_agent import RAMSSupervisorAgent

# Import data loaders
from rams_agents.data_loaders.uci_naval_loader import UCINavalPropulsionLoader
from rams_agents.data_loaders.navigation_loader import NavigationDataLoader


class RAMSOrchestrator:
    """
    Orchestrates the multi-agent RAMS system.
    
    Coordinates:
    - Data loading from hybrid sources
    - Agent lifecycle management
    - Inter-agent communication
    - Overall system execution
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Initialize agents
        self.reliability_agent = ReliabilityAgent()
        self.availability_agent = AvailabilityAgent()
        self.maintainability_agent = MaintainabilityAgent()
        self.safety_agent = SafetyAgent()
        self.supervisor = RAMSSupervisorAgent()
        
        # Register agents with supervisor
        self.supervisor.register_agent(self.reliability_agent)
        self.supervisor.register_agent(self.availability_agent)
        self.supervisor.register_agent(self.maintainability_agent)
        self.supervisor.register_agent(self.safety_agent)
        
        # Data loaders
        self.propulsion_loader: Optional[UCINavalPropulsionLoader] = None
        self.navigation_loader: Optional[NavigationDataLoader] = None
        
        # Timestep tracking
        self.current_step = 0
        self.operating_hours = 0.0
        
    def initialize_data(self, prefer_synthetic: bool = False, scenario: str = 'scenario16'):
        """
        Initialize data loaders.
        
        Args:
            prefer_synthetic: If True, use synthetic data without download
            scenario: Navigation scenario ID for AutoFerry data
        """
        if self.verbose:
            print("\n" + "="*60)
            print("RAMS Multi-Agent System - Data Sources")
            print("="*60)
        
        # Load propulsion data (UCI Naval)
        self.propulsion_loader = UCINavalPropulsionLoader()
        propulsion_data = self.propulsion_loader.load(prefer_synthetic=prefer_synthetic)
        
        if self.verbose:
            stats = self.propulsion_loader.get_degradation_statistics()
            print(f"\n[Propulsion Data] UCI Naval CBM Dataset")
            print(f"  - Samples: {len(propulsion_data)}")
            print(f"  - Compressor decay range: [{stats['compressor_decay']['min']:.4f}, {stats['compressor_decay']['max']:.4f}]")
            print(f"  - Turbine decay range: [{stats['turbine_decay']['min']:.4f}, {stats['turbine_decay']['max']:.4f}]")
        
        # Initialize reliability agent with population statistics (semi-supervised learning)
        import numpy as np
        if self.propulsion_loader.data is not None:
            kMc_all = self.propulsion_loader.data['compressor_decay'].values
            kMt_all = self.propulsion_loader.data['turbine_decay'].values
            pop_stats = self.reliability_agent.initialize_from_population(kMc_all, kMt_all)
            if self.verbose:
                print(f"  - RUL Estimator initialized from population (semi-supervised)")
                print(f"    * Degradation rate prior: {pop_stats['degradation_rate']['mean']:.6f} HI/hour")
        
        # Load navigation data (AutoFerry)
        self.navigation_loader = NavigationDataLoader()
        available_scenarios = self.navigation_loader.list_scenarios()
        
        if self.verbose:
            print(f"\n[Navigation Data] AutoFerry Sensor Fusion Dataset")
            print(f"  - Available scenarios: {available_scenarios}")
            print(f"  - Selected: {scenario}")
        
        return True
    
    def run_step(self, 
                 propulsion_data: Dict[str, Any],
                 navigation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one timestep of the RAMS multi-agent system.
        
        This demonstrates the agentic cycle:
        1. Each agent perceives its domain data
        2. Each agent reasons and forms beliefs
        3. Each agent acts and produces recommendations
        4. Supervisor coordinates and makes tradeoff decisions
        
        Args:
            propulsion_data: Propulsion system data
            navigation_data: Navigation environment data
        
        Returns:
            Integrated RAMS report from supervisor
        """
        self.current_step += 1
        timestamp = propulsion_data.get('timestamp', self.current_step)
        self.operating_hours = timestamp
        
        # Phase 1: Individual agent cycles
        # Each agent: perceive → reason → act
        
        # Reliability Agent: Processes propulsion degradation
        reliability_env = {'propulsion': propulsion_data, 'timestamp': timestamp}
        reliability_result = self.reliability_agent.run_cycle(reliability_env)
        
        # Availability Agent: Monitors operational capability
        availability_env = {'propulsion': propulsion_data, 'timestamp': timestamp}
        availability_result = self.availability_agent.run_cycle(availability_env)
        
        # Maintainability Agent: Manages maintenance scheduling
        maintenance_env = {
            'timestamp': timestamp,
            'operating_hours': self.operating_hours
        }
        # Pass messages from reliability agent
        for msg in self.reliability_agent.get_outgoing_messages():
            if msg.recipient == 'MaintainabilityAgent':
                self.maintainability_agent.receive_message(msg)
        maintenance_result = self.maintainability_agent.run_cycle(maintenance_env)
        
        # Safety Agent: Monitors navigation safety
        safety_env = {
            'navigation': navigation_data,
            'ownship_position': navigation_data.get('ownship_position', [0, 0]),
            'ownship_velocity': [5, 0],  # 5 m/s north
            'targets': navigation_data.get('targets', []),
            'timestamp': timestamp
        }
        safety_result = self.safety_agent.run_cycle(safety_env)
        
        # Phase 2: Supervisor coordination
        # Supervisor aggregates results and makes tradeoff decisions
        supervisor_env = {
            'timestamp': timestamp,
            'agent_reports': {
                'ReliabilityAgent': reliability_result,
                'AvailabilityAgent': availability_result,
                'MaintainabilityAgent': maintenance_result,
                'SafetyAgent': safety_result
            }
        }
        supervisor_result = self.supervisor.run_cycle(supervisor_env)
        
        return {
            'step': self.current_step,
            'timestamp': timestamp,
            'agents': {
                'reliability': reliability_result,
                'availability': availability_result,
                'maintainability': maintenance_result,
                'safety': safety_result
            },
            'supervisor': supervisor_result,
            'rams_metrics': self.supervisor.get_rams_metrics().to_dict()
        }
    
    def run_demonstration(self, 
                          n_steps: int = 30,
                          scenario_type: str = 'full',
                          delay: float = 0.1) -> List[Dict[str, Any]]:
        """
        Run the full RAMS demonstration.
        
        Args:
            n_steps: Number of simulation steps
            scenario_type: 'full', 'collision', or 'maintenance'
            delay: Delay between steps for visualization
        
        Returns:
            List of step results
        """
        results = []
        
        if self.verbose:
            print("\n" + "="*60)
            print("RAMS Multi-Agent System - Starting Demonstration")
            print("="*60)
            print(f"Scenario: {scenario_type}")
            print(f"Steps: {n_steps}")
            print("="*60)
        
        # Get data iterators
        propulsion_iter = self.propulsion_loader.iterate_timesteps(batch_size=1)
        navigation_iter = self.navigation_loader.iterate_timesteps('scenario16', time_step=1.0)
        
        for step in range(n_steps):
            # Get propulsion data
            try:
                propulsion_batch = next(propulsion_iter)
                propulsion_data = propulsion_batch[0].to_dict() if propulsion_batch else {}
            except StopIteration:
                # Restart iterator
                propulsion_iter = self.propulsion_loader.iterate_timesteps(batch_size=1)
                propulsion_batch = next(propulsion_iter)
                propulsion_data = propulsion_batch[0].to_dict() if propulsion_batch else {}
            
            # Get navigation data
            try:
                navigation_data = next(navigation_iter)
            except StopIteration:
                # Restart iterator
                navigation_iter = self.navigation_loader.iterate_timesteps('scenario16', time_step=1.0)
                navigation_data = next(navigation_iter)
            
            # Inject scenario-specific conditions
            if scenario_type == 'collision':
                propulsion_data, navigation_data = self._inject_collision_scenario(
                    propulsion_data, navigation_data, step, n_steps
                )
            elif scenario_type == 'maintenance':
                propulsion_data, navigation_data = self._inject_maintenance_scenario(
                    propulsion_data, navigation_data, step, n_steps
                )
            
            # Run step
            result = self.run_step(propulsion_data, navigation_data)
            results.append(result)
            
            # Display progress
            if self.verbose:
                self._display_step_result(result)
            
            # Delay for visualization
            if delay > 0:
                time.sleep(delay)
        
        # Final summary
        if self.verbose:
            self._display_summary(results)
        
        return results
    
    def _inject_collision_scenario(self,
                                    propulsion_data: Dict[str, Any],
                                    navigation_data: Dict[str, Any],
                                    step: int,
                                    total_steps: int) -> tuple:
        """
        Inject Scenario A: Collision risk + propulsion degradation.
        
        Simulates a situation where:
        - A vessel is on collision course
        - Propulsion system is degraded
        - Supervisor must balance safety maneuver vs propulsion stress
        """
        # Simulate degradation progressing through scenario
        progress = step / total_steps
        
        # Degrade propulsion components
        propulsion_data['compressor_decay'] = max(0.95, 1.0 - progress * 0.06)
        propulsion_data['turbine_decay'] = max(0.975, 1.0 - progress * 0.03)
        
        # Inject collision target at mid-scenario
        if 0.3 <= progress <= 0.7:
            # Target approaching from east
            approach_phase = (progress - 0.3) / 0.4
            target_east = 400 - approach_phase * 350  # Getting closer
            target_north = 100 + approach_phase * 50
            
            navigation_data['targets'] = [{
                'id': 1,
                'name': 'Approaching Vessel',
                'position': (target_north, target_east)
            }]
        
        return propulsion_data, navigation_data
    
    def _inject_maintenance_scenario(self,
                                      propulsion_data: Dict[str, Any],
                                      navigation_data: Dict[str, Any],
                                      step: int,
                                      total_steps: int) -> tuple:
        """
        Inject Scenario B: Maintenance vs availability tradeoff.
        
        Simulates a situation where:
        - Maintenance is overdue
        - Mission requires sustained availability
        - Supervisor must decide: maintenance now vs defer
        """
        progress = step / total_steps
        
        # Moderate degradation
        propulsion_data['compressor_decay'] = max(0.96, 1.0 - progress * 0.05)
        propulsion_data['turbine_decay'] = max(0.98, 1.0 - progress * 0.025)
        
        # Simulate increased maintenance need via operating hours
        propulsion_data['timestamp'] = step * 100  # Simulate 100h per step
        
        # No immediate collision risk (distant traffic only)
        navigation_data['targets'] = [{
            'id': 1,
            'name': 'Distant Vessel',
            'position': (1000, 1500)  # Far away
        }]
        
        return propulsion_data, navigation_data
    
    def _display_step_result(self, result: Dict[str, Any]) -> None:
        """Display step result to console."""
        step = result['step']
        metrics = result['rams_metrics']
        supervisor = result['supervisor']
        
        # Compact display
        R = metrics['reliability']
        A = metrics['availability']
        M = metrics['maintainability']
        S = metrics['safety_index']
        overall = metrics['overall_score']
        
        status = supervisor.get('system_status', 'UNKNOWN')
        decision = supervisor.get('current_decision', {})
        decision_type = decision.get('type', 'UNKNOWN')
        
        # Color-code status (using simple text markers)
        if status in ['EMERGENCY_RESPONSE', 'MISSION_ABORT']:
            status_marker = '[!!!]'
        elif status in ['REDUCED_OPERATIONS', 'MAINTENANCE_REQUIRED']:
            status_marker = '[!]'
        else:
            status_marker = '[OK]'
        
        print(f"\nStep {step:3d} {status_marker} | R:{R:.2f} A:{A:.2f} M:{M:.2f} S:{S:.0f} | Overall:{overall:.2f}")
        print(f"         Decision: {decision_type}")
        
        # Show tradeoffs if any
        tradeoffs = decision.get('tradeoffs', {})
        if tradeoffs:
            print(f"         Tradeoffs: {list(tradeoffs.keys())}")
        
        # Show recommendations
        recommendations = supervisor.get('recommendations', [])
        if recommendations:
            print(f"         Action: {recommendations[0][:60]}...")
    
    def _display_summary(self, results: List[Dict[str, Any]]) -> None:
        """Display demonstration summary."""
        print("\n" + "="*60)
        print("RAMS Multi-Agent System - Demonstration Summary")
        print("="*60)
        
        # Final metrics
        final = results[-1]
        metrics = final['rams_metrics']
        
        print(f"\nFinal RAMS Metrics:")
        print(f"  Reliability (R):      {metrics['reliability']:.4f}")
        print(f"  Availability (A):     {metrics['availability']:.4f}")
        print(f"  Maintainability (M):  {metrics['maintainability']:.4f}")
        print(f"  Safety Index (S):     {metrics['safety_index']:.1f}/100")
        print(f"  Overall Score:        {metrics['overall_score']:.4f}")
        
        # Decision summary
        decision_counts = {}
        for r in results:
            dt = r['supervisor'].get('current_decision', {}).get('type', 'UNKNOWN')
            decision_counts[dt] = decision_counts.get(dt, 0) + 1
        
        print(f"\nDecision Distribution:")
        for dt, count in sorted(decision_counts.items()):
            pct = count / len(results) * 100
            print(f"  {dt}: {count} ({pct:.1f}%)")
        
        # Tradeoffs made
        all_tradeoffs = set()
        for r in results:
            tradeoffs = r['supervisor'].get('current_decision', {}).get('tradeoffs', {})
            all_tradeoffs.update(tradeoffs.keys())
        
        if all_tradeoffs:
            print(f"\nTradeoffs Made:")
            for t in all_tradeoffs:
                print(f"  - {t}")
        
        # Data sources
        print(f"\nData Sources Used:")
        print("  - UCI Naval Propulsion CBM Dataset (CC BY 4.0)")
        print("    DOI: 10.24432/C5K31K")
        print("  - AutoFerry Sensor Fusion Dataset (MIT)")
        print("    https://github.com/Autoferry/sensor_fusion_dataset")
        
        print("\n" + "="*60)
        print("Demonstration Complete")
        print("="*60)


def main():
    """Main entry point for RAMS demonstration."""
    parser = argparse.ArgumentParser(
        description="RAMS Multi-Agent System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rams_demo.py                           # Full demonstration
  python rams_demo.py --scenario collision      # Collision avoidance scenario
  python rams_demo.py --scenario maintenance    # Maintenance tradeoff scenario
  python rams_demo.py --steps 100 --delay 0     # 100 steps, no delay

Data Sources:
  - UCI Naval Propulsion CBM Dataset (CC BY 4.0)
  - AutoFerry Sensor Fusion Dataset (MIT)
        """
    )
    
    parser.add_argument(
        '--scenario', '-s',
        choices=['full', 'collision', 'maintenance'],
        default='full',
        help='Scenario type to demonstrate'
    )
    
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=30,
        help='Number of simulation steps (default: 30)'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.1,
        help='Delay between steps in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data (no download required)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("\n" + "="*60)
        print("  RAMS Multi-Agent System for Maritime Vessels")
        print("  Demonstrating Agentic AI for RAMS")
        print("="*60)
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    # Initialize orchestrator
    orchestrator = RAMSOrchestrator(verbose=not args.quiet)
    
    # Initialize data
    orchestrator.initialize_data(
        prefer_synthetic=args.synthetic,
        scenario='scenario16'
    )
    
    # Run demonstration
    results = orchestrator.run_demonstration(
        n_steps=args.steps,
        scenario_type=args.scenario,
        delay=args.delay
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
