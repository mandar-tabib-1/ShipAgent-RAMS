"""
RAMS Supervisor Agent - Coordinates multi-agent RAMS decisions

The central coordinator for the RAMS multi-agent system.
Demonstrates true agentic behavior through goal arbitration,
cross-domain decision making, and tradeoff resolution.

Key Responsibilities:
- Coordinate all RAMS agents
- Arbitrate between conflicting goals
- Make tradeoff decisions (Safety vs Availability, etc.)
- Produce integrated RAMS assessment
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .base_agent import (
    RAMSBaseAgent, RAMSCategory, AgentBelief, AgentMessage,
    MessagePriority, RAMSMetrics
)


class DecisionType(Enum):
    """Types of supervisor decisions."""
    NORMAL_OPERATIONS = auto()
    MAINTENANCE_REQUIRED = auto()
    REDUCED_OPERATIONS = auto()
    EMERGENCY_RESPONSE = auto()
    MISSION_ABORT = auto()


@dataclass
class TradeoffDecision:
    """A tradeoff decision made by the supervisor."""
    decision_id: str
    decision_type: DecisionType
    description: str
    tradeoffs: Dict[str, str]  # What was traded off
    rationale: str
    actions: List[str]
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'type': self.decision_type.name,
            'description': self.description,
            'tradeoffs': self.tradeoffs,
            'rationale': self.rationale,
            'actions': self.actions
        }


class RAMSSupervisorAgent(RAMSBaseAgent):
    """
    Supervisor agent that coordinates all RAMS agents.
    
    RAMS Category: ALL (Coordinator)
    
    Agentic Behaviors:
    - Perceives: Messages and beliefs from all RAMS agents
    - Reasons: Applies goal arbitration and priority rules
    - Acts: Makes integrated decisions and issues directives
    - Communicates: Sends commands to subordinate agents
    
    Priority Hierarchy:
    1. Safety (S) - Highest priority, overrides others
    2. Reliability (R) - Second priority
    3. Availability (A) - Third priority
    4. Maintainability (M) - Lowest priority
    
    This demonstrates true agentic behavior:
    - Autonomous decision making
    - Goal-directed behavior with conflicting objectives
    - Cross-domain reasoning
    - Tradeoff resolution
    """
    
    # Priority weights for RAMS metrics
    RAMS_WEIGHTS = {
        RAMSCategory.SAFETY: 0.40,
        RAMSCategory.RELIABILITY: 0.30,
        RAMSCategory.AVAILABILITY: 0.20,
        RAMSCategory.MAINTAINABILITY: 0.10
    }
    
    # Thresholds for decisions
    SAFETY_CRITICAL_THRESHOLD = 30    # Safety index below this triggers emergency
    RELIABILITY_WARNING_THRESHOLD = 0.5
    AVAILABILITY_WARNING_THRESHOLD = 0.8
    
    def __init__(self):
        super().__init__(name="RAMSSupervisor", category=RAMSCategory.SAFETY)  # Safety is default
        
        # Subordinate agent references (will be set by orchestrator)
        self._agents: Dict[str, RAMSBaseAgent] = {}
        
        # Current RAMS metrics from agents
        self._agent_metrics: Dict[RAMSCategory, float] = {
            RAMSCategory.RELIABILITY: 1.0,
            RAMSCategory.AVAILABILITY: 1.0,
            RAMSCategory.MAINTAINABILITY: 1.0,
            RAMSCategory.SAFETY: 1.0
        }
        
        # Alert tracking
        self._active_alerts: List[AgentMessage] = []
        
        # Decision history
        self._decisions: List[TradeoffDecision] = []
        self._decision_counter = 0
        
        # Overall system state
        self._rams_metrics = RAMSMetrics()
        self._current_decision: Optional[TradeoffDecision] = None
        
        # Agent goals  
        self.goals = [
            "Ensure overall system safety",
            "Maximize operational availability",
            "Optimize maintenance scheduling",
            "Balance conflicting RAMS objectives"
        ]
    
    def register_agent(self, agent: RAMSBaseAgent) -> None:
        """Register a subordinate agent."""
        self._agents[agent.name] = agent
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive system state from all subordinate agents.
        
        Expected environment_data keys:
        - 'timestamp': Current simulation time
        - 'agent_reports': Dict of agent name -> agent action results
        """
        timestamp = environment_data.get('timestamp', 0.0)
        agent_reports = environment_data.get('agent_reports', {})
        
        # Collect metrics from agent reports
        for agent_name, report in agent_reports.items():
            self._process_agent_report(agent_name, report)
        
        # Process incoming messages from agents
        self._process_incoming_messages()
        
        # Update overall RAMS metrics
        self._rams_metrics = RAMSMetrics(
            reliability=self._agent_metrics.get(RAMSCategory.RELIABILITY, 1.0),
            availability=self._agent_metrics.get(RAMSCategory.AVAILABILITY, 1.0),
            maintainability=self._agent_metrics.get(RAMSCategory.MAINTAINABILITY, 1.0),
            safety_index=self._agent_metrics.get(RAMSCategory.SAFETY, 1.0) * 100,
            timestamp=datetime.now()
        )
        
        # Form integrated perception belief
        self.update_belief(AgentBelief(
            belief_type="system_state",
            value={
                'rams_metrics': self._rams_metrics.to_dict(),
                'active_alerts': len(self._active_alerts),
                'agent_count': len(self._agents)
            },
            confidence=0.95,
            source="multi_agent_fusion",
            evidence=[f"Reports from {len(agent_reports)} agents"]
        ))
    
    def reason(self) -> List[AgentBelief]:
        """
        Reason about system state and make integrated decisions.
        
        Applies priority rules and resolves tradeoffs.
        """
        new_beliefs = []
        
        # 1. Analyze alert severity
        critical_alerts = [a for a in self._active_alerts 
                         if a.priority == MessagePriority.CRITICAL]
        high_alerts = [a for a in self._active_alerts 
                      if a.priority == MessagePriority.HIGH]
        
        # 2. Check for conflicting situations requiring tradeoffs
        conflicts = self._detect_conflicts()
        if conflicts:
            new_beliefs.append(AgentBelief(
                belief_type="detected_conflicts",
                value=conflicts,
                confidence=0.9,
                source="conflict_detection",
                evidence=[f"{len(conflicts)} conflicts detected"]
            ))
        
        # 3. Determine required decision
        decision = self._make_decision(critical_alerts, high_alerts, conflicts)
        self._current_decision = decision
        self._decisions.append(decision)
        
        new_beliefs.append(AgentBelief(
            belief_type="supervisor_decision",
            value=decision.to_dict(),
            confidence=0.95,
            source="decision_engine",
            evidence=["RAMS priority hierarchy", "Conflict resolution"]
        ))
        
        # 4. Update integrated RAMS assessment
        rams_assessment = self._assess_overall_rams()
        new_beliefs.append(AgentBelief(
            belief_type="rams_assessment",
            value=rams_assessment,
            confidence=0.9,
            source="rams_integration",
            evidence=["Weighted RAMS metrics"]
        ))
        
        return new_beliefs
    
    def act(self) -> Dict[str, Any]:
        """
        Produce integrated RAMS report and directives.
        """
        if not self._current_decision:
            return {'status': 'NO_DECISION'}
        
        # Generate directives for agents
        directives = self._generate_directives()
        
        # Compile comprehensive report
        return {
            'rams_metrics': self._rams_metrics.to_dict(),
            'current_decision': self._current_decision.to_dict(),
            'system_status': self._get_system_status(),
            'active_alerts': self._format_alerts(),
            'directives': directives,
            'recommendations': self._current_decision.actions,
            'tradeoffs_made': self._current_decision.tradeoffs
        }
    
    def _process_agent_report(self, agent_name: str, report: Dict[str, Any]) -> None:
        """Process a report from a subordinate agent."""
        # Extract metrics based on agent type
        if 'reliability_metric' in report:
            self._agent_metrics[RAMSCategory.RELIABILITY] = report['reliability_metric']
        
        if 'availability_metric' in report:
            self._agent_metrics[RAMSCategory.AVAILABILITY] = report['availability_metric']
        
        if 'maintainability_metric' in report:
            self._agent_metrics[RAMSCategory.MAINTAINABILITY] = report['maintainability_metric']
        
        if 'safety_index' in report:
            self._agent_metrics[RAMSCategory.SAFETY] = report['safety_index'] / 100.0
    
    def _process_incoming_messages(self) -> None:
        """Process all incoming messages from agents."""
        # Get messages from registered agents
        for agent_name, agent in self._agents.items():
            messages = agent.get_outgoing_messages()
            for msg in messages:
                if msg.recipient in ['SUPERVISOR', 'BROADCAST', self.name]:
                    self._handle_agent_message(msg)
        
        # Also process direct inbox messages
        for msg in self.inbox:
            self._handle_agent_message(msg)
        self.inbox.clear()
    
    def _handle_agent_message(self, msg: AgentMessage) -> None:
        """Handle a message from a subordinate agent."""
        content = msg.content
        
        # Check for alerts
        if content.get('alert_type'):
            self._active_alerts.append(msg)
            
            # Keep alerts list manageable
            if len(self._active_alerts) > 50:
                # Remove old non-critical alerts
                self._active_alerts = [
                    a for a in self._active_alerts[-30:]
                    if a.priority.value <= MessagePriority.HIGH.value
                ] + self._active_alerts[-20:]
    
    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicts between RAMS objectives.
        
        Common conflicts:
        - Maintenance required but high availability needed
        - Safety maneuver impacts propulsion health
        - Reliability degradation vs mission completion
        """
        conflicts = []
        
        R = self._agent_metrics.get(RAMSCategory.RELIABILITY, 1.0)
        A = self._agent_metrics.get(RAMSCategory.AVAILABILITY, 1.0)
        M = self._agent_metrics.get(RAMSCategory.MAINTAINABILITY, 1.0)
        S = self._agent_metrics.get(RAMSCategory.SAFETY, 1.0)
        
        # Conflict: Low reliability but high availability demand
        if R < 0.7 and A > 0.9:
            conflicts.append({
                'type': 'RELIABILITY_vs_AVAILABILITY',
                'description': 'Propulsion degradation detected but high availability required',
                'reliability': R,
                'availability': A,
                'severity': 'HIGH' if R < 0.5 else 'MEDIUM'
            })
        
        # Conflict: Maintenance needed but cannot afford downtime
        if M < 0.5 and A > 0.8:
            conflicts.append({
                'type': 'MAINTENANCE_vs_AVAILABILITY',
                'description': 'Maintenance overdue but downtime not acceptable',
                'maintainability': M,
                'availability': A,
                'severity': 'MEDIUM'
            })
        
        # Conflict: Safety requires action that stresses propulsion
        safety_alerts = [a for a in self._active_alerts 
                        if a.category == RAMSCategory.SAFETY and
                        a.priority.value <= MessagePriority.HIGH.value]
        reliability_alerts = [a for a in self._active_alerts
                             if a.category == RAMSCategory.RELIABILITY and
                             a.priority.value <= MessagePriority.HIGH.value]
        
        if safety_alerts and reliability_alerts:
            conflicts.append({
                'type': 'SAFETY_vs_RELIABILITY',
                'description': 'Collision avoidance required but propulsion degraded',
                'safety_index': S * 100,
                'reliability': R,
                'severity': 'CRITICAL'
            })
        
        return conflicts
    
    def _make_decision(self, 
                       critical_alerts: List[AgentMessage],
                       high_alerts: List[AgentMessage],
                       conflicts: List[Dict[str, Any]]) -> TradeoffDecision:
        """
        Make integrated decision based on alerts and conflicts.
        
        Applies priority hierarchy: Safety > Reliability > Availability > Maintainability
        """
        self._decision_counter += 1
        decision_id = f"DEC-{self._decision_counter:04d}"
        
        # Check for safety-critical situations (always highest priority)
        safety_critical = any(a.category == RAMSCategory.SAFETY for a in critical_alerts)
        
        if safety_critical:
            return self._make_emergency_decision(decision_id, critical_alerts, conflicts)
        
        # Check for reliability critical
        reliability_critical = any(a.category == RAMSCategory.RELIABILITY for a in critical_alerts)
        if reliability_critical:
            return self._make_reliability_decision(decision_id, critical_alerts, conflicts)
        
        # Handle high-priority alerts
        if high_alerts:
            return self._make_high_priority_decision(decision_id, high_alerts, conflicts)
        
        # Handle conflicts without critical alerts
        if conflicts:
            return self._make_tradeoff_decision(decision_id, conflicts)
        
        # Normal operations
        return TradeoffDecision(
            decision_id=decision_id,
            decision_type=DecisionType.NORMAL_OPERATIONS,
            description="Normal RAMS operations - all metrics within acceptable limits",
            tradeoffs={},
            rationale="No conflicts detected, all RAMS metrics healthy",
            actions=["Continue normal operations", "Maintain monitoring levels"]
        )
    
    def _make_emergency_decision(self,
                                  decision_id: str,
                                  critical_alerts: List[AgentMessage],
                                  conflicts: List[Dict[str, Any]]) -> TradeoffDecision:
        """Make emergency decision for safety-critical situations."""
        # Identify the primary threat
        safety_alerts = [a for a in critical_alerts if a.category == RAMSCategory.SAFETY]
        
        primary_threat = "collision risk" if safety_alerts else "system failure"
        
        actions = [
            "IMMEDIATE: Execute emergency response procedures",
            "PRIORITY: Safety takes precedence over all other objectives"
        ]
        
        tradeoffs = {}
        
        # Check if we need to tradeoff propulsion health for safety
        if any(c['type'] == 'SAFETY_vs_RELIABILITY' for c in conflicts):
            tradeoffs['propulsion_stress'] = "Accept additional propulsion stress to execute safety maneuver"
            actions.append("ACCEPT: Additional propulsion wear from emergency maneuver")
        
        if any(c['type'] == 'MAINTENANCE_vs_AVAILABILITY' for c in conflicts):
            tradeoffs['maintenance_defer'] = "Defer all non-critical maintenance"
            actions.append("DEFER: Postpone scheduled maintenance until situation resolved")
        
        return TradeoffDecision(
            decision_id=decision_id,
            decision_type=DecisionType.EMERGENCY_RESPONSE,
            description=f"Emergency response: {primary_threat} detected",
            tradeoffs=tradeoffs,
            rationale="Safety is paramount - all other objectives subordinate to crew and vessel safety",
            actions=actions
        )
    
    def _make_reliability_decision(self,
                                    decision_id: str,
                                    critical_alerts: List[AgentMessage],
                                    conflicts: List[Dict[str, Any]]) -> TradeoffDecision:
        """Make decision for reliability-critical situations."""
        R = self._agent_metrics.get(RAMSCategory.RELIABILITY, 1.0)
        
        actions = [
            "REDUCE: Lower operational demands on degraded components",
            "SCHEDULE: Immediate maintenance planning required"
        ]
        
        tradeoffs = {}
        
        # May need to sacrifice availability
        if any(c['type'] == 'RELIABILITY_vs_AVAILABILITY' for c in conflicts):
            tradeoffs['reduced_availability'] = "Accept reduced availability to preserve reliability"
            actions.append(f"ACCEPT: Reduced capability ({R*100:.0f}% reliability)")
        
        return TradeoffDecision(
            decision_id=decision_id,
            decision_type=DecisionType.REDUCED_OPERATIONS,
            description=f"Reduced operations: Reliability degraded to {R*100:.1f}%",
            tradeoffs=tradeoffs,
            rationale="Reliability degradation requires protective measures to prevent failure",
            actions=actions
        )
    
    def _make_high_priority_decision(self,
                                      decision_id: str,
                                      high_alerts: List[AgentMessage],
                                      conflicts: List[Dict[str, Any]]) -> TradeoffDecision:
        """Make decision for high-priority situations."""
        # Categorize alerts
        categories = [a.category for a in high_alerts]
        
        actions = ["MONITOR: Increased vigilance on flagged systems"]
        tradeoffs = {}
        
        if RAMSCategory.SAFETY in categories:
            actions.append("PREPARE: Ready collision avoidance procedures")
        
        if RAMSCategory.RELIABILITY in categories:
            actions.append("PLAN: Schedule maintenance at next opportunity")
            tradeoffs['maintenance_planning'] = "Prioritize maintenance planning over other activities"
        
        if RAMSCategory.MAINTAINABILITY in categories:
            actions.append("SCHEDULE: Maintenance backlog requires attention")
        
        return TradeoffDecision(
            decision_id=decision_id,
            decision_type=DecisionType.MAINTENANCE_REQUIRED,
            description="Elevated attention required across RAMS dimensions",
            tradeoffs=tradeoffs,
            rationale="Multiple high-priority conditions require coordinated response",
            actions=actions
        )
    
    def _make_tradeoff_decision(self,
                                 decision_id: str,
                                 conflicts: List[Dict[str, Any]]) -> TradeoffDecision:
        """Make decision to resolve detected conflicts."""
        primary_conflict = conflicts[0]  # Handle most severe first
        
        actions = []
        tradeoffs = {}
        
        if primary_conflict['type'] == 'RELIABILITY_vs_AVAILABILITY':
            # Reliability takes precedence over availability
            tradeoffs['availability_reduction'] = f"Accept availability reduction to protect reliability"
            actions = [
                "REDUCE: Lower operational speed/power to reduce component stress",
                "MONITOR: Increase degradation monitoring frequency",
                "PLAN: Schedule maintenance window"
            ]
            rationale = "Reliability preservation prevents more costly failures"
            
        elif primary_conflict['type'] == 'MAINTENANCE_vs_AVAILABILITY':
            # Balance based on severity
            if primary_conflict['severity'] == 'HIGH':
                tradeoffs['scheduled_downtime'] = "Schedule maintenance downtime"
                actions = [
                    "SCHEDULE: Plan maintenance window within 24 hours",
                    "NOTIFY: Alert operations of upcoming downtime"
                ]
            else:
                tradeoffs['deferred_maintenance'] = "Defer maintenance with increased monitoring"
                actions = [
                    "DEFER: Postpone maintenance up to 48 hours",
                    "MONITOR: Increase system monitoring",
                    "PREPARE: Pre-stage maintenance resources"
                ]
            rationale = "Balance maintenance needs against operational commitments"
            
        else:
            actions = ["ASSESS: Continue monitoring situation"]
            rationale = "Conflict detected but no immediate action required"
        
        return TradeoffDecision(
            decision_id=decision_id,
            decision_type=DecisionType.MAINTENANCE_REQUIRED,
            description=f"Tradeoff resolution: {primary_conflict['type']}",
            tradeoffs=tradeoffs,
            rationale=rationale,
            actions=actions
        )
    
    def _assess_overall_rams(self) -> Dict[str, Any]:
        """Assess overall RAMS health."""
        R = self._agent_metrics.get(RAMSCategory.RELIABILITY, 1.0)
        A = self._agent_metrics.get(RAMSCategory.AVAILABILITY, 1.0)
        M = self._agent_metrics.get(RAMSCategory.MAINTAINABILITY, 1.0)
        S = self._agent_metrics.get(RAMSCategory.SAFETY, 1.0)
        
        # Weighted overall score
        overall = (
            self.RAMS_WEIGHTS[RAMSCategory.SAFETY] * S +
            self.RAMS_WEIGHTS[RAMSCategory.RELIABILITY] * R +
            self.RAMS_WEIGHTS[RAMSCategory.AVAILABILITY] * A +
            self.RAMS_WEIGHTS[RAMSCategory.MAINTAINABILITY] * M
        )
        
        # Determine overall status
        if S < 0.3 or R < 0.3:
            status = 'CRITICAL'
        elif S < 0.6 or R < 0.5 or A < 0.5:
            status = 'WARNING'
        elif overall < 0.7:
            status = 'CAUTION'
        else:
            status = 'HEALTHY'
        
        return {
            'overall_score': round(overall, 4),
            'status': status,
            'breakdown': {
                'safety': round(S, 4),
                'reliability': round(R, 4),
                'availability': round(A, 4),
                'maintainability': round(M, 4)
            },
            'weights': {k.name: v for k, v in self.RAMS_WEIGHTS.items()}
        }
    
    def _get_system_status(self) -> str:
        """Get overall system status string."""
        if self._current_decision:
            return self._current_decision.decision_type.name
        return 'UNKNOWN'
    
    def _format_alerts(self) -> List[Dict[str, Any]]:
        """Format active alerts for reporting."""
        return [
            {
                'sender': a.sender,
                'priority': a.priority.name,
                'category': a.category.name,
                'type': a.content.get('alert_type', 'unknown'),
                'summary': str(a.content)[:100]
            }
            for a in self._active_alerts[-10:]  # Last 10 alerts
        ]
    
    def _generate_directives(self) -> List[Dict[str, Any]]:
        """Generate directives for subordinate agents."""
        directives = []
        
        if not self._current_decision:
            return directives
        
        decision_type = self._current_decision.decision_type
        
        if decision_type == DecisionType.EMERGENCY_RESPONSE:
            directives.append({
                'target': 'SafetyAgent',
                'directive': 'EXECUTE_AVOIDANCE',
                'priority': 'CRITICAL'
            })
            directives.append({
                'target': 'ReliabilityAgent',
                'directive': 'ACCEPT_STRESS',
                'priority': 'HIGH'
            })
            
        elif decision_type == DecisionType.REDUCED_OPERATIONS:
            directives.append({
                'target': 'AvailabilityAgent',
                'directive': 'REDUCE_LOAD',
                'priority': 'HIGH'
            })
            directives.append({
                'target': 'MaintainabilityAgent',
                'directive': 'PREPARE_MAINTENANCE',
                'priority': 'HIGH'
            })
            
        elif decision_type == DecisionType.MAINTENANCE_REQUIRED:
            directives.append({
                'target': 'MaintainabilityAgent',
                'directive': 'SCHEDULE_MAINTENANCE',
                'priority': 'MEDIUM'
            })
        
        return directives
    
    def get_rams_metrics(self) -> RAMSMetrics:
        """Get current RAMS metrics."""
        return self._rams_metrics
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of decisions."""
        return [d.to_dict() for d in self._decisions[-20:]]
