"""
Maintainability Agent - Manages maintenance scheduling and optimization

Part of the RAMS multi-agent system demonstrating agentic AI.
Schedules maintenance based on reliability predictions and availability constraints.

Key Responsibilities:
- Schedule preventive and corrective maintenance
- Optimize maintenance timing based on RUL predictions
- Track maintenance history and costs
- Balance maintenance vs operational availability
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .base_agent import (
    RAMSBaseAgent, RAMSCategory, AgentBelief, AgentMessage,
    MessagePriority
)


class MaintenanceType(Enum):
    """Types of maintenance activities."""
    INSPECTION = auto()      # Visual/sensor inspection
    PREVENTIVE = auto()      # Scheduled preventive maintenance
    CORRECTIVE = auto()      # Repair after fault detection
    OVERHAUL = auto()        # Major maintenance/replacement


class MaintenancePriority(Enum):
    """Maintenance task priority levels."""
    CRITICAL = 1     # Must complete immediately
    HIGH = 2         # Complete within 24 hours
    MEDIUM = 3       # Complete within 1 week
    LOW = 4          # Schedule at convenience


@dataclass
class MaintenanceTask:
    """A scheduled maintenance task."""
    task_id: str
    component: str
    task_type: MaintenanceType
    priority: MaintenancePriority
    description: str
    estimated_duration_hours: float
    estimated_cost: float
    scheduled_time: Optional[float] = None
    completed: bool = False
    completion_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'component': self.component,
            'type': self.task_type.name,
            'priority': self.priority.name,
            'description': self.description,
            'duration_hours': self.estimated_duration_hours,
            'cost': self.estimated_cost,
            'scheduled': self.scheduled_time,
            'completed': self.completed
        }


# Maintenance parameters based on typical naval propulsion systems
MAINTENANCE_CATALOG = {
    'compressor': {
        MaintenanceType.INSPECTION: {'duration': 2, 'cost': 500, 'interval': 500},
        MaintenanceType.PREVENTIVE: {'duration': 8, 'cost': 5000, 'interval': 2000},
        MaintenanceType.CORRECTIVE: {'duration': 24, 'cost': 15000},
        MaintenanceType.OVERHAUL: {'duration': 168, 'cost': 100000, 'interval': 8000},
    },
    'turbine': {
        MaintenanceType.INSPECTION: {'duration': 2, 'cost': 500, 'interval': 500},
        MaintenanceType.PREVENTIVE: {'duration': 12, 'cost': 8000, 'interval': 2500},
        MaintenanceType.CORRECTIVE: {'duration': 48, 'cost': 25000},
        MaintenanceType.OVERHAUL: {'duration': 336, 'cost': 200000, 'interval': 10000},
    }
}


class MaintainabilityAgent(RAMSBaseAgent):
    """
    Agent responsible for maintenance management.
    
    RAMS Category: M (Maintainability)
    
    Agentic Behaviors:
    - Perceives: Maintenance triggers from Reliability/Availability agents
    - Reasons: Optimizes maintenance scheduling, estimates costs
    - Acts: Generates maintenance schedules and recommendations
    - Communicates: Coordinates with Supervisor for maintenance windows
    
    Maintainability M(t) = P(repair completed within time t)
    """
    
    # Repair rate (tasks per hour) - used for maintainability calculation
    REPAIR_RATE = 0.1  # Mean time to repair = 10 hours
    
    def __init__(self):
        super().__init__(name="MaintainabilityAgent", category=RAMSCategory.MAINTAINABILITY)
        
        # Maintenance tracking
        self._pending_tasks: List[MaintenanceTask] = []
        self._completed_tasks: List[MaintenanceTask] = []
        self._task_counter = 0
        
        # Operating hours tracking (for scheduled maintenance)
        self._operating_hours = 0.0
        self._last_maintenance: Dict[str, Dict[MaintenanceType, float]] = {
            'compressor': {},
            'turbine': {}
        }
        
        # Cost tracking
        self._total_maintenance_cost = 0.0
        self._total_downtime_hours = 0.0
        
        # Agent goals
        self.goals = [
            "Optimize maintenance scheduling",
            "Minimize maintenance costs",
            "Maximize operational availability"
        ]
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive maintenance-related information.
        
        Expected environment_data keys:
        - 'timestamp': Current simulation time
        - 'operating_hours': Total operating hours
        - 'propulsion': Propulsion data (for degradation awareness)
        """
        timestamp = environment_data.get('timestamp', 0.0)
        self._operating_hours = environment_data.get('operating_hours', timestamp)
        
        # Check for scheduled maintenance due
        due_maintenance = self._check_scheduled_maintenance()
        
        # Process any messages from other agents
        maintenance_triggers = []
        for msg in self.inbox:
            if msg.category == RAMSCategory.RELIABILITY:
                trigger = msg.content.get('maintenance_trigger')
                if trigger:
                    maintenance_triggers.append({
                        'source': msg.sender,
                        'trigger': trigger,
                        'component': msg.content.get('component'),
                        'severity': msg.content.get('severity', 'MEDIUM')
                    })
        
        # Form perception belief
        self.update_belief(AgentBelief(
            belief_type="maintenance_inputs",
            value={
                'operating_hours': self._operating_hours,
                'due_maintenance': due_maintenance,
                'triggers_received': len(maintenance_triggers),
                'pending_tasks': len(self._pending_tasks)
            },
            confidence=0.95,
            source="maintenance_monitoring",
            evidence=[f"Operating hours: {self._operating_hours:.1f}"]
        ))
        
        # Create tasks from triggers
        for trigger in maintenance_triggers:
            self._create_triggered_task(trigger)
    
    def reason(self) -> List[AgentBelief]:
        """
        Reason about maintenance priorities and scheduling.
        """
        new_beliefs = []
        
        # 1. Prioritize pending tasks
        sorted_tasks = self._prioritize_tasks()
        new_beliefs.append(AgentBelief(
            belief_type="prioritized_tasks",
            value=[t.to_dict() for t in sorted_tasks[:5]],  # Top 5
            confidence=0.9,
            source="task_prioritization",
            evidence=[f"{len(self._pending_tasks)} total pending tasks"]
        ))
        
        # 2. Calculate maintainability metric
        maintainability = self._calculate_maintainability()
        new_beliefs.append(AgentBelief(
            belief_type="maintainability_metric",
            value={
                'M_t': maintainability,
                'mean_repair_time': 1.0 / self.REPAIR_RATE,
                'total_cost': self._total_maintenance_cost,
                'total_downtime': self._total_downtime_hours
            },
            confidence=0.85,
            source="maintainability_calculation",
            evidence=["Exponential repair time distribution"]
        ))
        
        # 3. Check for overdue maintenance
        overdue = [t for t in self._pending_tasks 
                  if t.priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH]]
        if overdue:
            new_beliefs.append(AgentBelief(
                belief_type="overdue_maintenance",
                value={'count': len(overdue), 'tasks': [t.task_id for t in overdue]},
                confidence=1.0,
                source="overdue_check",
                evidence=["Priority CRITICAL or HIGH tasks pending"]
            ))
            self._alert_overdue_maintenance(overdue)
        
        return new_beliefs
    
    def act(self) -> Dict[str, Any]:
        """
        Produce maintenance schedule and recommendations.
        """
        maintainability = self._calculate_maintainability()
        sorted_tasks = self._prioritize_tasks()
        
        # Generate schedule recommendation
        schedule = self._generate_schedule(sorted_tasks[:10])
        
        # Cost forecast
        pending_cost = sum(t.estimated_cost for t in self._pending_tasks)
        pending_downtime = sum(t.estimated_duration_hours for t in self._pending_tasks)
        
        recommendations = []
        critical_count = len([t for t in self._pending_tasks 
                             if t.priority == MaintenancePriority.CRITICAL])
        high_count = len([t for t in self._pending_tasks 
                         if t.priority == MaintenancePriority.HIGH])
        
        if critical_count > 0:
            recommendations.append(f"CRITICAL: {critical_count} critical maintenance tasks require immediate attention")
        if high_count > 0:
            recommendations.append(f"HIGH: {high_count} high-priority tasks should be addressed within 24 hours")
        
        if pending_downtime > 48:
            recommendations.append(f"PLAN: Significant downtime required ({pending_downtime:.1f}h) - optimize scheduling")
        
        if not recommendations:
            recommendations.append("NORMAL: Maintenance status healthy")
        
        return {
            'maintainability_metric': maintainability,
            'pending_tasks_count': len(self._pending_tasks),
            'completed_tasks_count': len(self._completed_tasks),
            'schedule': schedule,
            'cost_forecast': {
                'pending_cost': pending_cost,
                'pending_downtime_hours': pending_downtime,
                'total_spent': self._total_maintenance_cost
            },
            'recommendations': recommendations
        }
    
    def _check_scheduled_maintenance(self) -> List[Dict[str, Any]]:
        """Check for scheduled maintenance that is due."""
        due_items = []
        
        for component, intervals in MAINTENANCE_CATALOG.items():
            for maint_type, params in intervals.items():
                if 'interval' not in params:
                    continue
                
                interval = params['interval']
                last_maint = self._last_maintenance.get(component, {}).get(maint_type, 0)
                
                hours_since = self._operating_hours - last_maint
                if hours_since >= interval:
                    due_items.append({
                        'component': component,
                        'type': maint_type,
                        'hours_overdue': hours_since - interval
                    })
                    
                    # Create task if not already pending
                    existing = [t for t in self._pending_tasks 
                               if t.component == component and t.task_type == maint_type]
                    if not existing:
                        self._create_scheduled_task(component, maint_type)
        
        return due_items
    
    def _create_scheduled_task(self, component: str, maint_type: MaintenanceType) -> MaintenanceTask:
        """Create a scheduled maintenance task."""
        self._task_counter += 1
        params = MAINTENANCE_CATALOG[component][maint_type]
        
        # Determine priority based on type
        if maint_type == MaintenanceType.OVERHAUL:
            priority = MaintenancePriority.MEDIUM
        elif maint_type == MaintenanceType.PREVENTIVE:
            priority = MaintenancePriority.MEDIUM
        else:
            priority = MaintenancePriority.LOW
        
        task = MaintenanceTask(
            task_id=f"MAINT-{self._task_counter:04d}",
            component=component,
            task_type=maint_type,
            priority=priority,
            description=f"Scheduled {maint_type.name.lower()} for {component}",
            estimated_duration_hours=params['duration'],
            estimated_cost=params['cost'],
            scheduled_time=self._operating_hours
        )
        
        self._pending_tasks.append(task)
        return task
    
    def _create_triggered_task(self, trigger: Dict[str, Any]) -> MaintenanceTask:
        """Create a maintenance task from reliability/availability trigger."""
        self._task_counter += 1
        
        component = trigger.get('component', 'unknown')
        severity = trigger.get('severity', 'MEDIUM')
        
        # Map severity to task type and priority
        if severity == 'CRITICAL':
            maint_type = MaintenanceType.CORRECTIVE
            priority = MaintenancePriority.CRITICAL
        elif severity == 'WARNING':
            maint_type = MaintenanceType.PREVENTIVE
            priority = MaintenancePriority.HIGH
        else:
            maint_type = MaintenanceType.INSPECTION
            priority = MaintenancePriority.MEDIUM
        
        # Get maintenance parameters
        if component in MAINTENANCE_CATALOG and maint_type in MAINTENANCE_CATALOG[component]:
            params = MAINTENANCE_CATALOG[component][maint_type]
        else:
            params = {'duration': 8, 'cost': 5000}
        
        task = MaintenanceTask(
            task_id=f"MAINT-{self._task_counter:04d}",
            component=component,
            task_type=maint_type,
            priority=priority,
            description=f"Triggered {maint_type.name.lower()} - {trigger.get('trigger', 'reliability issue')}",
            estimated_duration_hours=params['duration'],
            estimated_cost=params['cost'],
            scheduled_time=self._operating_hours
        )
        
        self._pending_tasks.append(task)
        return task
    
    def _prioritize_tasks(self) -> List[MaintenanceTask]:
        """Prioritize pending tasks."""
        return sorted(self._pending_tasks, 
                     key=lambda t: (t.priority.value, -t.scheduled_time if t.scheduled_time else 0))
    
    def _generate_schedule(self, tasks: List[MaintenanceTask]) -> List[Dict[str, Any]]:
        """Generate maintenance schedule."""
        schedule = []
        current_time = self._operating_hours
        
        for task in tasks:
            schedule.append({
                'task_id': task.task_id,
                'component': task.component,
                'type': task.task_type.name,
                'priority': task.priority.name,
                'recommended_start': current_time,
                'estimated_completion': current_time + task.estimated_duration_hours,
                'cost': task.estimated_cost
            })
            # Stack sequential tasks if critical/high priority
            if task.priority.value <= 2:
                current_time += task.estimated_duration_hours
        
        return schedule
    
    def _calculate_maintainability(self, target_time: float = 24.0) -> float:
        """
        Calculate maintainability metric M(t).
        
        M(t) = 1 - e^(-μt) where μ is repair rate
        
        This gives probability of completing repair within time t.
        """
        # Adjust repair rate based on task complexity
        if self._pending_tasks:
            avg_duration = np.mean([t.estimated_duration_hours for t in self._pending_tasks])
            effective_rate = 1.0 / max(1, avg_duration)
        else:
            effective_rate = self.REPAIR_RATE
        
        maintainability = 1 - np.exp(-effective_rate * target_time)
        return round(maintainability, 4)
    
    def _alert_overdue_maintenance(self, overdue_tasks: List[MaintenanceTask]) -> None:
        """Send alert for overdue critical maintenance."""
        msg = self.create_message(
            recipient='SUPERVISOR',
            content={
                'alert_type': 'overdue_maintenance',
                'task_count': len(overdue_tasks),
                'tasks': [t.to_dict() for t in overdue_tasks[:5]],
                'total_downtime_required': sum(t.estimated_duration_hours for t in overdue_tasks)
            },
            priority=MessagePriority.HIGH
        )
        self.communicate(msg)
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        for task in self._pending_tasks:
            if task.task_id == task_id:
                task.completed = True
                task.completion_time = self._operating_hours
                
                # Update last maintenance time
                if task.component in self._last_maintenance:
                    self._last_maintenance[task.component][task.task_type] = self._operating_hours
                
                # Track costs
                self._total_maintenance_cost += task.estimated_cost
                self._total_downtime_hours += task.estimated_duration_hours
                
                # Move to completed
                self._pending_tasks.remove(task)
                self._completed_tasks.append(task)
                
                return True
        return False
    
    def get_rams_contribution(self) -> float:
        """Get this agent's contribution to system maintainability metric."""
        return self._calculate_maintainability()
