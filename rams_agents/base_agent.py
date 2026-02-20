"""
Base Agent Framework for RAMS Demonstration

Implements the core agentic AI architecture with:
- Perception: Observe environment state
- Reasoning: Apply domain knowledge and ML models
- Action: Produce recommendations
- Communication: Inter-agent messaging

This framework demonstrates true agentic behavior vs simple data pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set
import uuid


class RAMSCategory(Enum):
    """RAMS categories for maritime systems."""
    RELIABILITY = auto()      # R - Probability of performing without failure
    AVAILABILITY = auto()     # A - Proportion of time system is operational
    MAINTAINABILITY = auto()  # M - Ease of maintaining/restoring function
    SAFETY = auto()           # S - Freedom from unacceptable risk


class MessagePriority(Enum):
    """Priority levels for inter-agent communication."""
    CRITICAL = 1    # Immediate action required
    HIGH = 2        # Urgent attention needed
    MEDIUM = 3      # Normal priority
    LOW = 4         # Informational


class AgentState(Enum):
    """Agent operational states."""
    IDLE = auto()
    PERCEIVING = auto()
    REASONING = auto()
    ACTING = auto()
    COMMUNICATING = auto()


@dataclass
class AgentMessage:
    """
    Message passed between agents.
    
    Enables true agentic behavior through inter-agent communication.
    """
    sender: str
    recipient: str  # Use 'BROADCAST' for all agents
    content: Dict[str, Any]
    priority: MessagePriority
    category: RAMSCategory
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __repr__(self) -> str:
        return (f"AgentMessage({self.sender}→{self.recipient}, "
                f"{self.category.name}, {self.priority.name})")


@dataclass
class AgentBelief:
    """
    Agent's belief about system state.
    
    Represents the agent's internal model of the world,
    updated through perception and reasoning.
    """
    belief_type: str  # e.g., 'propulsion_health', 'collision_risk'
    value: Any        # The belief value
    confidence: float  # 0.0 to 1.0
    source: str       # Where this belief came from
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if belief meets confidence threshold."""
        return self.confidence >= threshold


@dataclass
class RAMSMetrics:
    """
    Formal RAMS metrics for the system.
    
    Reliability: R(t) = P(no failure in [0,t])
    Availability: A = MTBF / (MTBF + MTTR)
    Maintainability: M(t) = P(repair completed in t)
    Safety: Risk index (0-100, lower is safer)
    """
    reliability: float = 1.0      # [0, 1]
    availability: float = 1.0     # [0, 1]
    maintainability: float = 1.0  # [0, 1]
    safety_index: float = 100.0   # [0, 100], 100 = safest
    timestamp: datetime = field(default_factory=datetime.now)
    
    def overall_rams_score(self) -> float:
        """Calculate weighted overall RAMS score."""
        # Safety weighted highest, then reliability, availability, maintainability
        weights = {'S': 0.35, 'R': 0.30, 'A': 0.20, 'M': 0.15}
        return (
            weights['R'] * self.reliability +
            weights['A'] * self.availability +
            weights['M'] * self.maintainability +
            weights['S'] * (self.safety_index / 100.0)
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'reliability': round(self.reliability, 4),
            'availability': round(self.availability, 4),
            'maintainability': round(self.maintainability, 4),
            'safety_index': round(self.safety_index, 2),
            'overall_score': round(self.overall_rams_score(), 4)
        }


class RAMSBaseAgent(ABC):
    """
    Abstract base class for RAMS agents.
    
    Implements the core agentic architecture:
    1. Perceive - Observe environment state
    2. Reason - Apply domain knowledge and ML
    3. Act - Produce recommendations/actions
    4. Communicate - Exchange messages with other agents
    
    This is what makes it an "agent" rather than a simple data processor:
    - Autonomous decision-making
    - Goal-directed behavior
    - Inter-agent communication
    - Belief maintenance
    """
    
    def __init__(self, name: str, category: RAMSCategory):
        self.name = name
        self.category = category
        self.state = AgentState.IDLE
        self.beliefs: Dict[str, AgentBelief] = {}
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.goals: List[str] = []
        self._message_handlers: Dict[str, callable] = {}
        
    @abstractmethod
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Observe the environment and update internal beliefs.
        
        This is the sensing/perception phase of the agent cycle.
        The agent interprets raw data and forms beliefs about system state.
        
        Args:
            environment_data: Raw sensor/operational data
        """
        pass
    
    @abstractmethod  
    def reason(self) -> List[AgentBelief]:
        """
        Apply domain knowledge and ML models to beliefs.
        
        This is the reasoning/deliberation phase.
        The agent processes beliefs, applies rules and models,
        and determines what actions to take.
        
        Returns:
            List of updated/new beliefs from reasoning
        """
        pass
    
    @abstractmethod
    def act(self) -> Dict[str, Any]:
        """
        Produce recommendations or actions based on reasoning.
        
        This is the action/execution phase.
        The agent determines what to do based on its beliefs and goals.
        
        Returns:
            Dictionary containing recommendations/actions
        """
        pass
    
    def communicate(self, message: AgentMessage) -> None:
        """
        Send a message to another agent or broadcast.
        
        Args:
            message: The message to send
        """
        self.state = AgentState.COMMUNICATING
        self.outbox.append(message)
    
    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent.
        
        Args:
            message: The incoming message
        """
        self.inbox.append(message)
        
        # Process message based on type
        handler = self._message_handlers.get(message.category.name)
        if handler:
            handler(message)
    
    def process_inbox(self) -> List[AgentBelief]:
        """
        Process all messages in inbox and form beliefs.
        
        Returns:
            List of beliefs formed from messages
        """
        new_beliefs = []
        for message in self.inbox:
            belief = AgentBelief(
                belief_type=f"message_{message.category.name.lower()}",
                value=message.content,
                confidence=0.9 if message.priority == MessagePriority.CRITICAL else 0.7,
                source=message.sender,
                evidence=[f"Message {message.message_id}"]
            )
            new_beliefs.append(belief)
            self.update_belief(belief)
        self.inbox.clear()
        return new_beliefs
    
    def update_belief(self, belief: AgentBelief) -> None:
        """Update or add a belief to the agent's belief set."""
        self.beliefs[belief.belief_type] = belief
    
    def get_belief(self, belief_type: str) -> Optional[AgentBelief]:
        """Retrieve a belief by type."""
        return self.beliefs.get(belief_type)
    
    def create_message(
        self,
        recipient: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        category: Optional[RAMSCategory] = None
    ) -> AgentMessage:
        """
        Create a new message to send.
        
        Args:
            recipient: Target agent name or 'BROADCAST'
            content: Message content dictionary
            priority: Message priority level
            category: RAMS category (defaults to agent's category)
        
        Returns:
            New AgentMessage ready to send
        """
        return AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            priority=priority,
            category=category or self.category
        )
    
    def run_cycle(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one full agent cycle: perceive → reason → act.
        
        This is the main agent loop that demonstrates agentic behavior.
        
        Args:
            environment_data: Current environment state
        
        Returns:
            Agent's actions/recommendations for this cycle
        """
        # Phase 1: Perceive
        self.state = AgentState.PERCEIVING
        self.perceive(environment_data)
        
        # Process any incoming messages
        self.process_inbox()
        
        # Phase 2: Reason
        self.state = AgentState.REASONING
        new_beliefs = self.reason()
        for belief in new_beliefs:
            self.update_belief(belief)
        
        # Phase 3: Act
        self.state = AgentState.ACTING
        actions = self.act()
        
        # Return to idle
        self.state = AgentState.IDLE
        
        return actions
    
    def get_outgoing_messages(self) -> List[AgentMessage]:
        """Get and clear outgoing messages."""
        messages = self.outbox.copy()
        self.outbox.clear()
        return messages
    
    def status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        return {
            'name': self.name,
            'category': self.category.name,
            'state': self.state.name,
            'num_beliefs': len(self.beliefs),
            'inbox_size': len(self.inbox),
            'outbox_size': len(self.outbox),
            'goals': self.goals
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}', {self.category.name})"
