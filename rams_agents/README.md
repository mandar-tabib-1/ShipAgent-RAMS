# RAMS Multi-Agent System for Maritime Vessels

**Demonstrating Agentic AI for Reliability, Availability, Maintainability, and Safety**

This module implements a multi-agent system that demonstrates true "agentic" AI behavior for maritime RAMS analysis. Unlike simple data pipelines, these agents operate autonomously, communicate with each other, and make goal-directed tradeoff decisions.

## What Makes This "Agentic"?

| Data Pipeline (Not Agentic) | Multi-Agent System (Agentic) |
|-------------------------------|------------------------------|
| Data → Process → Output | Each agent perceives, reasons, acts |
| Sequential processing | Parallel autonomous agents |
| No inter-component communication | Agents send/receive messages |
| Fixed logic | Goal-directed decision making |
| No conflict resolution | Tradeoff resolution between objectives |

### Agentic Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAMS Supervisor Agent                     │
│  • Coordinates all agents                                    │
│  • Resolves conflicts between RAMS objectives                │
│  • Makes tradeoff decisions (Safety > R > A > M)            │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
     ┌──────▼──────┐   ┌──────────────┐   ┌──────▼──────┐
     │ Reliability │   │ Availability │   │   Safety    │
     │   Agent     │◄──┤    Agent     ├──►│   Agent     │
     │             │   │              │   │             │
     └──────┬──────┘   └──────┬───────┘   └──────┬──────┘
            │                 │                   │
            │    ┌────────────▼───────────┐      │
            └───►│ Maintainability Agent  │◄─────┘
                 │                        │
                 └────────────────────────┘

    ▲                                              ▲
    │                                              │
UCI Naval Propulsion                    AutoFerry Sensor
   CBM Dataset                          Fusion Dataset
  (R, A, M)                                   (S)
```

## RAMS Framework

| Category | Metric | Agent | Data Source |
|----------|--------|-------|-------------|
| **R**eliability | $R(t) = P(\text{no failure in } [0,t])$ | `ReliabilityAgent` | UCI Naval CBM |
| **A**vailability | $A = \frac{MTBF}{MTBF + MTTR}$ | `AvailabilityAgent` | UCI Naval CBM |
| **M**aintainability | $M(t) = 1 - e^{-\mu t}$ | `MaintainabilityAgent` | Agent messages |
| **S**afety | Risk Index (CPA/TCPA/COLREGS) | `SafetyAgent` | AutoFerry |

## RL+PSF Collision Avoidance (NEW)

The Safety Agent now integrates **Reinforcement Learning** with a **Potential Safety Function (PSF)** for intelligent collision avoidance with formal safety guarantees.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SafetyAgent (RL+PSF)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Kalman Tracks ─► Observation Adapter ─► RL Policy (PPO)    │
│                          │                    │             │
│                          │                    ▼             │
│                          │              RL Action           │
│                          │          [speed, steering]       │
│                          │                    │             │
│                          │                    ▼             │
│                          └───────► PSF Filter (CBF)         │
│                                         │                   │
│                                         ▼                   │
│                               Safe Action (guaranteed)      │
│                                         │                   │
│                                         ▼                   │
│                            COLREGS-Compatible Maneuver      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **RL Policy** | Pre-trained PPO (Proximal Policy Optimization) from research |
| **PSF Filter** | Control Barrier Function ensuring φ(CPA,TCPA) > 0 |
| **Observation Adapter** | Converts Kalman tracks to RL input (virtual LiDAR scan) |
| **Fallback** | Rule-based COLREGS if RL unavailable |

### Safety Barrier Function

The PSF ensures safety using a Control Barrier Function:

```
φ(CPA, TCPA) = CPA - CPA_critical - k·max(0, TCPA_critical - TCPA) > 0
```

Where:
- `CPA_critical = 50m` (minimum safe closest approach)
- `TCPA_critical = 60s` (critical time horizon)
- `k = 0.5` (TCPA coupling coefficient)

If the RL action would violate φ > 0, PSF projects the action to the safe set.

### Usage

```python
from rams_agents.safety_agent import SafetyAgent

# Enable RL+PSF (default)
agent = SafetyAgent(use_rl_policy=True)

# Run cycle with collision scenario
result = agent.run_cycle({
    'targets': tracked_vessels,
    'ownship_position': (0, 0),
    'ownship_velocity': (5, 0)
})

# Check RL+PSF statistics
stats = agent.get_rl_psf_stats()
print(f"PSF intervention rate: {stats['intervention_rate']:.1%}")
```

### Attribution

The RL policy is based on:
- **Repository**: [Acmece/rl-collision-avoidance](https://github.com/Acmece/rl-collision-avoidance)
- **Paper**: "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning" (arXiv:1709.10082)

See [CREDITS.md](CREDITS.md) for full attribution.

## Quick Start

```bash
# Full demonstration (synthetic data, no download required)
python rams_demo.py --synthetic

# Collision avoidance + degradation scenario
python rams_demo.py --scenario collision --synthetic

# Maintenance tradeoff scenario  
python rams_demo.py --scenario maintenance --synthetic

# With real data (requires ucimlrepo package)
pip install ucimlrepo
python rams_demo.py
```

## Data Sources

### UCI Naval Propulsion CBM Dataset

**Domain:** Propulsion system health monitoring (Reliability, Availability, Maintainability)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/316)

**Citation:**
```bibtex
@misc{uci_cbm_naval_2014,
    author = {Coraddu, A. and Oneto, L. and Ghio, A. and Savio, S. and Anguita, D. and Figari, M.},
    title = {Condition Based Maintenance of Naval Propulsion Plants},
    year = {2014},
    publisher = {UCI Machine Learning Repository},
    doi = {10.24432/C5K31K}
}
```

**License:** CC BY 4.0

**Description:** 11,934 instances from a CODLAG frigate propulsion simulator with 16 operational features and 2 degradation coefficients (GT compressor decay, GT turbine decay).

**Key Features:**
- Lever position, Ship speed
- GT shaft torque, GT revolutions
- Turbine temperatures and pressures
- Fuel flow rate
- **Targets:** Compressor decay (kMc), Turbine decay (kMt)

### AutoFerry Sensor Fusion Dataset

**Domain:** Navigation safety (Safety)

**Source:** [GitHub - Autoferry](https://github.com/Autoferry/sensor_fusion_dataset)

**Citation:**
```bibtex
@misc{autoferry_sensor_fusion,
    author = {NTNU Autoferry Project},
    title = {Sensor Fusion Benchmark Dataset},
    year = {2021},
    url = {https://github.com/Autoferry/sensor_fusion_dataset}
}
```

**License:** MIT

**Description:** Multi-sensor maritime detections (LiDAR, Radar, IR/EO Camera) collected from the milliAmpere autonomous ferry in Trondheim, Norway.

## Demonstration Scenarios

### Scenario A: Collision Risk + Propulsion Degradation

Demonstrates agentic conflict resolution when:
1. **SafetyAgent** detects collision risk (TCPA=120s, CPA=80m)
2. **ReliabilityAgent** reports propulsion degradation (kMc=0.96)
3. **Supervisor** must balance: execute avoidance maneuver vs. stress degraded propulsion

**Example Decision:**
```
Supervisor Decision: REDUCED_OPERATIONS
Tradeoffs Made:
  - Accept additional propulsion stress for safety maneuver
  - Reduce operational load after maneuver complete
Rationale: Safety takes precedence - propulsion stress acceptable for collision avoidance
```

### Scenario B: Maintenance vs. Availability Tradeoff

Demonstrates goal arbitration when:
1. **MaintainabilityAgent** reports overdue maintenance
2. **AvailabilityAgent** reports mission requires sustained operations
3. **Supervisor** must decide: maintenance now vs. defer with risk

**Example Decision:**
```
Supervisor Decision: MAINTENANCE_REQUIRED
Tradeoffs Made:
  - Defer maintenance 6 hours
  - Accept 3% increased failure probability
Rationale: Complete current mission, mandatory maintenance upon completion
```

## Module Structure

```
rams_agents/
├── __init__.py                 # Package exports
├── base_agent.py               # RAMSBaseAgent, AgentMessage, RAMSMetrics
├── reliability_agent.py        # ReliabilityAgent (R)
├── availability_agent.py       # AvailabilityAgent (A)
├── maintainability_agent.py    # MaintainabilityAgent (M)
├── safety_agent.py             # SafetyAgent (S)
├── supervisor_agent.py         # RAMSSupervisorAgent (coordinator)
└── data_loaders/
    ├── __init__.py
    ├── uci_naval_loader.py     # UCI CBM dataset loader
    └── navigation_loader.py    # AutoFerry dataset loader
```

## Agent API

All agents implement the `RAMSBaseAgent` interface:

```python
class RAMSBaseAgent(ABC):
    @abstractmethod
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """Observe environment and update beliefs."""
        pass
    
    @abstractmethod
    def reason(self) -> List[AgentBelief]:
        """Apply domain knowledge, form new beliefs."""
        pass
    
    @abstractmethod
    def act(self) -> Dict[str, Any]:
        """Produce recommendations/actions."""
        pass
    
    def communicate(self, message: AgentMessage) -> None:
        """Send message to other agents."""
        pass
    
    def run_cycle(self, environment_data) -> Dict[str, Any]:
        """Execute full agent cycle: perceive → reason → act."""
        pass
```

## Extending the Framework

### Adding a New Agent

```python
from rams_agents.base_agent import RAMSBaseAgent, RAMSCategory

class MyCustomAgent(RAMSBaseAgent):
    def __init__(self):
        super().__init__(name="MyAgent", category=RAMSCategory.RELIABILITY)
    
    def perceive(self, environment_data):
        # Process sensor data
        pass
    
    def reason(self):
        # Apply domain logic
        return []
    
    def act(self):
        # Return recommendations
        return {'status': 'OK'}
```

### Using Different Data Sources

The data loaders are modular. To use your own data:

```python
from rams_agents.data_loaders.uci_naval_loader import UCINavalPropulsionLoader

# Load from local CSV
loader = UCINavalPropulsionLoader(data_path='my_propulsion_data.csv')

# Or generate synthetic
loader = UCINavalPropulsionLoader()
loader.load(prefer_synthetic=True)
```

## Dependencies

**Core (included in standard library):**
- `dataclasses`, `enum`, `abc`, `typing`
- `math`, `datetime`, `uuid`

**Scientific:**
- `numpy` - Numerical operations
- `pandas` - Data handling

**Optional (for real data):**
- `ucimlrepo` - Download UCI dataset
- `kagglehub` - Alternative download source

Install optional dependencies:
```bash
pip install ucimlrepo
# or
pip install kagglehub
```

## Methodology Note

This implementation demonstrates the **methodology** of agentic RAMS systems. It is:

- **Not specific to any vessel** - applicable to gas turbine, diesel-electric, or other propulsion
- **Based on real open-source datasets** - with proper academic attribution
- **Extensible** - easily adapted to other sensor data and operational contexts

The key contribution is showing how autonomous agents can:
1. Make domain-specific decisions independently
2. Communicate and coordinate through messages
3. Resolve conflicts between competing objectives (Safety vs. Availability, etc.)
4. Produce integrated system-level assessments

## License

This demonstration code is provided for educational and research purposes.

**Data License Requirements:**
- UCI Naval CBM Dataset: CC BY 4.0 - must cite original authors
- AutoFerry Dataset: MIT License

## References

1. Coraddu, A., Oneto, L., Ghio, A., et al. (2014). Condition Based Maintenance of Naval Propulsion Plants. UCI Machine Learning Repository.

2. NTNU Autoferry Project. (2021). Sensor Fusion Benchmark Dataset. GitHub.

3. IMO COLREGS (1972). International Regulations for Preventing Collisions at Sea.

4. IEC 61508. Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems.
