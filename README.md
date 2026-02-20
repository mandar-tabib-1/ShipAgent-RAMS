# Testing AI for Safety, maintenance, availability and reliability for Marine Vessel Autonomy.

## 🚀 Installation

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| [uv](https://docs.astral.sh/uv/) | latest |

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone & sync environment

```bash
git https://github.com/mandar-tabib-1/ShipAgent-RAMS.git
cd ShipAgent-RAMS

# Install all dependencies into .venv (exact versions from uv.lock)
uv sync
```

> **GPU / CUDA support** — for NVIDIA GPU inference:
> ```bash
> uv sync --extra-index-url https://download.pytorch.org/whl/cu121
> ```
> Replace `cu121` with your installed CUDA version (`cu118`, `cu124`, …).

### Environment variables (LLM Supervisor)

The Streamlit app supports four LLM providers. Set the key for whichever provider you want to use — only one is required. The provider and model are selected at runtime in the sidebar.

| Provider | Env variable | Available models |
|----------|-------------|-----------------|
| **NTNU HPC** | `NTNU_API_KEY` | Kimi-K2.5, Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, o3-mini |
| **Google Gemini** | `GOOGLE_API_KEY` | gemini-2.0-flash, gemini-2.5-pro, gemini-1.5-pro |
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |

```bash
# Linux / macOS — set one or more
export NTNU_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Windows PowerShell — set one or more
$env:NTNU_API_KEY = "your-key-here"
$env:OPENAI_API_KEY = "your-key-here"
$env:GOOGLE_API_KEY = "your-key-here"
$env:ANTHROPIC_API_KEY = "your-key-here"
```

The system runs in deterministic fallback mode when no API key is set.

### Download pre-trained RL weights (optional)

```bash
# PowerShell
Invoke-WebRequest `
  -Uri "https://github.com/Acmece/rl-collision-avoidance/raw/master/policy/stage2.pth" `
  -OutFile "rams_agents/ml_models/rl_policy_stage2.pth"

# bash / curl
curl -L "https://github.com/Acmece/rl-collision-avoidance/raw/master/policy/stage2.pth" \
     -o rams_agents/ml_models/rl_policy_stage2.pth
```

### Run the Streamlit app

```bash
uv run streamlit run streamlit_app.py
```

### Alternative: plain pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install .
streamlit run streamlit_app.py
```

---

## 🚢 Multi-Agent Predictive Maintenance & Navigation Framework

This agentic AI framework is specifically designed for testing vessel autonomy using publically available open-source dataset.



## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RAMS SUPERVISOR AGENT                                │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────────┐
    │                             │                                 │
    ▼                             ▼                                 ▼
┌────────────────┐      ┌─────────────────┐      ┌─────────────────────────┐
│  RELIABILITY   │      │  AVAILABILITY   │      │    MAINTAINABILITY      │
│     AGENT      │      │     AGENT       │      │        AGENT            │
│  (RUL + LSTM)  │      │ (Redundancy)    │      │  (Scheduling + Costs)   │
└────────────────┘      └─────────────────┘      └─────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
    ┌───────────────────┐               ┌───────────────────────────┐
    │   SENSOR FUSION   │               │      SAFETY AGENT         │
    │     (ML+KF)       │──────────────►│   (RL Policy + PSF +      │
    │                   │   Tracks      │    COLREGS Compliance)    │
    └───────────────────┘               └───────────────────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │  Avoidance Maneuvers  │
                                        │  CPA/TCPA + Risk      │
                                        └───────────────────────┘

        DATA FLOW:
        ─────────
        Radar/LiDAR/Cameras → Sensor Fusion (ML+Kalman) → Filtered Tracks
                                                               │
                                        ┌──────────────────────┴──────────────────────┐
                                        │                                             │
                                        ▼                                             ▼
                              RL Policy (PPO)                              COLREGS Rules 13-17
                              Virtual LiDAR Scan                             (Fallback)
                                        │
                                        ▼
                              PSF Safety Filter
                              (Control Barrier Function)
                                        │
                                        ▼
                              Safe Maneuver Commands
```

## 🎯 Streamlit GUI — Agent Summary

The interactive GUI (`streamlit_app.py`) runs a 6-step multi-agent pipeline:

| Step | Agent | AI / Model | Purpose |
|------|-------|------------|---------|
| 1 | **ML Sensor Fusion** | Kalman Filter + RandomForest Maneuver Detector + Sensor Reliability Model | Multi-sensor target tracking with adaptive noise |
| 2 | **Safety Agent** | Reinforcement Learning (PPO) + Policy Safety Filter (PSF / CBF) | Collision risk assessment & maneuver recommendation |
| 3 | **Reliability Agent** | Degradation rate analysis + RUL linear extrapolation | Propulsion component health & Remaining Useful Life |
| 4 | **Availability Agent** | Anomaly scoring + sensor status classification | Sensor & system operational capability monitoring |
| 5 | **Maintainability Agent** | Rule-based scheduling + cost & downtime estimation | Maintenance planning & cost forecasting |
| 6 | **LLM Supervisor** | Large Language Model (NTNU / OpenAI / Google / Anthropic) — COLREGS / DNV expert | Final integrated maritime safety recommendation |

## 🤖 Agents

### Navigation Agents (`agents/`)

#### 1. Sensor Fusion Agent
- Processes multi-sensor detections in **Autoferry benchmark format**
- Supports: Radar, LiDAR, EO Camera, IR Camera
- Implements **ML-Enhanced Kalman filtering** with track management
- NED (North-East-Down) coordinate frame
- **Maneuver-adaptive process noise** using Random Forest classifier
- **Sensor-specific measurement noise** learned from ground truth

#### 2. Collision Avoidance Agent
- **COLREGS Rules 13-17** implementation
- Head-on, crossing, and overtaking situation detection
- CPA/TCPA calculations
- Risk assessment with recommended maneuvers

### RAMS Agents (`rams_agents/`)

#### 5. Safety Agent (RL + PSF)
- **Pre-trained PPO Policy** for collision avoidance (arXiv:1709.10082)
- **Potential Safety Function (PSF)** using Control Barrier Functions
- **COLREGS-compliant** maneuver generation
- CPA > 50m safety constraint enforcement
- Fallback to rule-based COLREGS when RL unavailable

#### 6. Reliability Agent
- **LSTM-based RUL prediction** for propulsion components
- Degradation model tracking for diesel generators, thrusters
- UCI Naval Propulsion dataset training
- Failure probability forecasting

#### 7. Availability Agent
- **Sensor redundancy monitoring** (Radar, LiDAR, GPS/INS)
- System availability calculation
- Graceful degradation under sensor failures
- DP system availability tracking

#### 8. Maintainability Agent
- **Maintenance cost optimization**
- Scheduled vs emergency maintenance planning
- Component criticality ranking
- Work order generation

#### 9. Supervisor Agent
- **Orchestrates all RAMS agents**
- Risk aggregation and prioritization
- Cross-agent alert correlation
- Executive summary generation

## 📊 Open-Source Data Sources

### 1. Autoferry Sensor Fusion Benchmark
- **URL**: https://github.com/Autoferry/sensor_fusion_dataset
- **Description**: Multi-sensor tracking dataset with Vessel as target
- **Sensors**: Radar, LiDAR, EO/IR cameras
- **Format**: JSON with NED coordinates

### 2. Open Simulation Platform (OSP)
- **URL**: https://open-simulation-platform.github.io
- **Vessel Models**: DP control, path-following, vessel dynamics
- **Format**: FMU (Functional Mock-up Units)
- **Demo Cases**: Vessel-DP, Vessel-Path-following

### 3. MarineTraffic AIS Data
- **MMSI**: 258342000
- **Real-time tracking**: https://www.marinetraffic.com/en/ais/details/ships/shipid:311701

### 4. NTNU Digital Twin Project
- **URL**: https://github.com/Traversal2021/Gunnerus
- **Description**: Crane, engine, ship motion data visualization
- **Data**: MQTT-based real-time sensor streams

## 🧠 ML-Enhanced Sensor Fusion

The sensor fusion agent uses machine learning to **adaptively tune Kalman filter parameters** in real-time, improving tracking accuracy for maneuvering targets and accounting for sensor-specific noise characteristics.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ML-ENHANCED KALMAN FILTER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   MANEUVER      │    │    KALMAN       │    │    SENSOR       │  │
│  │   DETECTOR      │───►│    FILTER       │◄───│  RELIABILITY    │  │
│  │  (Q Adaptive)   │    │  (State Est.)   │    │  (R Adaptive)   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│         │                       │                       │           │
│    RandomForest           EKF State:              Per-sensor        │
│    Classifier          [N, E, vN, vE]           error stats         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### ML Components

#### 1. Maneuver Detector (Process Noise Q)

Predicts if a target is maneuvering based on velocity history:

| State | Process Noise Q | Description |
|-------|-----------------|-------------|
| **Steady** (prob < 0.5) | 0.1 | Smooth tracks, filters noise |
| **Maneuvering** (prob ≥ 0.5) | 2.0 | Responsive to accelerations |

- **Model**: Random Forest Classifier (50 trees, depth 8)
- **Features**: Sliding window of velocities, speeds, accelerations
- **Training**: Autoferry ground truth trajectories
- **Accuracy**: 89.7% (on training data)

#### 2. Sensor Reliability Estimator (Measurement Noise R)

Learns measurement noise from ground truth comparison:

| Sensor | Mean Error (m) | Reliability Score |
|--------|----------------|-------------------|
| LiDAR | 21.0 | 0.32 |
| Radar | 19.9 | 0.33 |
| IR Camera | - | 0.50 (default) |
| EO Camera | - | 0.50 (default) |

### Training/Test Split

| Environment | Scenarios | Usage |
|-------------|-----------|-------|
| **Env 1** (Training) | 2, 3, 4, 5, 6 | Model training |
| **Env 2** (Testing) | 13, 16, 17, 22 | Evaluation only |

### Usage

```python
# ML-enhanced tracking (default)
from agents.sensor_fusion_agent import SensorFusionAgentKalman

agent = SensorFusionAgentKalman(use_ml=True)  # Loads trained models
result = agent.process({"detections": [...], "ground_truth": [...]})

# Access ML status
for track in result['tracks']:
    print(f"Maneuver prob: {track['maneuver_probability']:.1%}")
    print(f"Process noise Q: {track['process_noise_Q']:.2f}")
```

### Visualization

```bash
# ML comparison visualization
python visualize_ml_fusion.py --scenario scenario16

# Shows: ML vs Fixed Kalman, maneuver detection, sensor reliability
```

### Retrain Models

```bash
# Retrain on all Environment 1 scenarios
python sensor_fusion_ml.py
```

## �️ RL+PSF Collision Avoidance

The Safety Agent combines **Reinforcement Learning** with a **Potential Safety Function** for collision avoidance that guarantees safety constraints.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RL + PSF COLLISION AVOIDANCE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │
│  │  KALMAN TRACKS  │    │   RL POLICY     │    │   PSF FILTER        │  │
│  │  (from Sensor   │───►│   (PPO)         │───►│   (CBF)             │  │
│  │   Fusion)       │    │                 │    │                     │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘  │
│         │                       │                       │               │
│    Track positions         Pre-trained            Control Barrier       │
│    + velocities           CNN policy              φ(CPA, TCPA) > 0     │
│    + CPA/TCPA                                                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              OBSERVATION ADAPTER                                 │    │
│  │  Converts Kalman tracks → Virtual 512-bin LiDAR scan            │    │
│  │  + Goal direction + Speed normalization                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **RL Policy** | Pre-trained PPO from Acmece/rl-collision-avoidance |
| **PSF Filter** | Control Barrier Function: φ = CPA - 50m - k·max(0, 60s - TCPA) |
| **Observation Adapter** | Converts tracks to 512-bin virtual LiDAR scan |
| **COLREGS Fallback** | Rule-based Rules 13-17 when RL unavailable |

### Safety Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| CPA_CRITICAL | 50m | Minimum closest point of approach |
| TCPA_CRITICAL | 60s | Time horizon for safety consideration |
| CBF_MARGIN | 10m | Safety margin before intervention |

### PSF Intervention Types

| Type | Trigger | Action |
|------|---------|--------|
| **None** | φ > margin | Pass RL action through |
| **Override** | 0 < φ < margin | Increase turn magnitude |
| **Speed** | Speed reduction safer | Reduce speed ratio |
| **Emergency** | φ ≤ 0 | Maximum evasive action |

### Usage

```python
from rams_agents.safety_agent import SafetyAgent

# Create with RL+PSF enabled
agent = SafetyAgent(use_rl_policy=True)

# Process navigation state
result = agent.process({
    'tracks': [...],  # Kalman-filtered tracks
    'own_position': (0, 0),
    'own_velocity': (5, 0),
    'own_heading': 0
})

# Get RL+PSF statistics
stats = agent.get_rl_psf_stats()
print(f"RL used: {stats['rl_used']} times")
print(f"PSF interventions: {stats['interventions']}")
```

### Pre-trained Weights

Download from [Acmece/rl-collision-avoidance](https://github.com/Acmece/rl-collision-avoidance):

```powershell
# Download pre-trained PPO weights
cd rams_agents/ml_models
Invoke-WebRequest -Uri "https://github.com/Acmece/rl-collision-avoidance/raw/master/policy/stage2.pth" -OutFile "rl_policy_stage2.pth"
```

### Demo

```bash
# Run RL+PSF collision avoidance demo
python rl_psf_demo.py
```

See [rams_agents/CREDITS.md](rams_agents/CREDITS.md) for full attribution.

## �🚀 Quick Start

```bash
cd ShipAgent-RAMS

# Run with synthetic data
python main_demo.py

# Run with real Autoferry data
python main_demo.py --real

# INTEGRATED NAVIGATION DISPLAY (Sensor Fusion + Collision Avoidance)
python integrated_navigation_display.py                  # Synthetic COLREGS scenarios
python integrated_navigation_display.py --real           # Real Autoferry data
python integrated_navigation_display.py --real --animate # Animated display

# ML comparison visualization
python visualize_ml_fusion.py --scenario scenario16

# Interactive sensor fusion GUI
python visualize_sensor_fusion.py --real --animate

# RAMS demonstration
python rams_demo.py                                      # Full RAMS system demo

# RL+PSF Collision Avoidance demo
python rl_psf_demo.py                                    # RL policy + PSF safety filter
```

## 📁 Project Structure

```
ShipAgent_ai_system/
├── __init__.py                        # Package info & vessel specs
├── orchestrator.py                    # Main orchestrator
├── main_demo.py                       # Demo script
├── rams_demo.py                       # RAMS system demo
├── rl_psf_demo.py                     # RL+PSF collision avoidance demo
├── llm_supervisor_demo.py             # LLM supervisor demo
├── integrated_navigation_display.py   # Integrated navigation GUI
├── streamlit_app.py                   # Streamlit web app
├── sensor_fusion_ml.py                # ML training module
├── visualize_sensor_fusion.py         # GUI visualization
├── visualize_ml_fusion.py             # ML comparison visualization
├── test_all_modules.py                # Module test suite
├── data.csv                           # Scenario data
├── pyproject.toml                     # Project configuration
├── uv.lock                            # Dependency lock file
│
├── agents/                            # Navigation agents
│   ├── __init__.py
│   ├── agents.py                      # Agent definitions
│   ├── base_agent.py                  # Base agent class
│   ├── sensor_fusion_agent.py         # Kalman filter + ML
│   └── collision_avoidance_agent.py   # COLREGS
│
├── config/                            # Configuration
│   ├── __init__.py
│   ├── gunnerus_config.py             # Vessel configuration
│   └── llm_config.yaml               # LLM configuration
│
├── rams_agents/                       # RAMS framework agents
│   ├── __init__.py
│   ├── base_agent.py                  # RAMS base agent
│   ├── supervisor_agent.py            # Orchestrates all RAMS agents
│   ├── llm_supervisor_agent.py        # LLM-based supervisor
│   ├── safety_agent.py                # RL+PSF collision avoidance
│   ├── reliability_agent.py           # RUL prediction
│   ├── availability_agent.py          # Redundancy monitoring
│   ├── dp_availability.py             # DP system availability
│   ├── maintainability_agent.py       # Maintenance optimization
│   ├── psf_filter.py                  # Potential Safety Function (CBF)
│   ├── rl_observation_adapter.py      # Tracks → RL observation
│   ├── rul_estimator.py               # RUL estimation models
│   ├── sensor_redundancy.py           # Sensor availability
│   ├── CREDITS.md                     # Attribution for RL model
│   ├── README.md                      # RAMS documentation
│   │
│   ├── ml_models/                     # ML models for RAMS
│   │   ├── rl_collision_policy.py     # Pre-trained PPO policy
│   │   ├── lstm_rul.py                # LSTM for RUL prediction
│   │   ├── autoencoder_anomaly.py     # Anomaly detection
│   │   ├── train_models.py            # Model training scripts
│   │   └── rl_policy_stage2.pth      # Pre-trained weights
│   │
│   └── data_loaders/                  # Data loaders
│       ├── uci_naval_loader.py        # UCI Naval dataset
│       └── navigation_loader.py       # Navigation data
│
├── data/
│   ├── __init__.py
│   ├── autoferry_loader.py            # Dataset loader
│   ├── README.md                      # Dataset documentation
│   └── sensor_fusion_dataset/         # Scenarios 2,3,4,5,6,13,16,17,22
│
└── models/
    ├── maneuver_detector.pkl          # Trained RF classifier
    ├── sensor_reliability.pkl         # Sensor error statistics
    ├── lstm_rul.pkl                   # LSTM RUL model
    ├── autoencoder_anomaly.pkl        # Anomaly detection model
    ├── test_X.npy                     # Test features
    ├── test_kMc.npy                   # Test Kalman covariance
    └── test_kMt.npy                   # Test Kalman targets
```

## 🔧 DNV RAMS Framework

For **Reliability, Availability, Maintainability, and Safety (RAMS)** per DNV standards:

| Aspect | Implementation | Agent |
|--------|----------------|-------|
| **Reliability** | LSTM RUL prediction, degradation models, failure forecasting | `ReliabilityAgent` |
| **Availability** | Sensor redundancy, DP availability, graceful degradation | `AvailabilityAgent` |
| **Maintainability** | Cost optimization, scheduled maintenance, work orders | `MaintainabilityAgent` |
| **Safety** | RL+PSF collision avoidance, COLREGS compliance, CPA>50m | `SafetyAgent` |

### Agent Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SUPERVISOR AGENT                               │
│                  (Orchestrates all agents)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ RELIABILITY │◄────►│ AVAILABILITY│◄────►│MAINTAINAB.  │
│   AGENT     │      │    AGENT    │      │   AGENT     │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       │     Degradation    │  Sensor Health     │  Cost Factors
       │     Forecasts      │  Status            │
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  SAFETY AGENT   │
                   │  (RL + PSF +    │
                   │   COLREGS)      │
                   └────────┬────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
RL Policy              PSF Filter            COLREGS Rules
(PPO)                  (CBF)                 (Fallback)
    │                       │                       │
    └───────────────────────┼───────────────────────┘
                            │
                            ▼
                   Safe Maneuver Commands
```

## 🖥️ Streamlit GUI — 6-Step Agent Pipeline

Launch the GUI with:

```bash
streamlit run streamlit_app.py
```

### Step 1 · ML Sensor Fusion (`SensorFusionAgentKalman`)

| | Detail |
|--|--------|
| **AI / Model** | 6-state Kalman Filter (x, y, vx, vy, ax, ay) with RandomForest Maneuver Detector (adaptive Q) and Sensor Reliability Model (adaptive R) |
| **Dataset** | AutoFerry multi-sensor benchmark — LiDAR, Radar, IR Camera, EO Camera in NED coordinates |
| **Inputs** | Multi-sensor detections with sensor IDs; scenario timestep index |
| **Outputs** | Confirmed Kalman tracks with position/velocity; adaptive noise Q & R; maneuver probability; position uncertainty (metres) |

### Step 2 · Safety Agent (`SafetyAgent`)

| | Detail |
|--|--------|
| **AI / Model** | Reinforcement Learning — PPO (Proximal Policy Optimisation) policy + Policy Safety Filter (PSF) implemented as a Control Barrier Function (CBF) |
| **Inputs** | Kalman-filtered target tracks; own-ship position, velocity, heading |
| **Outputs** | Risk level (LOW / MEDIUM / HIGH / CRITICAL); safety index (0–100 %); recommended collision-avoidance maneuvers; per-target COLREGS compliance |

### Step 3 · Reliability Agent (`ReliabilityAgent`)

| | Detail |
|--|--------|
| **AI / Model** | Degradation rate analysis on compressor and turbine decay coefficients; threshold-based health scoring; linear RUL extrapolation |
| **Dataset** | UCI Naval Propulsion Condition-Based Maintenance dataset (16 sensor features) |
| **Inputs** | Compressor decay coefficient; turbine decay coefficient; 16 propulsion sensor readings |
| **Outputs** | Overall health status (GOOD / WARNING / CRITICAL); per-component health %; Remaining Useful Life estimate (hours); alerts and warnings |

### Step 4 · Availability Agent (`AvailabilityAgent`)

| | Detail |
|--|--------|
| **AI / Model** | Z-score / threshold anomaly scoring; multi-sensor cross-validation; rule-based sensor status classifier |
| **Inputs** | 16 propulsion sensor readings; navigation sensor status flags; operating hours |
| **Outputs** | Operating mode (FULL / DEGRADED / LIMITED); system capability %; per-sensor health status; anomaly list |

### Step 5 · Maintainability Agent (`MaintainabilityAgent`)

| | Detail |
|--|--------|
| **AI / Model** | Rule-based maintenance scheduling engine; priority scoring (CRITICAL / HIGH / MEDIUM / LOW); cost and downtime estimation |
| **Inputs** | Reliability Agent findings (health scores, RUL); operating hours; component threshold definitions |
| **Outputs** | Prioritised maintenance task list; estimated cost per task (USD); estimated downtime per task (hours); actionable recommendations |

### Step 6 · LLM Supervisor (`LLMSupervisorAgent`)

| | Detail |
|--|--------|
| **AI / Model** | Large Language Model — selectable provider (NTNU HPC, OpenAI, Google Gemini, Anthropic); COLREGS Rules 8 & 16 knowledge base; DNV GL maritime safety standards; deterministic fallback |
| **Inputs** | Safety, Reliability, Availability, and Maintainability agent reports |
| **Outputs** | Integrated maritime safety recommendation; vessel readiness assessment; prioritised action items; COLREGS / DNV compliance verdict |

---

## ⚠️ Known Limitations

### Sensor Fusion: Observer Motion Not Compensated

**Current Limitation**: The sensor fusion agent assumes the observer vessel is stationary. Target positions and velocities are computed in the observer's body-relative frame without accounting for the observer's own motion.

**Impact**: When the observer vessel is moving:
- Target velocity estimates may be incorrect (observer motion is attributed to targets)
- Track positions are relative to a moving reference frame
- CPA/TCPA calculations may be inaccurate

```
Example Error Scenario:
────────────────────────
Observer moving East at 5 m/s
Target is stationary

Sensor reports: Target at (100m, 0m) at t=0
Sensor reports: Target at (100m, -5m) at t=1  ← Shifted due to observer motion

Current system interprets: Target velocity = (0, -5) m/s  ❌ WRONG
Reality: Target velocity = (0, 0) m/s, observer moved
```

### Solution: GPS/INS Integration

To correct this limitation, the sensor fusion agent needs the **observer vessel's own position and motion** from onboard navigation systems:

| Data Source | Provides | Update Rate |
|-------------|----------|-------------|
| **GPS** | Latitude, Longitude, SOG, COG | 1-10 Hz |
| **INS** | Position, Velocity, Heading, Roll/Pitch | 100+ Hz |
| **GPS/INS Fusion** | High-accuracy position + attitude | 100 Hz |

**Required Observer State Input**:
```python
observer_state = {
    "time": 1620114891.85,
    "x": 150.2,           # Observer North position (m) in world frame
    "y": -390.5,          # Observer East position (m) in world frame
    "heading": 0.52,      # Heading (radians, 0=North)
    "vx": 2.1,            # Velocity North (m/s)
    "vy": 0.3,            # Velocity East (m/s)
    "yaw_rate": 0.01      # Turn rate (rad/s)
}
```

**Correction Process**:
1. Receive detection in body-relative coordinates
2. Transform to world coordinates using observer position + heading
3. Run Kalman filter in world-fixed frame
4. Target velocities now correctly represent their true motion


### Autoferry Dataset Note

The Autoferry sensor fusion benchmark dataset:
- Was collected **from milliAmpere ferry's sensors** (not R/V Gunnerus)
- Contains Gunnerus **as a tracked target** in some scenarios
- Does **not include** milliAmpere's GPS/INS data in the public dataset
- Ownship position is provided per detection, but not velocity/heading

For production use on any vessel, real-time GPS/INS integration is required for accurate target tracking.

## 📚 References

1. Autoferry Project: https://autoferry.github.io
2. Open Simulation Platform: https://opensimulationplatform.com
3. NTNU R/V Gunnerus: https://www.ntnu.edu/gunnerus
4. VeSim Simulator: Hassani et al., 2015
5. Gunnerus Time Domain Model: ResearchGate
6. **RL Collision Avoidance**: Long et al., "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning", arXiv:1709.10082
7. **Implementation**: Tianyu Liu (2018), https://github.com/Acmece/rl-collision-avoidance
8. UCI Naval Propulsion Dataset: https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

## 📄 License

MIT License - Research and educational use.

See [rams_agents/CREDITS.md](rams_agents/CREDITS.md) for full attribution of external datasets and models.

---

**Built for NTNU Maritime Research** 🇳🇴
