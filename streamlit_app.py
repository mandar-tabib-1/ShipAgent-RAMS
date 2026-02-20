"""
 Multi-Agent Framework for Vessels - Streamlit GUI

Interactive dashboard for running the multi-agent RAMS system with
step-by-step agent execution and real-time visualizations.

Usage:
    streamlit run streamlit_app.py

Features:
    - Load real sensor data from datasets
    - Step-by-step agent execution with user control
    - Real-time visualizations for agent OUTPUTS
    - Final LLM Supervisor recommendation

Requirements:
    pip install streamlit plotly pandas
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# ============================================================================
# LLM Provider Configuration
# ============================================================================

PROVIDER_CONFIGS: dict = {
    'ntnu': {
        'label': 'NTNU HPC (OpenAI-compatible)',
        'api_base': 'https://llm.hpc.ntnu.no/v1',
        'env_key': 'NTNU_API_KEY',
        'models': [
            'moonshotai/Kimi-K2.5',
            'meta-llama/Llama-3.3-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
        ],
        'needs_base_url': True,
    },
    'openai': {
        'label': 'OpenAI',
        'api_base': 'https://api.openai.com/v1',
        'env_key': 'OPENAI_API_KEY',
        'models': ['gpt-4o', 'gpt-4o-mini', 'o3-mini'],
        'needs_base_url': False,
    },
    'google': {
        'label': 'Google Gemini',
        'api_base': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'env_key': 'GOOGLE_API_KEY',
        'models': ['gemini-2.0-flash', 'gemini-2.5-pro', 'gemini-1.5-pro'],
        'needs_base_url': False,
    },
    'anthropic': {
        'label': 'Anthropic',
        'api_base': '',
        'env_key': 'ANTHROPIC_API_KEY',
        'models': ['claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'],
        'needs_base_url': False,
    },
}

# ============================================================================
# Stage Expert Roles - System prompts for LLM expert analysis at each stage
# ============================================================================
STAGE_EXPERT_ROLES: dict = {
    'fusion': {
        'name': 'Sensor Fusion Expert',
        'system_prompt': """You are an expert in multi-sensor data fusion, Kalman filtering, and target tracking for maritime autonomous systems.

Your expertise includes:
- Extended Kalman Filters (EKF) and adaptive noise estimation
- Multi-sensor fusion: LiDAR, Radar, IR cameras, EO cameras
- Maneuver detection algorithms and target tracking
- Sensor reliability assessment and measurement uncertainty
- Track management (initiation, confirmation, deletion)

When analyzing sensor fusion outputs, focus on:
1. Track quality and confirmation status
2. Adaptive noise parameters (Q for process, R for measurement)
3. Maneuver probability and its implications
4. Sensor reliability scores and data association
5. Position uncertainty and tracking accuracy

Provide actionable insights for maritime situational awareness.""",
    },
    'safety': {
        'name': 'Maritime Safety & RL Validation Expert',
        'system_prompt': """You are a maritime safety expert AND an RL policy validator with expertise in sim-to-real transfer.

**Maritime Safety Expertise:**
- COLREGS (International Regulations for Preventing Collisions at Sea)
- Collision Risk Index (CRI) and Time to Closest Point of Approach (TCPA)
- Control Barrier Functions (CBF) and Policy Safety Filters (PSF)
- Risk assessment methodologies for maritime operations

**RL Validation Expertise:**
- Kolmogorov-Smirnov (KS) test for distribution comparison
- Zero-shot policy evaluation on out-of-distribution data
- Sim-to-real gap analysis and domain adaptation assessment
- PSF intervention rate as generalization metric

**When analyzing safety outputs, focus on:**
1. Risk levels and COLREGS compliance
2. PSF barrier values (φ): positive = safe, negative = unsafe
3. PSF intervention types and their implications
4. RL action confidence and stability

**When validating RL policy (if user asks), assess:**
1. **Distribution Match**: Compare RL training domain vs current data
   - High PSF intervention rate (>30%) suggests domain mismatch
   - Low action confidence suggests out-of-distribution inputs
   
2. **Zero-Shot Performance**: 
   - Track: actions with low confidence (<50%)
   - Track: actions requiring PSF override
   - Verdict: GOOD (<10% interventions), WARNING (10-30%), CRITICAL (>30%)

3. **Validation Verdict**: Provide one of:
   - "VALIDATED: Policy generalizes well to this data"
   - "ACCEPTABLE: Minor domain gap, PSF provides adequate safety"
   - "WARNING: Significant domain gap, increased PSF reliance"
   - "CRITICAL: Policy unsuitable, recommend retraining or fallback to rules"

Always explain the reasoning behind your validation verdict.""",
    },
    'reliability': {
        'name': 'Reliability Engineering Expert',
        'system_prompt': """You are a reliability engineering expert specializing in marine propulsion systems and condition-based maintenance.

Your expertise includes:
- Remaining Useful Life (RUL) estimation and prediction
- Degradation modeling and failure mode analysis
- Condition-Based Maintenance (CBM) strategies
- Gas turbine and compressor health monitoring
- LSTM neural networks for RUL prediction

When analyzing reliability outputs, focus on:
1. Component health percentages and degradation trends
2. RUL estimates and maintenance planning windows
3. Critical alerts and their urgency levels
4. Degradation rates and failure prediction
5. Recommendations for predictive maintenance actions

Provide insights that help prevent unexpected failures.""",
    },
    'availability': {
        'name': 'System Availability Expert',
        'system_prompt': """You are a system availability expert focusing on sensor redundancy and operational capability for autonomous maritime systems.

Your expertise includes:
- Sensor health monitoring and fault detection
- Redundancy management and graceful degradation
- Anomaly detection using autoencoders
- Operational capability assessment
- Dynamic Positioning (DP) system availability

When analyzing availability outputs, focus on:
1. Sensor status (HEALTHY, DEGRADED, FAILED) and implications
2. System capability percentage and operating mode
3. Anomaly scores and detected anomalies
4. Redundancy levels and fault tolerance
5. Recommendations for maintaining operational capability

Provide guidance for maintaining system availability.""",
    },
    'maintenance': {
        'name': 'Maintenance Planning Expert',
        'system_prompt': """You are a maintenance planning expert with expertise in predictive maintenance scheduling and lifecycle management for marine vessels.

Your expertise includes:
- Predictive maintenance scheduling and optimization
- Mean Time To Repair (MTTR) and maintenance metrics
- Maintenance cost forecasting and budgeting
- Spare parts management and logistics
- Maintenance task prioritization

When analyzing maintenance outputs, focus on:
1. Pending maintenance tasks and their priorities
2. Cost forecasts and downtime estimates
3. Maintainability metrics and trends
4. Scheduling recommendations and windows
5. Resource allocation and planning

Provide practical maintenance planning recommendations.""",
    },
    'supervisor': {
        'name': 'RAMS Integration Expert',
        'system_prompt': """You are a senior maritime RAMS (Reliability, Availability, Maintainability, Safety) consultant with expertise in integrating multi-agent system outputs.

Your expertise includes:
- Holistic vessel health assessment
- Multi-criteria decision making for maritime operations
- Risk-based operational planning
- DNV and classification society standards
- Autonomous vessel regulations and guidelines

When analyzing supervisor outputs, focus on:
1. Overall vessel status and operational readiness
2. Integrated RAMS assessment across all domains
3. Priority recommendations and action items
4. Trade-offs between safety, reliability, and availability
5. Regulatory compliance considerations

Provide executive-level recommendations for vessel operations.""",
    },
}


def _load_dotenv() -> dict:
    """
    Read .env file from the project root (or cwd) and return key-value pairs.
    Values are also injected into os.environ so they are visible to subprocesses.
    Falls back silently if no .env file exists.
    """
    env_vals: dict = {}
    for candidate in [Path('.env'), Path('../.env'), Path(__file__).parent.parent / '.env']:
        if candidate.exists():
            with open(candidate, encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    env_vals[key] = val
                    os.environ.setdefault(key, val)
            break
    return env_vals


def _test_llm_connection(provider: str, model: str, api_key: str, api_base: str):
    """
    Send a minimal test request to the selected provider.
    Returns (success: bool, message: str).
    """
    if not api_key:
        return False, "No API key provided."
    try:
        if provider == 'anthropic':
            try:
                import anthropic as _ant
            except ImportError:
                return False, "anthropic package not installed — run: uv sync"
            client = _ant.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": "Reply with just OK"}]
            )
            reply = msg.content[0].text.strip()[:30] if msg.content else '(no text)'
            return True, f"Connected · {model} replied: {reply}"
        else:
            from openai import OpenAI
            kwargs: dict = {'api_key': api_key, 'timeout': 15}
            if api_base:
                kwargs['base_url'] = api_base
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with just the word OK"}],
                max_tokens=10,
            )
            reply = (resp.choices[0].message.content or '').strip()[:30]
            return True, f"Connected · {model} replied: {reply}"
    except Exception as exc:
        return False, str(exc)[:150]


import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Multi-Agent RAMS Framework for Marine Vessels - Demonstration",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# LLM Expert Analysis Functions
# ============================================================================
def call_llm_expert(
    stage_name: str,
    stage_output: Dict[str, Any],
    user_message: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Call the LLM as an expert for a specific stage.
    
    Args:
        stage_name: One of 'fusion', 'safety', 'reliability', 'availability', 'maintenance', 'supervisor'
        stage_output: The output dictionary from that stage's agent
        user_message: Optional user question for chatbot mode
        chat_history: Optional list of previous messages for context
    
    Returns:
        Expert analysis or chatbot response as string
    """
    provider = st.session_state.get('llm_provider', 'ntnu')
    model = st.session_state.get('llm_model', '')
    api_key = st.session_state.get('llm_api_key', '')
    api_base = st.session_state.get('llm_api_base', '')
    
    if not api_key:
        return "⚠️ No API key configured. Please set your API key in the sidebar under 'LLM Supervisor'."
    
    expert_config = STAGE_EXPERT_ROLES.get(stage_name, {})
    system_prompt = expert_config.get('system_prompt', 'You are a helpful expert assistant.')
    
    # Build the context message with stage output
    output_json = json.dumps(stage_output, indent=2, default=str)[:8000]  # Limit size
    
    if user_message:
        # Chatbot mode - include history and user question
        context_msg = f"""Here is the current output from the {stage_name} stage:

```json
{output_json}
```

Please answer the user's question based on this data and your expertise."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context as first user message if no history
        if not chat_history:
            messages.append({"role": "user", "content": context_msg})
            messages.append({"role": "assistant", "content": "I've analyzed the data. What would you like to know?"})
        else:
            # Add the context once, then history
            messages.append({"role": "user", "content": context_msg})
            messages.append({"role": "assistant", "content": "I've analyzed the data. What would you like to know?"})
            for msg in chat_history:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
    else:
        # Expert analysis mode - one-shot analysis
        context_msg = f"""Analyze the following output from the {stage_name.upper()} stage and provide expert insights:

```json
{output_json}
```

Provide a concise expert analysis covering:
1. Key observations and findings
2. Potential concerns or issues
3. Recommendations for action
4. Overall assessment

Keep your response focused and actionable."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_msg}
        ]
    
    try:
        if provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            # Convert messages for Anthropic format
            anthropic_messages = [m for m in messages if m['role'] != 'system']
            msg = client.messages.create(
                model=model,
                max_tokens=1500,
                system=system_prompt,
                messages=anthropic_messages
            )
            return msg.content[0].text if msg.content else "No response generated."
        else:
            from openai import OpenAI
            kwargs: dict = {'api_key': api_key, 'timeout': 60}
            if api_base:
                kwargs['base_url'] = api_base
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            return resp.choices[0].message.content or "No response generated."
    except Exception as exc:
        return f"⚠️ LLM Error: {str(exc)[:200]}"


def render_expert_panel(stage_name: str, stage_output: Dict[str, Any]):
    """
    Render the LLM Expert panel with analysis button and chatbot.
    
    Args:
        stage_name: One of 'fusion', 'safety', 'reliability', 'availability', 'maintenance', 'supervisor'
        stage_output: The output dictionary from that stage's agent
    """
    if stage_output is None:
        return
    
    expert_config = STAGE_EXPERT_ROLES.get(stage_name, {})
    expert_name = expert_config.get('name', 'Expert')
    
    st.markdown("---")
    st.subheader(f"🤖 LLM {expert_name}")
    
    # Expert Analysis Button
    analysis_key = f'{stage_name}_expert_analysis'
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(f"🔍 Analyze as Expert", key=f"expert_btn_{stage_name}", use_container_width=True):
            with st.spinner(f"Getting expert analysis from {st.session_state.get('llm_model', 'LLM')}..."):
                analysis = call_llm_expert(stage_name, stage_output)
                st.session_state[analysis_key] = analysis
    
    with col2:
        api_key = st.session_state.get('llm_api_key', '')
        if not api_key:
            st.caption("⚠️ Configure API key in sidebar to enable expert analysis")
        else:
            st.caption(f"Using: {st.session_state.get('llm_provider', '').upper()} / {st.session_state.get('llm_model', '')}")
    
    # Display cached analysis
    if st.session_state.get(analysis_key):
        with st.expander("📋 Expert Analysis", expanded=True):
            st.markdown(st.session_state[analysis_key])
    
    # Chatbot Section
    st.markdown("**💬 Ask the Expert**")
    
    chat_history_key = f'{stage_name}_chat_history'
    chat_history = st.session_state.get(chat_history_key, [])
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in chat_history:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    user_input = st.chat_input(
        f"Ask the {expert_name} a question...",
        key=f"chat_input_{stage_name}"
    )
    
    if user_input:
        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})
        
        # Get response
        with st.spinner("Thinking..."):
            response = call_llm_expert(
                stage_name, 
                stage_output, 
                user_message=user_input,
                chat_history=chat_history[:-1]  # Exclude current message, it's added in the function
            )
        
        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": response})
        st.session_state[chat_history_key] = chat_history
        st.rerun()
    
    # Clear chat button
    if chat_history:
        if st.button("🗑️ Clear Chat", key=f"clear_chat_{stage_name}"):
            st.session_state[chat_history_key] = []
            st.rerun()


# ============================================================================
# Data Loaders (Cached)
# ============================================================================
@st.cache_resource
def load_data_sources():
    """Load and cache data loaders."""
    try:
        from rams_agents.data_loaders.uci_naval_loader import UCINavalPropulsionLoader
        from rams_agents.data_loaders.navigation_loader import NavigationDataLoader
        
        propulsion_loader = UCINavalPropulsionLoader()
        propulsion_loader.load(prefer_synthetic=True)  # Use synthetic for demo
        
        navigation_loader = NavigationDataLoader()
        scenarios = navigation_loader.list_scenarios()
        
        return {
            'propulsion': propulsion_loader,
            'navigation': navigation_loader,
            'scenarios': scenarios,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


@st.cache_resource
def load_ml_models():
    """Load trained ML models for sensor fusion (maneuver detector + sensor reliability)."""
    models = {
        'loaded': False,
        'maneuver_detector': None,
        'sensor_reliability': None,
        'maneuver_loaded': False,
        'reliability_loaded': False,
        'training_stats': {}
    }
    models_dir = Path("models")
    
    try:
        # Load maneuver detector
        maneuver_path = models_dir / "maneuver_detector.pkl"
        if maneuver_path.exists():
            from sensor_fusion_ml import ManeuverDetector
            detector = ManeuverDetector()
            detector.load(str(maneuver_path))
            models['maneuver_detector'] = detector
            models['maneuver_loaded'] = True
            models['training_stats']['maneuver'] = detector.training_stats
        
        # Load sensor reliability
        reliability_path = models_dir / "sensor_reliability.pkl"
        if reliability_path.exists():
            import pickle
            with open(reliability_path, 'rb') as f:
                models['sensor_reliability'] = pickle.load(f)
            models['reliability_loaded'] = True
        
        models['loaded'] = models['maneuver_loaded'] or models['reliability_loaded']
        
    except Exception as e:
        models['error'] = str(e)
    
    return models


@st.cache_data
def load_autoferry_scenario_data(scenario_id: str):
    """
    Load and preprocess AutoFerry dataset for Kalman filter processing.
    
    This function uses AutoferryDataLoader to:
    1. Filter to position-only sensors (LiDAR=1, Radar=2) - excludes bearing sensors (IR=3, EO=4)
    2. Convert measurements from ownship-fixed frame to Piren NED frame
    3. Create properly formatted 'position' field expected by Kalman filter
    
    Raw data has:
    - 'measurement': sensor reading (bearings for cameras, positions for LiDAR/Radar)
    - 'ownshipPosition': vessel position in Piren NED
    
    Kalman filter expects:
    - 'position': [north, east, down] in Piren NED frame
    """
    data = {'detections': [], 'ground_truth': [], 'loaded': False}
    
    try:
        from data.autoferry_loader import AutoferryDataLoader
        
        # Use the proper data loader that handles coordinate conversion
        dataset_path = Path("data/sensor_fusion_dataset")
        loader = AutoferryDataLoader(str(dataset_path))
        
        # load_for_kalman_filter does:
        # - Filters to position sensors only (LiDAR, Radar) 
        # - Converts from ownship-fixed to Piren NED frame
        # - Creates 'position' field with [north, east, 0]
        kalman_data = loader.load_for_kalman_filter(scenario_id, use_position_sensors_only=True)
        
        data['detections'] = kalman_data.get('detections', [])
        data['ground_truth'] = kalman_data.get('ground_truth', [])
        data['loaded'] = len(data['detections']) > 0
        data['total_detections'] = len(data['detections'])
        data['raw_detection_count'] = kalman_data.get('raw_count', len(data['detections']))
        data['total_ground_truth'] = len(data['ground_truth'])
        data['scenario_info'] = {
            'description': kalman_data.get('description', ''),
            'num_targets': kalman_data.get('num_targets', 0),
            'duration_seconds': kalman_data.get('duration_seconds', 0),
        }
        data['coordinate_frame'] = 'Piren NED (position sensors only: LiDAR + Radar)'
        
    except ImportError as e:
        # Fallback to raw loading if AutoferryDataLoader not available
        data['error'] = f"AutoferryDataLoader not available: {e}. Install scipy if needed."
        data['fallback_used'] = True
        
        # Legacy raw loading (won't work correctly for tracking!)
        scenario_dir = Path(f"data/sensor_fusion_dataset/{scenario_id}")
        det_file = scenario_dir / f"{scenario_id}_detections.json"
        if det_file.exists():
            with open(det_file, 'r') as f:
                raw_dets = json.load(f)
            # WARNING: This skips coordinate conversion - tracks will fail!
            data['detections'] = raw_dets
            data['total_detections'] = len(raw_dets)
            data['loaded'] = True
            data['warning'] = "Using RAW data without coordinate conversion - tracking will fail!"
            
    except Exception as e:
        data['error'] = str(e)
    
    return data


# ============================================================================
# Session State Initialization
# ============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Workflow state
        'current_step': 0,  # 0=config, 1=fusion, 2=safety, 3=reliability, 4=availability, 5=maintenance, 6=supervisor
        'workflow_started': False,
        'workflow_complete': False,
        
        # Agent results (OUTPUTS - computed by agents, not user inputs)
        'fusion_result': None,  # ML-enhanced Kalman filter output
        'base_kf_result': None,  # Base Kalman filter result (for comparison)
        'safety_result': None,
        'reliability_result': None,
        'availability_result': None,
        'maintenance_result': None,
        'supervisor_result': None,
        
        # AutoFerry data for sensor fusion
        'autoferry_data': None,
        
        # History for charts
        'safety_history': [],
        'reliability_history': [],
        'availability_history': [],
        'maintenance_history': [],
        
        # Simulation state
        'simulation_step': 0,
        'operating_hours': 0,
        
        # Loaded data (from datasets)
        'propulsion_data': None,
        'navigation_data': None,
        
        # User inputs (actual configuration)
        'selected_scenario': 'scenario16',
        'selected_timestep': 0,
        'data_loaded': False,
        'compare_kf_modes': False,  # Toggle to compare ML vs Base KF
        'kf_source_for_rlpsf': 'ML KF',  # Which KF result to feed to RL-PSF: 'ML KF' or 'Base KF'

        # LLM provider configuration (sidebar)
        'llm_provider': 'ntnu',
        'llm_model': PROVIDER_CONFIGS['ntnu']['models'][0],
        'llm_api_key': '',
        'llm_api_base': PROVIDER_CONFIGS['ntnu']['api_base'],
        '_llm_prev_provider': 'ntnu',
        
        # LLM Expert chat histories (per-stage)
        'fusion_chat_history': [],
        'safety_chat_history': [],
        'reliability_chat_history': [],
        'availability_chat_history': [],
        'maintenance_chat_history': [],
        'supervisor_chat_history': [],
        
        # LLM Expert analysis results (cached per stage)
        'fusion_expert_analysis': None,
        'safety_expert_analysis': None,
        'reliability_expert_analysis': None,
        'availability_expert_analysis': None,
        'maintenance_expert_analysis': None,
        'supervisor_expert_analysis': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================================
# Agent Initialization (Cached)
# ============================================================================
@st.cache_resource
def load_agents():
    """Load and cache RAMS agents including ML-enhanced sensor fusion."""
    try:
        from rams_agents import (
            SafetyAgent,
            ReliabilityAgent,
            AvailabilityAgent,
            MaintainabilityAgent,
        )
        
        # Load ML-enhanced sensor fusion agent
        try:
            from agents.sensor_fusion_agent import SensorFusionAgentKalman
            sensor_fusion = SensorFusionAgentKalman(use_ml=True)
            ml_fusion_available = sensor_fusion.use_ml
        except Exception as e:
            sensor_fusion = None
            ml_fusion_available = False
        
        # Check LLM supervisor availability (agent created dynamically from sidebar config)
        try:
            from rams_agents import LLMSupervisorAgent  # noqa: F401
            llm_available = True
        except ImportError:
            llm_available = False
        
        agents = {
            'sensor_fusion': sensor_fusion,  # ML-enhanced Kalman filter
            'safety': SafetyAgent(use_rl_policy=True),
            'reliability': ReliabilityAgent(),
            'availability': AvailabilityAgent(),
            'maintainability': MaintainabilityAgent(),
        }
        # NOTE: LLM supervisor is NOT cached here — it is created dynamically
        # from sidebar config via _get_or_create_llm_supervisor().
        return agents, llm_available, ml_fusion_available
    except ImportError as e:
        st.error(f"Failed to import agents: {e}")
        return None, False, False


def _get_or_create_llm_supervisor():
    """
    Return an LLMSupervisorAgent built from the current sidebar config.
    The instance is cached in session_state and recreated only when the
    provider / model / key / base-URL actually change.
    """
    from rams_agents.llm_supervisor_agent import LLMConfig, LLMSupervisorAgent

    provider  = st.session_state.get('llm_provider',  'ntnu')
    model     = st.session_state.get('llm_model',     PROVIDER_CONFIGS['ntnu']['models'][0])
    api_key   = st.session_state.get('llm_api_key',   '')
    api_base  = st.session_state.get('llm_api_base',  PROVIDER_CONFIGS['ntnu']['api_base'])

    config_sig = (provider, model, api_key, api_base)
    if (st.session_state.get('_llm_supervisor') is None
            or st.session_state.get('_llm_config_sig') != config_sig):
        llm_config = LLMConfig(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        st.session_state['_llm_supervisor'] = LLMSupervisorAgent(llm_config=llm_config)
        st.session_state['_llm_config_sig'] = config_sig

    return st.session_state['_llm_supervisor']


# ============================================================================
# Data Loading Functions (Load REAL data, not user-configured)
# ============================================================================
def load_scenario_data(data_sources: Dict, 
                       scenario_id: str,
                       timestep: int) -> Dict[str, Any]:
    """
    Load ACTUAL data from datasets - not user-configured values.
    
    The agents will analyze this data and produce their outputs.
    """
    propulsion_loader = data_sources['propulsion']
    navigation_loader = data_sources['navigation']
    
    # Get propulsion data at specified timestep
    if propulsion_loader.data is not None:
        # Get a specific row from the dataset
        idx = timestep % len(propulsion_loader.data)
        row = propulsion_loader.data.iloc[idx]
        
        propulsion_data = {
            'timestamp': timestep,
            'lever_position': row.get('lever_position', 5.0),
            'ship_speed': row.get('ship_speed', 12.0),
            'gt_shaft_torque': row.get('gt_shaft_torque', 200),
            'gt_speed': row.get('gt_speed', 2800),
            'gas_gen_speed': row.get('gas_gen_speed', 8500),
            'prop_torque': row.get('prop_torque', 180),
            'compressor_decay': row.get('compressor_decay', 0.98),
            'turbine_decay': row.get('turbine_decay', 0.99),
            'fuel_flow': row.get('fuel_flow', 0.08),
            't_compressor_in': row.get('t_compressor_in', 288),
            't_compressor_out': row.get('t_compressor_out', 550),
            'p_compressor_in': row.get('p_compressor_in', 101),
            'p_compressor_out': row.get('p_compressor_out', 1400),
            't_turbine_in': row.get('t_turbine_in', 1200),
            't_turbine_out': row.get('t_turbine_out', 800),
        }
    else:
        # Fallback synthetic data
        propulsion_data = {
            'timestamp': timestep,
            'compressor_decay': 0.95 - (timestep * 0.001),
            'turbine_decay': 0.97 - (timestep * 0.0005),
            'lever_position': 5.0,
            'ship_speed': 12.0,
        }
    
    # Get navigation data from selected scenario
    try:
        nav_iter = navigation_loader.iterate_timesteps(scenario_id, time_step=1.0)
        # Skip to desired timestep
        navigation_data = None
        for i, data in enumerate(nav_iter):
            if i >= timestep:
                navigation_data = data
                break
        
        if navigation_data is None:
            # Restart from beginning
            nav_iter = navigation_loader.iterate_timesteps(scenario_id, time_step=1.0)
            navigation_data = next(nav_iter)
    except Exception:
        # Fallback navigation data
        navigation_data = {
            'timestamp': timestep,
            'ownship_position': [63.4305, 10.3951],
            'ownship_velocity': [5, 0],
            'targets': [],
        }
    
    return {
        'propulsion': propulsion_data,
        'navigation': navigation_data,
        'scenario_id': scenario_id,
        'timestep': timestep
    }


# ============================================================================
# Agent Execution Functions (Agents ANALYZE data and produce OUTPUTS)
# ============================================================================
def run_safety_agent(agents: Dict, scenario_data: Dict, kf_result: Optional[Dict] = None, kf_source: str = 'ML KF') -> Dict[str, Any]:
    """
    Execute Safety Agent on confirmed KF tracks.
    
    The agent analyzes KF-filtered target tracks (not raw detections) and OUTPUTS risk assessment.
    Includes PSF safety guarantee details and RL validation metrics.
    
    Args:
        agents: Dictionary of initialized agents
        scenario_data: Raw scenario data (fallback)
        kf_result: Result from Kalman Filter (ML or Base), contains confirmed_tracks
        kf_source: Which KF produced the result ('ML KF' or 'Base KF')
    """
    safety_agent = agents['safety']
    
    # Extract confirmed tracks from KF result if available
    if kf_result and kf_result.get('status') == 'complete':
        confirmed_tracks = kf_result.get('confirmed_tracks', [])
        # Convert KF tracks to target format for safety agent
        targets = []
        for track in confirmed_tracks:
            state = track.get('state', [0, 0, 0, 0])
            targets.append({
                'id': track.get('track_id', 0),
                'position': [state[0], state[1]],  # north, east
                'velocity': [state[2], state[3]],  # vel_north, vel_east
                'source': f'{kf_source}',
                'maneuver_probability': track.get('maneuver_probability', 0.0),
                'measurement_noise_R': track.get('measurement_noise_R', None),
                'process_noise_Q': track.get('process_noise_Q', None),
            })
    else:
        # Fallback to raw navigation data
        nav_data = scenario_data.get('navigation', {})
        targets = nav_data.get('targets', [])
        kf_source = 'raw (no KF)'
    
    nav_data = scenario_data.get('navigation', {})
    env = {
        'navigation': {'targets': targets, **nav_data},
        'ownship_position': nav_data.get('ownship_position', [0, 0]),
        'ownship_velocity': nav_data.get('ownship_velocity', [5, 0]),
        'targets': targets,
        'timestamp': scenario_data.get('timestep', 0),
        'kf_source': kf_source,
    }
    
    # Agent analyzes the data and produces output
    result = safety_agent.run_cycle(env)
    
    # Extract agent's analysis (OUTPUTS)
    n_targets = len(targets)  # Use KF confirmed tracks count
    
    # Agent determines risk level based on target analysis
    actions = result.get('actions', {})
    maneuvers = actions.get('recommended_maneuvers', [])
    
    # Determine risk from agent's assessment
    if maneuvers and any(m.get('action') in ['EMERGENCY_STOP', 'HARD_TURN'] for m in maneuvers if isinstance(m, dict)):
        risk_level = 'CRITICAL'
        safety_index = 15
    elif n_targets > 2:
        risk_level = 'HIGH'
        safety_index = 35
    elif n_targets > 0:
        risk_level = 'MODERATE'
        safety_index = 65
    else:
        risk_level = 'LOW'
        safety_index = 95
    
    # ─────────────────────────────────────────────────────────────────────
    # Extract PSF Safety Guarantee Details
    # ─────────────────────────────────────────────────────────────────────
    rl_psf_details = {
        'rl_available': getattr(safety_agent, 'uses_rl_policy', False),
        'training_domain': 'Synthetic COLREGS encounters (custom Gym env)',
        'current_domain': 'AutoFerry multi-sensor data',
        'domain_mismatch': True,  # Always true - RL not trained on AutoFerry
    }
    
    # Extract from maneuvers if RL+PSF was used
    rl_maneuver = next((m for m in maneuvers if isinstance(m, dict) and m.get('source') == 'RL_PSF'), None)
    
    if rl_maneuver:
        rl_psf_details['rl_action'] = {
            'speed_ratio': rl_maneuver.get('speed_ratio', 0.8),
            'steering': rl_maneuver.get('steering', 0.0),
            'confidence': rl_maneuver.get('rl_confidence', 0.5),
        }
        rl_psf_details['psf_filter'] = {
            'intervention_type': rl_maneuver.get('psf_intervention', 'NONE'),
            'barrier_value': rl_maneuver.get('barrier_value', 1.0),
            'was_modified': rl_maneuver.get('psf_intervention', 'none').upper() not in ['NONE', ''],
        }
    else:
        # Fallback values when RL not used or no RL maneuver found
        rl_psf_details['rl_action'] = {'speed_ratio': 0.8, 'steering': 0.0, 'confidence': 0.6}
        rl_psf_details['psf_filter'] = {'intervention_type': 'NONE', 'barrier_value': 1.0, 'was_modified': False}
    
    # Get PSF statistics if available
    if hasattr(safety_agent, 'get_rl_psf_stats'):
        stats = safety_agent.get_rl_psf_stats()
        rl_psf_details['psf_stats'] = {
            'total_calls': stats.get('total_calls', 1),
            'interventions': stats.get('interventions', 0),
            'intervention_rate_pct': stats.get('intervention_rate', 0) * 100,
            'rl_used': stats.get('rl_used', 0),
        }
    else:
        rl_psf_details['psf_stats'] = {'total_calls': 1, 'interventions': 0, 'intervention_rate_pct': 0, 'rl_used': 0}
    
    # ─────────────────────────────────────────────────────────────────────
    # RL Validation Metrics (for LLM Expert)
    # ─────────────────────────────────────────────────────────────────────
    intervention_rate = rl_psf_details['psf_stats']['intervention_rate_pct']
    confidence = rl_psf_details['rl_action']['confidence']
    
    # Determine validation verdict based on PSF intervention rate and confidence
    if intervention_rate > 30 or confidence < 0.3:
        validation_verdict = 'CRITICAL'
        validation_reason = 'High PSF intervention rate or low confidence indicates significant domain mismatch'
    elif intervention_rate > 10 or confidence < 0.5:
        validation_verdict = 'WARNING'
        validation_reason = 'Moderate domain gap detected, PSF compensating for policy uncertainty'
    elif intervention_rate > 5:
        validation_verdict = 'ACCEPTABLE'
        validation_reason = 'Minor PSF adjustments, policy generalizing reasonably well'
    else:
        validation_verdict = 'VALIDATED'
        validation_reason = 'Low intervention rate and high confidence suggest good generalization'
    
    rl_validation_metrics = {
        'intervention_rate_pct': intervention_rate,
        'action_confidence': confidence,
        'validation_verdict': validation_verdict,
        'validation_reason': validation_reason,
        'ks_test_note': 'KS-test requires multiple episodes; PSF intervention rate used as proxy',
        'zero_shot_proxy': 'PSF intervention rate approximates zero-shot policy performance',
    }
    
    return {
        'status': 'complete',
        'risk_level': risk_level,
        'safety_index': safety_index,
        'active_tracks': n_targets,
        'kf_source': kf_source,  # Which KF provided the tracks
        'rl_psf_active': rl_psf_details['rl_available'],
        'maneuvers': maneuvers,
        'beliefs': result.get('beliefs', []),
        'raw_result': result,
        'input_data': {'targets': targets, 'kf_source': kf_source},
        'rl_psf_details': rl_psf_details,           # PSF details for visualization
        'rl_validation_metrics': rl_validation_metrics,  # For LLM expert validation
    }


def run_reliability_agent(agents: Dict, scenario_data: Dict) -> Dict[str, Any]:
    """
    Execute Reliability Agent on LOADED propulsion data.
    
    The agent analyzes sensor readings and OUTPUTS health assessment.
    User does NOT configure health - the agent measures it from data.
    """
    reliability_agent = agents['reliability']
    
    prop_data = scenario_data['propulsion']
    env = {
        'propulsion': prop_data,
        'timestamp': scenario_data.get('timestep', 0)
    }
    
    # Agent analyzes the data and produces output
    result = reliability_agent.run_cycle(env)
    
    # Extract agent's analysis (OUTPUTS from actual data)
    compressor_decay = prop_data.get('compressor_decay', 0.98)
    turbine_decay = prop_data.get('turbine_decay', 0.99)
    
    # Agent computes health from sensor data (OUTPUT)
    compressor_health = compressor_decay * 100
    turbine_health = turbine_decay * 100
    
    # Agent estimates RUL based on degradation analysis (OUTPUT)
    degradation_rate = (1 - compressor_decay) * 0.001
    if degradation_rate > 0:
        rul_hours = max(0, (compressor_decay - 0.85) / degradation_rate)
    else:
        rul_hours = 5000
    rul_hours = min(5000, rul_hours)
    
    # Agent determines health status (OUTPUT)
    if compressor_health > 85:
        overall_health = 'GOOD'
    elif compressor_health > 60:
        overall_health = 'DEGRADED'
    else:
        overall_health = 'CRITICAL'
    
    # Agent generates alerts based on analysis (OUTPUT)
    critical_alerts = []
    warnings = []
    
    if compressor_health < 50:
        critical_alerts.append(f'Compressor health critical: {compressor_health:.1f}%')
    elif compressor_health < 75:
        warnings.append(f'Compressor degradation detected: {compressor_health:.1f}%')
    
    if turbine_health < 60:
        critical_alerts.append(f'Turbine health low: {turbine_health:.1f}%')
    elif turbine_health < 80:
        warnings.append(f'Turbine showing wear: {turbine_health:.1f}%')
    
    return {
        'status': 'complete',
        'overall_health': overall_health,  # OUTPUT: Agent assessment
        'components': {
            'compressor': {'health': compressor_health, 'rul_hours': rul_hours},
            'turbine': {'health': turbine_health, 'rul_hours': rul_hours * 1.2}
        },
        'critical_alerts': critical_alerts,  # OUTPUT: Agent detections
        'warnings': warnings,  # OUTPUT: Agent warnings
        'beliefs': result.get('beliefs', []),
        'raw_result': result,
        'input_data': prop_data  # Show what data was analyzed
    }


def run_availability_agent(agents: Dict, scenario_data: Dict) -> Dict[str, Any]:
    """
    Execute Availability Agent on LOADED data.
    
    The agent monitors sensors and OUTPUTS availability assessment.
    Sensor status comes from agent analysis, not user configuration.
    """
    availability_agent = agents['availability']
    
    prop_data = scenario_data['propulsion']
    
    env = {
        'propulsion': prop_data,
        'timestamp': scenario_data.get('timestep', 0)
    }
    
    # Agent analyzes the data and produces output
    result = availability_agent.run_cycle(env)
    
    # Agent determines sensor status from its monitoring (OUTPUT)
    # In real system, this comes from sensor health checks
    anomaly_score = result.get('actions', {}).get('anomaly_score', 0.1)
    
    # Simulate sensor health based on propulsion data quality
    compressor_decay = prop_data.get('compressor_decay', 0.98)
    
    # Agent's sensor assessment (OUTPUT - not user input)
    if anomaly_score < 0.2:
        sensors = {'radar': 'HEALTHY', 'lidar': 'HEALTHY', 'camera': 'HEALTHY', 'gps': 'HEALTHY'}
    elif anomaly_score < 0.5:
        sensors = {'radar': 'HEALTHY', 'lidar': 'DEGRADED', 'camera': 'HEALTHY', 'gps': 'HEALTHY'}
    elif anomaly_score < 0.8:
        sensors = {'radar': 'HEALTHY', 'lidar': 'DEGRADED', 'camera': 'DEGRADED', 'gps': 'HEALTHY'}
    else:
        sensors = {'radar': 'DEGRADED', 'lidar': 'FAILED', 'camera': 'DEGRADED', 'gps': 'HEALTHY'}
    
    # Agent calculates capability (OUTPUT)
    healthy_sensors = sum(1 for s in sensors.values() if s == 'HEALTHY')
    total_sensors = len(sensors)
    sensor_capability = (healthy_sensors / total_sensors) * 100
    propulsion_capability = compressor_decay * 100
    overall_capability = (sensor_capability + propulsion_capability) / 2
    
    # Agent determines operating mode (OUTPUT)
    if overall_capability > 90:
        mode = 'FULL_POWER'
    elif overall_capability > 70:
        mode = 'DEGRADED'
    elif overall_capability > 50:
        mode = 'REDUCED'
    else:
        mode = 'EMERGENCY'
    
    # Agent detects anomalies (OUTPUT)
    anomalies = []
    for sensor, status in sensors.items():
        if status == 'FAILED':
            anomalies.append(f'{sensor.upper()} sensor failed')
        elif status == 'DEGRADED':
            anomalies.append(f'{sensor.upper()} showing degraded readings')
    
    return {
        'status': 'complete',
        'mode': mode,  # OUTPUT: Agent determination
        'capability_pct': overall_capability,  # OUTPUT: Computed
        'anomaly_score': anomaly_score,  # OUTPUT: Agent detection
        'sensors': sensors,  # OUTPUT: Agent assessment
        'anomalies': anomalies,  # OUTPUT: Agent detections
        'beliefs': result.get('beliefs', []),
        'raw_result': result,
        'input_data': prop_data
    }


def run_maintainability_agent(agents: Dict, scenario_data: Dict, reliability_result: Dict) -> Dict[str, Any]:
    """
    Execute Maintainability Agent based on reliability findings.
    
    Maintenance needs are OUTPUTS based on reliability agent's analysis,
    not user-configured backlog.
    """
    maintainability_agent = agents['maintainability']
    
    env = {
        'timestamp': scenario_data.get('timestep', 0),
        'operating_hours': st.session_state.operating_hours
    }
    
    # Agent analyzes and produces output
    result = maintainability_agent.run_cycle(env)
    
    # Agent determines maintenance needs from reliability data (OUTPUT)
    components = reliability_result.get('components', {})
    critical_alerts = reliability_result.get('critical_alerts', [])
    warnings = reliability_result.get('warnings', [])
    
    # Agent generates maintenance tasks (OUTPUT - not user input)
    pending_tasks = 0
    recommendations = []
    
    for comp_name, comp_data in components.items():
        health = comp_data.get('health', 100)
        rul = comp_data.get('rul_hours', 5000)
        
        if health < 50:
            pending_tasks += 1
            recommendations.append(f'CRITICAL: {comp_name.title()} requires immediate maintenance')
        elif health < 70:
            pending_tasks += 1
            recommendations.append(f'HIGH: {comp_name.title()} maintenance recommended within 100 hours')
        elif rul < 200:
            pending_tasks += 1
            recommendations.append(f'MEDIUM: Schedule {comp_name.title()} inspection (RUL: {rul:.0f}h)')
    
    # Add any from reliability alerts
    for alert in critical_alerts:
        pending_tasks += 1
        recommendations.append(f'CRITICAL: {alert}')
    
    if pending_tasks == 0:
        recommendations.append('LOW: All systems nominal - routine inspection in 500 hours')
    
    # Agent calculates cost forecast (OUTPUT)
    pending_cost = pending_tasks * 8000  # Estimated per-task cost
    pending_downtime = pending_tasks * 6  # Hours per task
    
    # Maintainability metric (OUTPUT)
    maintainability_metric = max(0.3, 1.0 - (pending_tasks * 0.15))
    
    return {
        'status': 'complete',
        'pending_tasks_count': pending_tasks,  # OUTPUT: Agent determines
        'maintainability_metric': maintainability_metric,  # OUTPUT: Computed
        'cost_forecast': {
            'pending_cost': pending_cost,
            'pending_downtime_hours': pending_downtime
        },
        'recommendations': recommendations,  # OUTPUT: Agent recommendations
        'beliefs': result.get('beliefs', []),
        'raw_result': result
    }


def run_llm_supervisor(safety_result: Dict,
                        reliability_result: Dict,
                        availability_result: Dict,
                        maintenance_result: Dict) -> Dict[str, Any]:
    """Execute LLM Supervisor and get final recommendation."""
    supervisor = _get_or_create_llm_supervisor()
    
    # Update supervisor with all agent reports
    supervisor.update_rams_state(
        safety_report={
            'risk_level': safety_result.get('risk_level', 'LOW'),
            'safety_index': safety_result.get('safety_index', 90),
            'active_tracks': safety_result.get('active_tracks', 0),
            'rl_psf_active': safety_result.get('rl_psf_active', False),
            'maneuvers': safety_result.get('maneuvers', [])
        },
        reliability_report={
            'overall_health': reliability_result.get('overall_health', 'GOOD'),
            'components': reliability_result.get('components', {}),
            'critical_alerts': reliability_result.get('critical_alerts', []),
            'warnings': reliability_result.get('warnings', [])
        },
        availability_report={
            'mode': availability_result.get('mode', 'FULL_POWER'),
            'capability_pct': availability_result.get('capability_pct', 100),
            'anomaly_score': availability_result.get('anomaly_score', 0),
            'sensors': availability_result.get('sensors', {}),
            'anomalies': availability_result.get('anomalies', [])
        },
        maintenance_report={
            'pending_tasks_count': maintenance_result.get('pending_tasks_count', 0),
            'maintainability_metric': maintenance_result.get('maintainability_metric', 1.0),
            'cost_forecast': maintenance_result.get('cost_forecast', {}),
            'recommendations': maintenance_result.get('recommendations', [])
        }
    )
    
    # Get LLM reasoning
    result = supervisor.act()
    
    return result


# ============================================================================
# ML-Enhanced Sensor Fusion Functions
# ============================================================================
def run_ml_sensor_fusion(agents: Dict, autoferry_data: Dict, timestep: int) -> Dict[str, Any]:
    """
    Run ML-enhanced Kalman filter sensor fusion on AutoFerry dataset.
    
    Uses trained models for:
    - Maneuver detection → adaptive process noise Q
    - Sensor reliability → adaptive measurement noise R
    
    Returns tracks with ML-enhanced state estimates.
    """
    sensor_fusion = agents.get('sensor_fusion')
    
    if sensor_fusion is None:
        return {
            'status': 'error',
            'error': 'Sensor fusion agent not available. Check if agents/sensor_fusion_agent.py is present and ML models are in models/ directory.',
            'tracks': [],
            'ml_enhanced': False
        }
    
    try:
        import time as time_module
        start_time = time_module.time()
        
        # Apply gate threshold from session state
        gate_threshold = st.session_state.get('gate_threshold', 16.0)
        if hasattr(sensor_fusion, 'set_gate_threshold'):
            sensor_fusion.set_gate_threshold(gate_threshold)
        else:
            # Direct attribute update for backward compatibility
            sensor_fusion.GATE_THRESHOLD = gate_threshold
        
        # Get detections up to current timestep
        all_detections = autoferry_data.get('detections', [])
        ground_truth = autoferry_data.get('ground_truth', [])
        
        # Filter detections by time
        detections_at_time = [d for d in all_detections if d.get('time', 0) <= timestep]
        
        # === DIAGNOSTIC: Coordinate Transformation Validation ===
        coord_warnings = []
        if detections_at_time:
            # Check for NaN/Inf in positions
            nan_count = sum(1 for d in detections_at_time 
                          if any(np.isnan(p) or np.isinf(p) for p in d.get('position', [0,0,0])[:2]))
            if nan_count > 0:
                coord_warnings.append(f"WARNING: {nan_count} detections have NaN/Inf coordinates")
            
            # Check for positions outside reasonable bounds (Piren NED frame ~10km)
            out_of_bounds = sum(1 for d in detections_at_time
                              if any(abs(p) > 10000 for p in d.get('position', [0,0,0])[:2]))
            if out_of_bounds > 0:
                coord_warnings.append(f"WARNING: {out_of_bounds} detections outside 10km bounds - check WGS-84/ENU projection")
        
        # === DIAGNOSTIC: Timestamp Synchronization Check ===
        timing_warnings = []
        if len(detections_at_time) > 10:
            times = sorted(set(d.get('time', 0) for d in detections_at_time))
            if len(times) > 1:
                time_gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
                max_gap = max(time_gaps) if time_gaps else 0
                min_gap = min(time_gaps) if time_gaps else 0
                if max_gap > 2.0:
                    timing_warnings.append(f"WARNING: Max time gap {max_gap:.2f}s - possible data dropout")
                if min_gap < 0.001:
                    timing_warnings.append(f"WARNING: Min time gap {min_gap*1000:.0f}μs - check PTP/IEEE 1588 sync")
        
        # Build fusion input
        fusion_input = {
            'detections': detections_at_time,
            'ground_truth': ground_truth,
            'current_time': timestep
        }
        
        # Run sensor fusion processing
        result = sensor_fusion.process(fusion_input)
        
        processing_time = (time_module.time() - start_time) * 1000
        
        # Extract tracks with ML fields
        tracks = result.get('tracks', [])
        
        # Get ML status
        ml_status = result.get('ml_status', {})
        
        # Build sensor statistics from tracks if not in result
        sensor_stats = ml_status.get('sensor_stats', {})
        if not sensor_stats:
            # Compute from detections
            sensor_stats = {}
            for det in detections_at_time:
                sid = str(det.get('sensorID', 0))
                if sid not in sensor_stats:
                    sensor_stats[sid] = {
                        'sensor_name': {1: 'LiDAR', 2: 'Radar', 3: 'IR Camera', 4: 'EO Camera'}.get(int(sid), f'Sensor {sid}'),
                        'count': 0,
                        'mean_error_m': 5.0,
                        'reliability_score': 0.7
                    }
                sensor_stats[sid]['count'] += 1
        
        return {
            'status': 'success',
            'tracks': tracks,
            'confirmed_tracks': len([t for t in tracks if t.get('status') == 'confirmed']),
            'tentative_tracks': len([t for t in tracks if t.get('status') == 'tentative']),
            'total_detections': len(detections_at_time),
            'all_detections': len(all_detections),
            'ml_enhanced': result.get('ml_enhanced', sensor_fusion.use_ml),
            'ml_status': {
                'sensor_stats': sensor_stats,
                'maneuver_detector_active': ml_status.get('maneuver_detector_active', False),
                'reliability_estimator_active': ml_status.get('reliability_estimator_active', False),
                'diagnostics': ml_status.get('diagnostics', {}),
                'gate_threshold_applied': gate_threshold,
            },
            'diagnostics': {
                'coordinate_warnings': coord_warnings,
                'timing_warnings': timing_warnings,
                'ml_diagnostics': ml_status.get('diagnostics', {}),
            },
            'processing_time_ms': processing_time,
            'raw_result': result
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'tracks': [],
            'ml_enhanced': False,
            'total_detections': 0
        }


def run_base_kalman_filter(autoferry_data: Dict, timestep: int) -> Dict[str, Any]:
    """
    Run BASE Kalman filter (no ML enhancement) for comparison.
    Uses fixed process noise Q and measurement noise R.
    """
    try:
        from agents.sensor_fusion_agent import SensorFusionAgentKalman
        import time as time_module
        
        start_time = time_module.time()
        
        # Create agent with ML disabled
        base_agent = SensorFusionAgentKalman(use_ml=False)
        
        # Apply same gate threshold as ML version
        gate_threshold = st.session_state.get('gate_threshold', 16.0)
        base_agent.GATE_THRESHOLD = gate_threshold
        
        # Get detections up to current timestep
        all_detections = autoferry_data.get('detections', [])
        ground_truth = autoferry_data.get('ground_truth', [])
        detections_at_time = [d for d in all_detections if d.get('time', 0) <= timestep]
        
        # Build fusion input
        fusion_input = {
            'detections': detections_at_time,
            'ground_truth': ground_truth,
            'current_time': timestep
        }
        
        # Run sensor fusion
        result = base_agent.process(fusion_input)
        processing_time = (time_module.time() - start_time) * 1000
        
        tracks = result.get('tracks', [])
        
        return {
            'status': 'success',
            'tracks': tracks,
            'confirmed_tracks': len([t for t in tracks if t.get('status') == 'confirmed']),
            'tentative_tracks': len([t for t in tracks if t.get('status') == 'tentative']),
            'total_detections': len(detections_at_time),
            'ml_enhanced': False,
            'processing_time_ms': processing_time,
            'kalman_params': {
                'process_noise_Q': base_agent.kalman.process_noise,
                'measurement_noise_R': base_agent.kalman.measurement_noise,
                'gate_threshold': gate_threshold
            },
            'quality_metrics': result.get('quality_metrics', {})
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'tracks': [],
            'ml_enhanced': False,
            'total_detections': 0
        }


def compute_kf_comparison_metrics(ml_result: Dict, base_result: Dict) -> Dict:
    """
    Compute comparison metrics between ML-enhanced and base Kalman filter.
    """
    ml_tracks = ml_result.get('tracks', [])
    base_tracks = base_result.get('tracks', [])
    
    ml_confirmed = [t for t in ml_tracks if t.get('status') == 'confirmed']
    base_confirmed = [t for t in base_tracks if t.get('status') == 'confirmed']
    
    # Compute average ML parameters from tracks
    ml_avg_maneuver_prob = 0.0
    ml_avg_Q = 0.0
    ml_avg_R = 0.0
    ml_avg_uncertainty = 0.0
    
    if ml_confirmed:
        ml_avg_maneuver_prob = np.mean([t.get('maneuver_probability', 0) for t in ml_confirmed])
        ml_avg_Q = np.mean([t.get('process_noise_Q', t.get('current_process_noise', 0.5)) for t in ml_confirmed])
        ml_avg_R = np.mean([t.get('measurement_noise_R', t.get('last_measurement_noise', 5.0)) for t in ml_confirmed])
        ml_avg_uncertainty = np.mean([t.get('position_uncertainty_m', 0) for t in ml_confirmed])
    
    base_avg_uncertainty = 0.0
    if base_confirmed:
        base_avg_uncertainty = np.mean([t.get('position_uncertainty_m', 0) for t in base_confirmed])
    
    # Base KF uses fixed parameters
    base_params = base_result.get('kalman_params', {})
    base_Q = base_params.get('process_noise_Q', 0.5)
    base_R = base_params.get('measurement_noise_R', 2.0)
    
    return {
        'ml': {
            'confirmed_tracks': len(ml_confirmed),
            'total_tracks': len(ml_tracks),
            'avg_maneuver_probability': ml_avg_maneuver_prob,
            'avg_process_noise_Q': ml_avg_Q,
            'avg_measurement_noise_R': ml_avg_R,
            'avg_position_uncertainty_m': ml_avg_uncertainty,
            'processing_time_ms': ml_result.get('processing_time_ms', 0),
            'adaptive_params': True
        },
        'base': {
            'confirmed_tracks': len(base_confirmed),
            'total_tracks': len(base_tracks),
            'avg_maneuver_probability': 0.0,  # Not computed
            'avg_process_noise_Q': base_Q,
            'avg_measurement_noise_R': base_R,
            'avg_position_uncertainty_m': base_avg_uncertainty,
            'processing_time_ms': base_result.get('processing_time_ms', 0),
            'adaptive_params': False
        },
        'comparison': {
            'track_difference': len(ml_confirmed) - len(base_confirmed),
            'uncertainty_improvement_pct': (
                ((base_avg_uncertainty - ml_avg_uncertainty) / base_avg_uncertainty * 100) 
                if base_avg_uncertainty > 0 else 0
            ),
            'Q_adaptation_range': f"{ml_avg_Q:.2f}" if ml_avg_Q > 0 else "N/A",
            'R_adaptation_range': f"{ml_avg_R:.1f}" if ml_avg_R > 0 else "N/A",
        }
    }


# ============================================================================
# ML Sensor Fusion Visualization Functions
# ============================================================================
def plot_ml_tracking_scenario(tracks: List[Dict], detections: List[Dict], 
                               ground_truth: List[Dict]) -> go.Figure:
    """
    Plot ML-enhanced tracking results with detections and ground truth.
    Shows sensor detections (colored), ground truth (dashed), and ML tracks (solid).
    """
    fig = go.Figure()
    
    # Sensor colors
    sensor_colors = {1: '#2ecc71', 2: '#3498db', 3: '#e74c3c', 4: '#f39c12'}
    sensor_names = {1: 'LiDAR', 2: 'Radar', 3: 'IR Camera', 4: 'EO Camera'}
    
    # Plot detections by sensor (faint markers)
    for sensor_id, color in sensor_colors.items():
        sensor_dets = [d for d in detections if d.get('sensorID') == sensor_id]
        if sensor_dets:
            easts = [d['position'][1] if len(d.get('position', [])) > 1 else 0 for d in sensor_dets]
            norths = [d['position'][0] if len(d.get('position', [])) > 0 else 0 for d in sensor_dets]
            fig.add_trace(go.Scatter(
                x=easts, y=norths,
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.3),
                name=f"{sensor_names[sensor_id]} detections",
                legendgroup=f"sensor_{sensor_id}"
            ))
    
    # Plot ground truth (dashed lines)
    # ground_truth may be a list of lists (one per timestep) or a flat list of dicts
    gt_by_target = {}
    for gt_entry in ground_truth:
        items = gt_entry if isinstance(gt_entry, list) else [gt_entry]
        for gt in items:
            if not isinstance(gt, dict):
                continue
            tid = gt.get('targetID', 0)
            if tid not in gt_by_target:
                gt_by_target[tid] = []
            pos = gt.get('position', [0, 0])
            gt_by_target[tid].append(pos)
    
    gt_colors = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#16a085']
    for i, (tid, positions) in enumerate(gt_by_target.items()):
        if len(positions) < 2:
            continue
        norths = [p[0] for p in positions]
        easts = [p[1] if len(p) > 1 else 0 for p in positions]
        fig.add_trace(go.Scatter(
            x=easts, y=norths,
            mode='lines',
            line=dict(color=gt_colors[i % len(gt_colors)], dash='dash', width=2),
            name=f"Ground Truth T{tid}",
            legendgroup=f"gt_{tid}"
        ))
    
    # Plot confirmed tracks (solid, thick lines)
    track_colors = px.colors.qualitative.Set2
    for i, track in enumerate(tracks):
        if track.get('status') != 'confirmed':
            continue
        
        history = track.get('history', [])
        if len(history) < 2:
            continue
        
        color = track_colors[i % len(track_colors)]
        norths = [h[0] if isinstance(h, (list, tuple)) else 0 for h in history]
        easts = [h[1] if isinstance(h, (list, tuple)) and len(h) > 1 else 0 for h in history]
        
        # Track trajectory
        fig.add_trace(go.Scatter(
            x=easts, y=norths,
            mode='lines',
            line=dict(color=color, width=3),
            name=track.get('target_name', f"Track {track.get('track_id', i)}"),
            legendgroup=f"track_{track.get('track_id', i)}"
        ))
        
        # Current position marker
        if easts and norths:
            fig.add_trace(go.Scatter(
                x=[easts[-1]], y=[norths[-1]],
                mode='markers+text',
                marker=dict(size=12, color=color, symbol='circle'),
                text=[f"T{track.get('track_id', i)}"],
                textposition='top right',
                showlegend=False
            ))
            
            # Velocity vector
            vel = track.get('velocity', [0, 0])
            if isinstance(vel, (list, tuple)) and len(vel) >= 2:
                scale = 5
                fig.add_trace(go.Scatter(
                    x=[easts[-1], easts[-1] + vel[1]*scale],
                    y=[norths[-1], norths[-1] + vel[0]*scale],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                ))
    
    # Observer position (origin)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=15, color='black', symbol='triangle-up'),
        text=['Ownship'],
        textposition='bottom center',
        name='Ownship'
    ))
    
    fig.update_layout(
        title="ML-Enhanced Kalman Filter Tracking",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        height=500
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def plot_ml_noise_adaptation(tracks: List[Dict]) -> go.Figure:
    """
    Plot adaptive process noise (Q) and measurement noise (R) per track.
    Shows how ML adjusts Kalman filter parameters based on maneuver detection.
    """
    confirmed = [t for t in tracks if t.get('status') == 'confirmed']
    
    if not confirmed:
        fig = go.Figure()
        fig.add_annotation(text="No confirmed tracks", x=0.5, y=0.5, 
                          xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig.update_layout(height=350)
        return fig
    
    # Create subplots for Q and R
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Process Noise Q (Maneuver-Adaptive)", "Measurement Noise R (Sensor-Specific)")
    )
    
    track_names = [t.get('target_name', f"T{t.get('track_id', i)}") for i, t in enumerate(confirmed)]
    
    # Process noise Q - higher when maneuvering
    q_values = [t.get('process_noise_Q', t.get('current_process_noise', 0.5)) for t in confirmed]
    maneuver_probs = [t.get('maneuver_probability', 0) for t in confirmed]
    
    # Color by maneuver probability
    colors_q = ['red' if p > 0.5 else 'orange' if p > 0.2 else 'green' for p in maneuver_probs]
    
    fig.add_trace(
        go.Bar(
            x=track_names,
            y=q_values,
            marker_color=colors_q,
            text=[f"P={p:.2f}" for p in maneuver_probs],
            textposition='outside',
            name='Process Noise Q'
        ),
        row=1, col=1
    )
    
    # Measurement noise R - varies by sensor
    r_values = [t.get('measurement_noise_R', t.get('current_measurement_noise', 5.0)) for t in confirmed]
    sensor_ids = [t.get('last_sensor_id', 0) for t in confirmed]
    sensor_names = {1: 'LiDAR', 2: 'Radar', 3: 'IR', 4: 'EO', 0: '?'}
    
    sensor_colors = {1: '#2ecc71', 2: '#3498db', 3: '#e74c3c', 4: '#f39c12', 0: 'gray'}
    colors_r = [sensor_colors.get(s, 'gray') for s in sensor_ids]
    
    fig.add_trace(
        go.Bar(
            x=track_names,
            y=r_values,
            marker_color=colors_r,
            text=[sensor_names.get(s, '?') for s in sensor_ids],
            textposition='outside',
            name='Measurement Noise R'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        title_text="ML-Adaptive Kalman Filter Parameters"
    )
    
    return fig


def plot_maneuver_detection(tracks: List[Dict]) -> go.Figure:
    """
    Plot maneuver probability gauges for each confirmed track.
    Shows output from trained RandomForest maneuver detector.
    """
    confirmed = [t for t in tracks if t.get('status') == 'confirmed']
    
    if not confirmed:
        fig = go.Figure()
        fig.add_annotation(text="No confirmed tracks", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig.update_layout(height=300)
        return fig
    
    # Create gauge for each track
    n_tracks = len(confirmed)
    
    fig = make_subplots(
        rows=1, cols=n_tracks,
        specs=[[{"type": "indicator"}] * n_tracks],
        subplot_titles=[t.get('target_name', f"Track {t.get('track_id', i)}") 
                       for i, t in enumerate(confirmed)]
    )
    
    for i, track in enumerate(confirmed):
        prob = track.get('maneuver_probability', 0)
        prob_pct = prob * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': 'red' if prob > 0.5 else 'orange' if prob > 0.2 else 'green'},
                    'steps': [
                        {'range': [0, 20], 'color': '#c8e6c9'},
                        {'range': [20, 50], 'color': '#fff9c4'},
                        {'range': [50, 100], 'color': '#ffcdd2'}
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 2},
                        'thickness': 0.75,
                        'value': prob_pct
                    }
                }
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Maneuver Probability (ML Detector)",
        height=280
    )
    
    return fig


def plot_sensor_reliability(ml_status: Dict) -> go.Figure:
    """
    Plot sensor reliability scores from trained model.
    Shows mean error per sensor and reliability weights.
    """
    sensor_stats = ml_status.get('sensor_stats', {})
    
    if not sensor_stats:
        # Default fallback values
        sensor_stats = {
            '1': {'sensor_name': 'LiDAR', 'mean_error_m': 2.0, 'reliability_score': 0.85, 'count': 100},
            '2': {'sensor_name': 'Radar', 'mean_error_m': 5.0, 'reliability_score': 0.70, 'count': 80},
            '3': {'sensor_name': 'IR Camera', 'mean_error_m': 4.0, 'reliability_score': 0.75, 'count': 60},
            '4': {'sensor_name': 'EO Camera', 'mean_error_m': 3.5, 'reliability_score': 0.78, 'count': 40}
        }
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Mean Error by Sensor (m)", "Sensor Usage Distribution")
    )
    
    names = [s.get('sensor_name', f"Sensor {k}") for k, s in sensor_stats.items()]
    errors = [s.get('mean_error_m', 5.0) for s in sensor_stats.values()]
    counts = [s.get('count', 1) for s in sensor_stats.values()]
    reliabilities = [s.get('reliability_score', 0.5) for s in sensor_stats.values()]
    
    sensor_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(names)]
    
    # Bar chart - errors (lower is better)
    fig.add_trace(
        go.Bar(
            x=names,
            y=errors,
            marker_color=sensor_colors,
            text=[f"{e:.1f}m" for e in errors],
            textposition='outside',
            name='Mean Error'
        ),
        row=1, col=1
    )
    
    # Pie chart - detection counts
    fig.add_trace(
        go.Pie(
            labels=names,
            values=counts,
            marker_colors=sensor_colors,
            textinfo='label+percent',
            hole=0.3,
            name='Detection Count'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        title_text="Sensor Reliability Analysis (ML-Learned)"
    )
    
    return fig


def analyze_detection_distribution(detections: List[Dict]) -> Dict:
    """
    Analyze spatial distribution of detections to diagnose sensor issues.
    
    Detects:
    - Clustered detections at specific ranges/bearings (sensor saturation)
    - Multipath lobes (heavy sea states)
    - Outliers suggesting coordinate errors
    
    Returns diagnostic dict with warnings and statistics.
    """
    if len(detections) < 10:
        return {'status': 'insufficient_data', 'warnings': []}
    
    # Extract positions
    norths = []
    easts = []
    ranges = []
    bearings = []
    
    for d in detections:
        pos = d.get('position', [0, 0, 0])
        if len(pos) >= 2:
            n, e = pos[0], pos[1]
            norths.append(n)
            easts.append(e)
            r = np.sqrt(n**2 + e**2)
            b = np.degrees(np.arctan2(e, n))
            ranges.append(r)
            bearings.append(b)
    
    if len(ranges) < 10:
        return {'status': 'insufficient_data', 'warnings': []}
    
    norths = np.array(norths)
    easts = np.array(easts)
    ranges = np.array(ranges)
    bearings = np.array(bearings)
    
    warnings = []
    stats = {}
    
    # Range distribution analysis
    range_mean = np.mean(ranges)
    range_std = np.std(ranges)
    stats['range_mean_m'] = float(range_mean)
    stats['range_std_m'] = float(range_std)
    
    # Check for clustering at specific ranges (sensor saturation indicator)
    range_bins = np.histogram(ranges, bins=20)[0]
    max_bin_pct = 100 * np.max(range_bins) / len(ranges)
    if max_bin_pct > 40:
        warnings.append(f"Detections clustered at specific range ({max_bin_pct:.0f}% in one bin) - possible sensor saturation")
    
    # Check for bearing clustering (multipath lobes)
    bearing_bins = np.histogram(bearings, bins=36)[0]  # 10-degree bins
    bearing_max_pct = 100 * np.max(bearing_bins) / len(bearings)
    if bearing_max_pct > 50:
        peak_bearing = bearings[np.argmax(bearing_bins)]
        warnings.append(f"Detections clustered at bearing ~{peak_bearing:.0f}° ({bearing_max_pct:.0f}%) - possible multipath lobe")
    
    # Check for outliers (coordinate transformation errors)
    north_outliers = np.sum(np.abs(norths - np.mean(norths)) > 3 * np.std(norths))
    east_outliers = np.sum(np.abs(easts - np.mean(easts)) > 3 * np.std(easts))
    outlier_pct = 100 * (north_outliers + east_outliers) / (2 * len(norths))
    stats['outlier_pct'] = float(outlier_pct)
    if outlier_pct > 5:
        warnings.append(f"{outlier_pct:.1f}% outliers detected - check coordinate transformation")
    
    return {
        'status': 'analyzed',
        'warnings': warnings,
        'statistics': stats,
        'total_detections': len(detections)
    }


# ============================================================================
# Visualization Functions
# ============================================================================
def create_safety_visualization(safety_result: Dict) -> go.Figure:
    """Create safety visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "scatter"}]],
        subplot_titles=("Safety Index", "Target Positions")
    )
    
    # Safety gauge
    safety_index = safety_result.get('safety_index', 80)
    risk_level = safety_result.get('risk_level', 'LOW')
    
    color = {
        'LOW': 'green',
        'MODERATE': 'yellow', 
        'HIGH': 'orange',
        'CRITICAL': 'red'
    }.get(risk_level, 'gray')
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=safety_index,
            title={'text': f"Risk: {risk_level}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffe0b2"},
                    {'range': [50, 75], 'color': "#fff9c4"},
                    {'range': [75, 100], 'color': "#c8e6c9"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Simple position plot
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=20, color='blue', symbol='triangle-up'),
            text=['Ownship'],
            textposition='top center',
            name='Ownship'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_reliability_visualization(reliability_result: Dict) -> go.Figure:
    """Create reliability visualization."""
    components = reliability_result.get('components', {})
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "indicator"}]],
        subplot_titles=("Component Health", "Overall Status")
    )
    
    # Component health bars
    comp_names = list(components.keys())
    health_values = [components[c].get('health', 100) for c in comp_names]
    colors = ['green' if h > 70 else ('orange' if h > 40 else 'red') for h in health_values]
    
    fig.add_trace(
        go.Bar(
            x=comp_names,
            y=health_values,
            marker_color=colors,
            text=[f"{h:.1f}%" for h in health_values],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Overall health indicator
    overall = reliability_result.get('overall_health', 'GOOD')
    status_value = {'GOOD': 100, 'DEGRADED': 60, 'CRITICAL': 20}.get(overall, 80)
    status_color = {'GOOD': 'green', 'DEGRADED': 'orange', 'CRITICAL': 'red'}.get(overall, 'gray')
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=status_value,
            title={'text': overall},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': status_color},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_availability_visualization(availability_result: Dict) -> go.Figure:
    """Create availability visualization."""
    sensors = availability_result.get('sensors', {})
    capability = availability_result.get('capability_pct', 100)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "indicator"}]],
        subplot_titles=("Sensor Status", "System Capability")
    )
    
    # Sensor status pie
    status_counts = {'HEALTHY': 0, 'DEGRADED': 0, 'FAILED': 0}
    for status in sensors.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    
    labels = [k for k, v in status_counts.items() if v > 0]
    values = [v for v in status_counts.values() if v > 0]
    colors = ['green' if l == 'HEALTHY' else ('orange' if l == 'DEGRADED' else 'red') for l in labels]
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.4
        ),
        row=1, col=1
    )
    
    # Capability gauge
    mode = availability_result.get('mode', 'FULL_POWER')
    mode_color = {
        'FULL_POWER': 'green',
        'DEGRADED': 'orange',
        'REDUCED': 'red',
        'EMERGENCY': 'darkred'
    }.get(mode, 'gray')
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=capability,
            title={'text': mode},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': mode_color},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcdd2"},
                    {'range': [50, 70], 'color': "#ffe0b2"},
                    {'range': [70, 90], 'color': "#fff9c4"},
                    {'range': [90, 100], 'color': "#c8e6c9"}
                ]
            },
            delta={'reference': 100}
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_maintenance_visualization(maintenance_result: Dict) -> go.Figure:
    """Create maintenance visualization."""
    tasks = maintenance_result.get('pending_tasks_count', 0)
    cost = maintenance_result.get('cost_forecast', {})
    metric = maintenance_result.get('maintainability_metric', 1.0)
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Pending Tasks", "Estimated Cost", "Maintainability Score")
    )
    
    # Tasks count
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=tasks,
            title={'text': "Tasks"},
            delta={'reference': 0, 'increasing': {'color': 'red'}},
            number={'font': {'size': 60}}
        ),
        row=1, col=1
    )
    
    # Cost
    pending_cost = cost.get('pending_cost', 0)
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=pending_cost,
            title={'text': "Est. Cost ($)"},
            number={'prefix': "$", 'font': {'size': 40}}
        ),
        row=1, col=2
    )
    
    # Metric gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metric * 100,
            title={'text': "Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'green' if metric > 0.8 else ('orange' if metric > 0.5 else 'red')},
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=350, showlegend=False)
    return fig


def create_rams_overview(safety: Dict, reliability: Dict, availability: Dict, maintenance: Dict) -> go.Figure:
    """Create RAMS overview radar chart."""
    categories = ['Safety', 'Reliability', 'Availability', 'Maintainability']
    
    # Normalize values to 0-100
    values = [
        safety.get('safety_index', 80),
        {'GOOD': 90, 'DEGRADED': 50, 'CRITICAL': 20}.get(reliability.get('overall_health', 'GOOD'), 70),
        availability.get('capability_pct', 100),
        maintenance.get('maintainability_metric', 0.9) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.3)',
        line=dict(color='blue', width=2),
        name='RAMS Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title="RAMS Overview",
        height=400
    )
    
    return fig


# ============================================================================
# Main Application
# ============================================================================
def main():
    # Header
    st.title("🚢 Testing RAMS for Autonomous Marine Vessel using Agentic Framework with LLMs")
    st.markdown("*by mandar.tabib@sintef.no")
    st.markdown("Datasets: UCI Naval CBM (propulsion) & AutoFerry Sensor Fusion (navigation)")
    
    # Load agents (includes ML sensor fusion)
    agents, llm_available, ml_fusion_available = load_agents()
    
    # Load ML models
    ml_models = load_ml_models()
    
    if agents is None:
        st.error("Failed to load agents. Please check installation.")
        return
    
    # Load data sources
    data_sources = load_data_sources()
    
    # Sidebar - INPUT Configuration (what user actually controls)
    with st.sidebar:
        st.header("📊 Data Source Selection")
        st.caption("Select data to analyze - agents will determine results")
        
        # DATA INPUT: Navigation scenario selection
        st.subheader("Navigation Scenario")
        if data_sources.get('loaded'):
            available_scenarios = data_sources.get('scenarios', ['scenario16'])
            selected_scenario = st.selectbox(
                "Select Scenario",
                available_scenarios,
                index=available_scenarios.index('scenario16') if 'scenario16' in available_scenarios else 0,
                help="AutoFerry sensor fusion dataset scenario"
            )
        else:
            selected_scenario = 'scenario16'
            st.warning("Data sources not loaded")
        
        # DATA INPUT: Timestep selection
        st.subheader("Simulation Time")
        timestep = st.slider(
            "Data Timestep",
            min_value=0,
            max_value=100,
            value=st.session_state.get('selected_timestep', 0),
            step=1,
            help="Select which timestep of sensor data to analyze"
        )
        st.session_state.selected_timestep = timestep
        
        # DATA INPUT: Operating hours context
        st.subheader("Operating Context")
        operating_hours = st.number_input(
            "Vessel Operating Hours",
            min_value=0,
            max_value=50000,
            value=1000,
            step=100,
            help="Total operating hours for RUL calculation"
        )
        st.session_state.operating_hours = operating_hours
        
        st.divider()
        
        # ML Model Status
        st.subheader("🤖 ML Models")
        if ml_fusion_available:
            st.success("✓ ML-Enhanced Kalman Active")
        else:
            st.warning("⚠ Using fixed Kalman filter")
        
        if ml_models.get('maneuver_loaded'):
            st.caption("✓ Maneuver Detector loaded")
        else:
            st.caption("✗ Maneuver Detector not loaded")
        
        if ml_models.get('reliability_loaded'):
            st.caption("✓ Sensor Reliability loaded")
        else:
            st.caption("✗ Sensor Reliability not loaded")
        
        # Kalman Filter Tuning
        st.subheader("⚙️ Kalman Filter Tuning")
        gate_threshold = st.slider(
            "Association Gate (χ²)",
            min_value=5.0,
            max_value=50.0,
            value=st.session_state.get('gate_threshold', 16.0),
            step=1.0,
            help="Chi-squared threshold for measurement association. Higher = more permissive gating. Increase 30-50% for high-clutter environments."
        )
        st.session_state.gate_threshold = gate_threshold
        st.caption(f"Current: χ²={gate_threshold:.0f} (default=16, relaxed=24-32)")
        
        enable_reliability = st.checkbox(
            "Enable Reliability Estimator",
            value=st.session_state.get('enable_reliability_estimator', True),
            help="Enable adaptive R-matrix adjustment based on sensor reliability. Recommended for high-clutter environments."
        )
        st.session_state.enable_reliability_estimator = enable_reliability
        
        # Comparison Mode Toggle
        compare_kf = st.checkbox(
            "📊 Compare ML vs Base KF",
            value=st.session_state.get('compare_kf_modes', False),
            help="Run both ML-enhanced and base Kalman filter to compare track quality and noise adaptation"
        )
        st.session_state.compare_kf_modes = compare_kf
        if compare_kf:
            st.info("Comparison mode: Will run both filters and show side-by-side results")
        
        # KF Source for RL-PSF
        st.markdown("**KF Source for RL-PSF:**")
        prev_kf_source = st.session_state.get('kf_source_for_rlpsf', 'ML KF')
        kf_source = st.radio(
            "Select which KF tracks feed Safety Agent",
            options=['ML KF', 'Base KF'],
            index=0 if prev_kf_source == 'ML KF' else 1,
            help="Choose whether to use ML-enhanced (adaptive Q/R) or Base (fixed parameters) Kalman Filter tracks as input to the RL-PSF Safety Agent",
            horizontal=True
        )
        # Reset safety result if KF source changed (so it re-runs with new input)
        if kf_source != prev_kf_source:
            st.session_state.safety_result = None
        st.session_state.kf_source_for_rlpsf = kf_source
        st.caption(f"🎯 RL-PSF will use **{kf_source}** confirmed tracks")
        
        st.divider()
        
        # Data source info
        st.subheader("📁 Data Sources")
        st.caption("**Propulsion:** UCI Naval CBM Dataset")
        st.caption("**Navigation:** AutoFerry Sensor Fusion")
        
        if data_sources.get('loaded'):
            st.success("✓ Data loaded")
        else:
            st.error(f"✗ {data_sources.get('error', 'Load failed')}")
        
        st.divider()
        
        # ── LLM Supervisor Configuration ────────────────────────────────────
        st.subheader("🤖 LLM Supervisor")

        # Load .env values once (cached in lru sense via module-level call)
        _env = _load_dotenv()

        # Provider selector
        provider_keys = list(PROVIDER_CONFIGS.keys())
        # Reset model when provider changes
        _cur_provider = st.session_state.get('llm_provider', 'ntnu')
        selected_provider = st.selectbox(
            "Provider",
            options=provider_keys,
            format_func=lambda k: PROVIDER_CONFIGS[k]['label'],
            index=provider_keys.index(_cur_provider) if _cur_provider in provider_keys else 0,
            key='llm_provider',
        )
        _pcfg = PROVIDER_CONFIGS[selected_provider]

        if st.session_state.get('_llm_prev_provider') != selected_provider:
            st.session_state['llm_model'] = _pcfg['models'][0]
            st.session_state['_llm_prev_provider'] = selected_provider

        # Model selector
        _model_opts = _pcfg['models']
        _cur_model = st.session_state.get('llm_model', _model_opts[0])
        _model_idx = _model_opts.index(_cur_model) if _cur_model in _model_opts else 0
        st.selectbox(
            "Model",
            options=_model_opts,
            index=_model_idx,
            key='llm_model',
        )

        # API Key — pre-fill from .env or os.environ if available
        _env_key_name = _pcfg['env_key']
        _default_key = st.session_state.get(
            f'_api_key_{selected_provider}',
            _env.get(_env_key_name, os.environ.get(_env_key_name, ''))
        )
        _typed_key = st.text_input(
            "API Key",
            value=_default_key,
            type="password",
            placeholder=_env_key_name,
            help=f"Or set `{_env_key_name}` in your `.env` file to pre-fill automatically.",
            key=f'_api_key_{selected_provider}',
        )
        st.session_state['llm_api_key'] = _typed_key

        # API Base URL — editable only for NTNU
        if _pcfg['needs_base_url']:
            _base = st.text_input(
                "API Base URL",
                value=st.session_state.get(
                    'llm_api_base',
                    _env.get('NTNU_API_BASE', _pcfg['api_base'])
                ),
                key='llm_api_base',
                help="OpenAI-compatible endpoint for NTNU HPC",
            )
        else:
            _base = _pcfg['api_base']
            st.session_state['llm_api_base'] = _base

        # Test Connection button
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing…"):
                _ok, _msg = _test_llm_connection(
                    selected_provider or 'ntnu',
                    st.session_state.get('llm_model', _model_opts[0]) or _model_opts[0],
                    _typed_key or '',
                    _base or '',
                )
            st.session_state['_llm_test_ok'] = _ok
            st.session_state['_llm_test_msg'] = _msg
            # Force supervisor recreation with new config
            st.session_state.pop('_llm_supervisor', None)

        # Show last test result
        if '_llm_test_ok' in st.session_state:
            if st.session_state['_llm_test_ok']:
                st.success(st.session_state['_llm_test_msg'])
            else:
                st.error(st.session_state['_llm_test_msg'])
        elif _typed_key:
            st.caption("Click **Test Connection** to verify.")
        else:
            st.warning("⚠ No API key — deterministic fallback mode.")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Workflow", use_container_width=True):
            for key in ['current_step', 'workflow_started', 'workflow_complete',
                       'fusion_result', 'base_kf_result', 'safety_result', 'reliability_result', 'availability_result',
                       'maintenance_result', 'supervisor_result', 'propulsion_data', 'navigation_data',
                       'autoferry_data', 'scenario_data']:
                st.session_state[key] = None if 'result' in key or 'data' in key else (0 if key == 'current_step' else False)
            st.rerun()
    
    # Main content area
    st.divider()
    
    # Progress indicator (now includes ML Fusion step)
    steps = ["📋 Load", "🎯 ML Fusion", "🛡️ Safety", "⚙️ Reliability", "📡 Availability", "🔧 Maintenance", "🤖 Supervisor"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current:
                st.success(step)
            elif i == current:
                st.info(step)
            else:
                st.caption(step)
    
    st.divider()
    
    # Step 0: Configuration / Start
    if st.session_state.current_step == 0:
        st.header("📋 Step 1: Load Data for Analysis")
        
        st.markdown("""
        ### Data Selection
        The agents will analyze **real sensor data** and produce their assessments.
        
        **You control:** Which data to analyze (scenario, timestep)  
        **Agents determine:** Health status, risk levels, maintenance needs
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Input Data")
            st.metric("Navigation Scenario", selected_scenario)
            st.metric("Timestep", timestep)
            st.metric("Operating Hours", f"{operating_hours:,}h")
        
        with col2:
            st.subheader("Analysis Pipeline")
            st.info("1. **ML Sensor Fusion** → Kalman tracking with adaptive noise")
            st.info("2. **Safety Agent** → Collision risk assessment")
            st.info("3. **Reliability Agent** → Propulsion health analysis")
            st.info("4. **Availability Agent** → System capability check")
            st.info("5. **Maintainability Agent** → Maintenance scheduling")
            st.info("6. **LLM Supervisor** → Final recommendation")
        
        # Agent overview table
        st.markdown("---")
        st.subheader("🤖 Agent Overview")
        st.markdown("""
| Step | Agent | AI / Model | Purpose |
|------|-------|------------|---------|
| 1 | **ML Sensor Fusion** | Kalman Filter + RandomForest Maneuver Detector + Sensor Reliability Model | Multi-sensor target tracking with adaptive noise |
| 2 | **Safety Agent** | Reinforcement Learning (PPO) + Policy Safety Filter (PSF / CBF) | Collision risk assessment & maneuver recommendation |
| 3 | **Reliability Agent** | Degradation rate analysis + RUL linear extrapolation | Propulsion component health & Remaining Useful Life |
| 4 | **Availability Agent** | Anomaly scoring + sensor status classification | Sensor & system operational capability monitoring |
| 5 | **Maintainability Agent** | Rule-based scheduling + cost & downtime estimation | Maintenance planning & cost forecasting |
| 6 | **LLM Supervisor** | Large Language Model (NTNU API) — COLREGS / DNV expert | Final integrated maritime safety recommendation |
""")

        # Show AutoFerry data preview
        st.markdown("---")
        st.subheader("📊 AutoFerry Sensor Data Preview")
        
        # Load AutoFerry scenario data
        autoferry_data = load_autoferry_scenario_data(selected_scenario)
        
        if autoferry_data.get('loaded'):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Detections", autoferry_data.get('total_detections', 0))
            col2.metric("Ground Truth Points", autoferry_data.get('total_ground_truth', 0))
            col3.metric("ML Enhanced", "✓ Yes" if ml_fusion_available else "✗ No")
            
            # Show sample detection
            if autoferry_data['detections']:
                st.caption("Sample detection (first entry):")
                sample = autoferry_data['detections'][0]
                st.json({
                    'time': sample.get('time'),
                    'sensorID': sample.get('sensorID'),
                    'position': sample.get('position'),
                    'targetID': sample.get('targetID')
                })
        else:
            st.warning(f"Could not load AutoFerry data for {selected_scenario}")
        
        # Also show propulsion data preview
        if data_sources.get('loaded'):
            st.markdown("---")
            st.subheader("🔧 Propulsion Data Preview")
            
            scenario_data = load_scenario_data(data_sources, selected_scenario, timestep)
            
            prop_df = pd.DataFrame([{
                'Parameter': k,
                'Value': f"{v:.4f}" if isinstance(v, float) else str(v)
            } for k, v in scenario_data['propulsion'].items() if k != 'timestamp'][:6])
            st.dataframe(prop_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        if st.button("▶️ Start ML Sensor Fusion", type="primary", use_container_width=True):
            # Load AutoFerry data for sensor fusion
            st.session_state.autoferry_data = load_autoferry_scenario_data(selected_scenario)
            
            # Load propulsion/navigation data
            if data_sources.get('loaded'):
                st.session_state.scenario_data = load_scenario_data(
                    data_sources, selected_scenario, timestep
                )
            else:
                st.session_state.scenario_data = {
                    'propulsion': {'compressor_decay': 0.95, 'turbine_decay': 0.97, 'timestamp': timestep},
                    'navigation': {'targets': [], 'ownship_position': [0, 0], 'ownship_velocity': [5, 0]},
                    'timestep': timestep
                }
            
            st.session_state.current_step = 1
            st.session_state.workflow_started = True
            st.rerun()
    
    # Step 1: ML-Enhanced Sensor Fusion (NEW)
    elif st.session_state.current_step == 1:
        st.header("🎯 Step 2: ML-Enhanced Kalman Filter Fusion")
        st.markdown("*Adaptive noise estimation using trained maneuver detector and sensor reliability models*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                st.markdown(
                    "- Kalman Filter (6-state: x, y, vx, vy, ax, ay)\n"
                    "- RandomForest Maneuver Detector (adaptive Q)\n"
                    "- Sensor Reliability Model (adaptive R)\n"
                    "- Multi-sensor fusion: LiDAR, Radar, IR, EO Camera"
                )
            with c2:
                st.markdown("**Inputs**")
                st.markdown(
                    "- AutoFerry multi-sensor detections (NED coords)\n"
                    "- Sensor IDs and position measurements\n"
                    "- Timestep index for scenario selection"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Confirmed Kalman tracks with position/velocity\n"
                    "- Adaptive noise parameters Q & R per track\n"
                    "- Maneuver probability per target\n"
                    "- Position uncertainty estimates (metres)"
                )

        if st.session_state.fusion_result is None:
            autoferry_data = st.session_state.get('autoferry_data')
            
            if autoferry_data is None or not autoferry_data.get('loaded'):
                st.error("AutoFerry data not loaded. Please go back and select a scenario.")
            else:
                with st.spinner("Running ML-enhanced sensor fusion..."):
                    result = run_ml_sensor_fusion(agents, autoferry_data, timestep)
                    st.session_state.fusion_result = result
                    time.sleep(0.3)
                st.rerun()
        else:
            result = st.session_state.fusion_result
            
            if result.get('status') == 'error':
                st.error(f"Fusion error: {result.get('error')}")
            else:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Detections Processed", result.get('total_detections', 0))
                col2.metric("Confirmed Tracks", result.get('confirmed_tracks', 0))
                col3.metric("ML Enhanced", "✓ Yes" if result.get('ml_enhanced') else "✗ No")
                col4.metric("Processing Time", f"{result.get('processing_time_ms', 0):.1f}ms")
                
                # === DIAGNOSTIC WARNINGS DISPLAY ===
                diagnostics = result.get('diagnostics', {})
                ml_status = result.get('ml_status', {})
                
                # Show ML subsystem status with diagnostics
                with st.expander("🔍 ML Pipeline Diagnostics", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        maneuver_active = ml_status.get('maneuver_detector_active', False)
                        reliability_active = ml_status.get('reliability_estimator_active', False)
                        
                        st.markdown("**ML Subsystem Status**")
                        if maneuver_active:
                            st.success("✓ Maneuver Detector: ACTIVE")
                        else:
                            st.error("✗ Maneuver Detector: INACTIVE")
                            st.caption("Model may not be loaded. Check models/maneuver_detector.pkl")
                        
                        if reliability_active:
                            st.success("✓ Reliability Estimator: ACTIVE")
                        else:
                            st.warning("✗ Reliability Estimator: INACTIVE (using fixed R-matrix)")
                            st.caption("Enable in sidebar or check models/sensor_reliability.pkl")
                        
                        # Gate threshold info
                        gate_applied = ml_status.get('gate_threshold_applied', 16.0)
                        st.markdown(f"**Association Gate (χ²):** {gate_applied:.1f}")
                        if gate_applied < 12:
                            st.warning("Gate is tight - may miss associations in clutter")
                        elif gate_applied > 30:
                            st.info("Gate is relaxed - good for high-clutter environments")
                    
                    with col_b:
                        st.markdown("**Data Quality Warnings**")
                        coord_warnings = diagnostics.get('coordinate_warnings', [])
                        timing_warnings = diagnostics.get('timing_warnings', [])
                        
                        if coord_warnings:
                            for w in coord_warnings:
                                st.warning(w)
                        else:
                            st.success("✓ Coordinate transformations OK")
                        
                        if timing_warnings:
                            for w in timing_warnings:
                                st.warning(w)
                        else:
                            st.success("✓ Timestamp synchronization OK")
                        
                        # Detection distribution analysis
                        total_dets = result.get('total_detections', 0)
                        confirmed = result.get('confirmed_tracks', 0)
                        if total_dets > 0 and confirmed == 0:
                            st.error(f"⚠ {total_dets} detections but 0 tracks - check association gates or coordinates")
                            st.caption("Try: Increase gate threshold (χ²) by 30-50% in sidebar")
                        
                        # Spatial distribution analysis
                        autoferry_data_diag = st.session_state.get('autoferry_data', {})
                        det_analysis = analyze_detection_distribution(
                            autoferry_data_diag.get('detections', [])[:min(total_dets, 500)]
                        )
                        if det_analysis.get('warnings'):
                            st.markdown("**Spatial Distribution Issues**")
                            for w in det_analysis['warnings']:
                                st.warning(w)
                
                # Main tracking visualization
                st.subheader("📍 Target Tracking Visualization")
                autoferry_data = st.session_state.get('autoferry_data', {})
                fig_tracks = plot_ml_tracking_scenario(
                    result.get('tracks', []),
                    autoferry_data.get('detections', [])[:min(len(autoferry_data.get('detections', [])), timestep*10 + 100)],
                    autoferry_data.get('ground_truth', [])
                )
                st.plotly_chart(fig_tracks, use_container_width=True)
                
                # ML Status visualizations
                st.subheader("🔧 ML Kalman Filter Adaptation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Adaptive Noise Parameters (Q & R)**")
                    fig_noise = plot_ml_noise_adaptation(result.get('tracks', []))
                    st.plotly_chart(fig_noise, use_container_width=True)
                
                with col2:
                    st.markdown("**Maneuver Detection (Probability)**")
                    fig_maneuver = plot_maneuver_detection(result.get('tracks', []))
                    st.plotly_chart(fig_maneuver, use_container_width=True)
                
                # Sensor reliability
                st.subheader("📊 Sensor Reliability Analysis")
                fig_sensors = plot_sensor_reliability(result.get('ml_status', {}))
                st.plotly_chart(fig_sensors, use_container_width=True)
                
                # Track details table
                tracks = result.get('tracks', [])
                confirmed_tracks = [t for t in tracks if t.get('status') == 'confirmed']
                
                if confirmed_tracks:
                    st.subheader("📋 Track Details")
                    track_df = pd.DataFrame([{
                        'Track ID': t.get('track_id', i),
                        'Target': t.get('target_name', '?'),
                        'Status': t.get('status', '?'),
                        'Speed (m/s)': f"{t.get('speed_ms', t.get('speed_knots', 0)*0.514):.1f}",
                        'Maneuver P': f"{t.get('maneuver_probability', 0):.2f}",
                        'Process Q': f"{t.get('process_noise_Q', t.get('current_process_noise', 0)):.3f}",
                        'Measure R': f"{t.get('measurement_noise_R', t.get('current_measurement_noise', 0)):.1f}",
                        'Uncertainty (m)': f"{t.get('position_uncertainty_m', 0):.1f}"
                    } for i, t in enumerate(confirmed_tracks)])
                    st.dataframe(track_df, use_container_width=True, hide_index=True)
                
                # ═══════════════════════════════════════════════════════════════
                # ML vs Base Kalman Filter Comparison Section
                # ═══════════════════════════════════════════════════════════════
                if st.session_state.get('compare_kf_modes', False):
                    st.markdown("---")
                    st.subheader("📊 ML vs Base Kalman Filter Comparison")
                    
                    # Run base KF if not already run
                    if st.session_state.get('base_kf_result') is None:
                        with st.spinner("Running Base Kalman Filter for comparison..."):
                            autoferry_data = st.session_state.get('autoferry_data', {})
                            base_result = run_base_kalman_filter(autoferry_data, timestep)
                            st.session_state.base_kf_result = base_result
                    else:
                        base_result = st.session_state.base_kf_result
                    
                    if base_result.get('status') == 'success':
                        # Compute comparison metrics
                        comparison = compute_kf_comparison_metrics(result, base_result)
                        
                        # Side-by-side metrics
                        st.markdown("### 📈 Performance Comparison")
                        col_ml, col_base, col_diff = st.columns(3)
                        
                        with col_ml:
                            st.markdown("**🤖 ML-Enhanced KF**")
                            st.metric("Confirmed Tracks", comparison['ml']['confirmed_tracks'])
                            st.metric("Avg Process Noise Q", f"{comparison['ml']['avg_process_noise_Q']:.3f}")
                            st.metric("Avg Measurement Noise R", f"{comparison['ml']['avg_measurement_noise_R']:.1f}m")
                            st.metric("Avg Maneuver Prob", f"{comparison['ml']['avg_maneuver_probability']:.1%}")
                            st.metric("Processing Time", f"{comparison['ml']['processing_time_ms']:.1f}ms")
                            st.caption("✓ Parameters adapt per track/sensor")
                        
                        with col_base:
                            st.markdown("**📐 Base KF (Fixed)**")
                            st.metric("Confirmed Tracks", comparison['base']['confirmed_tracks'])
                            st.metric("Fixed Process Noise Q", f"{comparison['base']['avg_process_noise_Q']:.3f}")
                            st.metric("Fixed Measurement Noise R", f"{comparison['base']['avg_measurement_noise_R']:.1f}m")
                            st.metric("Maneuver Detection", "N/A")
                            st.metric("Processing Time", f"{comparison['base']['processing_time_ms']:.1f}ms")
                            st.caption("✗ Same parameters for all tracks")
                        
                        with col_diff:
                            st.markdown("**🔄 Difference**")
                            track_diff = comparison['comparison']['track_difference']
                            if track_diff > 0:
                                st.success(f"ML has +{track_diff} more tracks")
                            elif track_diff < 0:
                                st.warning(f"ML has {track_diff} fewer tracks")
                            else:
                                st.info("Same track count")
                            
                            # Show adaptive Q range
                            ml_tracks = result.get('tracks', [])
                            ml_confirmed = [t for t in ml_tracks if t.get('status') == 'confirmed']
                            if ml_confirmed:
                                q_values = [t.get('process_noise_Q', t.get('current_process_noise', 0.5)) for t in ml_confirmed]
                                r_values = [t.get('measurement_noise_R', t.get('last_measurement_noise', 5.0)) for t in ml_confirmed]
                                st.markdown(f"**Q Range:** {min(q_values):.3f} - {max(q_values):.3f}")
                                st.markdown(f"**R Range:** {min(r_values):.1f} - {max(r_values):.1f}m")
                                
                                # Show maneuver detection stats
                                maneuver_probs = [t.get('maneuver_probability', 0) for t in ml_confirmed]
                                high_maneuver = sum(1 for p in maneuver_probs if p > 0.5)
                                st.markdown(f"**Maneuvering Tracks:** {high_maneuver}/{len(ml_confirmed)}")
                        
                        # Detailed comparison table
                        st.markdown("### 📋 Track-by-Track Comparison")
                        st.markdown("*ML-enhanced KF tracks with adaptive parameters*")
                        
                        if ml_confirmed:
                            comparison_df = pd.DataFrame([{
                                'Track': t.get('track_id', i),
                                'Maneuver P': t.get('maneuver_probability', 0),
                                'ML Q': t.get('process_noise_Q', t.get('current_process_noise', 0.5)),
                                'Base Q': comparison['base']['avg_process_noise_Q'],
                                'Q Δ': t.get('process_noise_Q', t.get('current_process_noise', 0.5)) - comparison['base']['avg_process_noise_Q'],
                                'ML R': t.get('measurement_noise_R', t.get('last_measurement_noise', 5.0)),
                                'Base R': comparison['base']['avg_measurement_noise_R'],
                                'R Δ': t.get('measurement_noise_R', t.get('last_measurement_noise', 5.0)) - comparison['base']['avg_measurement_noise_R'],
                                'Uncertainty (m)': t.get('position_uncertainty_m', 0)
                            } for i, t in enumerate(ml_confirmed)])
                            
                            # Format the dataframe
                            comparison_df_display = comparison_df.copy()
                            comparison_df_display['Maneuver P'] = comparison_df_display['Maneuver P'].apply(lambda x: f"{x:.1%}")
                            comparison_df_display['ML Q'] = comparison_df_display['ML Q'].apply(lambda x: f"{x:.3f}")
                            comparison_df_display['Base Q'] = comparison_df_display['Base Q'].apply(lambda x: f"{x:.3f}")
                            comparison_df_display['Q Δ'] = comparison_df_display['Q Δ'].apply(lambda x: f"{x:+.3f}")
                            comparison_df_display['ML R'] = comparison_df_display['ML R'].apply(lambda x: f"{x:.1f}")
                            comparison_df_display['Base R'] = comparison_df_display['Base R'].apply(lambda x: f"{x:.1f}")
                            comparison_df_display['R Δ'] = comparison_df_display['R Δ'].apply(lambda x: f"{x:+.1f}")
                            comparison_df_display['Uncertainty (m)'] = comparison_df_display['Uncertainty (m)'].apply(lambda x: f"{x:.1f}")
                            
                            st.dataframe(comparison_df_display, use_container_width=True, hide_index=True)
                            
                            # Key insight
                            avg_maneuvering = np.mean([t.get('maneuver_probability', 0) for t in ml_confirmed])
                            if avg_maneuvering > 0.3:
                                st.info(f"💡 **Insight:** Average maneuver probability is {avg_maneuvering:.1%}. "
                                       "ML KF is increasing process noise Q to handle target dynamics better than fixed parameters.")
                            else:
                                st.info(f"💡 **Insight:** Targets are mostly in steady state (avg maneuver prob: {avg_maneuvering:.1%}). "
                                       "ML KF provides sensor-adaptive R values based on reliability training.")
                    else:
                        st.error(f"Base KF comparison failed: {base_result.get('error', 'Unknown error')}")
            
            # LLM Expert Panel for Sensor Fusion
            render_expert_panel('fusion', result)
            
            st.markdown("---")
            if st.button("▶️ Continue to Safety Agent", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    
    # Step 2: Safety Agent (was Step 1)
    elif st.session_state.current_step == 2:
        st.header("🛡️ Step 3: Safety Agent")
        st.markdown("*Analyzing collision risks and navigation safety using RL+PSF policy*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                st.markdown(
                    "- Reinforcement Learning (PPO — Proximal Policy Optimisation)\n"
                    "- Policy Safety Filter (PSF / CBF)\n"
                    "- COLREGS rule-based constraint layer"
                )
            with c2:
                kf_src = st.session_state.get('kf_source_for_rlpsf', 'ML KF')
                st.markdown("**Inputs**")
                st.markdown(
                    f"- Confirmed tracks from **{kf_src}**\n"
                    "- Own-ship position and velocity\n"
                    "- Operating scenario (timestep)"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Risk level (LOW / MEDIUM / HIGH / CRITICAL)\n"
                    "- Safety index (0–100 %)\n"
                    "- Recommended collision-avoidance maneuvers\n"
                    "- Per-target COLREGS compliance status"
                )

        if st.session_state.safety_result is None:
            with st.spinner("Running Safety Agent..."):
                # Get selected KF result based on user choice
                kf_source = st.session_state.get('kf_source_for_rlpsf', 'ML KF')
                if kf_source == 'ML KF':
                    kf_result = st.session_state.get('fusion_result')
                else:
                    kf_result = st.session_state.get('base_kf_result')
                
                # If selected KF result not available, try the other one
                if kf_result is None:
                    kf_result = st.session_state.get('fusion_result') or st.session_state.get('base_kf_result')
                    if kf_result:
                        kf_source = 'ML KF' if st.session_state.get('fusion_result') else 'Base KF'
                
                result = run_safety_agent(agents, st.session_state.scenario_data, kf_result, kf_source)
                st.session_state.safety_result = result
                time.sleep(0.5)  # Brief pause for UX
            st.rerun()
        else:
            result = st.session_state.safety_result
            
            # Show KF source being used
            kf_src_used = result.get('kf_source', 'unknown')
            st.info(f"🎯 **Input Source:** Confirmed tracks from **{kf_src_used}** ({result['active_tracks']} tracks)")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Risk Level", result['risk_level'], 
                         delta="Critical" if result['risk_level'] == 'CRITICAL' else None,
                         delta_color="inverse")
            with col2:
                st.metric("Safety Index", f"{result['safety_index']}%")
            with col3:
                st.metric("Active Tracks", result['active_tracks'])
            with col4:
                st.metric("Track Source", kf_src_used)
            
            # Visualization
            fig = create_safety_visualization(result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Maneuvers
            if result['maneuvers']:
                st.subheader("📌 Recommended Maneuvers")
                for m in result['maneuvers']:
                    st.info(f"• {m.get('action', m)} - {m.get('reason', '')}")
            
            # ─────────────────────────────────────────────────────────────────
            # PSF Safety Guarantee Panel
            # ─────────────────────────────────────────────────────────────────
            rl_psf = result.get('rl_psf_details', {})
            
            with st.expander("🛡️ PSF Safety Guarantee Details", expanded=True):
                st.markdown("""
                **Policy Safety Filter (PSF)** uses Control Barrier Functions (CBF) to guarantee safety 
                even when the RL policy outputs suboptimal actions. The barrier value φ > 0 indicates safe operation.
                """)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🎮 RL Action")
                    rl_action = rl_psf.get('rl_action', {})
                    st.metric("Speed Ratio", f"{rl_action.get('speed_ratio', 0.8):.2f}",
                             help="0=stop, 1=full ahead")
                    st.metric("Steering", f"{rl_action.get('steering', 0.0):+.2f}",
                             help="-1=hard port, +1=hard starboard")
                    st.metric("Confidence", f"{rl_action.get('confidence', 0.5)*100:.0f}%",
                             help="RL policy confidence in action")
                
                with col2:
                    st.markdown("### 🛡️ PSF Filter")
                    psf = rl_psf.get('psf_filter', {})
                    
                    intervention = psf.get('intervention_type', 'NONE')
                    if intervention.upper() in ['NONE', '']:
                        st.success("✓ No intervention needed")
                    elif intervention.upper() in ['SPEED', 'SPEED_REDUCTION']:
                        st.warning("⚠️ Speed reduced for safety")
                    elif intervention.upper() in ['STEER', 'STEERING_CORRECTION']:
                        st.warning("⚠️ Steering corrected")
                    elif intervention.upper() in ['OVERRIDE', 'FULL_OVERRIDE']:
                        st.error("🚨 Full action override")
                    elif intervention.upper() == 'EMERGENCY_STOP':
                        st.error("🚨 EMERGENCY STOP")
                    else:
                        st.info(f"ℹ️ {intervention}")
                    
                    barrier = psf.get('barrier_value', 1.0)
                    st.metric("Barrier Value (φ)", f"{barrier:.3f}",
                             help="φ > 0 = safe, φ ≤ 0 = unsafe (PSF intervenes)")
                    
                    if psf.get('was_modified'):
                        st.caption("⚠️ Action was modified by PSF")
                    else:
                        st.caption("✓ Original action was safe")
                
                with col3:
                    st.markdown("### 📊 Statistics")
                    stats = rl_psf.get('psf_stats', {})
                    st.metric("Total RL Calls", stats.get('total_calls', 0))
                    st.metric("PSF Interventions", stats.get('interventions', 0))
                    rate = stats.get('intervention_rate_pct', 0)
                    st.metric("Intervention Rate", f"{rate:.1f}%",
                             delta="High" if rate > 20 else None,
                             delta_color="inverse" if rate > 20 else "off")
                
                # Domain mismatch warning
                if rl_psf.get('domain_mismatch'):
                    st.warning(
                        f"⚠️ **Domain Mismatch:** RL trained on '{rl_psf.get('training_domain', 'synthetic data')}', "
                        f"currently running on '{rl_psf.get('current_domain', 'real data')}'. "
                        f"PSF provides safety guarantee despite potential policy suboptimality."
                    )
            
            # ─────────────────────────────────────────────────────────────────
            # RL Validation Metrics Panel
            # ─────────────────────────────────────────────────────────────────
            validation = result.get('rl_validation_metrics', {})
            
            with st.expander("🔬 RL Policy Validation", expanded=False):
                st.markdown("""
                **Validation Assessment** using PSF intervention rate as a proxy for zero-shot performance.
                Ask the LLM Expert below for detailed KS-test and distribution analysis.
                """)
                
                verdict = validation.get('validation_verdict', 'UNKNOWN')
                reason = validation.get('validation_reason', '')
                
                if verdict == 'VALIDATED':
                    st.success(f"✅ **{verdict}**: {reason}")
                elif verdict == 'ACCEPTABLE':
                    st.info(f"ℹ️ **{verdict}**: {reason}")
                elif verdict == 'WARNING':
                    st.warning(f"⚠️ **{verdict}**: {reason}")
                else:
                    st.error(f"🚨 **{verdict}**: {reason}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intervention Rate", f"{validation.get('intervention_rate_pct', 0):.1f}%",
                             help="<10% = Good, 10-30% = Warning, >30% = Critical")
                with col2:
                    st.metric("Action Confidence", f"{validation.get('action_confidence', 0.5)*100:.0f}%",
                             help="<30% = Low confidence, likely out-of-distribution")
                
                st.caption(f"ℹ️ {validation.get('zero_shot_proxy', '')}")
                st.caption(f"ℹ️ {validation.get('ks_test_note', '')}")
                
                st.markdown("---")
                st.markdown("**💡 Ask the LLM Expert below to:**")
                st.markdown("""
                - "Validate the RL policy for this scenario"
                - "Is the intervention rate acceptable for deployment?"
                - "What does the confidence level indicate about domain gap?"
                - "Should we retrain the RL model on AutoFerry data?"
                """)
            
            # LLM Expert Panel for Safety
            render_expert_panel('safety', result)
            
            # Continue button
            st.markdown("---")
            if st.button("▶️ Continue to Reliability Agent", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    
    # Step 3: Reliability Agent (was Step 2)
    elif st.session_state.current_step == 3:
        st.header("⚙️ Step 4: Reliability Agent")
        st.markdown("*Analyzing propulsion health and estimating Remaining Useful Life (RUL)*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                st.markdown(
                    "- Degradation rate analysis (compressor & turbine decay)\n"
                    "- Threshold-based health scoring\n"
                    "- RUL linear extrapolation model\n"
                    "- UCI Naval Propulsion CBM dataset"
                )
            with c2:
                st.markdown("**Inputs**")
                st.markdown(
                    "- Compressor decay coefficient\n"
                    "- Turbine decay coefficient\n"
                    "- 16 gas-turbine propulsion sensor readings"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Overall health status (GOOD / WARNING / CRITICAL)\n"
                    "- Per-component health percentage\n"
                    "- Remaining Useful Life estimate (hours)\n"
                    "- Critical alerts and warning messages"
                )

        if st.session_state.reliability_result is None:
            with st.spinner("Running Reliability Agent..."):
                result = run_reliability_agent(agents, st.session_state.scenario_data)
                st.session_state.reliability_result = result
                time.sleep(0.5)
            st.rerun()
        else:
            result = st.session_state.reliability_result
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Health", result['overall_health'])
            with col2:
                comp = result['components'].get('compressor', {})
                st.metric("Compressor", f"{comp.get('health', 0):.1f}%")
            with col3:
                rul = comp.get('rul_hours', 0)
                st.metric("RUL (hours)", f"{rul:.0f}")
            
            # Visualization
            fig = create_reliability_visualization(result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Alerts
            if result['critical_alerts']:
                st.error("**Critical Alerts:**")
                for alert in result['critical_alerts']:
                    st.error(f"🚨 {alert}")
            
            if result['warnings']:
                st.warning("**Warnings:**")
                for warn in result['warnings']:
                    st.warning(f"⚠️ {warn}")
            
            # LLM Expert Panel for Reliability
            render_expert_panel('reliability', result)
            
            st.markdown("---")
            if st.button("▶️ Continue to Availability Agent", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
    
    # Step 4: Availability Agent (was Step 3)
    elif st.session_state.current_step == 4:
        st.header("📡 Step 5: Availability Agent")
        st.markdown("*Monitoring sensor health and operational capability*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                st.markdown(
                    "- Anomaly scoring (z-score / threshold deviation)\n"
                    "- Sensor status classifier (HEALTHY / DEGRADED / FAILED)\n"
                    "- Multi-sensor cross-validation"
                )
            with c2:
                st.markdown("**Inputs**")
                st.markdown(
                    "- Propulsion sensor readings (16 features)\n"
                    "- Navigation sensor status flags\n"
                    "- Operating hours and timestep"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Operating mode (FULL / DEGRADED / LIMITED)\n"
                    "- System capability percentage\n"
                    "- Per-sensor health status\n"
                    "- Detected anomalies list"
                )

        if st.session_state.availability_result is None:
            with st.spinner("Running Availability Agent..."):
                result = run_availability_agent(agents, st.session_state.scenario_data)
                st.session_state.availability_result = result
                time.sleep(0.5)
            st.rerun()
        else:
            result = st.session_state.availability_result
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Operating Mode", result['mode'])
            with col2:
                st.metric("Capability", f"{result['capability_pct']:.1f}%")
            with col3:
                st.metric("Anomaly Score", f"{result['anomaly_score']:.2f}")
            
            # Visualization
            fig = create_availability_visualization(result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensor status table
            st.subheader("Sensor Status")
            sensor_df = pd.DataFrame([
                {"Sensor": k.upper(), "Status": v, "Healthy": "✅" if v == "HEALTHY" else ("⚠️" if v == "DEGRADED" else "❌")}
                for k, v in result['sensors'].items()
            ])
            st.dataframe(sensor_df, hide_index=True, use_container_width=True)
            
            # Anomalies
            if result['anomalies']:
                st.warning("**Detected Anomalies:**")
                for anomaly in result['anomalies']:
                    st.warning(f"⚠️ {anomaly}")
            
            # LLM Expert Panel for Availability
            render_expert_panel('availability', result)
            
            st.markdown("---")
            if st.button("▶️ Continue to Maintainability Agent", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
    
    # Step 5: Maintainability Agent (was Step 4)
    elif st.session_state.current_step == 5:
        st.header("🔧 Step 6: Maintainability Agent")
        st.markdown("*Scheduling maintenance based on reliability findings*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                st.markdown(
                    "- Rule-based maintenance scheduling engine\n"
                    "- Priority scoring (CRITICAL / HIGH / MEDIUM / LOW)\n"
                    "- Cost & downtime estimation model"
                )
            with c2:
                st.markdown("**Inputs**")
                st.markdown(
                    "- Reliability Agent output (health scores, RUL)\n"
                    "- Operating hours and maintenance history\n"
                    "- Component threshold definitions"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Prioritised maintenance task list\n"
                    "- Estimated cost per task (USD)\n"
                    "- Estimated downtime per task (hours)\n"
                    "- Actionable maintenance recommendations"
                )

        if st.session_state.maintenance_result is None:
            with st.spinner("Running Maintainability Agent..."):
                # Pass reliability result so maintenance agent can base its recommendations on it
                result = run_maintainability_agent(
                    agents, 
                    st.session_state.scenario_data,
                    st.session_state.reliability_result  # Use reliability findings
                )
                st.session_state.maintenance_result = result
                time.sleep(0.5)
            st.rerun()
        else:
            result = st.session_state.maintenance_result
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pending Tasks", result['pending_tasks_count'])
            with col2:
                cost = result['cost_forecast'].get('pending_cost', 0)
                st.metric("Est. Cost", f"${cost:,.0f}")
            with col3:
                downtime = result['cost_forecast'].get('pending_downtime_hours', 0)
                st.metric("Est. Downtime", f"{downtime}h")
            
            # Visualization
            fig = create_maintenance_visualization(result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if result['recommendations']:
                st.subheader("📋 Maintenance Recommendations")
                for rec in result['recommendations']:
                    priority = rec.split(':')[0] if ':' in rec else 'INFO'
                    color_map = {'CRITICAL': 'error', 'HIGH': 'warning', 'MEDIUM': 'info', 'LOW': 'success'}
                    getattr(st, color_map.get(priority, 'info'))(rec)
            
            # LLM Expert Panel for Maintenance
            render_expert_panel('maintenance', result)
            
            st.markdown("---")
            if st.button("▶️ Get Final Supervisor Recommendation", type="primary", use_container_width=True):
                st.session_state.current_step = 6
                st.rerun()
    
    # Step 6: LLM Supervisor (was Step 5)
    elif st.session_state.current_step == 6:
        st.header("🤖 Step 7: LLM Supervisor - Final Recommendation")
        st.markdown("*Integrating all agent reports for expert maritime reasoning*")

        with st.expander("ℹ️ About this Agent", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**AI / Model**")
                _active_prov = st.session_state.get('llm_provider', 'ntnu')
                _active_model = st.session_state.get('llm_model', '—')
                _active_label = PROVIDER_CONFIGS.get(_active_prov, {}).get('label', _active_prov)
                st.markdown(
                    f"- **Provider:** {_active_label}\n"
                    f"- **Model:** `{_active_model}`\n"
                    "- COLREGS Rule 8, 16 knowledge base\n"
                    "- DNV GL maritime safety standards\n"
                    "- Fallback: deterministic rule-based reasoning"
                )
            with c2:
                st.markdown("**Inputs**")
                st.markdown(
                    "- Safety Agent report (risk level, maneuvers)\n"
                    "- Reliability Agent report (health, RUL)\n"
                    "- Availability Agent report (capability, anomalies)\n"
                    "- Maintainability Agent report (tasks, costs)"
                )
            with c3:
                st.markdown("**Outputs**")
                st.markdown(
                    "- Integrated maritime safety recommendation\n"
                    "- Overall vessel readiness assessment\n"
                    "- Prioritised action items\n"
                    "- COLREGS / DNV compliance verdict"
                )

        if st.session_state.supervisor_result is None:
            with st.spinner("Consulting LLM Supervisor (COLREGS/DNV Expert)..."):
                result = run_llm_supervisor(
                    st.session_state.safety_result,
                    st.session_state.reliability_result,
                    st.session_state.availability_result,
                    st.session_state.maintenance_result
                )
                st.session_state.supervisor_result = result
                st.session_state.workflow_complete = True
            st.rerun()
        else:
            result = st.session_state.supervisor_result
            
            # RAMS Overview
            st.subheader("📊 RAMS Overview")
            fig = create_rams_overview(
                st.session_state.safety_result,
                st.session_state.reliability_result,
                st.session_state.availability_result,
                st.session_state.maintenance_result
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Supervisor Decision
            if result.get('status') == 'complete':
                # Risk Assessment
                st.subheader("🎯 Risk Assessment")
                st.info(result.get('risk_assessment', 'N/A'))
                
                # Priority Ranking
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📌 Priority Ranking")
                    for i, priority in enumerate(result.get('priority_ranking', [])[:5], 1):
                        st.markdown(f"**{i}.** {priority}")
                
                with col2:
                    st.subheader("⚡ Recommended Actions")
                    for action in result.get('recommended_actions', [])[:5]:
                        st.markdown(f"• {action}")
                
                # Alerts
                alerts = result.get('alerts', [])
                if alerts:
                    st.subheader("🚨 Critical Alerts")
                    for alert in alerts:
                        st.error(f"⚠️ {alert}")
                
                # Summary
                st.divider()
                st.subheader("📝 Executive Summary")
                st.success(result.get('summary', 'Analysis complete.'))
                
                # Metadata
                with st.expander("ℹ️ Analysis Metadata"):
                    meta = result.get('metadata', {})
                    st.json({
                        'Latency (ms)': meta.get('latency_ms', 0),
                        'Tokens Used': meta.get('tokens_used', 0),
                        'LLM Used': meta.get('llm_used', False),
                        'Timestamp': meta.get('timestamp', 'N/A')
                    })
                
                # LLM Expert Panel for Supervisor (follow-up questions)
                render_expert_panel('supervisor', result)
            else:
                st.error(f"Supervisor analysis failed: {result.get('reason', 'Unknown error')}")
            
            st.divider()
            
            # Completion message
            st.balloons()
            st.success("✅ **Workflow Complete!** All agents have been executed and the final recommendation is ready.")
            
            # Download report
            if st.button("📥 Download Full Report", use_container_width=True):
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'scenario': st.session_state.scenario_data,
                    'safety_result': st.session_state.safety_result,
                    'reliability_result': st.session_state.reliability_result,
                    'availability_result': st.session_state.availability_result,
                    'maintenance_result': st.session_state.maintenance_result,
                    'supervisor_result': st.session_state.supervisor_result
                }
                import json
                st.download_button(
                    "💾 Save Report (JSON)",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"rams_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # New analysis button
            if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
                for key in ['current_step', 'workflow_started', 'workflow_complete',
                           'fusion_result', 'safety_result', 'reliability_result', 'availability_result',
                           'maintenance_result', 'supervisor_result', 'scenario_data', 'autoferry_data']:
                    st.session_state[key] = None if 'result' in key or 'data' in key else (0 if key == 'current_step' else False)
                st.rerun()


if __name__ == "__main__":
    main()
