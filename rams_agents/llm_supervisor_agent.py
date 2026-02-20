"""
LLM Supervisor Agent - Autonomous Vessel Expert

Uses an OpenAI-compatible LLM API to provide high-level reasoning,
risk prioritization, and natural language advisories for the RAMS framework.

Integrates with:
- SafetyAgent (collision avoidance)
- ReliabilityAgent (RUL prediction)
- AvailabilityAgent (sensor redundancy)
- MaintainabilityAgent (maintenance scheduling)
"""

import os
import json
import yaml
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# OpenAI-compatible client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Anthropic native SDK (optional)
try:
    import anthropic as _anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    _anthropic_sdk = None

from .base_agent import RAMSBaseAgent, RAMSCategory, AgentBelief, MessagePriority


@dataclass
class LLMConfig:
    """LLM configuration loaded from YAML."""
    api_key: str
    api_base: str
    model: str
    provider: str = 'ntnu'   # ntnu | openai | google | anthropic
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 4000.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass  
class SupervisorConfig:
    """Supervisor agent configuration."""
    reasoning_interval: float = 5.0
    safety_critical_threshold: float = 0.7
    rul_critical_hours: float = 100
    availability_threshold: float = 0.8
    verbose_mode: bool = True
    include_colregs_references: bool = True
    include_dnv_references: bool = True


@dataclass
class VesselContext:
    """Vessel information for context injection."""
    name: str = "Vessel"
    mmsi: int = 258342000
    imo: int = 9371361
    owner: str = "NTNU"
    dp_class: str = "SDP-11"
    operational_area: str = "Norwegian coastal waters"


@dataclass
class SupervisorDecision:
    """Structured output from LLM reasoning."""
    timestamp: float
    priority_ranking: List[str]
    recommended_actions: List[str]
    risk_assessment: str
    alerts: List[str]
    natural_language_summary: str
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'priority_ranking': self.priority_ranking,
            'recommended_actions': self.recommended_actions,
            'risk_assessment': self.risk_assessment,
            'alerts': self.alerts,
            'summary': self.natural_language_summary,
            'tokens': self.reasoning_tokens,
            'latency_ms': self.latency_ms
        }


class LLMSupervisorAgent(RAMSBaseAgent):
    """
    LLM-powered Supervisor Agent for autonomous vessel operations.
    
    Uses domain expertise to:
    - Prioritize competing RAMS concerns
    - Generate natural language advisories
    - Handle edge cases and ambiguous situations
    - Provide COLREGS and DNV-compliant reasoning
    
    The LLM operates at ~1-10 Hz for high-level reasoning while
    deterministic agents (RL+PSF, Kalman) handle real-time control.
    """
    
    SYSTEM_PROMPT = """You are an expert autonomous vessel supervisor for {vessel_name} (MMSI: {mmsi}).

Your responsibilities:
1. Analyze RAMS (Reliability, Availability, Maintainability, Safety) reports from specialized agents
2. Prioritize risks and recommend actions based on maritime best practices
3. Ensure COLREGS compliance for collision avoidance situations
4. Apply DNV maritime standards for safety-critical decisions
5. Communicate clearly and actionably with vessel operators

Vessel context:
- Name: {vessel_name}
- Owner: {owner}
- DP System: {dp_class}
- Operational Area: {operational_area}

Key COLREGS Rules to apply:
- Rule 13: Overtaking (give-way to vessel being overtaken)
- Rule 14: Head-on (both alter course to starboard)
- Rule 15: Crossing (give-way vessel on your starboard side)
- Rule 16: Action by give-way vessel (early, substantial action)
- Rule 17: Action by stand-on vessel (maintain course initially)

DNV Standards:
- DNVGL-ST-0033: Maritime simulator systems
- DNV-OS-D201: Electrical installations
- DNV-RU-SHIP: Rules for classification of ships

You must be:
- CONCISE: Prioritize actionable information
- SAFETY-FOCUSED: Always prioritize collision avoidance and crew safety
- SPECIFIC: Cite relevant rules and provide quantified recommendations
- PROACTIVE: Anticipate developing situations"""

    USER_PROMPT_TEMPLATE = """## Current RAMS Status (Timestamp: {timestamp})

{rams_sections}

## Required Output

Analyze the above RAMS status and provide your assessment as JSON:

```json
{{
    "priority_ranking": ["ordered list of concerns by urgency"],
    "recommended_actions": ["specific actionable recommendations"],
    "risk_assessment": "overall risk level (LOW/MEDIUM/HIGH/CRITICAL) with brief reasoning",
    "alerts": ["any immediate alerts requiring crew attention"],
    "summary": "2-3 sentence natural language summary for bridge display"
}}
```

Focus on:
1. Any collision risks requiring COLREGS maneuvers
2. Equipment degradation affecting mission capability
3. Sensor failures impacting situational awareness
4. Maintenance tasks that could prevent future failures"""

    def __init__(self, 
                 config_path: Optional[str] = None,
                 llm_config: Optional[LLMConfig] = None,
                 supervisor_config: Optional[SupervisorConfig] = None,
                 vessel_context: Optional[VesselContext] = None):
        """
        Initialize LLM Supervisor Agent.
        
        Args:
            config_path: Path to llm_config.yaml (default: config/llm_config.yaml)
            llm_config: Override LLM configuration programmatically
            supervisor_config: Override supervisor configuration
            vessel_context: Override vessel context information
        """
        super().__init__(name="LLMSupervisorAgent", category=RAMSCategory.SAFETY)
        
        # Load configuration
        self.llm_config, self.supervisor_config, self.vessel_context = \
            self._load_config(config_path, llm_config, supervisor_config, vessel_context)
        
        # Initialize LLM client (OpenAI-compatible or Anthropic native)
        self._client = None          # OpenAI-compatible client
        self._anthropic_client = None  # Anthropic native client
        self._active_provider: str = self.llm_config.provider
        self._init_client()
        
        # State tracking
        self._last_reasoning_time = 0.0
        self._decision_history: List[SupervisorDecision] = []
        self._current_rams_state: Dict[str, Any] = {}
        
        # Statistics
        self._total_decisions = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._errors: List[str] = []
        
        # Agent goals
        self.goals = [
            "Coordinate RAMS agents for unified decision-making",
            "Prioritize safety-critical decisions",
            "Provide expert maritime reasoning with COLREGS/DNV compliance",
            "Generate actionable advisories for vessel operators"
        ]
    
    def _load_config(self, 
                     config_path: Optional[str],
                     llm_config: Optional[LLMConfig],
                     supervisor_config: Optional[SupervisorConfig],
                     vessel_context: Optional[VesselContext]
                     ) -> tuple:
        """Load configuration from YAML or use provided configs."""
        
        # Default paths
        if config_path is None:
            module_dir = Path(__file__).parent.parent
            config_path = module_dir / "config" / "llm_config.yaml"
        else:
            config_path = Path(config_path)
        
        loaded_llm = llm_config
        loaded_supervisor = supervisor_config or SupervisorConfig()
        loaded_vessel = vessel_context or VesselContext()
        
        # Try loading YAML
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if not loaded_llm and 'llm' in config:
                    llm_cfg = config['llm']
                    # Resolve environment variable for API key
                    api_key = llm_cfg.get('api_key', '')
                    if api_key.startswith('${') and api_key.endswith('}'):
                        env_var = api_key[2:-1]
                        api_key = os.environ.get(env_var, '')
                    
                    loaded_llm = LLMConfig(
                        api_key=api_key,
                        api_base=llm_cfg.get('api_base', ''),
                        model=llm_cfg.get('model', 'gpt-4'),
                        temperature=llm_cfg.get('temperature', 0.1),
                        max_tokens=llm_cfg.get('max_tokens', 4096),
                        timeout=llm_cfg.get('timeout', 60.0),
                        max_retries=llm_cfg.get('max_retries', 3),
                        retry_delay=llm_cfg.get('retry_delay', 1.0)
                    )
                
                if 'supervisor' in config:
                    sup_cfg = config['supervisor']
                    loaded_supervisor = SupervisorConfig(
                        reasoning_interval=sup_cfg.get('reasoning_interval', 5.0),
                        safety_critical_threshold=sup_cfg.get('safety_critical_threshold', 0.7),
                        rul_critical_hours=sup_cfg.get('rul_critical_hours', 100),
                        availability_threshold=sup_cfg.get('availability_threshold', 0.8),
                        verbose_mode=sup_cfg.get('verbose_mode', True),
                        include_colregs_references=sup_cfg.get('include_colregs_references', True),
                        include_dnv_references=sup_cfg.get('include_dnv_references', True)
                    )
                
                if 'vessel' in config:
                    v_cfg = config['vessel']
                    loaded_vessel = VesselContext(
                        name=v_cfg.get('name', 'Vessel'),
                        mmsi=v_cfg.get('mmsi', 258342000),
                        imo=v_cfg.get('imo', 9371361),
                        owner=v_cfg.get('owner', 'NTNU'),
                        dp_class=v_cfg.get('dp_class', 'SDP-11'),
                        operational_area=v_cfg.get('operational_area', 'Norwegian coastal waters')
                    )
                    
                print(f"[LLMSupervisor] Loaded config from {config_path}")
                
            except Exception as e:
                print(f"[LLMSupervisor] Warning: Failed to load config: {e}")
        
        # Fallback if no config loaded
        if loaded_llm is None:
            loaded_llm = LLMConfig(
                api_key=os.environ.get('NTNU_API_KEY', ''),
                api_base=os.environ.get('NTNU_API_BASE', 'https://llm.hpc.ntnu.no/v1'),
                model='moonshotai/Kimi-K2.5'
            )
        
        return loaded_llm, loaded_supervisor, loaded_vessel
    
    def _init_client(self) -> None:
        """Initialize LLM client based on provider."""
        if not self.llm_config.api_key:
            print("[LLMSupervisor] No API key configured — running in fallback mode.")
            return

        provider = self.llm_config.provider

        if provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                print("[LLMSupervisor] 'anthropic' package not installed. "
                      "Run: pip install anthropic  (or: uv sync)")
                return
            try:
                self._anthropic_client = _anthropic_sdk.Anthropic(
                    api_key=self.llm_config.api_key
                )
                self._active_provider = 'anthropic'
                print(f"[LLMSupervisor] Anthropic client ready — model: {self.llm_config.model}")
            except Exception as e:
                print(f"[LLMSupervisor] Anthropic init failed: {e}")
                self._anthropic_client = None
        else:
            # NTNU / OpenAI / Google — all use the OpenAI-compatible SDK
            if not OPENAI_AVAILABLE:
                print("[LLMSupervisor] 'openai' package not available.")
                return
            try:
                kwargs: dict = {
                    'api_key': self.llm_config.api_key,
                    'timeout': self.llm_config.timeout,
                }
                if self.llm_config.api_base:
                    kwargs['base_url'] = self.llm_config.api_base
                self._client = OpenAI(**kwargs)
                self._active_provider = provider
                print(f"[LLMSupervisor] OpenAI-compatible client ready"
                      f" ({self.llm_config.api_base or 'api.openai.com'})"
                      f" — model: {self.llm_config.model}")
            except Exception as e:
                print(f"[LLMSupervisor] Client init failed: {e}")
                self._client = None

    @property
    def is_ready(self) -> bool:
        """Check if LLM client is ready for inference."""
        return self._client is not None or self._anthropic_client is not None
    
    def perceive(self, environment_data: Dict[str, Any]) -> None:
        """
        Perceive environment data from RAMS agents.
        
        Expected keys:
        - 'safety': SafetyAgent report
        - 'reliability': ReliabilityAgent report
        - 'availability': AvailabilityAgent report
        - 'maintenance': MaintainabilityAgent report
        """
        self.update_rams_state(
            safety_report=environment_data.get('safety'),
            reliability_report=environment_data.get('reliability'),
            availability_report=environment_data.get('availability'),
            maintenance_report=environment_data.get('maintenance')
        )
    
    def update_rams_state(self,
                          safety_report: Optional[Dict[str, Any]] = None,
                          reliability_report: Optional[Dict[str, Any]] = None,
                          availability_report: Optional[Dict[str, Any]] = None,
                          maintenance_report: Optional[Dict[str, Any]] = None) -> None:
        """
        Update current RAMS state from agent reports.
        
        Args:
            safety_report: Output from SafetyAgent.act()
            reliability_report: Output from ReliabilityAgent.act()
            availability_report: Output from AvailabilityAgent.act()
            maintenance_report: Output from MaintainabilityAgent.act()
        """
        if safety_report is not None:
            self._current_rams_state['safety'] = safety_report
        if reliability_report is not None:
            self._current_rams_state['reliability'] = reliability_report
        if availability_report is not None:
            self._current_rams_state['availability'] = availability_report
        if maintenance_report is not None:
            self._current_rams_state['maintenance'] = maintenance_report
        
        self._current_rams_state['timestamp'] = time.time()
    
    def _build_rams_sections(self) -> str:
        """Build RAMS status sections for the prompt."""
        state = self._current_rams_state
        sections = []
        
        # Safety section
        if 'safety' in state:
            safety = state['safety']
            sections.append(f"""### Safety (Collision Avoidance)
- Risk Level: {safety.get('risk_level', 'UNKNOWN')}
- Safety Index: {safety.get('safety_index', 'N/A')}/100
- Active Tracks: {safety.get('active_tracks', 0)}
- RL+PSF Active: {safety.get('rl_psf_active', False)}
- Recommended Maneuvers:
{json.dumps(safety.get('maneuvers', []), indent=2)}""")
        
        # Reliability section
        if 'reliability' in state:
            rel = state['reliability']
            sections.append(f"""### Reliability (RUL Prediction)
- Overall Health: {rel.get('overall_health', 'UNKNOWN')}
- Components Status:
{json.dumps(rel.get('components', {}), indent=2)}
- Critical Alerts: {rel.get('critical_alerts', [])}
- Degradation Warnings: {rel.get('warnings', [])}""")
        
        # Availability section
        if 'availability' in state:
            avail = state['availability']
            sections.append(f"""### Availability (Sensor Redundancy)
- Operational Mode: {avail.get('mode', 'UNKNOWN')}
- System Capability: {avail.get('capability_pct', 100)}%
- Anomaly Score: {avail.get('anomaly_score', 0):.2f}
- Sensor Status:
{json.dumps(avail.get('sensors', {}), indent=2)}
- Detected Anomalies: {avail.get('anomalies', [])}""")
        
        # Maintenance section
        if 'maintenance' in state:
            maint = state['maintenance']
            sections.append(f"""### Maintainability (Scheduling)
- Pending Tasks: {maint.get('pending_tasks_count', 0)}
- Maintainability Metric (24h): {maint.get('maintainability_metric', 1.0):.2f}
- Cost Forecast:
{json.dumps(maint.get('cost_forecast', {}), indent=2)}
- Current Recommendations: {maint.get('recommendations', [])}""")
        
        if not sections:
            sections.append("No RAMS data available.")
        
        return "\n\n".join(sections)
    
    def _build_user_prompt(self) -> str:
        """Build the complete user prompt."""
        return self.USER_PROMPT_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rams_sections=self._build_rams_sections()
        )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with vessel context."""
        return self.SYSTEM_PROMPT.format(
            vessel_name=self.vessel_context.name,
            mmsi=self.vessel_context.mmsi,
            owner=self.vessel_context.owner,
            dp_class=self.vessel_context.dp_class,
            operational_area=self.vessel_context.operational_area
        )
    
    def reason(self, force: bool = False) -> Optional[SupervisorDecision]:
        """
        Perform LLM-based reasoning on current RAMS state.
        
        Args:
            force: Force reasoning even if interval hasn't elapsed
            
        Returns:
            SupervisorDecision or None if skipped/failed
        """
        current_time = time.time()
        
        # Check reasoning interval (unless forced)
        if not force:
            elapsed = current_time - self._last_reasoning_time
            if elapsed < self.supervisor_config.reasoning_interval:
                return None
        
        # Use fallback if LLM not available
        if not self.is_ready:
            return self._fallback_reasoning()
        
        # Skip if no RAMS data
        if not self._current_rams_state or len(self._current_rams_state) <= 1:
            return None
        
        self._last_reasoning_time = current_time
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()
        
        # Call LLM with retries
        start_time = time.time()
        response_text = None
        tokens_used = 0
        
        for attempt in range(self.llm_config.max_retries):
            try:
                if self._active_provider == 'anthropic':
                    # Anthropic native SDK — system prompt is a top-level param
                    msg = self._anthropic_client.messages.create(
                        model=self.llm_config.model,
                        max_tokens=self.llm_config.max_tokens,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    response_text = msg.content[0].text
                    tokens_used = (msg.usage.input_tokens or 0) + (msg.usage.output_tokens or 0)
                else:
                    # OpenAI-compatible: NTNU / OpenAI / Google
                    response = self._client.chat.completions.create(
                        model=self.llm_config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.llm_config.temperature,
                        max_tokens=self.llm_config.max_tokens
                    )
                    response_text = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else 0
                break

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.llm_config.max_retries} failed: {e}"
                print(f"[LLMSupervisor] {error_msg}")
                self._errors.append(error_msg)
                
                if attempt < self.llm_config.max_retries - 1:
                    time.sleep(self.llm_config.retry_delay)
                else:
                    return self._fallback_reasoning()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        decision = self._parse_response(response_text, tokens_used, latency_ms)
        
        # Update statistics
        self._total_decisions += 1
        self._total_tokens += tokens_used
        self._total_latency_ms += latency_ms
        self._decision_history.append(decision)
        
        # Keep history bounded
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]
        
        return decision
    
    def _parse_response(self, 
                        response_text: str, 
                        tokens: int, 
                        latency_ms: float) -> SupervisorDecision:
        """Parse LLM response into structured decision."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response_text
            
            # Remove markdown code block if present
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                if end > start:
                    text = text[start:end].strip()
            elif '```' in text:
                start = text.find('```') + 3
                end = text.find('```', start)
                if end > start:
                    text = text[start:end].strip()
            
            # Find JSON object
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                
                return SupervisorDecision(
                    timestamp=time.time(),
                    priority_ranking=data.get('priority_ranking', []),
                    recommended_actions=data.get('recommended_actions', []),
                    risk_assessment=data.get('risk_assessment', 'Unable to assess'),
                    alerts=data.get('alerts', []),
                    natural_language_summary=data.get('summary', ''),
                    reasoning_tokens=tokens,
                    latency_ms=latency_ms
                )
                
        except json.JSONDecodeError as e:
            print(f"[LLMSupervisor] JSON parse error: {e}")
        except Exception as e:
            print(f"[LLMSupervisor] Parse error: {e}")
        
        # Fallback: use raw response as summary
        return SupervisorDecision(
            timestamp=time.time(),
            priority_ranking=[],
            recommended_actions=[],
            risk_assessment="Response parsing failed - see summary",
            alerts=[],
            natural_language_summary=response_text[:500] if response_text else "No response",
            reasoning_tokens=tokens,
            latency_ms=latency_ms
        )
    
    def _fallback_reasoning(self) -> SupervisorDecision:
        """
        Rule-based fallback reasoning when LLM is unavailable.
        
        Provides basic prioritization based on thresholds.
        """
        alerts = []
        actions = []
        priorities = []
        risk_level = "LOW"
        
        state = self._current_rams_state
        
        # Check safety (highest priority)
        if 'safety' in state:
            safety = state['safety']
            safety_risk = safety.get('risk_level', 'LOW')
            safety_index = safety.get('safety_index', 100)
            
            if safety_risk in ['CRITICAL']:
                risk_level = "CRITICAL"
                priorities.insert(0, f"SAFETY CRITICAL: Collision risk {safety_risk}")
                alerts.append(f"IMMEDIATE ACTION: Collision risk - execute COLREGS maneuver")
                for m in safety.get('maneuvers', [])[:2]:
                    actions.append(f"Execute: {m.get('action', 'maneuver')}")
                    
            elif safety_risk == 'HIGH' or safety_index < self.supervisor_config.safety_critical_threshold * 100:
                if risk_level != "CRITICAL":
                    risk_level = "HIGH"
                priorities.append(f"SAFETY HIGH: Risk level {safety_risk}, index {safety_index}")
                alerts.append(f"Collision risk elevated - monitor closely")
        
        # Check reliability
        if 'reliability' in state:
            rel = state['reliability']
            for alert in rel.get('critical_alerts', []):
                priorities.append(f"RELIABILITY: {alert}")
                alerts.append(f"Equipment alert: {alert}")
                if risk_level not in ["CRITICAL", "HIGH"]:
                    risk_level = "MEDIUM"
        
        # Check availability
        if 'availability' in state:
            avail = state['availability']
            cap = avail.get('capability_pct', 100)
            
            if cap < self.supervisor_config.availability_threshold * 100:
                priorities.append(f"AVAILABILITY: System at {cap}% capability")
                alerts.append(f"Reduced operational capability: {cap}%")
                actions.append("Consider mission abort if capability continues degrading")
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
        
        # Check maintenance
        if 'maintenance' in state:
            maint = state['maintenance']
            pending = maint.get('pending_tasks_count', 0)
            
            if pending > 0:
                priorities.append(f"MAINTENANCE: {pending} tasks pending")
                for rec in maint.get('recommendations', [])[:2]:
                    if 'CRITICAL' in rec.upper():
                        actions.append(rec)
        
        # Build summary
        if not priorities:
            summary = "All systems nominal. No immediate concerns."
        else:
            summary = f"Risk level: {risk_level}. {len(priorities)} concern(s) identified. " + \
                     "; ".join(priorities[:2])
        
        return SupervisorDecision(
            timestamp=time.time(),
            priority_ranking=priorities,
            recommended_actions=actions,
            risk_assessment=f"{risk_level} - Rule-based assessment (LLM unavailable)",
            alerts=alerts,
            natural_language_summary=summary,
            reasoning_tokens=0,
            latency_ms=0.0
        )
    
    def act(self) -> Dict[str, Any]:
        """
        Generate supervisor advisory.
        
        Returns:
            Dict with decision summary, recommendations, and metadata
        """
        decision = self.reason()
        
        if decision is None:
            return {
                'status': 'skipped',
                'reason': 'Reasoning interval not elapsed or no data available',
                'llm_ready': self.is_ready
            }
        
        return {
            'status': 'complete',
            'timestamp': datetime.fromtimestamp(decision.timestamp).isoformat(),
            'priority_ranking': decision.priority_ranking,
            'recommended_actions': decision.recommended_actions,
            'risk_assessment': decision.risk_assessment,
            'alerts': decision.alerts,
            'summary': decision.natural_language_summary,
            'metadata': {
                'tokens_used': decision.reasoning_tokens,
                'latency_ms': round(decision.latency_ms, 1),
                'llm_used': decision.reasoning_tokens > 0,
                'total_decisions': self._total_decisions
            }
        }
    
    def get_latest_decision(self) -> Optional[SupervisorDecision]:
        """Get the most recent decision."""
        return self._decision_history[-1] if self._decision_history else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get supervisor usage statistics."""
        avg_latency = self._total_latency_ms / max(1, self._total_decisions)
        avg_tokens = self._total_tokens / max(1, self._total_decisions)
        
        return {
            'total_decisions': self._total_decisions,
            'total_tokens': self._total_tokens,
            'avg_tokens_per_decision': round(avg_tokens, 1),
            'avg_latency_ms': round(avg_latency, 1),
            'llm_ready': self.is_ready,
            'model': self.llm_config.model,
            'api_base': self.llm_config.api_base,
            'errors': self._errors[-5:]  # Last 5 errors
        }
    
    def get_rams_contribution(self) -> float:
        """
        Get supervisor's contribution to RAMS metrics.
        
        Returns 1.0 if functioning normally, degraded otherwise.
        """
        if not self.is_ready:
            return 0.7  # Fallback mode available
        return 1.0
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self._total_decisions = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._errors.clear()
        self._decision_history.clear()


# Convenience function for quick initialization
def create_llm_supervisor(config_path: Optional[str] = None) -> LLMSupervisorAgent:
    """
    Factory function to create LLM Supervisor Agent.
    
    Args:
        config_path: Optional path to llm_config.yaml
        
    Returns:
        Configured LLMSupervisorAgent instance
    """
    return LLMSupervisorAgent(config_path=config_path)
