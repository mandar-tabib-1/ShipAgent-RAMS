# test_all_modules.py
import sys
print('=' * 60)
print('MODULE IMPORT & INFERENCE TESTS')
print('=' * 60)

results = {'pass': [], 'fail': []}

# Test 1: PyTorch
print('\n[1] PyTorch:')
try:
    import torch
    print(f'    ✓ Version: {torch.__version__}')
    results['pass'].append('PyTorch')
except Exception as e:
    print(f'    ✗ {e}')
    results['fail'].append(f'PyTorch: {e}')

# Test 2: Navigation Agents
print('\n[2] Navigation Agents:')
for name, path in [
    ('SensorFusionAgentKalman', 'agents.sensor_fusion_agent'),
    ('CollisionAvoidanceAgent', 'agents.collision_avoidance_agent'),
    ('DPAgent', 'agents.dp_agent'),
    ('PropulsionHealthAgent', 'agents.propulsion_health_agent'),
]:
    try:
        mod = __import__(path, fromlist=[name])
        getattr(mod, name)
        print(f'    ✓ {name}')
        results['pass'].append(name)
    except Exception as e:
        print(f'    ✗ {name}: {e}')
        results['fail'].append(f'{name}: {e}')

# Test 3: RAMS Agents
print('\n[3] RAMS Agents:')
for name, path in [
    ('SafetyAgent', 'rams_agents.safety_agent'),
    ('ReliabilityAgent', 'rams_agents.reliability_agent'),
    ('AvailabilityAgent', 'rams_agents.availability_agent'),
    ('MaintainabilityAgent', 'rams_agents.maintainability_agent'),
    ('SupervisorAgent', 'rams_agents.supervisor_agent'),
]:
    try:
        mod = __import__(path, fromlist=[name])
        getattr(mod, name)
        print(f'    ✓ {name}')
        results['pass'].append(name)
    except Exception as e:
        print(f'    ✗ {name}: {e}')
        results['fail'].append(f'{name}: {e}')

# Test 4: RL+PSF Components
print('\n[4] RL+PSF Components:')
for name, path in [
    ('RLCollisionPolicy', 'rams_agents.ml_models.rl_collision_policy'),
    ('PotentialSafetyFunction', 'rams_agents.psf_filter'),
    ('RLObservationAdapter', 'rams_agents.rl_observation_adapter'),
]:
    try:
        mod = __import__(path, fromlist=[name])
        getattr(mod, name)
        print(f'    ✓ {name}')
        results['pass'].append(name)
    except Exception as e:
        print(f'    ✗ {name}: {e}')
        results['fail'].append(f'{name}: {e}')

# Test 5: RL Inference
print('\n[5] RL+PSF Inference Test:')
try:
    from rams_agents.safety_agent import SafetyAgent
    agent = SafetyAgent(use_rl_policy=True)
    print(f'    ✓ SafetyAgent created with RL: {agent.uses_rl_policy}')
    results['pass'].append('RL Inference')
except Exception as e:
    print(f'    ✗ RL Inference: {e}')
    results['fail'].append(f'RL Inference: {e}')

# Summary
print('\n' + '=' * 60)
print(f'SUMMARY: {len(results["pass"])} passed, {len(results["fail"])} failed')
print('=' * 60)
if results['fail']:
    print('\nFailed modules:')
    for f in results['fail']:
        print(f'  - {f}')