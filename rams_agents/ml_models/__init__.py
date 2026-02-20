"""
ML Models for RAMS Agents

Provides deep learning models for:
- RUL Prediction (LSTM-based)
- Anomaly Detection (Autoencoder-based)
- RL-based Collision Avoidance (PPO policy)

Usage:
    from rams_agents.ml_models import (
        LSTMRULPredictor,
        AutoencoderAnomalyDetector,
        RLCollisionPolicy,
        load_trained_models
    )
    
    # Load pre-trained models
    rul_model, anomaly_model = load_trained_models()
    
    # Load RL policy for collision avoidance
    rl_policy = RLCollisionPolicy()
    rl_policy.load()
    
    # Or train new models
    python -m rams_agents.ml_models.train_models --model all

RL Collision Policy Attribution:
    Based on Acmece/rl-collision-avoidance (https://github.com/Acmece/rl-collision-avoidance)
    Paper: "Towards Optimally Decentralized Multi-Robot Collision Avoidance 
            via Deep Reinforcement Learning" (arXiv:1709.10082)
"""

import os
from pathlib import Path
from typing import Tuple, Optional

# Import models
from .lstm_rul import LSTMRULPredictor, RULPrediction, check_pytorch_available
from .autoencoder_anomaly import AutoencoderAnomalyDetector, AnomalyResult, BatchAnomalyResult

# Import RL collision policy (with fallback)
try:
    from .rl_collision_policy import RLCollisionPolicy, RLAction, get_rl_policy
    RL_POLICY_AVAILABLE = True
except ImportError:
    RL_POLICY_AVAILABLE = False
    RLCollisionPolicy = None
    RLAction = None
    get_rl_policy = None

__all__ = [
    'LSTMRULPredictor',
    'RULPrediction',
    'AutoencoderAnomalyDetector', 
    'AnomalyResult',
    'BatchAnomalyResult',
    'RLCollisionPolicy',
    'RLAction',
    'get_rl_policy',
    'load_trained_models',
    'check_pytorch_available',
    'RL_POLICY_AVAILABLE'
]


def get_models_dir() -> Path:
    """Get the models directory path."""
    # Look in project_root/models/
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    models_dir = project_root / "models"
    return models_dir


def load_trained_models(
    models_dir: Optional[Path] = None
) -> Tuple[Optional[LSTMRULPredictor], Optional[AutoencoderAnomalyDetector]]:
    """
    Load pre-trained ML models from disk.
    
    Args:
        models_dir: Optional custom models directory path
        
    Returns:
        Tuple of (LSTMRULPredictor or None, AutoencoderAnomalyDetector or None)
        Returns None for models that couldn't be loaded
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    rul_model = None
    anomaly_model = None
    
    # Load LSTM RUL model
    lstm_path = models_dir / "lstm_rul.pkl"
    if lstm_path.exists():
        try:
            rul_model = LSTMRULPredictor()
            if not rul_model.load(str(lstm_path)):
                rul_model = None
        except Exception as e:
            print(f"Warning: Failed to load LSTM model: {e}")
            rul_model = None
    
    # Load Autoencoder model
    ae_path = models_dir / "autoencoder_anomaly.pkl"
    if ae_path.exists():
        try:
            anomaly_model = AutoencoderAnomalyDetector()
            if not anomaly_model.load(str(ae_path)):
                anomaly_model = None
        except Exception as e:
            print(f"Warning: Failed to load Autoencoder model: {e}")
            anomaly_model = None
    
    return rul_model, anomaly_model


def models_available() -> dict:
    """
    Check which trained models are available.
    
    Returns:
        Dict with model availability status
    """
    models_dir = get_models_dir()
    
    return {
        'lstm_rul': (models_dir / "lstm_rul.pkl").exists(),
        'autoencoder_anomaly': (models_dir / "autoencoder_anomaly.pkl").exists(),
        'pytorch_available': check_pytorch_available()
    }
