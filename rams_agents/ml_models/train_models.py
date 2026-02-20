"""
ML Model Training Script for RAMS Agents

Trains and saves ML models for:
- LSTM RUL Predictor (Reliability/Maintainability)
- Autoencoder Anomaly Detector (Reliability/Availability)

Uses UCI Naval Propulsion dataset loaded via uci_naval_loader.

Usage:
    python -m rams_agents.ml_models.train_models --model all
    python -m rams_agents.ml_models.train_models --model lstm
    python -m rams_agents.ml_models.train_models --model autoencoder
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directories to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from rams_agents.ml_models.lstm_rul import LSTMRULPredictor, check_pytorch_available
from rams_agents.ml_models.autoencoder_anomaly import AutoencoderAnomalyDetector


def get_models_dir() -> Path:
    """Get models directory path."""
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def load_naval_data() -> Optional[Dict[str, np.ndarray]]:
    """
    Load UCI Naval Propulsion dataset.

    Returns:
        Dict with 'features', 'kMc', 'kMt' arrays or None if failed
    """
    try:
        from rams_agents.data_loaders.uci_naval_loader import UCINavalPropulsionLoader

        loader = UCINavalPropulsionLoader()
        df = loader.load()

        if df is None or len(df) == 0:
            print("Failed to load UCI Naval data")
            return None

        # Feature columns in standard order (16 operational measurements)
        feature_cols = [
            'lever_position', 'ship_speed', 'gt_shaft_torque', 'gt_revolutions',
            'gg_revolutions', 'starboard_torque', 'port_torque', 'hp_turbine_temp',
            'compressor_inlet_temp', 'compressor_outlet_temp', 'hp_turbine_pressure',
            'compressor_inlet_pressure', 'compressor_outlet_pressure', 'exhaust_pressure',
            'turbine_injection_control', 'fuel_flow'
        ]

        available = [c for c in feature_cols if c in df.columns]
        if len(available) < 14:
            print(f"Insufficient feature columns found ({len(available)}): {available}")
            return None

        features: np.ndarray = np.array(df[available].values, dtype=np.float32)
        kMc: np.ndarray = np.array(df['compressor_decay'].values, dtype=np.float32)
        kMt: np.ndarray = np.array(df['turbine_decay'].values, dtype=np.float32)

        return {
            'features': features,
            'kMc': kMc,
            'kMt': kMt
        }

    except ImportError as e:
        print(f"Import error: {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def generate_synthetic_data(n_samples: int = 5000) -> Dict[str, np.ndarray]:
    """
    Generate synthetic naval propulsion data for testing.
    
    Used when UCI data is not available.
    """
    print("Generating synthetic data for training...")
    
    np.random.seed(42)
    
    # Feature ranges (approximate UCI ranges)
    features = np.zeros((n_samples, 16), dtype=np.float32)
    
    # Operating condition progression
    time = np.linspace(0, 1, n_samples)
    
    # Lever position: varies with operation
    features[:, 0] = np.clip(3 + 6 * np.sin(2 * np.pi * time * 5) + np.random.randn(n_samples) * 0.5, 1, 9)
    
    # Ship speed: correlates with lever
    features[:, 1] = 3 + features[:, 0] * 2.5 + np.random.randn(n_samples) * 0.5
    
    # GT shaft torque
    features[:, 2] = 10 + features[:, 0] * 8 + np.random.randn(n_samples) * 2
    
    # GT revolutions
    features[:, 3] = 2500 + features[:, 0] * 200 + np.random.randn(n_samples) * 50
    
    # GG revolutions
    features[:, 4] = 7000 + features[:, 0] * 400 + np.random.randn(n_samples) * 100
    
    # Prop torques
    features[:, 5] = 50 + features[:, 0] * 30 + np.random.randn(n_samples) * 5
    features[:, 6] = 50 + features[:, 0] * 30 + np.random.randn(n_samples) * 5
    
    # HP turbine temp
    features[:, 7] = 900 + features[:, 0] * 30 + np.random.randn(n_samples) * 10
    
    # Compressor temps
    features[:, 8] = 280 + features[:, 0] * 5 + np.random.randn(n_samples) * 3
    features[:, 9] = 400 + features[:, 0] * 10 + np.random.randn(n_samples) * 5
    
    # Pressures
    features[:, 10] = 0.5 + features[:, 0] * 0.05 + np.random.randn(n_samples) * 0.02
    features[:, 11] = 1.0 + np.random.randn(n_samples) * 0.01
    features[:, 12] = 10 + features[:, 0] * 1.5 + np.random.randn(n_samples) * 0.3
    features[:, 13] = 1.02 + np.random.randn(n_samples) * 0.01
    
    # Injection control
    features[:, 14] = features[:, 0] / 9.0 * 0.15 + np.random.randn(n_samples) * 0.01
    
    # Fuel flow
    features[:, 15] = 0.1 + features[:, 0] * 0.15 + np.random.randn(n_samples) * 0.02
    
    # Degradation coefficients (gradually decrease)
    base_kMc = 1.0 - 0.00001 * np.arange(n_samples)
    base_kMt = 1.0 - 0.000005 * np.arange(n_samples)
    
    # Add some variability
    kMc = base_kMc + np.random.randn(n_samples) * 0.001
    kMt = base_kMt + np.random.randn(n_samples) * 0.0005
    
    kMc = np.clip(kMc, 0.95, 1.0)
    kMt = np.clip(kMt, 0.975, 1.0)
    
    return {
        'features': features,
        'kMc': kMc.astype(np.float32),
        'kMt': kMt.astype(np.float32)
    }


def train_lstm_rul(data: Dict[str, np.ndarray], 
                   save_path: Path,
                   epochs: int = 50,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Train LSTM RUL predictor.
    
    Args:
        data: Dict with 'features', 'kMc', 'kMt'
        save_path: Path to save trained model
        epochs: Training epochs
        verbose: Print progress
        
    Returns:
        Training history dict
    """
    print("\n" + "="*60)
    print("Training LSTM RUL Model")
    print("="*60)
    
    if not check_pytorch_available():
        print("ERROR: PyTorch not available. Install with: pip install torch")
        return {'error': 'PyTorch not available'}
    
    predictor = LSTMRULPredictor(
        sequence_length=50,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"Data shape: {data['features'].shape}")
    print(f"kMc range: [{data['kMc'].min():.4f}, {data['kMc'].max():.4f}]")
    print(f"kMt range: [{data['kMt'].min():.4f}, {data['kMt'].max():.4f}]")
    
    try:
        history = predictor.train(
            data=data['features'],
            kMc=data['kMc'],
            kMt=data['kMt'],
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            val_split=0.2,
            verbose=verbose
        )
        
        # Save model
        predictor.save(str(save_path))
        print(f"\nModel saved to: {save_path}")
        print(f"Best validation loss: {history['best_val_loss']:.6f}")
        
        return history
        
    except Exception as e:
        print(f"Training failed: {e}")
        return {'error': str(e)}


def train_autoencoder(data: Dict[str, np.ndarray],
                      save_path: Path,
                      epochs: int = 50,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    Train autoencoder anomaly detector.
    
    Args:
        data: Dict with 'features'
        save_path: Path to save trained model
        epochs: Training epochs
        verbose: Print progress
        
    Returns:
        Training history dict
    """
    print("\n" + "="*60)
    print("Training Autoencoder Anomaly Detector")
    print("="*60)
    
    if not check_pytorch_available():
        print("ERROR: PyTorch not available. Install with: pip install torch")
        return {'error': 'PyTorch not available'}
    
    detector = AutoencoderAnomalyDetector(
        input_size=16,
        encoding_dims=[8, 4],
        dropout=0.1,
        threshold_percentile=95.0
    )
    
    print(f"Data shape: {data['features'].shape}")
    
    try:
        history = detector.fit(
            data=data['features'],
            epochs=epochs,
            batch_size=64,
            learning_rate=0.001,
            verbose=verbose
        )
        
        # Save model
        detector.save(str(save_path))
        print(f"\nModel saved to: {save_path}")
        print(f"Final loss: {history['final_loss']:.6f}")
        print(f"Anomaly threshold: {history['threshold']:.6f}")
        
        # Test detection
        test_result = detector.detect_batch(data['features'][:1000])
        print(f"Anomaly rate on training data: {test_result.anomaly_rate:.2%}")
        
        return history
        
    except Exception as e:
        print(f"Training failed: {e}")
        return {'error': str(e)}


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train RAMS ML models'
    )
    parser.add_argument(
        '--model', 
        choices=['lstm', 'autoencoder', 'all'],
        default='all',
        help='Model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of UCI dataset'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Check PyTorch
    if not check_pytorch_available():
        print("="*60)
        print("WARNING: PyTorch not installed!")
        print("Install with: pip install torch")
        print("="*60)
        return 1
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    if args.synthetic:
        data = generate_synthetic_data(5000)
    else:
        data = load_naval_data()
        if data is None:
            print("UCI data not available, using synthetic data")
            data = generate_synthetic_data(5000)
    
    print(f"Loaded {len(data['features'])} samples")
    
    # Get models directory
    models_dir = get_models_dir()

    # Save a held-out test split (last 20%) before training
    n_total = len(data['features'])
    n_test = int(n_total * 0.2)
    test_data = {
        'features': data['features'][-n_test:],
        'kMc': data['kMc'][-n_test:],
        'kMt': data['kMt'][-n_test:]
    }
    train_data = {
        'features': data['features'][:-n_test],
        'kMc': data['kMc'][:-n_test],
        'kMt': data['kMt'][:-n_test]
    }

    np.save(str(models_dir / 'test_X.npy'), test_data['features'])
    np.save(str(models_dir / 'test_kMc.npy'), test_data['kMc'])
    np.save(str(models_dir / 'test_kMt.npy'), test_data['kMt'])
    print(f"\nTest data saved: {n_test} samples -> {models_dir}/test_X.npy, test_kMc.npy, test_kMt.npy")
    print(f"Train data: {len(train_data['features'])} samples")

    # Train models
    results = {}

    if args.model in ['lstm', 'all']:
        lstm_path = models_dir / 'lstm_rul.pkl'
        results['lstm'] = train_lstm_rul(
            train_data, lstm_path,
            epochs=args.epochs,
            verbose=not args.quiet
        )

    if args.model in ['autoencoder', 'all']:
        ae_path = models_dir / 'autoencoder_anomaly.pkl'
        results['autoencoder'] = train_autoencoder(
            train_data, ae_path,
            epochs=args.epochs,
            verbose=not args.quiet
        )

    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    for model_name, result in results.items():
        if 'error' in result:
            print(f"  {model_name}: FAILED - {result['error']}")
        else:
            print(f"  {model_name}: SUCCESS")

    print(f"\nAll artifacts saved to: {models_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
