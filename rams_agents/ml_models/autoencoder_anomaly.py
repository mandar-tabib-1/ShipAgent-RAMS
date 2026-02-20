"""
Autoencoder-based Anomaly Detector for Naval Propulsion Systems

Deep learning model for multivariate anomaly detection using reconstruction error.
Learns normal operational patterns and detects deviations.

Architecture:
- Encoder: 16 → 8 → 4 (bottleneck)
- Decoder: 4 → 8 → 16
- Anomaly score = reconstruction error (MSE)

Part of RAMS ML enhancement for Reliability and Availability agents.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os
import warnings

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Autoencoder will use fallback mode.")


@dataclass
class AnomalyResult:
    """Anomaly detection result for a single sample."""
    is_anomaly: bool
    anomaly_score: float  # Reconstruction error
    threshold: float
    confidence: float  # How far above/below threshold
    top_anomalous_features: List[Tuple[str, float]]  # (feature_name, error)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'threshold': self.threshold,
            'confidence': self.confidence,
            'top_features': self.top_anomalous_features
        }


@dataclass
class BatchAnomalyResult:
    """Batch anomaly detection results."""
    n_anomalies: int
    n_total: int
    anomaly_rate: float
    mean_score: float
    max_score: float
    scores: np.ndarray
    anomaly_indices: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_anomalies': self.n_anomalies,
            'n_total': self.n_total,
            'anomaly_rate': self.anomaly_rate,
            'mean_score': self.mean_score,
            'max_score': self.max_score,
            'anomaly_indices': self.anomaly_indices
        }


if TORCH_AVAILABLE:
    class AutoencoderNetwork(nn.Module):
        """Autoencoder neural network."""
        
        def __init__(self, 
                     input_size: int = 16,
                     encoding_dims: List[int] = [8, 4],
                     dropout: float = 0.1):
            super().__init__()
            
            self.input_size = input_size
            self.encoding_dims = encoding_dims
            
            # Build encoder
            encoder_layers = []
            prev_dim = input_size
            for dim in encoding_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = dim
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Build decoder (reverse of encoder)
            decoder_layers = []
            decoding_dims = encoding_dims[-2::-1] + [input_size]  # Reverse minus bottleneck
            prev_dim = encoding_dims[-1]  # Start from bottleneck
            for i, dim in enumerate(decoding_dims):
                decoder_layers.append(nn.Linear(prev_dim, dim))
                if i < len(decoding_dims) - 1:  # No activation on output
                    decoder_layers.extend([
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                prev_dim = dim
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def encode(self, x):
            return self.encoder(x)


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector for Naval Propulsion data.
    
    Training approach:
    1. Train on normal operational data only
    2. Learn to reconstruct normal patterns
    3. Anomalies have high reconstruction error
    
    Threshold estimation:
    - Use percentile of training reconstruction errors
    - Default: 95th percentile
    
    Features (UCI Naval dataset):
    - 16 operational features
    - Excludes degradation coefficients (those are targets)
    """
    
    FEATURE_NAMES = [
        'lever_position',
        'ship_speed',
        'gt_shaft_torque',
        'gt_revolutions',
        'gg_revolutions',
        'stbd_prop_torque',
        'port_prop_torque',
        'hp_turbine_temp',
        'gt_compressor_inlet_temp',
        'gt_compressor_outlet_temp',
        'hp_turbine_pressure',
        'gt_compressor_inlet_pressure',
        'gt_compressor_outlet_pressure',
        'gt_exhaust_pressure',
        'turbine_injection_control',
        'fuel_flow'
    ]
    
    def __init__(self,
                 input_size: int = 16,
                 encoding_dims: List[int] = [8, 4],
                 dropout: float = 0.1,
                 threshold_percentile: float = 95.0):
        """
        Initialize autoencoder anomaly detector.
        
        Args:
            input_size: Number of input features
            encoding_dims: Dimensions of encoder layers (last is bottleneck)
            dropout: Dropout probability
            threshold_percentile: Percentile for anomaly threshold
        """
        self.input_size = input_size
        self.encoding_dims = encoding_dims
        self.dropout = dropout
        self.threshold_percentile = threshold_percentile
        
        # Model
        self.model: Optional['AutoencoderNetwork'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.is_trained = False
        
        # Normalization
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
        # Threshold
        self.threshold: float = 0.0
        self.train_errors: Optional[np.ndarray] = None
        
        # Training history
        self.train_losses: List[float] = []
    
    def fit(self,
            data: np.ndarray,
            epochs: int = 50,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Train the autoencoder on normal data.
        
        Args:
            data: (N, input_size) array of normal operational data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        
        # Use only first input_size features
        if data.shape[1] > self.input_size:
            data = data[:, :self.input_size]
        
        # Normalize
        self.feature_mean = np.mean(data, axis=0)
        self.feature_std = np.std(data, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        
        normalized_data = (data - self.feature_mean) / self.feature_std
        
        # Create dataset
        tensor_data = torch.FloatTensor(normalized_data)
        dataset = TensorDataset(tensor_data, tensor_data)  # Input = target
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = AutoencoderNetwork(
            input_size=self.input_size,
            encoding_dims=self.encoding_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            epoch_loss /= len(dataset)
            self.train_losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
        
        # Compute reconstruction errors on training data
        self.model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(normalized_data).to(self.device)
            reconstructed = self.model(all_data).cpu().numpy()
            
            # Per-sample reconstruction error (MSE)
            self.train_errors = np.mean((normalized_data - reconstructed) ** 2, axis=1)
        
        # Set threshold
        self.threshold = np.percentile(self.train_errors, self.threshold_percentile)
        
        self.is_trained = True
        
        return {
            'train_losses': self.train_losses,
            'final_loss': self.train_losses[-1],
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'mean_train_error': np.mean(self.train_errors),
            'std_train_error': np.std(self.train_errors),
            'n_samples': len(data)
        }
    
    def detect(self, sample: np.ndarray) -> AnomalyResult:
        """
        Detect if a single sample is anomalous.
        
        Args:
            sample: (input_size,) or (1, input_size) array
            
        Returns:
            AnomalyResult
        """
        if not self.is_trained:
            return self._fallback_detection(sample)
        
        if not TORCH_AVAILABLE:
            return self._fallback_detection(sample)
        
        # Ensure 2D
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Use only required features
        if sample.shape[1] > self.input_size:
            sample = sample[:, :self.input_size]
        
        # Normalize
        normalized = (sample - self.feature_mean) / self.feature_std
        
        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            tensor_sample = torch.FloatTensor(normalized).to(self.device)
            reconstructed = self.model(tensor_sample).cpu().numpy()
        
        # Compute errors
        per_feature_error = (normalized - reconstructed) ** 2
        total_error = np.mean(per_feature_error)
        
        # Check anomaly
        is_anomaly = total_error > self.threshold
        
        # Confidence (how far from threshold)
        if is_anomaly:
            confidence = min(1.0, (total_error - self.threshold) / self.threshold)
        else:
            confidence = 1.0 - (total_error / self.threshold)
        
        # Top anomalous features
        feature_errors = per_feature_error.flatten()
        sorted_indices = np.argsort(feature_errors)[::-1]
        top_features = [
            (self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"feature_{i}",
             float(feature_errors[i]))
            for i in sorted_indices[:5]
        ]
        
        return AnomalyResult(
            is_anomaly=bool(is_anomaly),
            anomaly_score=float(total_error),
            threshold=self.threshold,
            confidence=float(confidence),
            top_anomalous_features=top_features
        )
    
    def detect_batch(self, data: np.ndarray) -> BatchAnomalyResult:
        """
        Detect anomalies in a batch of samples.
        
        Args:
            data: (N, input_size) array
            
        Returns:
            BatchAnomalyResult
        """
        if not self.is_trained or not TORCH_AVAILABLE:
            # Fallback
            scores = np.zeros(len(data))
            return BatchAnomalyResult(
                n_anomalies=0,
                n_total=len(data),
                anomaly_rate=0.0,
                mean_score=0.0,
                max_score=0.0,
                scores=scores,
                anomaly_indices=[]
            )
        
        # Use only required features
        if data.shape[1] > self.input_size:
            data = data[:, :self.input_size]
        
        # Normalize
        normalized = (data - self.feature_mean) / self.feature_std
        
        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(normalized).to(self.device)
            reconstructed = self.model(tensor_data).cpu().numpy()
        
        # Compute errors
        errors = np.mean((normalized - reconstructed) ** 2, axis=1)
        
        # Find anomalies
        anomaly_mask = errors > self.threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        return BatchAnomalyResult(
            n_anomalies=int(np.sum(anomaly_mask)),
            n_total=len(data),
            anomaly_rate=float(np.mean(anomaly_mask)),
            mean_score=float(np.mean(errors)),
            max_score=float(np.max(errors)),
            scores=errors,
            anomaly_indices=anomaly_indices
        )
    
    def get_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """
        Get reconstruction errors for data.
        
        Args:
            data: (N, input_size) array
            
        Returns:
            (N,) array of reconstruction errors
        """
        if not self.is_trained or not TORCH_AVAILABLE:
            return np.zeros(len(data))
        
        # Use only required features
        if data.shape[1] > self.input_size:
            data = data[:, :self.input_size]
        
        # Normalize
        normalized = (data - self.feature_mean) / self.feature_std
        
        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(normalized).to(self.device)
            reconstructed = self.model(tensor_data).cpu().numpy()
        
        # Compute per-sample MSE
        return np.mean((normalized - reconstructed) ** 2, axis=1)
    
    def _fallback_detection(self, sample: np.ndarray) -> AnomalyResult:
        """Fallback detection using z-score."""
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        if sample.shape[1] > self.input_size:
            sample = sample[:, :self.input_size]
        
        # Simple z-score based detection
        if self.feature_mean is not None:
            z_scores = np.abs((sample - self.feature_mean) / (self.feature_std + 1e-6))
            max_z = np.max(z_scores)
            is_anomaly = max_z > 3.0
            score = float(max_z / 3.0)
        else:
            is_anomaly = False
            score = 0.0
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            threshold=1.0,
            confidence=0.5,
            top_anomalous_features=[]
        )
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_dict = {
            'input_size': self.input_size,
            'encoding_dims': self.encoding_dims,
            'dropout': self.dropout,
            'threshold_percentile': self.threshold_percentile,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'threshold': self.threshold,
            'is_trained': self.is_trained,
            'train_losses': self.train_losses
        }
        
        if TORCH_AVAILABLE and self.model is not None:
            save_dict['model_state'] = self.model.state_dict()
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.input_size = save_dict['input_size']
        self.encoding_dims = save_dict['encoding_dims']
        self.dropout = save_dict['dropout']
        self.threshold_percentile = save_dict['threshold_percentile']
        self.feature_mean = save_dict['feature_mean']
        self.feature_std = save_dict['feature_std']
        self.threshold = save_dict['threshold']
        self.is_trained = save_dict['is_trained']
        self.train_losses = save_dict.get('train_losses', [])
        
        if TORCH_AVAILABLE and 'model_state' in save_dict:
            self.model = AutoencoderNetwork(
                input_size=self.input_size,
                encoding_dims=self.encoding_dims,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(save_dict['model_state'])
        
        return True
    
    def update_threshold(self, new_percentile: float) -> float:
        """
        Update anomaly threshold to new percentile.
        
        Args:
            new_percentile: New percentile (0-100)
            
        Returns:
            New threshold value
        """
        if self.train_errors is not None:
            self.threshold_percentile = new_percentile
            self.threshold = np.percentile(self.train_errors, new_percentile)
        return self.threshold


def check_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE
