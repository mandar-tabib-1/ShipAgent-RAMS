"""
LSTM-based Remaining Useful Life (RUL) Predictor

Deep learning model for RUL estimation using UCI Naval Propulsion data.
Uses LSTM layers to capture temporal degradation patterns.

Architecture:
- Input: Sequence of operational features over time window
- LSTM: 2 layers with 64 hidden units
- Output: RUL estimate with uncertainty (via Monte Carlo Dropout)

Part of RAMS ML enhancement for Reliability and Maintainability agents.
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
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. LSTM RUL model will use fallback mode.")


@dataclass
class RULPrediction:
    """RUL prediction result with uncertainty."""
    rul_hours: float
    uncertainty: float  # Standard deviation
    confidence_interval: Tuple[float, float]  # 95% CI
    confidence_level: str  # 'low', 'medium', 'high'
    model_type: str = "lstm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rul_hours': self.rul_hours,
            'uncertainty': self.uncertainty,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'model_type': self.model_type
        }


if TORCH_AVAILABLE:
    class NavalPropulsionDataset(Dataset):
        """PyTorch Dataset for UCI Naval Propulsion sequences."""
        
        def __init__(self, 
                     sequences: np.ndarray, 
                     rul_targets: np.ndarray):
            """
            Args:
                sequences: (N, seq_len, features) array
                rul_targets: (N,) array of RUL values
            """
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(rul_targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
    
    
    class LSTMRULNetwork(nn.Module):
        """LSTM network for RUL prediction."""
        
        def __init__(self, 
                     input_size: int = 18,
                     hidden_size: int = 64,
                     num_layers: int = 2,
                     dropout: float = 0.2):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            
            # LSTM forward
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last hidden state
            last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
            
            # FC layers
            out = self.fc(last_hidden)
            
            return out.squeeze(-1)


class LSTMRULPredictor:
    """
    LSTM-based RUL Predictor for Naval Propulsion Systems.
    
    Uses UCI Naval Propulsion dataset format:
    - 16 operational features
    - 2 degradation coefficients (kMc, kMt) → used to compute target RUL
    
    Training approach:
    - Create sequences of operational data
    - Compute RUL target from degradation trajectory
    - Train LSTM to predict RUL from sequence
    
    Inference:
    - Monte Carlo Dropout for uncertainty estimation
    - Returns RUL with confidence interval
    """
    
    # UCI Naval dataset feature columns
    FEATURE_COLUMNS = [
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
        'fuel_flow',
        'compressor_decay',  # kMc
        'turbine_decay'      # kMt
    ]
    
    def __init__(self,
                 sequence_length: int = 50,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 max_rul_hours: float = 5000.0):
        """
        Initialize LSTM RUL predictor.
        
        Args:
            sequence_length: Number of timesteps in input sequence
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            max_rul_hours: Maximum RUL cap for normalization
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_rul_hours = max_rul_hours
        
        # Model and training components
        self.model: Optional['LSTMRULNetwork'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.is_trained = False
        
        # Normalization parameters
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def _create_sequences(self, 
                          data: np.ndarray, 
                          rul_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time-series data.
        
        Args:
            data: (N, features) array of operational data
            rul_values: (N,) array of RUL at each point
            
        Returns:
            sequences: (num_sequences, seq_len, features)
            targets: (num_sequences,) RUL at end of each sequence
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = rul_values[i + self.sequence_length - 1]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def _compute_rul_targets(self, 
                             kMc: np.ndarray, 
                             kMt: np.ndarray) -> np.ndarray:
        """
        Compute RUL targets from degradation coefficients.
        
        Approach: Linear interpolation to failure threshold
        - kMc failure: 0.95
        - kMt failure: 0.975
        """
        # Health indices
        kMc_healthy, kMc_failure = 1.0, 0.95
        kMt_healthy, kMt_failure = 1.0, 0.975
        
        hi_c = (kMc - kMc_failure) / (kMc_healthy - kMc_failure)
        hi_t = (kMt - kMt_failure) / (kMt_healthy - kMt_failure)
        
        # Combined health index (minimum)
        hi = np.minimum(hi_c, hi_t)
        hi = np.clip(hi, 0, 1)
        
        # Estimate degradation rate from trajectory differences
        if len(hi) > 10:
            # Use moving average of degradation rate
            hi_diff = np.diff(hi)
            # Filter out positive changes (recovery/noise)
            degradation = np.where(hi_diff < 0, -hi_diff, 0)
            avg_rate = np.mean(degradation) if np.mean(degradation) > 0 else 1e-5
        else:
            avg_rate = 1e-4
        
        # RUL = remaining_hi / degradation_rate * hours_per_sample
        hours_per_sample = 1.0  # UCI approximation
        rul = (hi / avg_rate) * hours_per_sample
        rul = np.clip(rul, 0, self.max_rul_hours)
        
        return rul
    
    def train(self,
              data: np.ndarray,
              kMc: np.ndarray,
              kMt: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              val_split: float = 0.2,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM RUL model.
        
        Args:
            data: (N, 16) operational features
            kMc: (N,) compressor decay coefficients
            kMt: (N,) turbine decay coefficients
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            verbose: Print training progress
            
        Returns:
            Training history dict
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        
        # Include degradation coefficients as features
        full_data = np.column_stack([data, kMc, kMt])
        
        # Normalize features
        self.feature_mean = np.mean(full_data, axis=0)
        self.feature_std = np.std(full_data, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
        
        normalized_data = (full_data - self.feature_mean) / self.feature_std
        
        # Compute RUL targets
        rul_targets = self._compute_rul_targets(kMc, kMt)
        rul_normalized = rul_targets / self.max_rul_hours  # Normalize to [0, 1]
        
        # Create sequences
        sequences, targets = self._create_sequences(normalized_data, rul_normalized)
        
        if len(sequences) < 100:
            raise ValueError(f"Insufficient data: only {len(sequences)} sequences created")
        
        # Split train/val
        n_val = int(len(sequences) * val_split)
        indices = np.random.permutation(len(sequences))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_dataset = NavalPropulsionDataset(sequences[train_idx], targets[train_idx])
        val_dataset = NavalPropulsionDataset(sequences[val_idx], targets[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        n_features = full_data.shape[1]
        self.model = LSTMRULNetwork(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            
            for batch_seq, batch_target in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_seq)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_seq)
            
            train_loss /= len(train_dataset)
            self.train_losses.append(train_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    output = self.model(batch_seq)
                    loss = criterion(output, batch_target)
                    val_loss += loss.item() * len(batch_seq)
            
            val_loss /= len(val_dataset)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epochs,
            'n_sequences': len(sequences)
        }
    
    def predict(self, 
                sequence: np.ndarray,
                n_samples: int = 30) -> RULPrediction:
        """
        Predict RUL with uncertainty estimation.
        
        Uses Monte Carlo Dropout: run inference multiple times with dropout
        enabled to get uncertainty estimate.
        
        Args:
            sequence: (seq_len, features) or (seq_len, 16) array
                     If 16 features, kMc/kMt must be appended
            n_samples: Number of MC samples for uncertainty
            
        Returns:
            RULPrediction with uncertainty
        """
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(sequence)
        
        if not TORCH_AVAILABLE:
            return self._fallback_prediction(sequence)
        
        # Ensure we have all features
        if sequence.shape[1] < 18 and sequence.shape[1] == 16:
            # Assume last two are kMc, kMt
            pass
        
        # Normalize
        if self.feature_mean is not None:
            sequence = (sequence - self.feature_mean) / self.feature_std
        
        # Pad/trim sequence
        if len(sequence) < self.sequence_length:
            # Pad with zeros at beginning
            padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
        elif len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Monte Carlo Dropout
        self.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(seq_tensor)
                predictions.append(pred.item())
        
        predictions = np.array(predictions)
        
        # Denormalize
        predictions *= self.max_rul_hours
        
        # Statistics
        mean_rul = np.mean(predictions)
        std_rul = np.std(predictions)
        
        # 95% CI
        ci_low = np.percentile(predictions, 2.5)
        ci_high = np.percentile(predictions, 97.5)
        
        # Confidence level
        cv = std_rul / (mean_rul + 1e-6)  # Coefficient of variation
        if cv < 0.1:
            confidence = 'high'
        elif cv < 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return RULPrediction(
            rul_hours=max(0, mean_rul),
            uncertainty=std_rul,
            confidence_interval=(max(0, ci_low), ci_high),
            confidence_level=confidence,
            model_type='lstm'
        )
    
    def _fallback_prediction(self, sequence: np.ndarray) -> RULPrediction:
        """Fallback prediction when model not available."""
        # Use simple linear extrapolation on last column (assume it's decay)
        if sequence.shape[1] >= 2:
            decay = sequence[:, -1]  # Assume last column is decay
            if len(decay) > 1:
                rate = np.mean(np.diff(decay))
                if rate < 0:
                    rul = abs(decay[-1] / rate)
                else:
                    rul = self.max_rul_hours
            else:
                rul = self.max_rul_hours / 2
        else:
            rul = self.max_rul_hours / 2
        
        return RULPrediction(
            rul_hours=min(rul, self.max_rul_hours),
            uncertainty=rul * 0.5,
            confidence_interval=(rul * 0.25, rul * 1.5),
            confidence_level='low',
            model_type='fallback'
        )
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        save_dict = {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_rul_hours': self.max_rul_hours,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'is_trained': self.is_trained,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if TORCH_AVAILABLE and self.model is not None:
            save_dict['model_state'] = self.model.state_dict()
            save_dict['input_size'] = self.model.lstm.input_size
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.sequence_length = save_dict['sequence_length']
        self.hidden_size = save_dict['hidden_size']
        self.num_layers = save_dict['num_layers']
        self.dropout = save_dict['dropout']
        self.max_rul_hours = save_dict['max_rul_hours']
        self.feature_mean = save_dict['feature_mean']
        self.feature_std = save_dict['feature_std']
        self.is_trained = save_dict['is_trained']
        self.train_losses = save_dict.get('train_losses', [])
        self.val_losses = save_dict.get('val_losses', [])
        
        if TORCH_AVAILABLE and 'model_state' in save_dict:
            input_size = save_dict.get('input_size', 18)
            self.model = LSTMRULNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(save_dict['model_state'])
        
        return True


def check_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE
