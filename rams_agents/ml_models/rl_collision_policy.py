"""
RL-Based Collision Avoidance Policy

Pre-trained PPO policy for multi-robot collision avoidance, adapted for
maritime vessel navigation.

Based on: Acmece/rl-collision-avoidance
Paper: "Towards Optimally Decentralized Multi-Robot Collision Avoidance 
        via Deep Reinforcement Learning" (arXiv:1709.10082)

This module provides inference-only functionality using pre-trained weights.
No training required - uses stage2.pth checkpoint.

Architecture (CNNPolicy):
- Input: 1D scan (512 bins) + goal vector (2) + speed vector (2)
- Conv1D layers: 32 filters, kernel 5/3, stride 2
- FC layers: 256 -> 128
- Output: [speed_ratio (0-1), steering (-1 to 1)]

Part of RAMS ML enhancement for Safety agent.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
import os

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL policy will use fallback mode.")


@dataclass
class RLAction:
    """RL policy action output."""
    speed_ratio: float      # 0-1, fraction of max speed
    steering: float         # -1 to 1, normalized turn rate
    raw_action: np.ndarray  # Raw network output
    confidence: float       # Action confidence (inverse of std)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'speed_ratio': float(self.speed_ratio),
            'steering': float(self.steering),
            'confidence': float(self.confidence)
        }


if TORCH_AVAILABLE:
    def log_normal_density(x: 'torch.Tensor', mean: 'torch.Tensor', 
                           std: 'torch.Tensor', log_std: 'torch.Tensor') -> 'torch.Tensor':
        """Compute log probability density of Gaussian."""
        var = std.pow(2)
        log_density = -0.5 * ((x - mean).pow(2) / var + 2 * log_std + math.log(2 * math.pi))
        return log_density.sum(-1)


    class CNNPolicy(nn.Module):
        """
        CNN-based policy network for collision avoidance.
        
        Architecture matches Acmece/rl-collision-avoidance for compatibility
        with pre-trained weights.
        
        Input:
            x: (batch, frames, scan_bins) - 1D scan observations
            goal: (batch, 2) - [cos(theta_goal), sin(theta_goal)]
            speed: (batch, 2) - [vx, vy] normalized
            
        Output:
            value: (batch,) - state value estimate
            action: (batch, 2) - [speed_ratio, steering]
            logprob: (batch,) - log probability of action
            mean: (batch, 2) - action mean
        """
        
        def __init__(self, frames: int = 3, action_space: int = 2, scan_bins: int = 512):
            super().__init__()
            self.frames = frames
            self.action_space = action_space
            self.scan_bins = scan_bins
            
            # Log standard deviation for action distribution
            self.logstd = nn.Parameter(torch.zeros(action_space))
            
            # Actor network (policy)
            self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                         kernel_size=5, stride=2, padding=1)
            self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                         kernel_size=3, stride=2, padding=1)
            
            # Calculate conv output size
            # After conv1: (512-5+2)/2 + 1 = 255.5 -> 255
            # After conv2: (255-3+2)/2 + 1 = 128
            conv_out_size = self._get_conv_output_size(scan_bins)
            
            self.act_fc1 = nn.Linear(conv_out_size * 32, 256)
            self.act_fc2 = nn.Linear(256 + 2 + 2, 128)  # +goal+speed
            self.actor1 = nn.Linear(128, 1)  # speed ratio (sigmoid)
            self.actor2 = nn.Linear(128, 1)  # steering (tanh)
            
            # Critic network (value function)
            self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32,
                                         kernel_size=5, stride=2, padding=1)
            self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32,
                                         kernel_size=3, stride=2, padding=1)
            self.crt_fc1 = nn.Linear(conv_out_size * 32, 256)
            self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
            self.critic = nn.Linear(128, 1)
        
        def _get_conv_output_size(self, input_size: int) -> int:
            """Calculate output size after conv layers."""
            # Conv1: stride 2, kernel 5, padding 1
            size = (input_size - 5 + 2) // 2 + 1
            # Conv2: stride 2, kernel 3, padding 1
            size = (size - 3 + 2) // 2 + 1
            return size
        
        def forward(self, x: torch.Tensor, goal: torch.Tensor, 
                   speed: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            """
            Forward pass returning value, action, logprob, mean.
            
            Args:
                x: (batch, frames, scan_bins) scan observations
                goal: (batch, 2) goal direction [cos, sin]
                speed: (batch, 2) velocity [vx, vy]
            """
            # Actor path
            a = F.relu(self.act_fea_cv1(x))
            a = F.relu(self.act_fea_cv2(a))
            a = a.view(a.shape[0], -1)
            a = F.relu(self.act_fc1(a))
            a = torch.cat((a, goal, speed), dim=-1)
            a = F.relu(self.act_fc2(a))
            
            # Action output
            mean1 = torch.sigmoid(self.actor1(a))  # speed ratio [0, 1]
            mean2 = torch.tanh(self.actor2(a))      # steering [-1, 1]
            mean = torch.cat((mean1, mean2), dim=-1)
            
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            action = torch.normal(mean, std)
            
            # Log probability
            logprob = log_normal_density(action, mean, std=std, log_std=logstd)
            
            # Critic path
            v = F.relu(self.crt_fea_cv1(x))
            v = F.relu(self.crt_fea_cv2(v))
            v = v.view(v.shape[0], -1)
            v = F.relu(self.crt_fc1(v))
            v = torch.cat((v, goal, speed), dim=-1)
            v = F.relu(self.crt_fc2(v))
            v = self.critic(v)
            
            return v.squeeze(-1), action, logprob, mean
        
        def get_action_deterministic(self, x: torch.Tensor, goal: torch.Tensor,
                                     speed: torch.Tensor) -> torch.Tensor:
            """Get deterministic action (mean only, no sampling)."""
            a = F.relu(self.act_fea_cv1(x))
            a = F.relu(self.act_fea_cv2(a))
            a = a.view(a.shape[0], -1)
            a = F.relu(self.act_fc1(a))
            a = torch.cat((a, goal, speed), dim=-1)
            a = F.relu(self.act_fc2(a))
            
            mean1 = torch.sigmoid(self.actor1(a))
            mean2 = torch.tanh(self.actor2(a))
            return torch.cat((mean1, mean2), dim=-1)


class RLCollisionPolicy:
    """
    Wrapper for RL-based collision avoidance policy.
    
    Provides inference-only functionality using pre-trained PPO weights.
    Falls back to rule-based default action if PyTorch unavailable.
    
    Usage:
        policy = RLCollisionPolicy()
        policy.load('path/to/stage2.pth')
        action = policy.predict(scan_obs, goal, speed)
        
    Attribution:
        Based on Acmece/rl-collision-avoidance (MIT License)
        Paper: arXiv:1709.10082
    """
    
    def __init__(self, 
                 frames: int = 3,
                 scan_bins: int = 512,
                 device: str = "cpu"):
        # Configuration
        self.frames = frames
        self.scan_bins = scan_bins
        self.device = device
        
        # Model
        self._model: Optional[nn.Module] = None
        self._is_loaded = False
        self._model_path: Optional[str] = None
        
        # Frame buffer for temporal stacking
        self._frame_buffer: List[np.ndarray] = []
    
    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self._model is not None
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load pre-trained model weights.
        
        Args:
            model_path: Path to .pth file. If None, looks in default location.
            
        Returns:
            True if loaded successfully
        """
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Using fallback mode.")
            return False
        
        # Default model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'rl_policy_stage2.pth')
        
        if not os.path.exists(model_path):
            warnings.warn(f"Model file not found: {model_path}. Using fallback mode.")
            # Create a placeholder model for demo purposes
            self._model = CNNPolicy(frames=self.frames, scan_bins=self.scan_bins)
            self._is_loaded = True
            print("[RLCollisionPolicy] Using untrained model (demo mode)")
            return True
        
        try:
            self._model = CNNPolicy(frames=self.frames, scan_bins=self.scan_bins)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model.to(self.device)
            self._model.eval()
            
            self._is_loaded = True
            self._model_path = model_path
            print(f"[RLCollisionPolicy] Loaded model from {model_path}")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}")
            return False
    
    def predict(self, 
                scan: np.ndarray,
                goal: np.ndarray,
                speed: np.ndarray,
                deterministic: bool = True) -> RLAction:
        """
        Predict collision avoidance action.
        
        Args:
            scan: (scan_bins,) or (frames, scan_bins) range scan
            goal: (2,) goal direction [cos(theta), sin(theta)]
            speed: (2,) velocity [vx, vy] normalized
            deterministic: If True, return mean action; else sample
            
        Returns:
            RLAction with speed_ratio and steering
        """
        if not TORCH_AVAILABLE or not self.is_ready:
            return self._fallback_action(goal)
        
        # Prepare input
        scan = np.asarray(scan, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        speed = np.asarray(speed, dtype=np.float32)
        
        # Handle frame stacking
        if scan.ndim == 1:
            self._frame_buffer.append(scan)
            if len(self._frame_buffer) > self.frames:
                self._frame_buffer.pop(0)
            
            # Pad if not enough frames
            while len(self._frame_buffer) < self.frames:
                self._frame_buffer.insert(0, scan.copy())
            
            scan = np.stack(self._frame_buffer, axis=0)
        
        # Convert to tensors
        with torch.no_grad():
            x = torch.FloatTensor(scan).unsqueeze(0).to(self.device)
            g = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            s = torch.FloatTensor(speed).unsqueeze(0).to(self.device)
            
            if deterministic:
                action = self._model.get_action_deterministic(x, g, s)
            else:
                _, action, _, _ = self._model.forward(x, g, s)
            
            action_np = action.cpu().numpy()[0]
            
            # Get confidence from log std
            logstd = self._model.logstd.detach().cpu().numpy()
            confidence = 1.0 / (np.exp(logstd).mean() + 0.1)
        
        return RLAction(
            speed_ratio=float(np.clip(action_np[0], 0, 1)),
            steering=float(np.clip(action_np[1], -1, 1)),
            raw_action=action_np,
            confidence=float(confidence)
        )
    
    def reset(self) -> None:
        """Reset frame buffer for new episode."""
        self._frame_buffer.clear()
    
    def _fallback_action(self, goal: np.ndarray) -> RLAction:
        """
        Fallback action when model not available.
        
        Simple rule: steer toward goal, maintain moderate speed.
        """
        steering = float(np.arctan2(goal[1], goal[0]) / math.pi) if len(goal) >= 2 else 0.0
        steering = np.clip(steering, -1, 1)
        
        return RLAction(
            speed_ratio=0.6,  # Moderate speed for safety
            steering=float(steering),
            raw_action=np.array([0.6, steering]),
            confidence=0.3  # Low confidence for fallback
        )


# Module-level singleton for convenience
_default_policy: Optional[RLCollisionPolicy] = None


def get_rl_policy() -> RLCollisionPolicy:
    """Get or create default RL policy instance."""
    global _default_policy
    if _default_policy is None:
        _default_policy = RLCollisionPolicy()
        _default_policy.load()
    return _default_policy
