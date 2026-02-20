"""
Data-Driven RUL Estimator for Naval Propulsion Systems.

Uses semi-supervised/unsupervised methods on UCI Naval Propulsion dataset
to estimate Remaining Useful Life from degradation coefficients.

Methodology:
1. Health Index (HI): Normalize degradation coefficients to [0,1] scale
2. Degradation Model: Fit exponential decay or Wiener process to HI trajectory
3. RUL Estimation: Predict time to failure threshold using fitted model
4. ENHANCED: LSTM deep learning model for sequence-based prediction

References:
    - Coraddu, A. et al. (2016). UCI Naval Propulsion Dataset
    - Si, X. et al. (2011). Remaining Useful Life Estimation - A Review
    - Gebraeel, N. et al. (2005). Residual Life Distributions from Degradation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import curve_fit
import warnings

# Import LSTM model if available
try:
    from .ml_models import LSTMRULPredictor, load_trained_models
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


class DegradationModel(Enum):
    """Degradation model types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    WIENER = "wiener"  # Stochastic
    LSTM = "lstm"       # Deep learning (if available)


@dataclass
class HealthIndex:
    """
    Health Index computed from degradation coefficients.
    
    HI = 1.0 means perfect health
    HI = 0.0 means at failure threshold
    """
    compressor_hi: float  # Compressor health [0, 1]
    turbine_hi: float     # Turbine health [0, 1]
    system_hi: float      # Overall system health (min of components)
    timestamp: float = 0.0
    
    @classmethod
    def from_decay_coefficients(cls, 
                                 kMc: float, 
                                 kMt: float,
                                 timestamp: float = 0.0,
                                 kMc_healthy: float = 1.0,
                                 kMc_failure: float = 0.95,
                                 kMt_healthy: float = 1.0,
                                 kMt_failure: float = 0.975) -> 'HealthIndex':
        """
        Create Health Index from raw decay coefficients.
        
        Normalizes decay coefficients to [0, 1] health scale:
        - kMc: 1.0 (healthy) → 0.95 (failure) maps to HI: 1.0 → 0.0
        - kMt: 1.0 (healthy) → 0.975 (failure) maps to HI: 1.0 → 0.0
        """
        # Normalize to [0, 1] health index
        compressor_hi = (kMc - kMc_failure) / (kMc_healthy - kMc_failure)
        turbine_hi = (kMt - kMt_failure) / (kMt_healthy - kMt_failure)
        
        # Clamp to valid range
        compressor_hi = max(0.0, min(1.0, compressor_hi))
        turbine_hi = max(0.0, min(1.0, turbine_hi))
        
        # System HI is limited by weakest component
        system_hi = min(compressor_hi, turbine_hi)
        
        return cls(
            compressor_hi=compressor_hi,
            turbine_hi=turbine_hi,
            system_hi=system_hi,
            timestamp=timestamp
        )


@dataclass
class RULEstimate:
    """RUL estimation result with uncertainty."""
    rul_hours: float
    confidence_interval: Tuple[float, float]  # 95% CI
    confidence_level: str  # 'low', 'medium', 'high'
    model_used: str
    limiting_component: str
    health_index: float


@dataclass  
class DegradationTrajectory:
    """Stores degradation trajectory for a component."""
    timestamps: List[float] = field(default_factory=list)
    health_indices: List[float] = field(default_factory=list)
    
    def add_point(self, timestamp: float, hi: float) -> None:
        """Add observation point to trajectory."""
        self.timestamps.append(timestamp)
        self.health_indices.append(hi)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays for fitting."""
        return np.array(self.timestamps), np.array(self.health_indices)
    
    def __len__(self) -> int:
        return len(self.timestamps)


class RULEstimator:
    """
    Semi-supervised RUL Estimator using UCI Naval Propulsion data.
    
    Approach:
    1. Convert decay coefficients (kMc, kMt) to Health Index (HI)
    2. Learn degradation pattern from historical HI trajectory
    3. Fit parametric degradation model (exponential, Wiener)
    4. Extrapolate to failure threshold (HI = 0)
    5. Provide uncertainty bounds via bootstrap or model variance
    
    This is "semi-supervised" because:
    - We don't have explicit failure labels
    - But we use domain knowledge (decay coefficient semantics)
    - And learn degradation dynamics from data distribution
    """
    
    # Failure thresholds (from UCI dataset documentation)
    KMC_HEALTHY = 1.0
    KMC_FAILURE = 0.95  # Compressor failure threshold
    KMT_HEALTHY = 1.0
    KMT_FAILURE = 0.975  # Turbine failure threshold (more sensitive)
    
    # Operating hour conversion (UCI data doesn't have explicit time)
    # Each sample represents approximately 1 hour of operation
    HOURS_PER_SAMPLE = 1.0
    
    def __init__(self, 
                 preferred_model: DegradationModel = DegradationModel.EXPONENTIAL,
                 min_observations: int = 5,
                 max_rul_hours: float = 10000.0,
                 use_lstm: bool = True):
        """
        Initialize RUL Estimator.
        
        Args:
            preferred_model: Degradation model to fit
            min_observations: Minimum data points needed for reliable estimate
            max_rul_hours: Maximum RUL cap
            use_lstm: Whether to use LSTM model if available
        """
        self.preferred_model = preferred_model
        self.min_observations = min_observations
        self.max_rul_hours = max_rul_hours
        
        # Trajectory storage
        self.compressor_trajectory = DegradationTrajectory()
        self.turbine_trajectory = DegradationTrajectory()
        
        # Fitted model parameters
        self._compressor_params: Optional[Dict[str, float]] = None
        self._turbine_params: Optional[Dict[str, float]] = None
        
        # Population statistics (learned from full dataset)
        self._population_stats: Optional[Dict[str, Any]] = None
        
        # Current time counter
        self._current_time: float = 0.0
        
        # ===== ENHANCED: LSTM Model =====
        self._lstm_predictor: Optional['LSTMRULPredictor'] = None
        self._feature_history: List[np.ndarray] = []  # For LSTM sequence
        self._lstm_sequence_length = 50
        
        if use_lstm and LSTM_AVAILABLE:
            try:
                self._lstm_predictor, _ = load_trained_models()
                if self._lstm_predictor:
                    self._lstm_sequence_length = self._lstm_predictor.sequence_length
            except Exception:
                self._lstm_predictor = None
    
    def learn_population_statistics(self, 
                                     kMc_samples: np.ndarray,
                                     kMt_samples: np.ndarray) -> Dict[str, Any]:
        """
        Learn degradation statistics from full UCI dataset (unsupervised).
        
        This provides prior distributions for Bayesian RUL estimation.
        
        Args:
            kMc_samples: All compressor decay values from dataset
            kMt_samples: All turbine decay values from dataset
            
        Returns:
            Population statistics dictionary
        """
        # Convert to health indices
        compressor_hi = (kMc_samples - self.KMC_FAILURE) / (self.KMC_HEALTHY - self.KMC_FAILURE)
        turbine_hi = (kMt_samples - self.KMT_FAILURE) / (self.KMT_HEALTHY - self.KMT_FAILURE)
        
        # Fit distributions
        comp_mean, comp_std = np.mean(compressor_hi), np.std(compressor_hi)
        turb_mean, turb_std = np.mean(turbine_hi), np.std(turbine_hi)
        
        # Estimate degradation rate distribution
        # Assuming samples are roughly ordered by operating time
        n_samples = len(kMc_samples)
        if n_samples > 100:
            # Estimate typical degradation rate per "sample"
            # Using differences between percentiles
            hi_p90 = np.percentile(compressor_hi, 90)
            hi_p10 = np.percentile(compressor_hi, 10)
            
            # Typical HI drop over dataset span
            hi_range = hi_p90 - hi_p10
            
            # Estimate degradation rate (HI units per hour)
            # This is approximate since we don't have explicit time ordering
            est_rate_mean = hi_range / n_samples * self.HOURS_PER_SAMPLE
            est_rate_std = est_rate_mean * 0.5  # Assume 50% CV
        else:
            est_rate_mean = 0.00005  # Default: ~20k hours to failure
            est_rate_std = 0.00002
        
        self._population_stats = {
            'compressor': {
                'hi_mean': comp_mean,
                'hi_std': comp_std,
                'hi_min': np.min(compressor_hi),
                'hi_max': np.max(compressor_hi),
            },
            'turbine': {
                'hi_mean': turb_mean,
                'hi_std': turb_std,
                'hi_min': np.min(turbine_hi),
                'hi_max': np.max(turbine_hi),
            },
            'degradation_rate': {
                'mean': est_rate_mean,
                'std': est_rate_std
            },
            'n_samples': n_samples
        }
        
        return self._population_stats
    
    def update(self, kMc: float, kMt: float, timestamp: Optional[float] = None) -> HealthIndex:
        """
        Update trajectories with new observation.
        
        Args:
            kMc: Compressor decay coefficient
            kMt: Turbine decay coefficient
            timestamp: Optional explicit timestamp (hours)
            
        Returns:
            Current Health Index
        """
        if timestamp is None:
            timestamp = self._current_time
            self._current_time += self.HOURS_PER_SAMPLE
        else:
            self._current_time = timestamp
        
        # Compute Health Index
        hi = HealthIndex.from_decay_coefficients(
            kMc, kMt, timestamp,
            self.KMC_HEALTHY, self.KMC_FAILURE,
            self.KMT_HEALTHY, self.KMT_FAILURE
        )
        
        # Update trajectories
        self.compressor_trajectory.add_point(timestamp, hi.compressor_hi)
        self.turbine_trajectory.add_point(timestamp, hi.turbine_hi)
        
        # Refit models if enough data
        if len(self.compressor_trajectory) >= self.min_observations:
            self._fit_degradation_model()
        
        return hi
    
    def _fit_degradation_model(self) -> None:
        """Fit degradation model to observed trajectories."""
        # Fit compressor model
        self._compressor_params = self._fit_single_trajectory(
            self.compressor_trajectory, 'compressor'
        )
        
        # Fit turbine model
        self._turbine_params = self._fit_single_trajectory(
            self.turbine_trajectory, 'turbine'
        )
    
    def _fit_single_trajectory(self, 
                                trajectory: DegradationTrajectory,
                                component: str) -> Dict[str, float]:
        """
        Fit degradation model to single component trajectory.
        
        Returns model parameters and goodness of fit.
        """
        if len(trajectory) < self.min_observations:
            return {'status': 'insufficient_data'}
        
        t, hi = trajectory.get_arrays()
        
        # Normalize time to start from 0
        t_norm = t - t[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.preferred_model == DegradationModel.LINEAR:
                return self._fit_linear(t_norm, hi)
            elif self.preferred_model == DegradationModel.EXPONENTIAL:
                return self._fit_exponential(t_norm, hi)
            elif self.preferred_model == DegradationModel.POLYNOMIAL:
                return self._fit_polynomial(t_norm, hi)
            else:  # WIENER
                return self._fit_wiener(t_norm, hi)
    
    def _fit_linear(self, t: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
        """
        Fit linear degradation model: HI(t) = HI_0 - rate * t
        """
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, hi)
            
            return {
                'model': 'linear',
                'intercept': intercept,  # Initial HI
                'slope': slope,          # Degradation rate (negative)
                'r_squared': r_value**2,
                'std_err': std_err,
                'status': 'fitted'
            }
        except Exception:
            return {'status': 'fit_failed', 'model': 'linear'}
    
    def _fit_exponential(self, t: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
        """
        Fit exponential degradation model: HI(t) = HI_0 * exp(-lambda * t)
        
        This models gradual degradation that accelerates over time.
        """
        try:
            # Define exponential decay function
            def exp_decay(t, hi_0, lam):
                return hi_0 * np.exp(-lam * t)
            
            # Initial guess
            p0 = [hi[0], 0.0001]
            
            # Fit with bounds
            popt, pcov = curve_fit(
                exp_decay, t, hi, 
                p0=p0, 
                bounds=([0.5, 0], [1.5, 0.01]),
                maxfev=1000
            )
            
            # Calculate R-squared
            hi_pred = exp_decay(t, *popt)
            ss_res = np.sum((hi - hi_pred)**2)
            ss_tot = np.sum((hi - np.mean(hi))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            return {
                'model': 'exponential',
                'hi_0': popt[0],      # Initial health index
                'lambda': popt[1],     # Decay rate constant
                'r_squared': r_squared,
                'std_err': np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0],
                'status': 'fitted'
            }
        except Exception:
            # Fall back to linear
            return self._fit_linear(t, hi)
    
    def _fit_polynomial(self, t: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
        """
        Fit quadratic degradation model: HI(t) = a + b*t + c*t^2
        """
        try:
            coeffs = np.polyfit(t, hi, 2)
            hi_pred = np.polyval(coeffs, t)
            
            ss_res = np.sum((hi - hi_pred)**2)
            ss_tot = np.sum((hi - np.mean(hi))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            return {
                'model': 'polynomial',
                'coeffs': coeffs.tolist(),  # [a, b, c] for at^2 + bt + c
                'r_squared': r_squared,
                'status': 'fitted'
            }
        except Exception:
            return {'status': 'fit_failed', 'model': 'polynomial'}
    
    def _fit_wiener(self, t: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
        """
        Fit Wiener process degradation model.
        
        Wiener process: X(t) = μt + σW(t)
        where W(t) is standard Brownian motion.
        
        For degradation: HI(t) = HI_0 - μt + σW(t)
        """
        try:
            # Estimate drift (μ) from linear trend
            dt = np.diff(t)
            dhi = np.diff(hi)
            
            # Avoid division by zero
            valid = dt > 0
            if not np.any(valid):
                return self._fit_linear(t, hi)
            
            increments = dhi[valid] / dt[valid]
            
            # Drift estimate (mean rate of change)
            mu = np.mean(increments)
            
            # Diffusion estimate (volatility)
            sigma = np.std(increments) * np.sqrt(np.mean(dt[valid]))
            
            return {
                'model': 'wiener',
                'hi_0': hi[0],
                'mu': mu,        # Drift (degradation rate)
                'sigma': sigma,  # Diffusion (uncertainty)
                'status': 'fitted'
            }
        except Exception:
            return self._fit_linear(t, hi)
    
    def estimate_rul(self, 
                      current_hi: Optional[HealthIndex] = None,
                      failure_threshold: float = 0.0) -> RULEstimate:
        """
        Estimate Remaining Useful Life.
        
        Args:
            current_hi: Current health index (uses latest if None)
            failure_threshold: HI value considered failure (default 0)
            
        Returns:
            RULEstimate with point estimate and uncertainty bounds
        """
        # Get current HI if not provided
        if current_hi is None:
            if len(self.compressor_trajectory) == 0:
                return self._default_rul_estimate()
            
            comp_hi = self.compressor_trajectory.health_indices[-1]
            turb_hi = self.turbine_trajectory.health_indices[-1]
            current_time = self.compressor_trajectory.timestamps[-1]
        else:
            comp_hi = current_hi.compressor_hi
            turb_hi = current_hi.turbine_hi
            current_time = current_hi.timestamp
        
        # Estimate RUL for each component
        comp_rul, comp_ci = self._estimate_component_rul(
            comp_hi, self._compressor_params, current_time, failure_threshold
        )
        
        turb_rul, turb_ci = self._estimate_component_rul(
            turb_hi, self._turbine_params, current_time, failure_threshold
        )
        
        # System RUL is minimum of components
        if comp_rul <= turb_rul:
            system_rul = comp_rul
            limiting = 'compressor'
            ci = comp_ci
            model = self._compressor_params.get('model', 'unknown') if self._compressor_params else 'unknown'
        else:
            system_rul = turb_rul
            limiting = 'turbine'
            ci = turb_ci
            model = self._turbine_params.get('model', 'unknown') if self._turbine_params else 'unknown'
        
        # Cap RUL
        system_rul = min(system_rul, self.max_rul_hours)
        ci = (min(ci[0], self.max_rul_hours), min(ci[1], self.max_rul_hours))
        
        # Determine confidence level
        n_obs = len(self.compressor_trajectory)
        if n_obs >= 50:
            confidence = 'high'
        elif n_obs >= 20:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return RULEstimate(
            rul_hours=round(system_rul, 1),
            confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
            confidence_level=confidence,
            model_used=model,
            limiting_component=limiting,
            health_index=min(comp_hi, turb_hi)
        )
    
    def _estimate_component_rul(self,
                                 current_hi: float,
                                 params: Optional[Dict[str, float]],
                                 current_time: float,
                                 threshold: float) -> Tuple[float, Tuple[float, float]]:
        """
        Estimate RUL for single component using fitted model.
        
        Returns (point_estimate, (lower_bound, upper_bound))
        """
        if params is None or params.get('status') != 'fitted':
            # Fall back to population statistics or default
            return self._fallback_rul_estimate(current_hi, threshold)
        
        model = params.get('model', 'linear')
        
        if model == 'linear':
            # HI(t) = intercept + slope * t
            # Solve for t when HI = threshold
            slope = params['slope']
            if slope >= 0:  # Not degrading
                return (self.max_rul_hours, (self.max_rul_hours, self.max_rul_hours))
            
            # Time to threshold from current HI
            rul = (threshold - current_hi) / slope
            
            # Confidence interval from std_err
            std_err = params.get('std_err', 0)
            ci_factor = 1.96  # 95% CI
            rul_std = abs(rul * std_err / slope) if slope != 0 else 0
            
            return (max(0, rul), (max(0, rul - ci_factor * rul_std), rul + ci_factor * rul_std))
        
        elif model == 'exponential':
            # HI(t) = HI_0 * exp(-lambda * t)
            # Solve for t when HI = threshold
            hi_0 = params['hi_0']
            lam = params['lambda']
            
            if lam <= 0 or hi_0 <= threshold:
                return (self.max_rul_hours, (self.max_rul_hours, self.max_rul_hours))
            
            # Current elapsed time (from start of observation)
            t_elapsed = current_time - self.compressor_trajectory.timestamps[0] if self.compressor_trajectory.timestamps else 0
            
            # Time to threshold from current
            # current_hi = hi_0 * exp(-lam * t_elapsed)
            # threshold = hi_0 * exp(-lam * (t_elapsed + rul))
            # rul = (1/lam) * ln(current_hi / threshold)
            
            if current_hi <= threshold:
                return (0.0, (0.0, 0.0))
            
            rul = (1 / lam) * np.log(current_hi / max(threshold, 0.001))
            
            # Uncertainty from lambda std_err
            std_err = params.get('std_err', [0, 0])
            lam_std = std_err[1] if len(std_err) > 1 else 0
            
            # Propagate uncertainty
            if lam > 0 and lam_std > 0:
                rul_low = (1 / (lam + lam_std)) * np.log(current_hi / max(threshold, 0.001))
                rul_high = (1 / max(lam - lam_std, 0.0001)) * np.log(current_hi / max(threshold, 0.001))
            else:
                rul_low, rul_high = rul * 0.8, rul * 1.2
            
            return (max(0, rul), (max(0, rul_low), rul_high))
        
        elif model == 'wiener':
            # Wiener process: HI(t) = HI_0 - mu*t + sigma*W(t)
            # Expected hitting time to threshold
            mu = params['mu']
            sigma = params.get('sigma', 0)
            
            if mu >= 0:  # Not degrading on average
                return (self.max_rul_hours, (self.max_rul_hours, self.max_rul_hours))
            
            # Expected RUL (ignoring diffusion for point estimate)
            rul = (threshold - current_hi) / mu
            
            # Wiener hitting time has inverse Gaussian distribution
            # Use sigma to estimate uncertainty
            if sigma > 0:
                # Approximate CI based on diffusion
                rul_std = abs(sigma / mu) * np.sqrt(rul) if mu != 0 else 0
                ci_factor = 1.96
                rul_low = max(0, rul - ci_factor * rul_std)
                rul_high = rul + ci_factor * rul_std
            else:
                rul_low, rul_high = rul * 0.8, rul * 1.2
            
            return (max(0, rul), (rul_low, rul_high))
        
        elif model == 'polynomial':
            # Numerical solution for polynomial
            coeffs = params.get('coeffs', [0, -0.0001, 1])
            # Shift to solve: poly(t) = threshold
            shifted_coeffs = coeffs.copy()
            shifted_coeffs[-1] -= threshold
            
            roots = np.roots(shifted_coeffs)
            positive_real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
            
            if positive_real_roots:
                rul = min(positive_real_roots)
                return (rul, (rul * 0.8, rul * 1.2))
            else:
                return (self.max_rul_hours, (self.max_rul_hours, self.max_rul_hours))
        
        return self._fallback_rul_estimate(current_hi, threshold)
    
    def _fallback_rul_estimate(self, 
                                current_hi: float, 
                                threshold: float) -> Tuple[float, Tuple[float, float]]:
        """
        Fallback RUL estimate using population statistics or defaults.
        """
        if self._population_stats:
            rate = self._population_stats['degradation_rate']['mean']
            rate_std = self._population_stats['degradation_rate']['std']
        else:
            # Conservative default: 0.00005 HI/hour → 20k hours from HI=1 to HI=0
            rate = 0.00005
            rate_std = 0.00002
        
        if rate <= 0:
            return (self.max_rul_hours, (self.max_rul_hours, self.max_rul_hours))
        
        remaining_hi = current_hi - threshold
        rul = remaining_hi / rate
        
        # CI from rate uncertainty
        rul_low = remaining_hi / (rate + rate_std)
        rul_high = remaining_hi / max(rate - rate_std, 0.00001)
        
        return (max(0, rul), (max(0, rul_low), rul_high))
    
    def _default_rul_estimate(self) -> RULEstimate:
        """Return default RUL when no data available."""
        return RULEstimate(
            rul_hours=self.max_rul_hours,
            confidence_interval=(self.max_rul_hours * 0.5, self.max_rul_hours),
            confidence_level='low',
            model_used='none',
            limiting_component='unknown',
            health_index=1.0
        )
    
    def get_health_trend(self) -> Dict[str, Any]:
        """
        Get current health trend analysis.
        
        Returns dictionary with degradation rates and predictions.
        """
        result = {
            'compressor': self._get_component_trend(self._compressor_params, 'compressor'),
            'turbine': self._get_component_trend(self._turbine_params, 'turbine'),
            'n_observations': len(self.compressor_trajectory)
        }
        
        # Overall assessment
        comp_rate = result['compressor'].get('degradation_rate', 0)
        turb_rate = result['turbine'].get('degradation_rate', 0)
        
        if comp_rate > 0.001 or turb_rate > 0.001:
            result['status'] = 'RAPID_DEGRADATION'
        elif comp_rate > 0.0001 or turb_rate > 0.0001:
            result['status'] = 'NORMAL_DEGRADATION'
        elif comp_rate > 0:
            result['status'] = 'SLOW_DEGRADATION'
        else:
            result['status'] = 'STABLE'
        
        return result
    
    def _get_component_trend(self, 
                              params: Optional[Dict[str, float]], 
                              component: str) -> Dict[str, Any]:
        """Get trend for single component."""
        if params is None or params.get('status') != 'fitted':
            return {'status': 'insufficient_data', 'degradation_rate': 0}
        
        model = params.get('model', 'linear')
        
        if model == 'linear':
            rate = abs(params.get('slope', 0))
        elif model == 'exponential':
            rate = params.get('lambda', 0)
        elif model == 'wiener':
            rate = abs(params.get('mu', 0))
        else:
            rate = 0
        
        return {
            'model': model,
            'degradation_rate': rate,
            'r_squared': params.get('r_squared', 0),
            'status': 'fitted'
        }
    
    # =========================================================================
    # ENHANCED: LSTM-based RUL Prediction
    # =========================================================================
    
    def add_features_for_lstm(self, features: np.ndarray, kMc: float, kMt: float) -> None:
        """
        Add operational features for LSTM prediction.
        
        Call this with each observation to build up sequence for LSTM.
        
        Args:
            features: (16,) array of operational features
            kMc: Compressor decay coefficient
            kMt: Turbine decay coefficient
        """
        # Combine features with decay coefficients
        full_features = np.concatenate([features, [kMc, kMt]])
        self._feature_history.append(full_features)
        
        # Trim history to max sequence length
        if len(self._feature_history) > self._lstm_sequence_length * 2:
            self._feature_history = self._feature_history[-self._lstm_sequence_length:]
    
    def estimate_rul_lstm(self) -> Optional[RULEstimate]:
        """
        Estimate RUL using LSTM model.
        
        Returns:
            RULEstimate from LSTM or None if not available
        """
        if self._lstm_predictor is None:
            return None
        
        if len(self._feature_history) < 5:  # Need some history
            return None
        
        try:
            # Build sequence
            sequence = np.array(self._feature_history[-self._lstm_sequence_length:])
            
            # Get prediction
            prediction = self._lstm_predictor.predict(sequence)
            
            # Convert to RULEstimate format
            # Determine limiting component from last decay values
            if len(self._feature_history) > 0:
                last_features = self._feature_history[-1]
                kMc = last_features[-2]
                kMt = last_features[-1]
                
                # Compute health indices
                hi_c = (kMc - self.KMC_FAILURE) / (self.KMC_HEALTHY - self.KMC_FAILURE)
                hi_t = (kMt - self.KMT_FAILURE) / (self.KMT_HEALTHY - self.KMT_FAILURE)
                
                limiting = 'compressor' if hi_c < hi_t else 'turbine'
                health_index = min(hi_c, hi_t)
            else:
                limiting = 'unknown'
                health_index = 1.0
            
            return RULEstimate(
                rul_hours=prediction.rul_hours,
                confidence_interval=prediction.confidence_interval,
                confidence_level=prediction.confidence_level,
                model_used='lstm',
                limiting_component=limiting,
                health_index=health_index
            )
            
        except Exception as e:
            # Fallback to traditional method
            return None
    
    def has_lstm_model(self) -> bool:
        """Check if LSTM model is available."""
        return self._lstm_predictor is not None and self._lstm_predictor.is_trained


def demo_rul_estimation():
    """Demonstrate RUL estimation with synthetic degradation data."""
    print("=" * 60)
    print("RUL Estimator Demo - Data-Driven Approach")
    print("=" * 60)
    
    # Create estimator
    estimator = RULEstimator(
        preferred_model=DegradationModel.EXPONENTIAL,
        min_observations=5
    )
    
    # Simulate population data (like UCI dataset)
    np.random.seed(42)
    n_population = 1000
    kMc_population = np.random.uniform(0.95, 1.0, n_population)
    kMt_population = np.random.uniform(0.975, 1.0, n_population)
    
    # Learn from population
    pop_stats = estimator.learn_population_statistics(kMc_population, kMt_population)
    print(f"\nLearned from {pop_stats['n_samples']} population samples")
    print(f"  Compressor HI mean: {pop_stats['compressor']['hi_mean']:.3f}")
    print(f"  Turbine HI mean: {pop_stats['turbine']['hi_mean']:.3f}")
    
    # Simulate degradation trajectory
    print("\n--- Simulating Degradation Trajectory ---")
    
    kMc_start = 0.995
    kMt_start = 0.995
    
    # Exponential decay simulation
    decay_rate_c = 0.0002
    decay_rate_t = 0.00015
    
    for t in range(50):
        # Add noise
        kMc = kMc_start * np.exp(-decay_rate_c * t) + np.random.normal(0, 0.0005)
        kMt = kMt_start * np.exp(-decay_rate_t * t) + np.random.normal(0, 0.0003)
        
        # Clamp to valid range
        kMc = max(0.95, min(1.0, kMc))
        kMt = max(0.975, min(1.0, kMt))
        
        hi = estimator.update(kMc, kMt, timestamp=float(t))
        
        if t % 10 == 0:
            rul = estimator.estimate_rul()
            print(f"  t={t:3d}: HI={hi.system_hi:.3f}, RUL={rul.rul_hours:.0f}h "
                  f"[{rul.confidence_interval[0]:.0f}-{rul.confidence_interval[1]:.0f}] "
                  f"({rul.confidence_level})")
    
    # Final estimate
    print("\n--- Final RUL Estimate ---")
    final_rul = estimator.estimate_rul()
    print(f"  RUL: {final_rul.rul_hours:.1f} hours")
    print(f"  95% CI: [{final_rul.confidence_interval[0]:.1f}, {final_rul.confidence_interval[1]:.1f}]")
    print(f"  Confidence: {final_rul.confidence_level}")
    print(f"  Limiting Component: {final_rul.limiting_component}")
    print(f"  Model Used: {final_rul.model_used}")
    print(f"  Current HI: {final_rul.health_index:.3f}")
    
    # Health trend
    print("\n--- Health Trend ---")
    trend = estimator.get_health_trend()
    print(f"  Status: {trend['status']}")
    print(f"  Compressor rate: {trend['compressor'].get('degradation_rate', 0):.6f} HI/hour")
    print(f"  Turbine rate: {trend['turbine'].get('degradation_rate', 0):.6f} HI/hour")


if __name__ == "__main__":
    demo_rul_estimation()
