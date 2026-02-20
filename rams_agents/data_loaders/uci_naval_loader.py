"""
UCI Naval Propulsion CBM Dataset Loader

Loads the Condition Based Maintenance of Naval Propulsion Plants dataset
from UCI Machine Learning Repository.

Dataset Citation:
    Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
    Condition Based Maintenance of Naval Propulsion Plants [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.24432/C5K31K

License: CC BY 4.0

Features (16 operational measurements):
    - Lever position (lp) [-]
    - Ship speed (v) [knots]
    - GT shaft torque (GTT) [kN m]
    - GT rate of revolutions (GTn) [rpm]
    - Gas Generator rate of revolutions (GGn) [rpm]
    - Starboard Propeller Torque (Ts) [kN]
    - Port Propeller Torque (Tp) [kN]
    - HP Turbine exit temperature (T48) [C]
    - GT Compressor inlet air temperature (T1) [C]
    - GT Compressor outlet air temperature (T2) [C]
    - HP Turbine exit pressure (P48) [bar]
    - GT Compressor inlet air pressure (P1) [bar]
    - GT Compressor outlet air pressure (P2) [bar]
    - GT exhaust gas pressure (Pexh) [bar]
    - Turbine Injection Control (TIC) [%]
    - Fuel flow (mf) [kg/s]

Targets (degradation coefficients):
    - GT Compressor decay state coefficient (kMc) [0.95-1.0]
    - GT Turbine decay state coefficient (kMt) [0.975-1.0]
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class PropulsionDataPoint:
    """Single propulsion data observation."""
    # Operational parameters
    lever_position: float       # lp [-]
    ship_speed: float          # v [knots]
    gt_shaft_torque: float     # GTT [kN m]
    gt_revolutions: float      # GTn [rpm]
    gg_revolutions: float      # GGn [rpm]
    starboard_torque: float    # Ts [kN]
    port_torque: float         # Tp [kN]
    hp_turbine_temp: float     # T48 [C]
    compressor_inlet_temp: float   # T1 [C]
    compressor_outlet_temp: float  # T2 [C]
    hp_turbine_pressure: float     # P48 [bar]
    compressor_inlet_pressure: float   # P1 [bar]
    compressor_outlet_pressure: float  # P2 [bar]
    exhaust_pressure: float    # Pexh [bar]
    turbine_injection_control: float  # TIC [%]
    fuel_flow: float           # mf [kg/s]
    
    # Degradation coefficients (targets)
    compressor_decay: float    # kMc [0.95-1.0]
    turbine_decay: float       # kMt [0.975-1.0]
    
    # Metadata
    timestamp: float = 0.0     # Simulated time
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'lever_position': self.lever_position,
            'ship_speed': self.ship_speed,
            'gt_shaft_torque': self.gt_shaft_torque,
            'gt_revolutions': self.gt_revolutions,
            'gg_revolutions': self.gg_revolutions,
            'starboard_torque': self.starboard_torque,
            'port_torque': self.port_torque,
            'hp_turbine_temp': self.hp_turbine_temp,
            'compressor_inlet_temp': self.compressor_inlet_temp,
            'compressor_outlet_temp': self.compressor_outlet_temp,
            'hp_turbine_pressure': self.hp_turbine_pressure,
            'compressor_inlet_pressure': self.compressor_inlet_pressure,
            'compressor_outlet_pressure': self.compressor_outlet_pressure,
            'exhaust_pressure': self.exhaust_pressure,
            'turbine_injection_control': self.turbine_injection_control,
            'fuel_flow': self.fuel_flow,
            'compressor_decay': self.compressor_decay,
            'turbine_decay': self.turbine_decay,
            'timestamp': self.timestamp
        }
    
    def get_features(self) -> np.ndarray:
        """Get operational features as numpy array."""
        return np.array([
            self.lever_position, self.ship_speed, self.gt_shaft_torque,
            self.gt_revolutions, self.gg_revolutions, self.starboard_torque,
            self.port_torque, self.hp_turbine_temp, self.compressor_inlet_temp,
            self.compressor_outlet_temp, self.hp_turbine_pressure,
            self.compressor_inlet_pressure, self.compressor_outlet_pressure,
            self.exhaust_pressure, self.turbine_injection_control, self.fuel_flow
        ])
    
    def get_targets(self) -> np.ndarray:
        """Get degradation targets as numpy array."""
        return np.array([self.compressor_decay, self.turbine_decay])


# Column names matching UCI dataset
FEATURE_COLUMNS = [
    'lever_position', 'ship_speed', 'gt_shaft_torque', 'gt_revolutions',
    'gg_revolutions', 'starboard_torque', 'port_torque', 'hp_turbine_temp',
    'compressor_inlet_temp', 'compressor_outlet_temp', 'hp_turbine_pressure',
    'compressor_inlet_pressure', 'compressor_outlet_pressure', 'exhaust_pressure',
    'turbine_injection_control', 'fuel_flow'
]

TARGET_COLUMNS = ['compressor_decay', 'turbine_decay']


class UCINavalPropulsionLoader:
    """
    Loader for UCI Naval Propulsion CBM Dataset.
    
    Supports multiple loading methods:
    1. From local CSV file
    2. From ucimlrepo package
    3. From kagglehub
    4. Synthetic generation (for demo without download)
    
    The dataset contains 11,934 instances representing different
    degradation states of a CODLAG frigate propulsion system.
    """
    
    # Auto-detect data.csv in common locations
    DEFAULT_DATA_PATHS = [
        'data.csv',
        'data/data.csv',
        '../data.csv',
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            data_path: Optional path to local CSV file.
                      If None, will attempt to auto-detect or load from packages.
        """
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        self._current_idx = 0
        self._loaded = False
        
        # Auto-detect data file if not specified
        if not self.data_path:
            for path in self.DEFAULT_DATA_PATHS:
                if os.path.exists(path):
                    self.data_path = path
                    break
        
    def load(self, prefer_synthetic: bool = False) -> pd.DataFrame:
        """
        Load the dataset.
        
        Args:
            prefer_synthetic: If True, use synthetic data without attempting download
        
        Returns:
            DataFrame with propulsion data
        """
        if prefer_synthetic:
            self.data = self._generate_synthetic_data()
            self._loaded = True
            return self.data
        
        # Try local file first (check again in case data_path wasn't found at init)
        if not self.data_path:
            for path in self.DEFAULT_DATA_PATHS:
                if os.path.exists(path):
                    self.data_path = path
                    break
        
        # If we have a local file, use it
        if self.data_path and os.path.exists(self.data_path):
            print(f"[UCINavalLoader] Loading from local file: {self.data_path}")
            self.data = self._load_from_csv(self.data_path)
            self._loaded = True
            return self.data
        
        # Otherwise try package-based loading
        self.data = self._try_load_from_packages()
        
        self._loaded = True
        return self.data
    
    def _load_from_csv(self, path: str) -> pd.DataFrame:
        """Load from local CSV file."""
        df = pd.read_csv(path)
        
        # Normalize column names (handle various formats)
        # Map from possible column names to standard names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Match columns by key phrases
            if 'lever' in col_lower and 'position' in col_lower:
                col_mapping[col] = 'lever_position'
            elif 'ship' in col_lower and 'speed' in col_lower:
                col_mapping[col] = 'ship_speed'
            elif 'shaft' in col_lower and 'torque' in col_lower:
                col_mapping[col] = 'gt_shaft_torque'
            elif 'gt' in col_lower and 'rate' in col_lower and 'revolution' in col_lower:
                col_mapping[col] = 'gt_revolutions'
            elif ('gas generator' in col_lower or 'gg' in col_lower) and 'revolution' in col_lower:
                col_mapping[col] = 'gg_revolutions'
            elif 'starboard' in col_lower and 'torque' in col_lower:
                col_mapping[col] = 'starboard_torque'
            elif 'port' in col_lower and 'torque' in col_lower:
                col_mapping[col] = 'port_torque'
            elif 't48' in col_lower or ('hp' in col_lower and 'turbine' in col_lower and 'temp' in col_lower):
                col_mapping[col] = 'hp_turbine_temp'
            elif 't1' in col_lower or ('compressor' in col_lower and 'inlet' in col_lower and 'temp' in col_lower):
                col_mapping[col] = 'compressor_inlet_temp'
            elif 't2' in col_lower or ('compressor' in col_lower and 'outlet' in col_lower and 'temp' in col_lower):
                col_mapping[col] = 'compressor_outlet_temp'
            elif 'p48' in col_lower or ('hp' in col_lower and 'turbine' in col_lower and 'pressure' in col_lower):
                col_mapping[col] = 'hp_turbine_pressure'
            elif 'p1' in col_lower or ('compressor' in col_lower and 'inlet' in col_lower and 'pressure' in col_lower):
                col_mapping[col] = 'compressor_inlet_pressure'
            elif 'p2' in col_lower or ('compressor' in col_lower and 'outlet' in col_lower and 'pressure' in col_lower):
                col_mapping[col] = 'compressor_outlet_pressure'
            elif 'exhaust' in col_lower and 'pressure' in col_lower:
                col_mapping[col] = 'exhaust_pressure'
            elif 'tic' in col_lower or ('injection' in col_lower and 'control' in col_lower):
                col_mapping[col] = 'turbine_injection_control'
            elif 'fuel' in col_lower and 'flow' in col_lower:
                col_mapping[col] = 'fuel_flow'
            elif 'compressor' in col_lower and 'decay' in col_lower:
                col_mapping[col] = 'compressor_decay'
            elif 'turbine' in col_lower and 'decay' in col_lower:
                col_mapping[col] = 'turbine_decay'
        
        # Apply mapping
        df = df.rename(columns=col_mapping)
        
        # Drop non-essential columns (like index)
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        return df
    
    def _try_load_from_packages(self) -> pd.DataFrame:
        """Try to load from ucimlrepo or kagglehub packages."""
        
        # Try ucimlrepo
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=316)
            X = dataset.data.features
            y = dataset.data.targets
            
            df = pd.concat([X, y], axis=1)
            df.columns = FEATURE_COLUMNS + TARGET_COLUMNS
            print("[UCINavalLoader] Loaded from ucimlrepo package")
            return df
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"ucimlrepo failed: {e}")
        
        # Try kagglehub
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "thedevastator/improving-naval-vessel-condition-through-machine"
            )
            csv_path = os.path.join(path, 'data.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Kaggle version has different column names, map them
                df = self._normalize_kaggle_columns(df)
                print(f"[UCINavalLoader] Loaded from kagglehub: {csv_path}")
                return df
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"kagglehub failed: {e}")
        
        # Fall back to synthetic
        print("[UCINavalLoader] Using synthetic data (install ucimlrepo or kagglehub for real data)")
        return self._generate_synthetic_data()
    
    def _normalize_kaggle_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Kaggle dataset column names to standard format."""
        # Kaggle column mapping (approximate)
        col_mapping = {
            'Lever position': 'lever_position',
            'Ship speed': 'ship_speed',
            'GT shaft torque': 'gt_shaft_torque',
            'GT rate of revolutions': 'gt_revolutions',
            'GG rate of revolutions': 'gg_revolutions',
            'Starboard Propeller Torque': 'starboard_torque',
            'Port Propeller Torque': 'port_torque',
            'HP Turbine exit temperature': 'hp_turbine_temp',
            'GT Compressor inlet air temperature': 'compressor_inlet_temp',
            'GT Compressor outlet air temperature': 'compressor_outlet_temp',
            'HP Turbine exit pressure': 'hp_turbine_pressure',
            'GT Compressor inlet air pressure': 'compressor_inlet_pressure',
            'GT Compressor outlet air pressure': 'compressor_outlet_pressure',
            'GT exhaust gas pressure': 'exhaust_pressure',
            'Turbine Injection Control': 'turbine_injection_control',
            'Fuel flow': 'fuel_flow',
            'GT Compressor decay state coefficient': 'compressor_decay',
            'GT Turbine decay state coefficient': 'turbine_decay'
        }
        
        df = df.rename(columns=col_mapping)
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data matching UCI dataset characteristics.
        
        This enables demonstration without requiring dataset download.
        The synthetic data mimics the statistical properties of the real dataset.
        """
        np.random.seed(42)
        
        # Simulate degradation over time
        # kMc ranges from 1.0 (new) to 0.95 (degraded)
        # kMt ranges from 1.0 (new) to 0.975 (degraded)
        compressor_decay = np.linspace(1.0, 0.95, n_samples) + np.random.normal(0, 0.002, n_samples)
        turbine_decay = np.linspace(1.0, 0.975, n_samples) + np.random.normal(0, 0.001, n_samples)
        
        # Clamp to valid ranges
        compressor_decay = np.clip(compressor_decay, 0.95, 1.0)
        turbine_decay = np.clip(turbine_decay, 0.975, 1.0)
        
        # Ship speed varies between 3-27 knots
        ship_speed = np.random.choice([3, 6, 9, 12, 15, 18, 21, 24, 27], n_samples)
        
        # Lever position correlates with speed
        lever_position = ship_speed / 27.0 + np.random.normal(0, 0.02, n_samples)
        lever_position = np.clip(lever_position, 0, 1)
        
        # Operational parameters vary with speed and degradation
        data = {
            'lever_position': lever_position,
            'ship_speed': ship_speed,
            'gt_shaft_torque': ship_speed * 2.5 + np.random.normal(0, 2, n_samples),
            'gt_revolutions': 3000 + ship_speed * 100 + np.random.normal(0, 50, n_samples),
            'gg_revolutions': 8000 + ship_speed * 200 + np.random.normal(0, 100, n_samples),
            'starboard_torque': ship_speed * 50 * compressor_decay + np.random.normal(0, 10, n_samples),
            'port_torque': ship_speed * 50 * compressor_decay + np.random.normal(0, 10, n_samples),
            'hp_turbine_temp': 1000 + ship_speed * 10 / turbine_decay + np.random.normal(0, 20, n_samples),
            'compressor_inlet_temp': 20 + np.random.normal(0, 2, n_samples),
            'compressor_outlet_temp': 300 + ship_speed * 5 / compressor_decay + np.random.normal(0, 10, n_samples),
            'hp_turbine_pressure': 3.0 + ship_speed * 0.1 + np.random.normal(0, 0.1, n_samples),
            'compressor_inlet_pressure': 1.0 + np.random.normal(0, 0.01, n_samples),
            'compressor_outlet_pressure': 10 + ship_speed * 0.3 * compressor_decay + np.random.normal(0, 0.5, n_samples),
            'exhaust_pressure': 1.0 + ship_speed * 0.01 + np.random.normal(0, 0.02, n_samples),
            'turbine_injection_control': 60 + ship_speed + np.random.normal(0, 3, n_samples),
            'fuel_flow': 0.5 + ship_speed * 0.05 / (compressor_decay * turbine_decay) + np.random.normal(0, 0.05, n_samples),
            'compressor_decay': compressor_decay,
            'turbine_decay': turbine_decay
        }
        
        return pd.DataFrame(data)
    
    def get_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features (X) and targets (y) as numpy arrays.
        
        Returns:
            Tuple of (features, targets) numpy arrays
        """
        if not self._loaded:
            self.load()
        
        X = self.data[FEATURE_COLUMNS].values
        y = self.data[TARGET_COLUMNS].values
        
        return X, y
    
    def iterate_timesteps(self, 
                          batch_size: int = 1,
                          simulate_time: bool = True) -> Generator[List[PropulsionDataPoint], None, None]:
        """
        Iterate through data as simulated time steps.
        
        This enables streaming simulation for RAMS demonstration.
        
        Args:
            batch_size: Number of data points per timestep
            simulate_time: Whether to add simulated timestamps
        
        Yields:
            List of PropulsionDataPoint objects for each timestep
        """
        if not self._loaded:
            self.load()
        
        n_samples = len(self.data)
        time_step = 0
        
        for idx in range(0, n_samples, batch_size):
            end_idx = min(idx + batch_size, n_samples)
            batch = self.data.iloc[idx:end_idx]
            
            points = []
            for _, row in batch.iterrows():
                point = PropulsionDataPoint(
                    lever_position=row['lever_position'],
                    ship_speed=row['ship_speed'],
                    gt_shaft_torque=row['gt_shaft_torque'],
                    gt_revolutions=row['gt_revolutions'],
                    gg_revolutions=row['gg_revolutions'],
                    starboard_torque=row['starboard_torque'],
                    port_torque=row['port_torque'],
                    hp_turbine_temp=row['hp_turbine_temp'],
                    compressor_inlet_temp=row['compressor_inlet_temp'],
                    compressor_outlet_temp=row['compressor_outlet_temp'],
                    hp_turbine_pressure=row['hp_turbine_pressure'],
                    compressor_inlet_pressure=row['compressor_inlet_pressure'],
                    compressor_outlet_pressure=row['compressor_outlet_pressure'],
                    exhaust_pressure=row['exhaust_pressure'],
                    turbine_injection_control=row['turbine_injection_control'],
                    fuel_flow=row['fuel_flow'],
                    compressor_decay=row['compressor_decay'],
                    turbine_decay=row['turbine_decay'],
                    timestamp=time_step if simulate_time else 0
                )
                points.append(point)
                time_step += 1
            
            yield points
    
    def get_degradation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about degradation in the dataset."""
        if not self._loaded:
            self.load()
        
        return {
            'compressor_decay': {
                'min': self.data['compressor_decay'].min(),
                'max': self.data['compressor_decay'].max(),
                'mean': self.data['compressor_decay'].mean(),
                'std': self.data['compressor_decay'].std()
            },
            'turbine_decay': {
                'min': self.data['turbine_decay'].min(),
                'max': self.data['turbine_decay'].max(),
                'mean': self.data['turbine_decay'].mean(),
                'std': self.data['turbine_decay'].std()
            }
        }
    
    @staticmethod
    def get_citation() -> str:
        """Get the proper citation for this dataset."""
        return """
Citation:
    Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
    Condition Based Maintenance of Naval Propulsion Plants [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.24432/C5K31K

BibTeX:
    @misc{uci_cbm_naval_2014,
        author = {Coraddu, A. and Oneto, L. and Ghio, A. and Savio, S. and Anguita, D. and Figari, M.},
        title = {Condition Based Maintenance of Naval Propulsion Plants},
        year = {2014},
        publisher = {UCI Machine Learning Repository},
        doi = {10.24432/C5K31K}
    }

License: CC BY 4.0 (Creative Commons Attribution 4.0 International)
"""
