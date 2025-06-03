"""
Parameter management for PACT reconstruction.

This module handles loading, validation, and management of configuration parameters
for the photoacoustic computed tomography reconstruction pipeline. It provides
functionality to load parameters from YAML files and ensure they meet the
required specifications for different reconstruction algorithms.

Key functionality:
- Parameter loading from configuration files
- Default parameter management
- Parameter validation and type checking
- Configuration merging and overrides
"""

import os
import yaml
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

from .validation import validate_all_parameters, ConfigValidationError


class ConfigManager:
    """Centralized configuration manager for PACT parameters.
    
    This class provides a clean, object-oriented interface to access configuration
    parameters while maintaining backward compatibility with the global variable
    approach used in util.py.
    
    Attributes:
        _params: Raw parameter dictionary loaded from YAML
        _validated: Whether parameters have been validated
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, validate: bool = True):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses 'params.yaml'
            validate: Whether to validate parameters on load
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If parameter validation fails
        """
        self._params: Dict[str, Any] = {}
        self._validated: bool = False
        
        if config_path is None:
            config_path = "params.yaml"
        
        self.load_config(config_path, validate=validate)
    
    def load_config(self, config_path: Union[str, Path], validate: bool = True) -> None:
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            validate: Whether to validate parameters after loading
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If parameter validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._params = yaml.safe_load(f)
        
        if validate:
            self.validate()
    
    def validate(self) -> None:
        """Validate all parameters.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        validate_all_parameters(self._params)
        self._validated = True
    
    def get_raw_params(self) -> Dict[str, Any]:
        """Get the raw parameter dictionary.
        
        Returns:
            Raw parameter dictionary
        """
        return self._params.copy()
    
    # File parameters
    @property
    def data_path(self) -> str:
        """Get the data path."""
        return self._params["file"]["data_path"]
    
    # Generate data parameters
    @property
    def batch_size(self) -> int:
        """Get the batch size for data generation."""
        return self._params["generate_data"]["batch_size"]
    
    @property
    def N(self) -> Tuple[int, int, int]:
        """Get the grid dimensions (including margins)."""
        return tuple(self._params["generate_data"]["N"])
    
    @property
    def shrink_factor(self) -> int:
        """Get the shrink factor for data generation."""
        return self._params["generate_data"]["shrink_factor"]
    
    @property
    def dims(self) -> int:
        """Get the number of spatial dimensions (2 or 3)."""
        return self._params["generate_data"]["dims"]
    
    @property
    def dx(self) -> Tuple[float, float, float]:
        """Get the spatial grid spacing."""
        return tuple(self._params["generate_data"]["dx"])
    
    @property
    def c(self) -> float:
        """Get the background speed of sound."""
        return self._params["generate_data"]["c"]
    
    @property
    def c_periodicity(self) -> int:
        """Get the speed of sound periodicity parameter."""
        return self._params["generate_data"]["c_periodicity"]
    
    @property
    def c_variation_amplitude(self) -> float:
        """Get the speed of sound variation amplitude."""
        return self._params["generate_data"]["c_variation_amplitude"]
    
    @property
    def c_blood(self) -> float:
        """Get the blood speed of sound."""
        return self._params["generate_data"]["c_blood"]
    
    @property
    def cfl(self) -> float:
        """Get the CFL condition parameter."""
        return self._params["generate_data"]["cfl"]
    
    @property
    def pml_margin(self) -> Tuple[int, int, int]:
        """Get the PML margin on each side."""
        return tuple(self._params["generate_data"]["pml_margin"])
    
    @property
    def tissue_margin(self) -> Tuple[int, int, int]:
        """Get the tissue margin."""
        return tuple(self._params["generate_data"]["tissue_margin"])
    
    @property
    def sensor_margin(self) -> Tuple[int, int, int]:
        """Get the sensor margin."""
        return tuple(self._params["generate_data"]["sensor_margin"])
    
    @property
    def num_sensors(self) -> int:
        """Get the number of sensors."""
        return self._params["generate_data"]["num_sensors"]
    
    @property
    def noise_amplitude(self) -> float:
        """Get the noise amplitude."""
        return self._params["generate_data"]["noise_amplitude"]
    
    # Lighting parameters
    @property
    def lighting_attenuation(self) -> bool:
        """Get whether lighting attenuation is enabled."""
        return self._params["lighting"]["lighting_attenuation"]
    
    @property
    def num_lighting_angles(self) -> int:
        """Get the number of lighting angles."""
        return self._params["lighting"]["num_lighting_angles"]
    
    @property
    def attenuation(self) -> float:
        """Get the attenuation coefficient."""
        return self._params["lighting"]["attenuation"]
    
    # Reconstruction parameters
    @property
    def recon_iterations(self) -> int:
        """Get the number of reconstruction iterations."""
        return self._params["reconstruct"]["recon_iterations"]
    
    @property
    def lr_mu_r(self) -> float:
        """Get the learning rate for mu reconstruction."""
        return self._params["reconstruct"]["lr_mu_r"]
    
    @property
    def lr_c_r(self) -> float:
        """Get the learning rate for c reconstruction."""
        return self._params["reconstruct"]["lr_c_r"]
    
    @property
    def recon_file_start(self) -> int:
        """Get the reconstruction file start index."""
        return self._params["reconstruct"]["recon_file_start"]
    
    @property
    def recon_file_end(self) -> int:
        """Get the reconstruction file end index."""
        return self._params["reconstruct"]["recon_file_end"]
    
    # Training parameters
    @property
    def checkpoint_file_index(self) -> int:
        """Get the checkpoint file index."""
        return self._params["train"]["checkpoint_index"]
    
    @property
    def lr_r_mu(self) -> float:
        """Get the learning rate for R_mu training."""
        return self._params["train"]["lr_R_mu"]
    
    @property
    def lr_r_c(self) -> float:
        """Get the learning rate for R_c training."""
        return self._params["train"]["lr_R_c"]
    
    @property
    def dropout(self) -> float:
        """Get the dropout rate."""
        return self._params["train"]["dropout"]
    
    @property
    def train_file_start(self) -> int:
        """Get the training file start index."""
        return self._params["train"]["train_file_start"]
    
    @property
    def train_file_end(self) -> int:
        """Get the training file end index."""
        return self._params["train"]["train_file_end"]
    
    # Derived paths (matching util.py interface)
    @property
    def mu_path(self) -> str:
        """Get the mu data path."""
        return os.path.join(self.data_path, "mu")
    
    @property
    def angles_path(self) -> str:
        """Get the angles data path."""
        return os.path.join(self.data_path, "angles")
    
    @property
    def att_masks_path(self) -> str:
        """Get the ATT masks data path."""
        return os.path.join(self.data_path, "ATT_masks")
    
    @property
    def p0_path(self) -> str:
        """Get the P_0 data path."""
        return os.path.join(self.data_path, "P_0")
    
    @property
    def c_path(self) -> str:
        """Get the c data path."""
        return os.path.join(self.data_path, "c")
    
    @property
    def sensors_path(self) -> str:
        """Get the sensors data path."""
        return os.path.join(self.data_path, "sensors")
    
    @property
    def P_0_path(self) -> str:
        """Get the P_0 data path (alias for p0_path)."""
        return self.p0_path
    
    @property
    def P_data_path(self) -> str:
        """Get the P_data path."""
        return os.path.join(self.data_path, "P_data")
    
    @property
    def P_data_noisy_path(self) -> str:
        """Get the noisy P_data path."""
        return os.path.join(self.data_path, "P_data_noisy")
    
    @property
    def mu_r_path(self) -> str:
        """Get the reconstructed mu path."""
        return os.path.join(self.data_path, "mu_r")
    
    @property
    def c_r_path(self) -> str:
        """Get the reconstructed c path."""
        return os.path.join(self.data_path, "c_r")
    
    @property
    def params_R_mu_path(self) -> str:
        """Get the R_mu parameters path."""
        return os.path.join(self.data_path, "checkpoints", "params_R_mu")
    
    @property
    def params_R_c_path(self) -> str:
        """Get the R_c parameters path."""
        return os.path.join(self.data_path, "checkpoints", "params_R_c")
    
    @property
    def checkpoints_path(self) -> str:
        """Get the checkpoints path."""
        return os.path.abspath(
            os.path.join(self.data_path, "checkpoints", str(self.checkpoint_file_index))
        )
    
    @property
    def state_path(self) -> str:
        """Get the state path."""
        return os.path.join(self.data_path, "state")
    
    @property
    def profile_dir(self) -> str:
        """Get the profile directory path."""
        return os.path.join(self.data_path, "profile")
    
    @property
    def pres_path(self) -> str:
        """Get the presentation figures path."""
        return os.path.abspath('../../Presentation/figures/')
    
    def create_directories(self) -> None:
        """Create all necessary directories for the data paths.
        
        This method creates all the directories that would be created in util.py,
        ensuring the file system is properly set up for the PACT pipeline.
        """
        directories = [
            self.data_path,
            self.mu_path,
            self.angles_path,
            self.c_path,
            self.P_data_path,
            self.P_data_noisy_path,
            self.sensors_path,
            self.P_0_path,
            self.mu_r_path,
            self.c_r_path,
            self.att_masks_path,
            self.profile_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def file(self, path: str, index: int, iteration: Optional[int] = None) -> str:
        """Generate a file path with index and optional iteration.
        
        This method replicates the file() function from util.py.
        
        Args:
            path: Base path for the file
            index: File index
            iteration: Optional iteration number
            
        Returns:
            Complete file path
        """
        if iteration is not None:
            return os.path.join(path, f"{index}_{iteration}.npy")
        else:
            return os.path.join(path, f"{index}.npy")
    
    def max_file_index(self, path: str) -> int:
        """Get the maximum file index in a directory.
        
        This method replicates the max_file_index() function from util.py.
        
        Args:
            path: Directory path to search
            
        Returns:
            Maximum file index found
        """
        if not os.path.exists(path):
            return -1
        
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        if not files:
            return -1
        
        indices = []
        for f in files:
            try:
                # Extract index from filename (before first underscore or .npy)
                name = f.replace('.npy', '')
                if '_' in name:
                    index = int(name.split('_')[0])
                else:
                    index = int(name)
                indices.append(index)
            except ValueError:
                continue
        
        return max(indices) if indices else -1
    
    def get_parameter_group(self, group: str) -> Dict[str, Any]:
        """Get a specific parameter group.
        
        Args:
            group: Parameter group name ('file', 'generate_data', 'lighting', 
                  'reconstruct', 'train')
                  
        Returns:
            Parameter group dictionary
            
        Raises:
            KeyError: If group doesn't exist
        """
        if group not in self._params:
            raise KeyError(f"Parameter group '{group}' not found")
        
        return self._params[group].copy()
    
    def update_parameter(self, group: str, key: str, value: Any, validate: bool = True) -> None:
        """Update a specific parameter.
        
        Args:
            group: Parameter group name
            key: Parameter key
            value: New parameter value
            validate: Whether to validate after update
            
        Raises:
            KeyError: If group or key doesn't exist
            ConfigValidationError: If validation fails
        """
        if group not in self._params:
            raise KeyError(f"Parameter group '{group}' not found")
        
        if key not in self._params[group]:
            raise KeyError(f"Parameter '{key}' not found in group '{group}'")
        
        self._params[group][key] = value
        self._validated = False
        
        if validate:
            self.validate()
    
    def __repr__(self) -> str:
        """String representation of the ConfigManager."""
        return f"ConfigManager(data_path='{self.data_path}', validated={self._validated})"


def create_default_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Factory function to create a default ConfigManager instance.
    
    Args:
        config_path: Path to configuration file. If None, uses 'params.yaml'
        
    Returns:
        Configured ConfigManager instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If parameter validation fails
    """
    config = ConfigManager(config_path)
    config.create_directories()
    return config


# Global instance for backward compatibility (optional)
# This can be used to maintain compatibility with existing code
_default_config: Optional[ConfigManager] = None


def get_default_config() -> ConfigManager:
    """Get the default global configuration instance.
    
    Returns:
        Default ConfigManager instance
    """
    global _default_config
    if _default_config is None:
        _default_config = create_default_config()
    return _default_config


def reset_default_config() -> None:
    """Reset the default global configuration instance."""
    global _default_config
    _default_config = None