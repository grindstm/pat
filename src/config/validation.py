"""
Configuration validation utilities for PACT.

This module provides validation functions to ensure that configuration parameters
are valid and consistent for photoacoustic reconstruction. It includes checks for
parameter ranges, dependencies, and compatibility between different settings.

Key functionality:
- Parameter range validation
- Cross-parameter dependency checking
- Configuration consistency verification
- Error reporting and suggestions
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_positive_number(value: Union[int, float], name: str, min_value: float = 0) -> None:
    """Validate that a value is a positive number.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (exclusive)
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{name} must be a number, got {type(value).__name__}")
    
    if value <= min_value:
        raise ConfigValidationError(f"{name} must be greater than {min_value}, got {value}")


def validate_positive_integer(value: int, name: str, min_value: int = 0) -> None:
    """Validate that a value is a positive integer.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (exclusive)
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ConfigValidationError(f"{name} must be an integer, got {type(value).__name__}")
    
    if value <= min_value:
        raise ConfigValidationError(f"{name} must be greater than {min_value}, got {value}")


def validate_tuple_length(value: Tuple, name: str, expected_length: int) -> None:
    """Validate that a tuple has the expected length.
    
    Args:
        value: The tuple to validate
        name: Parameter name for error messages
        expected_length: Expected tuple length
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, (tuple, list)):
        raise ConfigValidationError(f"{name} must be a tuple or list, got {type(value).__name__}")
    
    if len(value) != expected_length:
        raise ConfigValidationError(
            f"{name} must have length {expected_length}, got {len(value)}"
        )


def validate_power_of_two(value: int, name: str) -> None:
    """Validate that a value is a power of 2.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ConfigValidationError(f"{name} must be an integer, got {type(value).__name__}")
    
    if value <= 0 or (value & (value - 1)) != 0:
        raise ConfigValidationError(f"{name} must be a power of 2, got {value}")


def validate_range(value: Union[int, float], name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a specified range.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{name} must be a number, got {type(value).__name__}")
    
    if not (min_val <= value <= max_val):
        raise ConfigValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def validate_file_parameters(params: Dict[str, Any]) -> None:
    """Validate file-related parameters.
    
    Args:
        params: File parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if "data_path" not in params:
        raise ConfigValidationError("data_path is required in file parameters")
    
    if not isinstance(params["data_path"], str):
        raise ConfigValidationError("data_path must be a string")


def validate_generate_data_parameters(params: Dict[str, Any]) -> None:
    """Validate data generation parameters.
    
    Args:
        params: Generate data parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    required_params = [
        "batch_size", "N", "shrink_factor", "dims", "dx", "c", "c_periodicity",
        "c_variation_amplitude", "c_blood", "cfl", "pml_margin", "tissue_margin",
        "sensor_margin", "num_sensors", "noise_amplitude"
    ]
    
    for param in required_params:
        if param not in params:
            raise ConfigValidationError(f"{param} is required in generate_data parameters")
    
    # Validate individual parameters
    validate_positive_integer(params["batch_size"], "batch_size")
    validate_positive_integer(params["shrink_factor"], "shrink_factor")
    validate_range(params["dims"], "dims", 2, 3)
    
    # Validate N dimensions
    N = params["N"]
    validate_tuple_length(N, "N", 3)
    for i, n in enumerate(N):
        validate_power_of_two(n, f"N[{i}]")
    
    # Validate dx
    dx = params["dx"]
    validate_tuple_length(dx, "dx", 3)
    for i, d in enumerate(dx):
        validate_positive_number(d, f"dx[{i}]")
    
    # Validate speed of sound parameters
    validate_positive_number(params["c"], "c")
    validate_positive_number(params["c_blood"], "c_blood")
    validate_positive_integer(params["c_periodicity"], "c_periodicity")
    validate_positive_number(params["c_variation_amplitude"], "c_variation_amplitude")
    
    # Validate CFL condition
    validate_range(params["cfl"], "cfl", 0.0, 1.0)
    
    # Validate margins
    for margin_name in ["pml_margin", "tissue_margin", "sensor_margin"]:
        margin = params[margin_name]
        validate_tuple_length(margin, margin_name, 3)
        for i, m in enumerate(margin):
            validate_positive_integer(m, f"{margin_name}[{i}]")
    
    # Validate sensors
    validate_positive_integer(params["num_sensors"], "num_sensors")
    validate_positive_number(params["noise_amplitude"], "noise_amplitude", min_value=-1)


def validate_lighting_parameters(params: Dict[str, Any]) -> None:
    """Validate lighting parameters.
    
    Args:
        params: Lighting parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    required_params = ["lighting_attenuation", "num_lighting_angles", "attenuation"]
    
    for param in required_params:
        if param not in params:
            raise ConfigValidationError(f"{param} is required in lighting parameters")
    
    if not isinstance(params["lighting_attenuation"], bool):
        raise ConfigValidationError("lighting_attenuation must be a boolean")
    
    validate_positive_integer(params["num_lighting_angles"], "num_lighting_angles")
    validate_positive_number(params["attenuation"], "attenuation")


def validate_reconstruct_parameters(params: Dict[str, Any]) -> None:
    """Validate reconstruction parameters.
    
    Args:
        params: Reconstruction parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    required_params = [
        "recon_iterations", "lr_mu_r", "lr_c_r", "recon_file_start", "recon_file_end"
    ]
    
    for param in required_params:
        if param not in params:
            raise ConfigValidationError(f"{param} is required in reconstruct parameters")
    
    validate_positive_integer(params["recon_iterations"], "recon_iterations")
    validate_positive_number(params["lr_mu_r"], "lr_mu_r")
    validate_positive_number(params["lr_c_r"], "lr_c_r")
    
    # Validate file range
    start = params["recon_file_start"]
    end = params["recon_file_end"]
    validate_positive_integer(start, "recon_file_start", min_value=-1)
    validate_positive_integer(end, "recon_file_end", min_value=-1)
    
    if start >= end:
        raise ConfigValidationError(
            f"recon_file_start ({start}) must be less than recon_file_end ({end})"
        )


def validate_train_parameters(params: Dict[str, Any]) -> None:
    """Validate training parameters.
    
    Args:
        params: Training parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    required_params = [
        "checkpoint_index", "lr_R_mu", "lr_R_c", "dropout", "train_file_start", "train_file_end"
    ]
    
    for param in required_params:
        if param not in params:
            raise ConfigValidationError(f"{param} is required in train parameters")
    
    validate_positive_integer(params["checkpoint_index"], "checkpoint_index", min_value=-1)
    validate_positive_number(params["lr_R_mu"], "lr_R_mu")
    validate_positive_number(params["lr_R_c"], "lr_R_c")
    validate_range(params["dropout"], "dropout", 0.0, 1.0)
    
    # Validate file range
    start = params["train_file_start"]
    end = params["train_file_end"]
    validate_positive_integer(start, "train_file_start", min_value=-1)
    validate_positive_integer(end, "train_file_end", min_value=-1)
    
    if start >= end:
        raise ConfigValidationError(
            f"train_file_start ({start}) must be less than train_file_end ({end})"
        )


def validate_parameter_consistency(params: Dict[str, Any]) -> None:
    """Validate consistency between different parameter groups.
    
    Args:
        params: Complete parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Check N dimensions vs dims setting
    if "generate_data" in params:
        gen_params = params["generate_data"]
        dims = gen_params.get("dims", 3)
        N = gen_params.get("N", [128, 128, 128])
        
        # N should always be a 3-element array regardless of dims
        # dims refers to the dimensionality of the input images (2D or 3D)
        # but the grid N is always 3D for the simulation
        if not (dims == 2 or dims == 3):
            raise ConfigValidationError(f"dims must be 2 or 3, got {dims}")
        
        # Check that margins don't exceed grid size
        for margin_name in ["pml_margin", "tissue_margin", "sensor_margin"]:
            if margin_name in gen_params:
                margin = gen_params[margin_name]
                for i, (m, n) in enumerate(zip(margin, N)):
                    if 2 * m >= n:
                        raise ConfigValidationError(
                            f"{margin_name}[{i}] ({m}) is too large for N[{i}] ({n}). "
                            f"Total margin (2*{m}={2*m}) must be less than grid size ({n})"
                        )


def validate_all_parameters(params: Dict[str, Any]) -> None:
    """Validate all parameter groups and their consistency.
    
    Args:
        params: Complete parameters dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Validate individual parameter groups
    if "file" in params:
        validate_file_parameters(params["file"])
    
    if "generate_data" in params:
        validate_generate_data_parameters(params["generate_data"])
    
    if "lighting" in params:
        validate_lighting_parameters(params["lighting"])
    
    if "reconstruct" in params:
        validate_reconstruct_parameters(params["reconstruct"])
    
    if "train" in params:
        validate_train_parameters(params["train"])
    
    # Validate cross-parameter consistency
    validate_parameter_consistency(params)