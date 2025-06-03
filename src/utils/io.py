"""
Input/Output utilities for photoacoustic computed tomography.

This module provides file I/O operations, path management, and data persistence
utilities used throughout the PACT reconstruction pipeline.
"""

import os
import pickle
import shutil
from typing import Optional, Union, Any, List
import numpy as np
import jax.numpy as jnp


def file_path(base_path: str, index: int, iteration: Optional[int] = None) -> str:
    """
    Generate standardized file path for data files.

    Parameters
    ----------
    base_path : str
        Base directory path
    index : int
        File index
    iteration : int, optional
        Iteration number for reconstruction results

    Returns
    -------
    str
        Complete file path

    Examples
    --------
    >>> file_path("/data/mu", 42)
    '/data/mu/42.npy'
    >>> file_path("/data/mu_r", 42, 10)
    '/data/mu_r/42_10.npy'
    """
    if iteration is not None:
        return os.path.join(base_path, f"{index}_{iteration}.npy")
    else:
        return os.path.join(base_path, f"{index}.npy")


def max_file_index(path: str) -> int:
    """
    Find the maximum file index in a directory.

    Parameters
    ----------
    path : str
        Directory path to search

    Returns
    -------
    int
        Maximum file index found, or 0 if no files exist

    Examples
    --------
    >>> max_file_index("/data/mu")  # Contains 0.npy, 1.npy, 2.npy
    3
    """
    if not os.path.exists(path):
        return 0
    
    try:
        indices = [
            int(filename.split("_")[0])
            for filename in os.listdir(path)
            if filename.split("_")[0].isdigit()
        ]
        return max(indices, default=-1) + 1
    except (ValueError, IndexError):
        return 0


def max_iteration(path: str) -> int:
    """
    Find the maximum iteration number in a reconstruction results directory.

    Parameters
    ----------
    path : str
        Directory path containing reconstruction files

    Returns
    -------
    int
        Maximum iteration number found, or 0 if no files exist

    Examples
    --------
    >>> max_iteration("/data/mu_r")  # Contains 0_0.npy, 0_1.npy, 0_2.npy
    2
    """
    if not os.path.exists(path):
        return 0
    
    try:
        iterations = [
            int(filename.split("_")[1].split(".")[0])
            for filename in os.listdir(path)
            if len(filename.split("_")) > 1 and filename.split("_")[1].split(".")[0].isdigit()
        ]
        return max(iterations, default=0)
    except (ValueError, IndexError):
        return 0


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def safe_save_array(array: Union[np.ndarray, jnp.ndarray], filepath: str) -> None:
    """
    Safely save array to file with directory creation.

    Parameters
    ----------
    array : np.ndarray or jnp.ndarray
        Array to save
    filepath : str
        Complete file path including filename
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        ensure_directory(directory)
    
    # Convert JAX array to numpy if needed
    if isinstance(array, jnp.ndarray):
        array = np.array(array)
    
    # Save array
    np.save(filepath, array)


def safe_load_array(filepath: str, default_value: Optional[Any] = None) -> Optional[np.ndarray]:
    """
    Safely load array from file with error handling.

    Parameters
    ----------
    filepath : str
        Path to the file to load
    default_value : Any, optional
        Value to return if file doesn't exist

    Returns
    -------
    np.ndarray or None
        Loaded array, or default_value if file doesn't exist
    """
    try:
        if os.path.exists(filepath):
            return np.load(filepath)
        else:
            return default_value
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return default_value


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file with directory creation.

    Parameters
    ----------
    obj : Any
        Object to save
    filepath : str
        Complete file path including filename
    """
    directory = os.path.dirname(filepath)
    if directory:
        ensure_directory(directory)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str, default_value: Optional[Any] = None) -> Any:
    """
    Load object from pickle file with error handling.

    Parameters
    ----------
    filepath : str
        Path to the pickle file
    default_value : Any, optional
        Value to return if file doesn't exist

    Returns
    -------
    Any
        Loaded object, or default_value if file doesn't exist
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return default_value
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return default_value


def cleanup_old_files(directory: str, pattern: str, keep_latest: int = 5) -> None:
    """
    Clean up old files in directory, keeping only the latest N files.

    Parameters
    ----------
    directory : str
        Directory to clean up
    pattern : str
        File pattern to match (e.g., "*.npy")
    keep_latest : int, optional
        Number of latest files to keep
    """
    if not os.path.exists(directory):
        return
    
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old files
    for filepath in files[keep_latest:]:
        try:
            os.remove(filepath)
            print(f"Removed old file: {filepath}")
        except Exception as e:
            print(f"Warning: Could not remove {filepath}: {e}")


def get_directory_size(directory: str) -> int:
    """
    Get total size of directory in bytes.

    Parameters
    ----------
    directory : str
        Directory path

    Returns
    -------
    int
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def list_data_files(directory: str, extension: str = ".npy") -> List[str]:
    """
    List all data files in directory with given extension.

    Parameters
    ----------
    directory : str
        Directory to search
    extension : str, optional
        File extension to filter by

    Returns
    -------
    list
        List of file paths
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(os.path.join(directory, filename))
    
    return sorted(files)


def validate_file_integrity(filepath: str) -> bool:
    """
    Validate that a numpy file can be loaded successfully.

    Parameters
    ----------
    filepath : str
        Path to the file to validate

    Returns
    -------
    bool
        True if file is valid, False otherwise
    """
    try:
        if not os.path.exists(filepath):
            return False
        
        # Try to load the file
        data = np.load(filepath)
        
        # Check if data is valid
        if data is None or data.size == 0:
            return False
        
        return True
    except Exception:
        return False


def backup_file(filepath: str, backup_dir: str = "backups") -> str:
    """
    Create a backup copy of a file.

    Parameters
    ----------
    filepath : str
        Path to file to backup
    backup_dir : str, optional
        Directory to store backups

    Returns
    -------
    str
        Path to backup file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Create backup directory
    directory = os.path.dirname(filepath)
    backup_path = os.path.join(directory, backup_dir)
    ensure_directory(backup_path)
    
    # Generate backup filename with timestamp
    import time
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = int(time.time())
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_filepath = os.path.join(backup_path, backup_filename)
    
    # Copy file
    shutil.copy2(filepath, backup_filepath)
    return backup_filepath