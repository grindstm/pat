"""
JAX-specific utilities for photoacoustic computed tomography.

This module provides JAX-related utility functions including performance timing,
memory management, and common JAX operations used throughout the PACT pipeline.
"""

import os
import time
import functools
from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp


# Set JAX environment variables for optimal performance
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"


def timer(func: Callable) -> Callable:
    """
    Decorator that prints the execution time of the decorated function.
    
    This decorator is particularly useful for profiling JAX-compiled functions
    and understanding performance bottlenecks in the reconstruction pipeline.

    Parameters
    ----------
    func : callable
        Function to be timed

    Returns
    -------
    callable
        Wrapped function that returns (result, execution_time)

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     return jnp.sum(jnp.ones((1000, 1000)))
    >>> result, exec_time = slow_function()
    Finished 'slow_function' in 0.1234 secs
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper_timer


def clear_jax_cache():
    """Clear JAX compilation cache to free memory."""
    jax.clear_caches()


def get_device_info() -> dict:
    """
    Get information about available JAX devices.

    Returns
    -------
    dict
        Dictionary containing device information
    """
    devices = jax.devices()
    return {
        'num_devices': len(devices),
        'device_types': [d.device_kind for d in devices],
        'default_backend': jax.default_backend(),
        'devices': devices
    }


def ensure_jax_array(x: Any) -> jnp.ndarray:
    """
    Ensure input is a JAX array.

    Parameters
    ----------
    x : Any
        Input to convert to JAX array

    Returns
    -------
    jnp.ndarray
        JAX array
    """
    if not isinstance(x, jnp.ndarray):
        return jnp.array(x)
    return x


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Perform safe division with epsilon to avoid division by zero.

    Parameters
    ----------
    numerator : jnp.ndarray
        Numerator array
    denominator : jnp.ndarray
        Denominator array
    eps : float, optional
        Small epsilon value to add to denominator

    Returns
    -------
    jnp.ndarray
        Result of safe division
    """
    return numerator / (denominator + eps)


def normalize_array(x: jnp.ndarray, axis: int = None, eps: float = 1e-8) -> jnp.ndarray:
    """
    Normalize array to have unit norm.

    Parameters
    ----------
    x : jnp.ndarray
        Input array
    axis : int, optional
        Axis along which to normalize
    eps : float, optional
        Small epsilon for numerical stability

    Returns
    -------
    jnp.ndarray
        Normalized array
    """
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return safe_divide(x, norm, eps)


def clip_gradients(grads: Any, max_norm: float = 1.0) -> Any:
    """
    Clip gradients by global norm.

    Parameters
    ----------
    grads : Any
        Gradient tree structure
    max_norm : float, optional
        Maximum gradient norm

    Returns
    -------
    Any
        Clipped gradients
    """
    global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads)))
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    return jax.tree_map(lambda g: g * clip_factor, grads)


def tree_norm(tree: Any) -> float:
    """
    Compute the L2 norm of a pytree.

    Parameters
    ----------
    tree : Any
        PyTree structure (e.g., parameters, gradients)

    Returns
    -------
    float
        L2 norm of the tree
    """
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_leaves(tree)))


def tree_zeros_like(tree: Any) -> Any:
    """
    Create a tree of zeros with the same structure as input tree.

    Parameters
    ----------
    tree : Any
        Input tree structure

    Returns
    -------
    Any
        Tree of zeros with same structure
    """
    return jax.tree_map(jnp.zeros_like, tree)


def tree_ones_like(tree: Any) -> Any:
    """
    Create a tree of ones with the same structure as input tree.

    Parameters
    ----------
    tree : Any
        Input tree structure

    Returns
    -------
    Any
        Tree of ones with same structure
    """
    return jax.tree_map(jnp.ones_like, tree)


def check_finite(tree: Any, name: str = "tree") -> bool:
    """
    Check if all values in a tree are finite.

    Parameters
    ----------
    tree : Any
        Tree structure to check
    name : str, optional
        Name for error reporting

    Returns
    -------
    bool
        True if all values are finite

    Raises
    ------
    ValueError
        If non-finite values are found
    """
    def is_finite(x):
        return jnp.all(jnp.isfinite(x))
    
    finite_checks = jax.tree_map(is_finite, tree)
    all_finite = all(jax.tree_leaves(finite_checks))
    
    if not all_finite:
        raise ValueError(f"Non-finite values found in {name}")
    
    return all_finite


def memory_efficient_scan(f: Callable, init: Any, xs: Any, length: int = None) -> Tuple[Any, Any]:
    """
    Memory-efficient scan operation that periodically clears cache.

    Parameters
    ----------
    f : callable
        Function to scan over
    init : Any
        Initial carry value
    xs : Any
        Input sequence
    length : int, optional
        Length of sequence

    Returns
    -------
    tuple
        (final_carry, outputs)
    """
    def scan_fn(carry, x):
        # Clear cache periodically to prevent memory buildup
        if hasattr(x, 'shape') and x.shape[0] % 100 == 0:
            jax.clear_caches()
        return f(carry, x)
    
    return jax.lax.scan(scan_fn, init, xs, length=length)


def print_tree_structure(tree: Any, name: str = "tree", max_depth: int = 3) -> None:
    """
    Print the structure of a pytree for debugging.

    Parameters
    ----------
    tree : Any
        Tree structure to print
    name : str, optional
        Name of the tree
    max_depth : int, optional
        Maximum depth to print
    """
    def print_node(node, depth=0, prefix=""):
        if depth > max_depth:
            print(f"{prefix}...")
            return
            
        if isinstance(node, (list, tuple)):
            print(f"{prefix}{type(node).__name__}[{len(node)}]")
            for i, item in enumerate(node):
                print_node(item, depth + 1, f"{prefix}  [{i}] ")
        elif isinstance(node, dict):
            print(f"{prefix}dict[{len(node)}]")
            for key, value in node.items():
                print_node(value, depth + 1, f"{prefix}  {key}: ")
        elif hasattr(node, 'shape'):
            print(f"{prefix}{type(node).__name__}{node.shape} {node.dtype}")
        else:
            print(f"{prefix}{type(node).__name__}: {node}")
    
    print(f"\n{name} structure:")
    print_node(tree)
    print()