"""
Loss functions for photoacoustic computed tomography reconstruction.

This module provides various loss functions used in PACT reconstruction including
data fidelity terms, regularization losses, and composite loss functions for
multi-parameter optimization.
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad
from typing import Callable, Tuple, Optional
from functools import partial

from jwave.geometry import Medium, FourierSeries
from jaxdf.operators import compose
from jaxdf.operators.differential import gradient, diag_jacobian
from jaxdf.operators.functions import sum_over_dims
from flax import linen as nn


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Mean squared error loss.

    Parameters
    ----------
    pred : jnp.ndarray
        Predicted values
    target : jnp.ndarray
        Target values

    Returns
    -------
    float
        MSE loss value
    """
    return jnp.mean((pred - target) ** 2)


def l2_loss(x: jnp.ndarray, alpha: float) -> float:
    """
    L2 regularization loss.

    Parameters
    ----------
    x : jnp.ndarray
        Input array
    alpha : float
        Regularization strength

    Returns
    -------
    float
        L2 loss value
    """
    return alpha * jnp.mean(x ** 2)


def l1_loss(x: jnp.ndarray, alpha: float) -> float:
    """
    L1 regularization loss.

    Parameters
    ----------
    x : jnp.ndarray
        Input array
    alpha : float
        Regularization strength

    Returns
    -------
    float
        L1 loss value
    """
    return alpha * jnp.mean(jnp.abs(x))


def total_variation_loss(x: jnp.ndarray, alpha: float) -> float:
    """
    Total variation regularization loss.

    Parameters
    ----------
    x : jnp.ndarray
        Input array (2D or 3D)
    alpha : float
        Regularization strength

    Returns
    -------
    float
        TV loss value
    """
    if x.ndim == 2:
        dx = jnp.diff(x, axis=0)
        dy = jnp.diff(x, axis=1)
        tv = jnp.mean(jnp.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2))
    elif x.ndim == 3:
        dx = jnp.diff(x, axis=0)
        dy = jnp.diff(x, axis=1)
        dz = jnp.diff(x, axis=2)
        tv = jnp.mean(jnp.sqrt(dx[:-1, :-1, :] ** 2 + dy[:-1, :, :-1] ** 2 + dz[:, :-1, :-1] ** 2))
    else:
        raise ValueError("Total variation loss only supports 2D and 3D arrays")
    
    return alpha * tv


def get_sound_speed(params: FourierSeries) -> FourierSeries:
    """
    Get the sound speed from the parameterized field.
    
    Parameters
    ----------
    params : FourierSeries
        Parameterized sound speed field

    Returns
    -------
    FourierSeries
        Sound speed field in m/s
    """
    return 1420.0 + 140.0 * compose(params)(nn.sigmoid)


def get_mu(params: FourierSeries) -> FourierSeries:
    """
    Get the attenuation coefficient from the parameterized field.
    
    Parameters
    ----------
    params : FourierSeries
        Parameterized attenuation field

    Returns
    -------
    FourierSeries
        Attenuation coefficient field
    """
    return compose(params)(nn.softplus)


# Anisotropic Diffusion Functions for regularization
def divergence(u: FourierSeries, stagger: list) -> FourierSeries:
    """
    Compute divergence operator for anisotropic diffusion.

    Parameters
    ----------
    u : FourierSeries
        Input field
    stagger : list
        Stagger parameters

    Returns
    -------
    FourierSeries
        Divergence of input field
    """
    return sum_over_dims(diag_jacobian(u, stagger=stagger))


def conductivity_kernel(u: FourierSeries) -> FourierSeries:
    """
    Diffusion conductivity kernel for edge-preserving regularization.

    Parameters
    ----------
    u : FourierSeries
        Input field

    Returns
    -------
    FourierSeries
        Conductivity field
    """
    kernel = lambda x: 1 / (1 + (x / 0.03) ** 2)
    return compose(u)(kernel)


def norm(u: FourierSeries) -> FourierSeries:
    """
    Compute gradient magnitude.

    Parameters
    ----------
    u : FourierSeries
        Input field

    Returns
    -------
    FourierSeries
        Gradient magnitude
    """
    z = sum_over_dims(u ** 2)
    return compose(z)(jnp.sqrt)


@jax.jit
def anisotropic_diffusion(u: FourierSeries, stagger: list = [0.5]) -> FourierSeries:
    """
    Anisotropic diffusion regularization operator.

    Parameters
    ----------
    u : FourierSeries
        Input field
    stagger : list, optional
        Stagger parameters for gradient computation

    Returns
    -------
    FourierSeries
        Anisotropic diffusion result
    """
    grad_u = gradient(u, stagger=stagger)
    mod_gradient = norm(grad_u)
    c = conductivity_kernel(mod_gradient)
    return divergence(c * grad_u, stagger=[-0.5])


def data_fidelity_loss(
    mu_p: FourierSeries,
    c_p: FourierSeries,
    p_data: jnp.ndarray,
    forward_model: Callable,
    illumination_fn: Optional[Callable] = None,
    angles: Optional[jnp.ndarray] = None,
    att_masks: Optional[jnp.ndarray] = None
) -> float:
    """
    Compute data fidelity loss for photoacoustic reconstruction.

    Parameters
    ----------
    mu_p : FourierSeries
        Parameterized absorption coefficient
    c_p : FourierSeries
        Parameterized sound speed
    p_data : jnp.ndarray
        Measured pressure data
    forward_model : callable
        Forward simulation function
    illumination_fn : callable, optional
        Illumination function for computing initial pressure
    angles : jnp.ndarray, optional
        Illumination angles
    att_masks : jnp.ndarray, optional
        Attenuation masks

    Returns
    -------
    float
        Data fidelity loss value
    """
    mu_r = get_mu(mu_p)
    c_r = get_sound_speed(c_p)
    
    # Compute initial pressure based on illumination method
    if illumination_fn is not None and angles is not None:
        p0 = illumination_fn(mu_r.squeeze(), angles)
        p0 = FourierSeries(jnp.expand_dims(p0, -1), mu_r.domain)
    elif att_masks is not None:
        p0 = mu_r * att_masks
    else:
        raise ValueError("Either illumination_fn+angles or att_masks must be provided")
    
    # Forward simulation
    p_pred = forward_model(c_r, p0)
    
    # Data fidelity term
    return mse_loss(p_pred.squeeze(), p_data)


def composite_loss(
    mu_p: FourierSeries,
    c_p: FourierSeries,
    p_data: jnp.ndarray,
    forward_model: Callable,
    l2_alpha: float = 0.0,
    c_alpha: float = 0.0,
    tv_alpha: float = 0.0,
    c_baseline: float = 1450.0,
    illumination_fn: Optional[Callable] = None,
    angles: Optional[jnp.ndarray] = None,
    att_masks: Optional[jnp.ndarray] = None
) -> float:
    """
    Composite loss function with data fidelity and regularization terms.

    Parameters
    ----------
    mu_p : FourierSeries
        Parameterized absorption coefficient
    c_p : FourierSeries
        Parameterized sound speed
    p_data : jnp.ndarray
        Measured pressure data
    forward_model : callable
        Forward simulation function
    l2_alpha : float, optional
        L2 regularization strength for mu
    c_alpha : float, optional
        Sound speed regularization strength
    tv_alpha : float, optional
        Total variation regularization strength
    c_baseline : float, optional
        Baseline sound speed value
    illumination_fn : callable, optional
        Illumination function
    angles : jnp.ndarray, optional
        Illumination angles
    att_masks : jnp.ndarray, optional
        Attenuation masks

    Returns
    -------
    float
        Total loss value
    """
    # Data fidelity term
    data_loss = data_fidelity_loss(
        mu_p, c_p, p_data, forward_model,
        illumination_fn, angles, att_masks
    )
    
    # Regularization terms
    total_loss = data_loss
    
    if l2_alpha > 0:
        total_loss += l2_loss(mu_p.on_grid, l2_alpha)
    
    if c_alpha > 0:
        c_r = get_sound_speed(c_p)
        c_reg = mse_loss(c_r.on_grid, jnp.ones_like(c_r.on_grid) * c_baseline)
        total_loss += c_alpha * c_reg
    
    if tv_alpha > 0:
        mu_r = get_mu(mu_p)
        total_loss += total_variation_loss(mu_r.on_grid, tv_alpha)
    
    return total_loss


def create_loss_and_grad_fn(
    forward_model: Callable,
    loss_type: str = "data_fidelity",
    **loss_kwargs
) -> Callable:
    """
    Create a loss function with gradients for optimization.

    Parameters
    ----------
    forward_model : callable
        Forward simulation function
    loss_type : str, optional
        Type of loss function ("data_fidelity" or "composite")
    **loss_kwargs
        Additional arguments for loss function

    Returns
    -------
    callable
        Function that returns (loss, gradients)
    """
    if loss_type == "data_fidelity":
        loss_fn = partial(data_fidelity_loss, forward_model=forward_model, **loss_kwargs)
    elif loss_type == "composite":
        loss_fn = partial(composite_loss, forward_model=forward_model, **loss_kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    def loss_and_grad_fn(mu_p, c_p, p_data):
        """Compute loss and gradients with respect to mu_p and c_p."""
        def loss_wrapper(mu_p, c_p):
            return loss_fn(mu_p, c_p, p_data)
        
        loss_value, (grad_mu, grad_c) = value_and_grad(loss_wrapper, (0, 1))(mu_p, c_p)
        return loss_value, grad_mu, grad_c
    
    return loss_and_grad_fn


class LossTracker:
    """
    Utility class for tracking and analyzing loss values during training.
    """
    
    def __init__(self):
        self.losses = []
        self.components = {}
    
    def add_loss(self, total_loss: float, **components):
        """
        Add a loss value and its components.

        Parameters
        ----------
        total_loss : float
            Total loss value
        **components
            Individual loss components (e.g., data_loss=0.1, l2_loss=0.01)
        """
        self.losses.append(total_loss)
        for name, value in components.items():
            if name not in self.components:
                self.components[name] = []
            self.components[name].append(value)
    
    def get_losses(self) -> jnp.ndarray:
        """Get array of total losses."""
        return jnp.array(self.losses)
    
    def get_component(self, name: str) -> jnp.ndarray:
        """Get array of specific loss component."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found")
        return jnp.array(self.components[name])
    
    def get_latest_loss(self) -> float:
        """Get the most recent loss value."""
        return self.losses[-1] if self.losses else 0.0
    
    def is_converged(self, tolerance: float = 1e-6, window: int = 10) -> bool:
        """
        Check if training has converged based on loss stability.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance
        window : int, optional
            Number of recent iterations to check

        Returns
        -------
        bool
            True if converged
        """
        if len(self.losses) < window:
            return False
        
        recent_losses = jnp.array(self.losses[-window:])
        return jnp.std(recent_losses) < tolerance
    
    def clear(self):
        """Clear all stored losses."""
        self.losses.clear()
        self.components.clear()