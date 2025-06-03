"""
Reconstruction solvers for photoacoustic computed tomography.

This module provides various reconstruction algorithms including gradient-based
optimization, learned regularization, and multi-parameter reconstruction methods.
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad
import optax
from typing import Dict, List, Tuple, Callable, Optional, Any
from collections import defaultdict
import numpy as np

from jwave.geometry import FourierSeries, Domain, Medium
from ..models.losses import (
    get_mu, get_sound_speed, mse_loss, 
    create_loss_and_grad_fn, LossTracker
)
from ..data.generation import illuminate_2d_vmap


class ReconstructionSolver:
    """
    Base class for photoacoustic reconstruction solvers.
    """
    
    def __init__(self, config, domain: Domain, forward_model: Callable):
        """
        Initialize reconstruction solver.

        Parameters
        ----------
        config : ConfigManager
            Configuration object
        domain : Domain
            Simulation domain
        forward_model : callable
            Forward simulation function
        """
        self.config = config
        self.domain = domain
        self.forward_model = forward_model
        self.loss_tracker = LossTracker()
    
    def initialize_parameters(self, shape: Tuple[int, ...]) -> Tuple[FourierSeries, FourierSeries]:
        """
        Initialize reconstruction parameters.

        Parameters
        ----------
        shape : tuple
            Shape of the reconstruction domain

        Returns
        -------
        tuple
            (mu_p, c_p) - Initial parameter estimates
        """
        # Initialize with small values
        mu_p = FourierSeries(jnp.zeros(shape) + 1.0, self.domain)
        c_p = FourierSeries(jnp.zeros(shape) - 5.0, self.domain)
        return mu_p, c_p
    
    def setup_optimizers(
        self, 
        mu_lr: float, 
        c_lr: float
    ) -> Tuple[optax.GradientTransformation, optax.GradientTransformation, Any, Any]:
        """
        Set up optimizers for reconstruction parameters.

        Parameters
        ----------
        mu_lr : float
            Learning rate for absorption coefficient
        c_lr : float
            Learning rate for sound speed

        Returns
        -------
        tuple
            (opt_mu, opt_c, opt_mu_state, opt_c_state)
        """
        opt_mu = optax.adam(learning_rate=mu_lr)
        opt_c = optax.adam(learning_rate=c_lr)
        return opt_mu, opt_c
    
    def reconstruct(
        self,
        data: Dict[str, Any],
        num_iterations: int,
        learning_rates: List[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform reconstruction (to be implemented by subclasses).

        Parameters
        ----------
        data : dict
            Data dictionary from PADataset
        num_iterations : int
            Number of reconstruction iterations
        learning_rates : list
            Learning rates [mu_lr, c_lr]
        **kwargs
            Additional solver-specific arguments

        Returns
        -------
        dict
            Reconstruction results
        """
        raise NotImplementedError("Subclasses must implement reconstruct method")


class GradientDescentSolver(ReconstructionSolver):
    """
    Standard gradient descent reconstruction solver.
    """
    
    def reconstruct(
        self,
        data: Dict[str, Any],
        num_iterations: int,
        learning_rates: List[float],
        regularization: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform gradient descent reconstruction.

        Parameters
        ----------
        data : dict
            Data dictionary containing measurements and illumination info
        num_iterations : int
            Number of iterations
        learning_rates : list
            Learning rates [mu_lr, c_lr]
        regularization : dict, optional
            Regularization parameters
        **kwargs
            Additional arguments

        Returns
        -------
        dict
            Reconstruction results including parameter evolution
        """
        jax.clear_caches()
        
        # Extract data
        p_data = data["P_data"]
        angles = data.get("angles")
        att_masks = data.get("ATT_masks")
        
        # Initialize parameters
        im_shape = (self.config.n[0], self.config.n[1], 1)
        mu_p, c_p = self.initialize_parameters(im_shape)
        
        # Set up optimizers
        opt_mu = optax.adam(learning_rate=learning_rates[0])
        opt_c = optax.adam(learning_rate=learning_rates[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)
        
        # Create loss function
        if angles is not None:
            loss_fn = self._create_illumination_loss_fn(angles, p_data, regularization)
        else:
            loss_fn = self._create_attenuation_loss_fn(att_masks, p_data, regularization)
        
        # Reconstruction loop
        mu_rs = []
        c_rs = []
        losses = {"data": [], "mu": [], "c": [], "total": []}
        
        for i in range(num_iterations):
            # Compute loss and gradients
            loss_value, grad_mu, grad_c = loss_fn(mu_p, c_p)
            
            # Update parameters
            updates_mu, opt_mu_state = opt_mu.update(grad_mu, opt_mu_state)
            updates_c, opt_c_state = opt_c.update(grad_c, opt_c_state)
            
            mu_p = optax.apply_updates(mu_p, updates_mu)
            c_p = optax.apply_updates(c_p, updates_c)
            
            # Store results
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            mu_rs.append(mu_r.on_grid.squeeze())
            c_rs.append(c_r.on_grid.squeeze())
            losses["data"].append(float(loss_value))
            losses["total"].append(float(loss_value))
            
            self.loss_tracker.add_loss(float(loss_value))
        
        return {
            "mu_rs": mu_rs,
            "c_rs": c_rs,
            "losses": losses,
            "final_mu": mu_rs[-1],
            "final_c": c_rs[-1],
            "converged": self.loss_tracker.is_converged()
        }
    
    def _create_illumination_loss_fn(
        self, 
        angles: jnp.ndarray, 
        p_data: jnp.ndarray,
        regularization: Optional[Dict[str, float]] = None
    ) -> Callable:
        """Create loss function for illumination-based reconstruction."""
        
        def loss_fn(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            # Compute initial pressure using illumination
            p0 = illuminate_2d_vmap(mu_r.squeeze(), angles, self.config.attenuation)
            p0 = FourierSeries(jnp.expand_dims(p0, -1), self.domain)
            
            # Forward simulation
            medium = Medium(
                domain=self.domain, 
                sound_speed=c_r[0], 
                pml_size=self.config.pml_margin[0]
            )
            p_pred = self.forward_model(medium, p0)
            
            # Data fidelity loss
            data_loss = mse_loss(p_pred.squeeze(), p_data)
            
            # Add regularization if specified
            total_loss = data_loss
            if regularization:
                if "l2_alpha" in regularization:
                    total_loss += regularization["l2_alpha"] * jnp.mean(mu_p.on_grid ** 2)
                if "c_alpha" in regularization:
                    c_baseline = regularization.get("c_baseline", self.config.c)
                    total_loss += regularization["c_alpha"] * mse_loss(
                        c_r.on_grid, jnp.ones_like(c_r.on_grid) * c_baseline
                    )
            
            return total_loss
        
        return jax.jit(value_and_grad(loss_fn, (0, 1)))
    
    def _create_attenuation_loss_fn(
        self, 
        att_masks: jnp.ndarray, 
        p_data: jnp.ndarray,
        regularization: Optional[Dict[str, float]] = None
    ) -> Callable:
        """Create loss function for attenuation mask-based reconstruction."""
        
        att_masks_fs = FourierSeries(att_masks, self.domain)
        
        def loss_fn(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            # Compute initial pressure using attenuation masks
            p0 = mu_r * att_masks_fs
            
            # Forward simulation
            medium = Medium(
                domain=self.domain, 
                sound_speed=c_r[0], 
                pml_size=self.config.pml_margin[0]
            )
            p_pred = self.forward_model(medium, p0)
            
            return mse_loss(p_pred.squeeze(), p_data)
        
        return jax.jit(value_and_grad(loss_fn, (0, 1)))


class LearnedRegularizationSolver(ReconstructionSolver):
    """
    Reconstruction solver with learned regularization networks.
    """
    
    def __init__(self, config, domain: Domain, forward_model: Callable, regularizer_states: List):
        """
        Initialize solver with learned regularization.

        Parameters
        ----------
        config : ConfigManager
            Configuration object
        domain : Domain
            Simulation domain
        forward_model : callable
            Forward simulation function
        regularizer_states : list
            List of trained regularizer states
        """
        super().__init__(config, domain, forward_model)
        self.regularizer_states = regularizer_states
    
    def reconstruct(
        self,
        data: Dict[str, Any],
        num_iterations: int,
        learning_rates: List[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform reconstruction with learned regularization.

        Parameters
        ----------
        data : dict
            Data dictionary
        num_iterations : int
            Number of iterations
        learning_rates : list
            Learning rates [mu_lr, c_lr]
        **kwargs
            Additional arguments

        Returns
        -------
        dict
            Reconstruction results
        """
        jax.clear_caches()
        
        # Extract data
        p_data = data["P_data"]
        angles = data.get("angles")
        att_masks = data.get("ATT_masks")
        
        # Initialize parameters
        im_shape = (self.config.n[0], self.config.n[1], 1)
        mu_p, c_p = self.initialize_parameters(im_shape)
        
        # Set up optimizers
        opt_mu = optax.adam(learning_rate=learning_rates[0])
        opt_c = optax.adam(learning_rate=learning_rates[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)
        
        # Create base loss function
        if angles is not None:
            base_loss_fn = self._create_illumination_step_fn(angles, p_data)
        else:
            base_loss_fn = self._create_attenuation_step_fn(att_masks, p_data)
        
        # Reconstruction loop
        mu_rs = []
        c_rs = []
        losses = {"data": [], "mu": [], "c": [], "total": []}
        
        for i in range(min(num_iterations, len(self.regularizer_states))):
            # Compute base gradients
            loss_value, grad_mu, grad_c = base_loss_fn(mu_p, c_p)
            
            # Apply learned regularization to gradients
            if i < len(self.regularizer_states):
                regularized_grad_c = self.regularizer_states[i].apply_fn(
                    self.regularizer_states[i].params,
                    mu_p.on_grid,
                    grad_mu.on_grid,
                    c_p.on_grid,
                    grad_c.on_grid,
                    train=False
                )
                grad_c = FourierSeries(regularized_grad_c, self.domain)
            
            # Update parameters
            updates_mu, opt_mu_state = opt_mu.update(grad_mu, opt_mu_state)
            updates_c, opt_c_state = opt_c.update(grad_c, opt_c_state)
            
            mu_p = optax.apply_updates(mu_p, updates_mu)
            c_p = optax.apply_updates(c_p, updates_c)
            
            # Store results
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            mu_rs.append(mu_r.on_grid.squeeze())
            c_rs.append(c_r.on_grid.squeeze())
            losses["data"].append(float(loss_value))
            losses["total"].append(float(loss_value))
            
            self.loss_tracker.add_loss(float(loss_value))
        
        return {
            "mu_rs": mu_rs,
            "c_rs": c_rs,
            "losses": losses,
            "final_mu": mu_rs[-1],
            "final_c": c_rs[-1],
            "converged": self.loss_tracker.is_converged()
        }
    
    def _create_illumination_step_fn(self, angles: jnp.ndarray, p_data: jnp.ndarray) -> Callable:
        """Create step function for illumination-based reconstruction."""
        
        def step_fn(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            p0 = illuminate_2d_vmap(mu_r.squeeze(), angles, self.config.attenuation)
            p0 = FourierSeries(jnp.expand_dims(p0, -1), self.domain)
            
            medium = Medium(
                domain=self.domain, 
                sound_speed=c_r[0], 
                pml_size=self.config.pml_margin[0]
            )
            p_pred = self.forward_model(medium, p0)
            
            return mse_loss(p_pred.squeeze(), p_data)
        
        return jax.jit(value_and_grad(step_fn, (0, 1)))
    
    def _create_attenuation_step_fn(self, att_masks: jnp.ndarray, p_data: jnp.ndarray) -> Callable:
        """Create step function for attenuation mask-based reconstruction."""
        
        att_masks_fs = FourierSeries(att_masks, self.domain)
        
        def step_fn(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            
            p0 = mu_r * att_masks_fs
            
            medium = Medium(
                domain=self.domain, 
                sound_speed=c_r[0], 
                pml_size=self.config.pml_margin[0]
            )
            p_pred = self.forward_model(medium, p0)
            
            return mse_loss(p_pred.squeeze(), p_data)
        
        return jax.jit(value_and_grad(step_fn, (0, 1)))


class MultiParameterSolver(ReconstructionSolver):
    """
    Solver for simultaneous reconstruction of multiple parameters.
    """
    
    def reconstruct(
        self,
        data: Dict[str, Any],
        num_iterations: int,
        learning_rates: List[float],
        alternating: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform multi-parameter reconstruction.

        Parameters
        ----------
        data : dict
            Data dictionary
        num_iterations : int
            Number of iterations
        learning_rates : list
            Learning rates for each parameter
        alternating : bool, optional
            Whether to alternate parameter updates
        **kwargs
            Additional arguments

        Returns
        -------
        dict
            Reconstruction results
        """
        # Implementation for multi-parameter reconstruction
        # This would include logic for handling multiple parameters
        # and potentially alternating optimization schemes
        
        return self._standard_multiparameter_reconstruction(
            data, num_iterations, learning_rates, alternating
        )
    
    def _standard_multiparameter_reconstruction(
        self,
        data: Dict[str, Any],
        num_iterations: int,
        learning_rates: List[float],
        alternating: bool
    ) -> Dict[str, Any]:
        """Standard multi-parameter reconstruction implementation."""
        
        # This would contain the actual implementation
        # For now, delegate to gradient descent solver
        gradient_solver = GradientDescentSolver(self.config, self.domain, self.forward_model)
        return gradient_solver.reconstruct(data, num_iterations, learning_rates)


def create_solver(
    solver_type: str,
    config,
    domain: Domain,
    forward_model: Callable,
    **kwargs
) -> ReconstructionSolver:
    """
    Factory function to create reconstruction solvers.

    Parameters
    ----------
    solver_type : str
        Type of solver ("gradient_descent", "learned_regularization", "multi_parameter")
    config : ConfigManager
        Configuration object
    domain : Domain
        Simulation domain
    forward_model : callable
        Forward simulation function
    **kwargs
        Additional solver-specific arguments

    Returns
    -------
    ReconstructionSolver
        Configured solver instance
    """
    if solver_type == "gradient_descent":
        return GradientDescentSolver(config, domain, forward_model)
    elif solver_type == "learned_regularization":
        regularizer_states = kwargs.get("regularizer_states", [])
        return LearnedRegularizationSolver(config, domain, forward_model, regularizer_states)
    elif solver_type == "multi_parameter":
        return MultiParameterSolver(config, domain, forward_model)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def batch_reconstruct(
    solver: ReconstructionSolver,
    dataset,
    file_indices: List[int],
    num_iterations: int,
    learning_rates: List[float],
    save_results: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform batch reconstruction on multiple files.

    Parameters
    ----------
    solver : ReconstructionSolver
        Reconstruction solver to use
    dataset : PADataset
        Dataset containing the data
    file_indices : list
        List of file indices to reconstruct
    num_iterations : int
        Number of iterations per reconstruction
    learning_rates : list
        Learning rates
    save_results : bool, optional
        Whether to save reconstruction results
    **kwargs
        Additional arguments

    Returns
    -------
    dict
        Batch reconstruction results
    """
    batch_results = {
        "file_indices": file_indices,
        "individual_results": {},
        "summary_stats": {}
    }
    
    total_losses = []
    convergence_rates = []
    
    for file_idx in file_indices:
        print(f"Reconstructing file {file_idx}")
        
        # Load data
        data = dataset[file_idx]
        
        # Perform reconstruction
        result = solver.reconstruct(data, num_iterations, learning_rates, **kwargs)
        
        # Store results
        batch_results["individual_results"][file_idx] = result
        total_losses.extend(result["losses"]["total"])
        convergence_rates.append(1 if result["converged"] else 0)
        
        # Save results if requested
        if save_results:
            _save_reconstruction_result(solver.config, file_idx, result)
    
    # Compute summary statistics
    batch_results["summary_stats"] = {
        "mean_final_loss": float(np.mean([
            r["losses"]["total"][-1] for r in batch_results["individual_results"].values()
        ])),
        "convergence_rate": float(np.mean(convergence_rates)),
        "total_files": len(file_indices),
        "successful_reconstructions": len(batch_results["individual_results"])
    }
    
    return batch_results


def _save_reconstruction_result(config, file_idx: int, result: Dict[str, Any]) -> None:
    """Save reconstruction results to disk."""
    from ..utils.io import safe_save_array
    
    mu_rs = result["mu_rs"]
    c_rs = result["c_rs"]
    
    # Save each iteration
    for i, (mu_r, c_r) in enumerate(zip(mu_rs, c_rs)):
        mu_file = config.file(config.mu_r_path, file_idx, i)
        c_file = config.file(config.c_r_path, file_idx, i)
        
        safe_save_array(mu_r, mu_file)
        safe_save_array(c_r, c_file)