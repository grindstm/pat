"""
Training utilities and state management for PACT reconstruction.

This module provides training state management, checkpoint handling, and training
loops for neural network-based regularization in photoacoustic reconstruction.
"""

import os
import pickle
import shutil
import signal
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
import flax.serialization
import msgpack

from flax import linen as nn
from flax.training import train_state
from flax.training import orbax_utils

from jwave.geometry import FourierSeries
from ..models.losses import LossTracker


class TrainState(train_state.TrainState):
    """
    Extended training state with additional fields for PACT reconstruction.
    
    Attributes
    ----------
    key : jax.Array
        Random key for stochastic operations
    batch_stats : Any
        Batch normalization statistics
    losses : dict
        Dictionary to store loss history
    """
    key: jax.Array
    batch_stats: Any
    losses: Dict = defaultdict(dict)


def create_train_state(
    key: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    shapes: List[Tuple],
    num_steps: int,
) -> List[TrainState]:
    """
    Create a list of training states for iterative training.
    
    This function creates separate training states for each iteration,
    which is useful when using different parameters for each step.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initialization
    model : nn.Module
        Model to train
    learning_rate : float
        Learning rate for optimizer
    shapes : list
        List of input shapes for model initialization
    num_steps : int
        Number of training steps to create states for

    Returns
    -------
    list
        List of TrainState objects
    """
    states = []
    key, key_init, *keys = random.split(key, 2 + num_steps)

    for step in range(num_steps):
        # Create dummy inputs for initialization
        inputs = tuple(random.normal(keys[step], shape) for shape in shapes)

        # Initialize model
        variables = model.init(key_init, *inputs)
        
        batch_stats = variables.get("batch_stats", None)
        tx = optax.adam(learning_rate)
        
        train_state_obj = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=tx,
            batch_stats=batch_stats,
            key=keys[step]
        )

        states.append(train_state_obj)

    return states


def save_checkpoint(
    checkpoint_path: str, 
    step: int, 
    state: Dict[str, Any], 
    keep: int = 4
) -> None:
    """
    Save training state to checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint directory
    step : int
        Current training step
    state : dict
        State dictionary to save
    keep : int, optional
        Number of checkpoints to keep
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Clean up old checkpoints
    try:
        all_steps = sorted(
            int(f.split('.')[0]) 
            for f in os.listdir(checkpoint_path) 
            if f.endswith('.msgpack')
        )
        if len(all_steps) >= keep:
            for old_step in all_steps[:-keep+1]:
                old_file = os.path.join(checkpoint_path, f"{old_step}.msgpack")
                if os.path.exists(old_file):
                    os.remove(old_file)
    except (ValueError, FileNotFoundError):
        pass

    # Serialize training states
    state_to_save = {
        "state_r": [
            flax.serialization.to_bytes(ts) for ts in state["state_r"]
        ] if "state_r" in state else [],
        "losses_batch": state.get("losses_batch", {})
    }
    
    # Save checkpoint
    checkpoint_file = os.path.join(checkpoint_path, f"{step}.msgpack")
    with open(checkpoint_file, "wb") as f:
        f.write(msgpack.packb(state_to_save, use_bin_type=True))


def restore_checkpoint(
    checkpoint_path: str, 
    step: int
) -> Optional[Dict[str, Any]]:
    """
    Restore training state from checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint directory
    step : int
        Step number to restore

    Returns
    -------
    dict or None
        Restored state dictionary, or None if restoration fails
    """
    checkpoint_file = os.path.join(checkpoint_path, f"{step}.msgpack")
    
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, "rb") as f:
            state_data = msgpack.unpackb(f.read(), raw=False)
        
        # Deserialize training states
        restored_state = {
            "state_r": [
                flax.serialization.from_bytes(None, state_bytes)
                for state_bytes in state_data["state_r"]
            ],
            "losses_batch": state_data["losses_batch"]
        }
        
        return restored_state
    
    except Exception as e:
        print(f"Failed to restore checkpoint {checkpoint_file}: {e}")
        return None


def get_latest_checkpoint_step(checkpoint_path: str) -> Optional[int]:
    """
    Get the latest checkpoint step number.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint directory

    Returns
    -------
    int or None
        Latest step number, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        steps = [
            int(f.split('.')[0])
            for f in os.listdir(checkpoint_path)
            if f.endswith('.msgpack')
        ]
        return max(steps) if steps else None
    except (ValueError, FileNotFoundError):
        return None


class TrainingManager:
    """
    Manager class for handling training workflows and state management.
    """
    
    def __init__(
        self, 
        config,
        model: nn.Module,
        dataset,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize training manager.

        Parameters
        ----------
        config : ConfigManager
            Configuration object
        model : nn.Module
            Model to train
        dataset : PADataset
            Dataset for training
        checkpoint_dir : str, optional
            Directory for checkpoints
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir or config.checkpoints_path
        self.loss_tracker = LossTracker()
        self.exit_flag = False
        
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal for graceful exit."""
        self.exit_flag = True
        print("Exit signal received, finishing current task...")
    
    def create_training_states(
        self, 
        key: jax.random.PRNGKey,
        learning_rate: float,
        input_shapes: List[Tuple],
        num_iterations: int
    ) -> List[TrainState]:
        """
        Create training states for all iterations.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key
        learning_rate : float
            Learning rate
        input_shapes : list
            Input shapes for model
        num_iterations : int
            Number of iterations

        Returns
        -------
        list
            List of training states
        """
        return create_train_state(
            key, self.model, learning_rate, input_shapes, num_iterations
        )
    
    def setup_optimizers(
        self, 
        mu_lr: float, 
        c_lr: float
    ) -> Tuple[optax.GradientTransformation, optax.GradientTransformation]:
        """
        Set up optimizers for mu and c parameters.

        Parameters
        ----------
        mu_lr : float
            Learning rate for mu optimizer
        c_lr : float
            Learning rate for c optimizer

        Returns
        -------
        tuple
            (mu_optimizer, c_optimizer)
        """
        opt_mu = optax.adam(learning_rate=mu_lr)
        opt_c = optax.adam(learning_rate=c_lr)
        return opt_mu, opt_c
    
    def train_regularizer(
        self,
        num_illuminations: int = 10,
        learning_rates: List[float] = None,
        num_iterations: int = None,
        continue_training: bool = False,
        file_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Train the regularization network.

        Parameters
        ----------
        num_illuminations : int, optional
            Number of illumination angles to use
        learning_rates : list, optional
            Learning rates [mu_lr, c_lr]
        num_iterations : int, optional
            Number of iterations
        continue_training : bool, optional
            Whether to continue from checkpoint
        file_range : tuple, optional
            (start, end) file indices to train on

        Returns
        -------
        dict
            Training results and statistics
        """
        if learning_rates is None:
            learning_rates = [self.config.lr_mu_r, self.config.lr_c_r]
        
        if num_iterations is None:
            num_iterations = self.config.recon_iterations
        
        if file_range is None:
            file_range = (self.config.train_file_start, self.config.train_file_end)
        
        # Clear JAX cache
        jax.clear_caches()
        
        # Initialize random key
        key = random.PRNGKey(59)
        key, key_train_state = random.split(key)
        
        # Create training states
        im_shape = (self.config.n[0], self.config.n[1], 1)
        state_r = self.create_training_states(
            key_train_state,
            self.config.lr_r_c,
            [im_shape, im_shape],
            num_iterations
        )
        
        # Handle checkpointing
        latest_step = get_latest_checkpoint_step(self.checkpoint_dir)
        start_file = file_range[0]
        
        if continue_training and latest_step is not None:
            restored = restore_checkpoint(self.checkpoint_dir, latest_step)
            if restored:
                state_r = restored["state_r"]
                start_file = latest_step
                print(f"Restored checkpoint from step {latest_step}")
            else:
                print("Failed to restore checkpoint, starting fresh")
                self._clean_checkpoint_dir()
        else:
            self._clean_checkpoint_dir()
        
        # Training loop
        training_results = {
            "losses": [],
            "checkpoints_saved": [],
            "files_processed": []
        }
        
        for file_index in range(start_file, file_range[1]):
            if self.exit_flag:
                break
            
            print(f"Training on file {file_index}")
            
            # Load data
            file_results = self._train_on_file(
                file_index, state_r, num_illuminations, 
                learning_rates, num_iterations, key
            )
            
            training_results["losses"].extend(file_results["losses"])
            training_results["files_processed"].append(file_index)
            
            # Save checkpoint
            state = {
                "state_r": state_r,
                "losses_batch": file_results["losses"]
            }
            save_checkpoint(self.checkpoint_dir, file_index, state)
            training_results["checkpoints_saved"].append(file_index)
        
        return training_results
    
    def _train_on_file(
        self,
        file_index: int,
        state_r: List[TrainState],
        num_illuminations: int,
        learning_rates: List[float],
        num_iterations: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """
        Train on a single file.

        Parameters
        ----------
        file_index : int
            File index to train on
        state_r : list
            List of training states
        num_illuminations : int
            Number of illuminations
        learning_rates : list
            Learning rates
        num_iterations : int
            Number of iterations
        key : jax.random.PRNGKey
            Random key

        Returns
        -------
        dict
            Training results for this file
        """
        # Select illumination angles
        num_angles = self.dataset.num_angles
        if num_illuminations < num_angles:
            illum_indices = np.linspace(
                0, num_angles, num_illuminations, endpoint=False
            ).astype(int)
        else:
            illum_indices = np.arange(num_angles)
        
        # Load data
        data = self.dataset[file_index, illum_indices]
        
        # Initialize parameters
        im_shape = (self.config.n[0], self.config.n[1], 1)
        domain = self.config.domain  # This would need to be added to config
        
        c_p = FourierSeries(jnp.zeros(im_shape) - 5, domain)
        mu_p = jnp.zeros(im_shape) + 1
        
        # Set up optimizers
        opt_mu, opt_c = self.setup_optimizers(*learning_rates)
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)
        
        # Training iterations
        losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
        
        for i in range(num_iterations):
            if self.exit_flag:
                break
            
            # This would need the actual reconstruction step implementation
            # For now, this is a placeholder structure
            loss_value = self._training_step(
                i, mu_p, c_p, data, state_r[i], 
                opt_mu_state, opt_c_state, key
            )
            
            losses["data"].append(loss_value)
            self.loss_tracker.add_loss(loss_value)
        
        return {"losses": losses}
    
    def _training_step(
        self,
        iteration: int,
        mu_p: jnp.ndarray,
        c_p: FourierSeries,
        data: Dict[str, Any],
        state: TrainState,
        opt_mu_state: Any,
        opt_c_state: Any,
        key: jax.random.PRNGKey
    ) -> float:
        """
        Perform a single training step.
        
        This is a placeholder for the actual training step implementation.
        """
        # Placeholder implementation
        return 0.1  # Would return actual loss value
    
    def _clean_checkpoint_dir(self):
        """Clean checkpoint directory."""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("Created empty checkpoint folder")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.

        Returns
        -------
        dict
            Training summary statistics
        """
        losses = self.loss_tracker.get_losses()
        
        if len(losses) == 0:
            return {"status": "No training data"}
        
        return {
            "total_iterations": len(losses),
            "final_loss": float(losses[-1]),
            "best_loss": float(jnp.min(losses)),
            "loss_reduction": float(losses[0] - losses[-1]) if len(losses) > 1 else 0.0,
            "converged": self.loss_tracker.is_converged(),
            "loss_history": losses.tolist()
        }


def print_training_losses(
    file_index: int, 
    losses: Dict[str, List[float]], 
    num_to_print: int = 5
) -> None:
    """
    Print recent training losses.

    Parameters
    ----------
    file_index : int
        Current file index
    losses : dict
        Dictionary of loss arrays
    num_to_print : int, optional
        Number of recent losses to print
    """
    print(f"\nFile {file_index} - Recent losses:")
    for loss_name, loss_values in losses.items():
        if loss_values:
            recent = loss_values[-num_to_print:]
            print(f"  {loss_name}: {recent}")


def create_progress_callback(
    mu_gt: jnp.ndarray,
    c_gt: jnp.ndarray,
    save_frequency: int = 10
) -> Callable:
    """
    Create a callback function for saving training progress.

    Parameters
    ----------
    mu_gt : jnp.ndarray
        Ground truth absorption coefficient
    c_gt : jnp.ndarray
        Ground truth sound speed
    save_frequency : int, optional
        How often to save progress

    Returns
    -------
    callable
        Progress callback function
    """
    mu_rs = []
    c_rs = []
    losses = []
    
    def save_progress(iteration: int, loss_value: float, mu_r: jnp.ndarray, c_r: jnp.ndarray):
        """Save current progress."""
        losses.append(loss_value)
        mu_rs.append(mu_r)
        c_rs.append(c_r)
        
        if iteration % save_frequency == 0:
            print(f"Iteration {iteration}: Loss = {loss_value:.6f}")
    
    return save_progress