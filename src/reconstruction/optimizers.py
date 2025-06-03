"""
Optimization utilities for photoacoustic computed tomography reconstruction.

This module provides specialized optimizers, learning rate schedules, and
optimization strategies tailored for PACT reconstruction problems.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Callable, Optional, Tuple, NamedTuple
from functools import partial


class OptimizationState(NamedTuple):
    """State for custom optimization algorithms."""
    params: Any
    opt_state: Any
    iteration: int
    best_loss: float
    best_params: Any
    patience_counter: int


def create_adam_optimizer(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0
) -> optax.GradientTransformation:
    """
    Create Adam optimizer with optional weight decay.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    beta1 : float, optional
        Exponential decay rate for first moment estimates
    beta2 : float, optional
        Exponential decay rate for second moment estimates
    eps : float, optional
        Small constant for numerical stability
    weight_decay : float, optional
        Weight decay coefficient

    Returns
    -------
    optax.GradientTransformation
        Configured Adam optimizer
    """
    if weight_decay > 0:
        return optax.adamw(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        return optax.adam(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=eps
        )


def create_sgd_optimizer(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = True
) -> optax.GradientTransformation:
    """
    Create SGD optimizer with momentum.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    momentum : float, optional
        Momentum coefficient
    nesterov : bool, optional
        Whether to use Nesterov momentum

    Returns
    -------
    optax.GradientTransformation
        Configured SGD optimizer
    """
    return optax.sgd(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )


def create_rmsprop_optimizer(
    learning_rate: float,
    decay: float = 0.9,
    eps: float = 1e-8
) -> optax.GradientTransformation:
    """
    Create RMSprop optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    decay : float, optional
        Decay rate for moving average
    eps : float, optional
        Small constant for numerical stability

    Returns
    -------
    optax.GradientTransformation
        Configured RMSprop optimizer
    """
    return optax.rmsprop(
        learning_rate=learning_rate,
        decay=decay,
        eps=eps
    )


def exponential_decay_schedule(
    init_value: float,
    decay_rate: float,
    decay_steps: int,
    staircase: bool = False
) -> optax.Schedule:
    """
    Create exponential decay learning rate schedule.

    Parameters
    ----------
    init_value : float
        Initial learning rate
    decay_rate : float
        Decay rate
    decay_steps : int
        Number of steps between decay
    staircase : bool, optional
        Whether to apply decay in discrete steps

    Returns
    -------
    optax.Schedule
        Learning rate schedule
    """
    return optax.exponential_decay(
        init_value=init_value,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0
) -> optax.Schedule:
    """
    Create cosine decay learning rate schedule.

    Parameters
    ----------
    init_value : float
        Initial learning rate
    decay_steps : int
        Number of steps for complete decay
    alpha : float, optional
        Minimum learning rate as fraction of init_value

    Returns
    -------
    optax.Schedule
        Learning rate schedule
    """
    return optax.cosine_decay_schedule(
        init_value=init_value,
        decay_steps=decay_steps,
        alpha=alpha
    )


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> optax.Schedule:
    """
    Create warmup followed by cosine decay schedule.

    Parameters
    ----------
    init_value : float
        Initial learning rate
    peak_value : float
        Peak learning rate after warmup
    warmup_steps : int
        Number of warmup steps
    decay_steps : int
        Number of decay steps
    end_value : float, optional
        Final learning rate

    Returns
    -------
    optax.Schedule
        Learning rate schedule
    """
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value
    )


def create_gradient_clipping(
    max_norm: float,
    norm_type: str = "global"
) -> optax.GradientTransformation:
    """
    Create gradient clipping transformation.

    Parameters
    ----------
    max_norm : float
        Maximum gradient norm
    norm_type : str, optional
        Type of norm ("global" or "per_param")

    Returns
    -------
    optax.GradientTransformation
        Gradient clipping transformation
    """
    if norm_type == "global":
        return optax.clip_by_global_norm(max_norm)
    elif norm_type == "per_param":
        return optax.clip(max_norm)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_composite_optimizer(
    base_optimizer: optax.GradientTransformation,
    gradient_clip_norm: Optional[float] = None,
    weight_decay: Optional[float] = None
) -> optax.GradientTransformation:
    """
    Create composite optimizer with additional transformations.

    Parameters
    ----------
    base_optimizer : optax.GradientTransformation
        Base optimizer
    gradient_clip_norm : float, optional
        Maximum gradient norm for clipping
    weight_decay : float, optional
        Weight decay coefficient

    Returns
    -------
    optax.GradientTransformation
        Composite optimizer
    """
    transformations = []
    
    if gradient_clip_norm is not None:
        transformations.append(create_gradient_clipping(gradient_clip_norm))
    
    transformations.append(base_optimizer)
    
    if weight_decay is not None:
        transformations.append(optax.add_decayed_weights(weight_decay))
    
    return optax.chain(*transformations)


class AdaptiveLearningRateOptimizer:
    """
    Optimizer with adaptive learning rate based on loss progress.
    """
    
    def __init__(
        self,
        base_optimizer: optax.GradientTransformation,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-8,
        threshold: float = 1e-4
    ):
        """
        Initialize adaptive learning rate optimizer.

        Parameters
        ----------
        base_optimizer : optax.GradientTransformation
            Base optimizer
        patience : int, optional
            Number of steps to wait before reducing learning rate
        factor : float, optional
            Factor by which to reduce learning rate
        min_lr : float, optional
            Minimum learning rate
        threshold : float, optional
            Threshold for measuring improvement
        """
        self.base_optimizer = base_optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
    def init(self, params: Any) -> OptimizationState:
        """Initialize optimization state."""
        base_state = self.base_optimizer.init(params)
        return OptimizationState(
            params=params,
            opt_state=base_state,
            iteration=0,
            best_loss=float('inf'),
            best_params=params,
            patience_counter=0
        )
    
    def update(
        self,
        gradients: Any,
        state: OptimizationState,
        current_loss: float
    ) -> Tuple[Any, OptimizationState]:
        """
        Update parameters with adaptive learning rate.

        Parameters
        ----------
        gradients : Any
            Gradients
        state : OptimizationState
            Current optimization state
        current_loss : float
            Current loss value

        Returns
        -------
        tuple
            (updates, new_state)
        """
        # Check if loss improved
        if current_loss < state.best_loss - self.threshold:
            # Loss improved
            best_loss = current_loss
            best_params = state.params
            patience_counter = 0
        else:
            # No improvement
            best_loss = state.best_loss
            best_params = state.best_params
            patience_counter = state.patience_counter + 1
        
        # Update base optimizer
        updates, new_opt_state = self.base_optimizer.update(
            gradients, state.opt_state, state.params
        )
        
        # Reduce learning rate if needed
        if patience_counter >= self.patience:
            # Scale down the learning rate in the optimizer state
            new_opt_state = self._scale_learning_rate(new_opt_state, self.factor)
            patience_counter = 0
        
        new_state = OptimizationState(
            params=state.params,
            opt_state=new_opt_state,
            iteration=state.iteration + 1,
            best_loss=best_loss,
            best_params=best_params,
            patience_counter=patience_counter
        )
        
        return updates, new_state
    
    def _scale_learning_rate(self, opt_state: Any, factor: float) -> Any:
        """Scale learning rate in optimizer state."""
        # This is a simplified implementation
        # In practice, this would need to handle different optimizer types
        return opt_state


class EarlyStopping:
    """
    Early stopping utility for reconstruction optimization.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.

        Parameters
        ----------
        patience : int, optional
            Number of iterations to wait for improvement
        min_delta : float, optional
            Minimum change to qualify as improvement
        restore_best_weights : bool, optional
            Whether to restore best parameters when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_params = None
        self.wait = 0
        self.stopped_iteration = 0
    
    def __call__(self, current_loss: float, params: Any) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        current_loss : float
            Current loss value
        params : Any
            Current parameters

        Returns
        -------
        bool
            True if training should stop
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_params = params
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_iteration = self.wait
            return True
        
        return False
    
    def get_best_params(self) -> Any:
        """Get best parameters found during optimization."""
        return self.best_params if self.restore_best_weights else None


def create_multi_optimizer(
    optimizers: Dict[str, optax.GradientTransformation]
) -> Callable:
    """
    Create multi-optimizer for different parameter groups.

    Parameters
    ----------
    optimizers : dict
        Dictionary mapping parameter names to optimizers

    Returns
    -------
    callable
        Multi-optimizer function
    """
    def multi_opt_update(gradients: Dict[str, Any], opt_states: Dict[str, Any], params: Dict[str, Any]):
        """Update multiple parameter groups with different optimizers."""
        updates = {}
        new_opt_states = {}
        
        for param_name in optimizers.keys():
            if param_name in gradients:
                update, new_state = optimizers[param_name].update(
                    gradients[param_name], opt_states[param_name], params[param_name]
                )
                updates[param_name] = update
                new_opt_states[param_name] = new_state
        
        return updates, new_opt_states
    
    return multi_opt_update


def create_alternating_optimizer(
    optimizers: Dict[str, optax.GradientTransformation],
    schedule: Dict[str, int]
) -> Callable:
    """
    Create alternating optimizer that updates different parameters on different iterations.

    Parameters
    ----------
    optimizers : dict
        Dictionary mapping parameter names to optimizers
    schedule : dict
        Dictionary mapping parameter names to update frequencies

    Returns
    -------
    callable
        Alternating optimizer function
    """
    def alternating_update(
        iteration: int,
        gradients: Dict[str, Any], 
        opt_states: Dict[str, Any], 
        params: Dict[str, Any]
    ):
        """Update parameters according to alternating schedule."""
        updates = {}
        new_opt_states = {}
        
        for param_name in optimizers.keys():
            if iteration % schedule[param_name] == 0:
                # Update this parameter group
                update, new_state = optimizers[param_name].update(
                    gradients[param_name], opt_states[param_name], params[param_name]
                )
                updates[param_name] = update
                new_opt_states[param_name] = new_state
            else:
                # Keep current state
                updates[param_name] = jax.tree_map(jnp.zeros_like, params[param_name])
                new_opt_states[param_name] = opt_states[param_name]
        
        return updates, new_opt_states
    
    return alternating_update


def create_optimizer_from_config(config: Dict[str, Any]) -> optax.GradientTransformation:
    """
    Create optimizer from configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary with optimizer settings

    Returns
    -------
    optax.GradientTransformation
        Configured optimizer
    """
    optimizer_type = config.get("type", "adam")
    learning_rate = config.get("learning_rate", 1e-3)
    
    # Create learning rate schedule if specified
    if "schedule" in config:
        schedule_config = config["schedule"]
        schedule_type = schedule_config.get("type", "constant")
        
        if schedule_type == "exponential":
            learning_rate = exponential_decay_schedule(
                init_value=learning_rate,
                decay_rate=schedule_config.get("decay_rate", 0.9),
                decay_steps=schedule_config.get("decay_steps", 1000)
            )
        elif schedule_type == "cosine":
            learning_rate = cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=schedule_config.get("decay_steps", 1000),
                alpha=schedule_config.get("alpha", 0.0)
            )
        elif schedule_type == "warmup_cosine":
            learning_rate = warmup_cosine_decay_schedule(
                init_value=learning_rate * 0.1,
                peak_value=learning_rate,
                warmup_steps=schedule_config.get("warmup_steps", 100),
                decay_steps=schedule_config.get("decay_steps", 1000)
            )
    
    # Create base optimizer
    if optimizer_type == "adam":
        base_optimizer = create_adam_optimizer(
            learning_rate=learning_rate,
            beta1=config.get("beta1", 0.9),
            beta2=config.get("beta2", 0.999),
            weight_decay=config.get("weight_decay", 0.0)
        )
    elif optimizer_type == "sgd":
        base_optimizer = create_sgd_optimizer(
            learning_rate=learning_rate,
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", True)
        )
    elif optimizer_type == "rmsprop":
        base_optimizer = create_rmsprop_optimizer(
            learning_rate=learning_rate,
            decay=config.get("decay", 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Add gradient clipping if specified
    gradient_clip = config.get("gradient_clip")
    if gradient_clip:
        base_optimizer = create_composite_optimizer(
            base_optimizer=base_optimizer,
            gradient_clip_norm=gradient_clip
        )
    
    return base_optimizer