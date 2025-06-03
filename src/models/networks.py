"""
Neural network architectures for PACT reconstruction.

This module contains neural network models and architectures used for
photoacoustic reconstruction. It includes convolutional neural networks
with encoder-decoder architectures, skip connections, and specialized
networks for multi-field processing.

Key functionality:
- ConvBlock: Basic convolutional building block with batch normalization and dropout
- EncoderBlock/DecoderBlock: Encoder and decoder components for U-Net style architectures
- TreeNet: Multi-field network with skip connections for 4-input processing
- TreeNet_P0: Specialized TreeNet variant for illumination-aware processing
- YNet: Two-field network for dual input processing
- ConcatNet: Concatenation-based network for multi-field outputs
- StepNet: Iterative step network for optimization-based reconstruction
- RegNet: Regression network for encoding to scalar outputs
- Network factory function for easy instantiation
"""

from functools import partial
from typing import Any, Dict, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block with batch normalization, activation, and dropout.
    
    Attributes:
        dropout: Dropout rate for regularization
        features: Number of output features/channels
        activation: Activation function name (default: "relu")
        train: Whether in training mode (affects batch norm and dropout)
    """
    dropout: float
    features: int = None
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", kernel_init=nn.initializers.he_normal())(x)
        
        # mode='fan_out', distribution='truncated_normal'))(x)

        # x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", kernel_init=nn.initializers.constant(.1))(x)

        x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.train)

        return x


class EncoderBlock(nn.Module):
    """
    Encoder block with two convolutional blocks followed by max pooling.
    
    Attributes:
        features: Number of output features/channels
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
        train: Whether in training mode
    """
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x = Conver(features=self.features)(x)
        x = Conver(features=self.features)(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling followed by two convolutional blocks.
    
    Attributes:
        features: Number of output features/channels
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
        train: Whether in training mode
    """
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x = jax.image.resize(
            x,
            (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
            method="bilinear",
        )

        x = Conver(features=self.features)(x)
        x = Conver(features=self.features)(x)

        return x


class TreeNet(nn.Module):
    """
    Multi-field network that combines 4 fields into a single field output using skip connections.
    
    This network processes four input fields through separate encoder paths, concatenates
    them at the bottleneck, and uses skip connections in the decoder path.
    
    Expected input fields (by convention):
    - x0: mu_r (absorption coefficient reference)
    - x1: d_mu (absorption coefficient gradient)
    - x2: c_r (speed of sound reference)
    - x3: d_c (speed of sound gradient)
    
    Output: Modified gradient d_c_r
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        # 
        x0 = jnp.broadcast_to(x2, (x0.shape))
        # 
        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        e1_3 = Encoder(features=f * 8)(e1_2)

        e2_1 = Encoder(features=f * 2)(x2)
        e2_2 = Encoder(features=f * 4)(e2_1)
        e2_3 = Encoder(features=f * 8)(e2_2)

        e3_1 = Encoder(features=f * 2)(x3)
        e3_2 = Encoder(features=f * 4)(e3_1)
        e3_3 = Encoder(features=f * 8)(e3_2)
               
        c = jnp.concatenate([e0_3, e1_3, e2_3, e3_3], axis=-1)

        c = Conver(features=f * 8)(c)
        c = Conver(features=f * 8)(c)

        d2 = Decoder(features=f * 4)(c)
        d2 = jnp.concatenate([d2, e0_2, e1_2, e2_2, e3_2], axis=-1)
        d1 = Decoder(features=f * 2)(d2)
        d1 = jnp.concatenate([d1, e0_1, e1_1, e2_1, e3_1], axis=-1)
        d0 = Decoder(features=f)(d1)

        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)

        return o


class TreeNet_P0(nn.Module):
    """
    Specialized TreeNet variant that accommodates batch processing for illuminations.
    
    This network combines 4 fields into a single field output using skip connections.
    The first field accommodates a batch (of illuminations) with proper broadcasting.
    
    Expected input fields (by convention):
    - x0: P0_r (initial pressure reference, batched)
    - x1: d_mu (absorption coefficient gradient)
    - x2: c_r (speed of sound reference)
    - x3: d_c (speed of sound gradient)
    
    Output: Modified gradient d_c_r
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        # 
        x0 = jnp.broadcast_to(x2, (x0.shape))
        # 
        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        e1_3 = Encoder(features=f * 8)(e1_2)

        e2_1 = Encoder(features=f * 2)(x2)
        e2_2 = Encoder(features=f * 4)(e2_1)
        e2_3 = Encoder(features=f * 8)(e2_2)

        e3_1 = Encoder(features=f * 2)(x3)
        e3_2 = Encoder(features=f * 4)(e3_1)
        e3_3 = Encoder(features=f * 8)(e3_2)
        
        e1_1 = jnp.broadcast_to(e1_1, e0_1.shape)
        e2_1 = jnp.broadcast_to(e2_1, e0_1.shape)
        e3_1 = jnp.broadcast_to(e3_1, e0_1.shape)
        
        e1_2 = jnp.broadcast_to(e1_2, e0_2.shape)
        e2_2 = jnp.broadcast_to(e2_2, e0_2.shape)
        e3_2 = jnp.broadcast_to(e3_2, e0_2.shape)
        
        e1_3 = jnp.broadcast_to(e1_3, e0_3.shape)
        e2_3 = jnp.broadcast_to(e2_3, e0_3.shape)
        e3_3 = jnp.broadcast_to(e3_3, e0_3.shape)
       
        c = jnp.concatenate([e0_3, e1_3, e2_3, e3_3], axis=-1)

        c = Conver(features=f * 8)(c)
        c = Conver(features=f * 8)(c)

        d2 = Decoder(features=f * 4)(c)
        d2 = jnp.concatenate([d2, e0_2, e1_2, e2_2, e3_2], axis=-1)
        d1 = Decoder(features=f * 2)(d2)
        d1 = jnp.concatenate([d1, e0_1, e1_1, e2_1, e3_1], axis=-1)
        d0 = Decoder(features=f)(d1)

        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)
        o = jnp.permute_dims(o, (3, 1, 2, 0)) 
        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o)
    
        return o


class YNet(nn.Module):
    """
    Y-shaped network that combines 2 fields into a single field output using skip connections.
    
    This is a simplified version of TreeNet for dual-input processing.
    All fields must have the same batch size.
    
    Expected input fields (by convention):
    - x0: d_mu (absorption coefficient gradient)
    - x1: d_c (speed of sound gradient)
    
    Output: Modified gradient d_c_r
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )

        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        # e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        # e1_3 = Encoder(features=f * 8)(e1_2)
        
        c = jnp.concatenate([e0_2, e1_2], axis=-1)
        c = Conver(features=f * 4)(c)
        c = Conver(features=f * 4)(c)
        # c = jnp.concatenate([e0_3, e1_3], axis=-1)
        # c = Conver(features=f * 8)(c)
        # c = Conver(features=f * 8)(c)


        # d2 = Decoder(features=f * 4)(c)
        # d2 = jnp.concatenate([d2, e0_2, e1_2], axis=-1)
        # d1 = Decoder(features=f * 2)(d2)

        d1 = Decoder(features=f * 2)(c)
        d1 = jnp.concatenate([d1, e0_1, e1_1], axis=-1)
        d0 = Decoder(features=f)(d1)
    
        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)

        return o0


class ConcatNet(nn.Module):
    """
    Concatenation-based network that combines 4 fields into 2 outputs without skip connections.
    
    This network processes multiple fields through concatenation and produces dual outputs.
    All fields must have the same batch size.
    
    Expected input fields (by convention):
    - x0: P0_r (initial pressure reference)
    - x1: d_P0 (initial pressure gradient)
    - x2: c_r (speed of sound reference)
    - x3: d_c (speed of sound gradient)
    
    Output: Modified gradients d_c_r, d_P0_r
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
        train: Whether in training mode
    """
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x0, x1, x2, x3):
        f = self.features
        Encoder = partial(
            EncoderBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )
        Decoder = partial(
            DecoderBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x2_b = jnp.broadcast_to(x2, (x0.shape))
        x3_b = jnp.broadcast_to(x3, (x0.shape))

        x0 = Conver(features=f // 2)(x0)
        x1 = Conver(features=f // 2)(x1)
        x2_b = Conver(features=f // 2)(x2_b)
        x3_b = Conver(features=f // 2)(x3_b)

        c = jnp.concatenate([x0, x1, x2_b, x3_b], axis=-1)

        x = Encoder(features=f * 2)(c)
        x = Encoder(features=f * 4)(x)
        x = Encoder(features=f * 8)(x)

        x = Conver(features=f * 8)(x)

        x = jnp.split(x, 2, axis=-1)

        o0 = Conver(features=f * 8)(x[0])
        o1 = Conver(features=f * 8)(x[1])

        o0 = Decoder(features=f * 8)(o0)
        o0 = Decoder(features=f * 4)(o0)
        o0 = Decoder(features=f * 2)(o0)

        o1 = Decoder(features=f * 8)(o1)
        o1 = Decoder(features=f * 4)(o1)
        o1 = Decoder(features=f * 2)(o1)

        return o0, o1


class StepNet(nn.Module):
    """
    Iterative step network for optimization-based reconstruction.
    
    This network implements a learnable optimization step:
    x_next = x_previous - alpha * dx - R(x)
    where R is a residual network defined by the network parameter.
    
    Expected inputs from iteration k:
    - x0: mu_r (absorption coefficient reference)
    - x1: d_mu (absorption coefficient gradient)
    - x2: c_r (speed of sound reference)
    - x3: d_c (speed of sound gradient)
    
    Outputs for iteration k+1:
    - o0: mu_r (updated absorption coefficient reference)
    - o1: c_r (updated speed of sound reference)
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        network: The residual network module to use
        activation: Activation function name (default: "relu")
    """
    features: int
    dropout: float
    network: nn.Module
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        R = self.network(
            features=self.features,
            dropout=self.dropout,
            activation=self.activation,
            train=train,
        )
        r = R(x0, x1, x2, x3)
   
        alpha_0 = self.param("alpha_0", nn.initializers.ones, ())
        o0 = x0 - alpha_0 * x1 - r[0]
        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o0)
        o0 = jnp.permute_dims(o0, (3, 1, 2, 0))
        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o0)

        alpha_1 = self.param("alpha_1", nn.initializers.ones, ())
        o1 = x2 - alpha_1 * x3 - r[1]
        o1 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o1)
        o1 = jnp.permute_dims(o1, (3, 1, 2, 0))
        o1 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o1)

        return o0, o1


class RegNet(nn.Module):
    """
    Regression network that encodes input fields down to a single scalar output.
    
    This network progressively downsamples the input through multiple encoder blocks
    and produces a scalar output through global average pooling and a dense layer.
    
    Attributes:
        features: Base number of features for the network
        dropout: Dropout rate for regularization
        activation: Activation function name (default: "relu")
        train: Whether in training mode
    """
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x0, x1, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        c = jnp.concatenate([x0, x1], axis=-1)

        x = Encoder(features=f * 2)(c)
        x = Encoder(features=f * 4)(x)
        x = Encoder(features=f * 8)(x)    
        x = Encoder(features=f * 16)(x)    
        x = Encoder(features=f * 32)(x)    
        x = Encoder(features=f * 64)(x)    

        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        x = nn.Dense(features=1)(x)

        return x


# Network factory function for easy instantiation
def create_network(network_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create neural networks by name.
    
    Args:
        network_type: Name of the network type to create
        **kwargs: Additional keyword arguments to pass to the network constructor
        
    Returns:
        Instantiated network module
        
    Raises:
        ValueError: If network_type is not recognized
    """
    network_registry = {
        'convblock': ConvBlock,
        'encoder': EncoderBlock,
        'decoder': DecoderBlock,
        'treenet': TreeNet,
        'treenet_p0': TreeNet_P0,
        'ynet': YNet,
        'concatnet': ConcatNet,
        'stepnet': StepNet,
        'regnet': RegNet,
    }
    
    network_type_lower = network_type.lower()
    if network_type_lower not in network_registry:
        available_types = ', '.join(network_registry.keys())
        raise ValueError(f"Unknown network type '{network_type}'. Available types: {available_types}")
    
    return network_registry[network_type_lower](**kwargs)


def get_available_networks() -> Dict[str, str]:
    """
    Get a dictionary of available network types and their descriptions.
    
    Returns:
        Dictionary mapping network names to descriptions
    """
    return {
        'convblock': 'Basic convolutional block with batch normalization and dropout',
        'encoder': 'Encoder block with convolutions and max pooling',
        'decoder': 'Decoder block with upsampling and convolutions',
        'treenet': 'Multi-field network with skip connections (4 inputs → 1 output)',
        'treenet_p0': 'TreeNet variant for illumination-aware processing',
        'ynet': 'Y-shaped network for dual input processing (2 inputs → 1 output)',
        'concatnet': 'Concatenation-based network (4 inputs → 2 outputs)',
        'stepnet': 'Iterative step network for optimization-based reconstruction',
        'regnet': 'Regression network encoding to scalar output',
    }