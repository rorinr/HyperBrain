"""
Positional Encoding Module for 2D Images in PyTorch.

This module contains the PositionalEncoding class, which implements a sinusoidal positional encoding
mechanism for 2D images. The encoding is designed to provide models, especially those lacking an inherent
notion of position (like Transformers), with spatial context. This positional information is crucial for
tasks involving image processing and other scenarios where the relative positioning of data points is
significant.

The PositionalEncoding class generates a grid of sine and cosine values with varying frequencies, which
can be added to the input feature maps of a neural network. This addition enhances the model's ability to
interpret the position of pixels or features in the image.

Classes:
    PositionalEncoding: Implements sinusoidal positional encoding for 2D images.

Key Functionalities:
    - Generates a positional encoding tensor based on sine and cosine functions.
    - Supports variable dimensionality and maximum shape for the encoding grid.
    - Easily integrates with PyTorch models, adding positional context to the input tensors.

Example Usage:
    To use the PositionalEncoding class, initialize it with the desired number of channels (dimensionality)
    and the maximum shape for the positional encoding grid. Then, simply call the forward method with
    an input tensor of shape [N, C, H, W].

Dependencies:
    - torch
    - math
"""

import math
from typing import Tuple
from torch import nn
import torch


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for 2D images.

    This class generates a positional encoding based on sine and cosine functions
    with different frequencies. The encoding is applied to the input tensor to provide
    positional information to the model.

    Attributes:
        positional_encoding (torch.Tensor): A tensor holding the precomputed sinusoidal positional encodings.

    Args:
        dimensionality (int): The number of channels in the position encoding tensor.
        maximum_shape (Tuple[int, int]): The maximum shape (height, width) for the positional encoding grid.
                                        For a feature map that is 1/8th the size of the original image,
                                        a maximum_shape of (256, 256) corresponds to an original image size of
                                        (2048, 2048) pixels.
    """

    def __init__(
        self, dimensionality: int, maximum_shape: Tuple[int, int] = (256, 256)
    ) -> None:
        super().__init__()

        # Initialize a positional encoding (PE) tensor with zeros of shape [C, H, W]
        positional_encoding = torch.zeros((dimensionality, *maximum_shape))

        # Generate position indices for the x and y dimensions
        y_position = (
            torch.ones(maximum_shape).cumsum(0).float().unsqueeze(0)
        )  # Cumulative sum along y-axis
        x_position = (
            torch.ones(maximum_shape).cumsum(1).float().unsqueeze(0)
        )  # Cumulative sum along x-axis

        # Create the division term used in the sinusoidal function
        div_term = torch.exp(
            torch.arange(0, dimensionality // 2, 2).float()
            * (-math.log(10000.0) / (dimensionality // 2))
        )
        div_term = div_term[:, None, None]  # Expand dimensions for broadcasting

        # Compute the sinusoidal encodings
        positional_encoding[0::4, :, :] = torch.sin(
            x_position * div_term
        )  # Sinusoidal encoding for x-coordinates
        positional_encoding[1::4, :, :] = torch.cos(
            x_position * div_term
        )  # Cosine encoding for x-coordinates
        positional_encoding[2::4, :, :] = torch.sin(
            y_position * div_term
        )  # Sinusoidal encoding for y-coordinates
        positional_encoding[3::4, :, :] = torch.cos(
            y_position * div_term
        )  # Cosine encoding for y-coordinates

        # Register the PE tensor as a buffer in the module (not a model parameter)
        self.register_buffer(
            "positional_encoding", positional_encoding.unsqueeze(0), persistent=False
        )  # Add a batch dimension [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (Tensor): The input tensor with shape [N, C, H, W].

        Returns:
            Tensor: The input tensor with positional encodings added.
        """
        # Add positional encoding to the input tensor, adjusting to match the input size
        return (
            x + self.positional_encoding[:, :, : x.size(2), : x.size(3)]
        )  # Apply positional_encoding to each element in the batch
