"""
This module implements the Linear Attention mechanism, as part of a transformer model.
Additionally, the module includes a modified Exponential Linear Unit (ELU) activation function
"""


import torch
from torch import nn


def elu_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Exponential Linear Unit (ELU) activation function to the input tensor and adds 1.
    Args:
        x (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The tensor after applying the ELU activation function and adding 1.
    """
    return nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    """
    A Multi-Head Linear Attention mechanism as proposed in "Transformers are RNNs".
    This class provides an efficient attention mechanism for transformer models by applying
    linearized attention computation.

    This implementation uses the ELU (Exponential Linear Unit) activation function as part of
    its attention computation. The ELU activation is modified slightly by adding 1 to its output.
    Can be used for both, self-attention and cross-attention.

    Attributes:
        elu_activation (function): A function that applies the modified ELU activation.
        epsilon (float): A small value to avoid division by zero in attention computations.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        """
        Initializes the LinearAttention module.
        Args:
            epsilon (float, optional): A small value added for numerical stability in attention
                                       computation to prevent division by zero. Default: 1e-6.
        """

        super().__init__()
        self.elu_activation = elu_activation
        self.epsilon = epsilon

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of Linear Attention.

        L, S = Number of features per image, eg 1600 (40x40) for 1/16th scale. Note that key and value
                have the same number of features since they came from the same input image.
        Args:
            queries: [Batchsize, L, Number of heads, Head dimension]
            keys: [Batchsize, S, Number of heads, Head dimension]
            values: [Batchsize, S, Number of heads, Head dimension]
        Returns:
            queried_values: (Batchsize, L, Number of heads, Head dimension)
        """
        Q = self.elu_activation(queries)
        K = self.elu_activation(keys)

        values_length = values.size(1)
        values = values / values_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.epsilon)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * values_length

        return queried_values.contiguous()
