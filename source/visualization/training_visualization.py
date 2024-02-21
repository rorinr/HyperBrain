from typing import Iterator, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def plot_grad_flow_hist(named_parameters: Iterator[Tuple[str, torch.Tensor]]) -> None:
    """Plots the gradients flowing through different layers in the network during training.

    This function can be used for checking possible gradient vanishing/exploding problems.

    Args:
        named_parameters: An iterator over model parameters, as returned by `model.named_parameters()`.

    Usage:
        Call this function after `loss.backward()` to visualize the gradient flow,
        e.g., `plot_grad_flow_hist(model.named_parameters())`.
    """
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in named_parameters:
        if param.requires_grad and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.cpu().abs().mean().item())
            max_grads.append(param.grad.cpu().abs().max().item())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0, top=0.02)  # Zoom in on the lower gradient regions.
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["Max-Gradient", "Mean-Gradient", "Zero-Gradient"],
    )
    plt.show()
