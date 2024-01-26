import torch
from typing import Tuple


def generate_image_grid_coordinates(image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Generate a 3D grid of pixel coordinates for an image, maintaining the image shape.

    This function creates a grid of (x, y) coordinates corresponding to each pixel location in an image,
    and arranges them in a 3D tensor that mirrors the image's dimensions.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions of the image (height, width).

    Returns:
    - grid_coordinates (torch.Tensor): A tensor of shape (height, width, 2), where each element [i, j]
                                       contains the (x, y) coordinates of the pixel at position (i, j) in the image.
    """
    height, width = image_size
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    grid_coordinates = torch.stack((x_coordinates, y_coordinates), dim=2)

    return grid_coordinates.float()
