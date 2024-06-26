import torch
from typing import Tuple
from source.data_processing.patch_processing import get_patch_coordinates
from source.data_processing.transformations import (
    translate_fine_to_coarse,
    translate_coarse_to_fine,
)


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


def get_patch_mid_coordinates(
    match_matrix: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the midpoints of the match matrix.

    Args:
        match_matrix (torch.Tensor): A tensor representing the (predicted) match matrix.
        patch_size (int): The size of each patch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the coordinates of the midpoints of the matched patches. Both of shape (N, 2).
    """

    # Extract indices of matched patches
    crop_1_patch_indices = match_matrix[0].nonzero()[:, 0].cpu()
    crop_2_patch_indices = match_matrix[0].nonzero()[:, 1].cpu()

    half_patch_size = patch_size // 2

    # Calculate midpoints of matched patches
    num_patches_per_side = int(match_matrix.shape[-1] ** 0.5)
    crop_1_patch_mid_coordinates = (
        get_patch_coordinates(patch_indices=crop_1_patch_indices, patch_size=patch_size, num_patches_per_side=num_patches_per_side)
        + torch.Tensor([half_patch_size, half_patch_size]).long()
    )
    crop_2_patch_mid_coordinates = (
        get_patch_coordinates(patch_indices=crop_2_patch_indices, patch_size=patch_size, num_patches_per_side=num_patches_per_side)
        + torch.Tensor([half_patch_size, half_patch_size]).long()
    )

    return crop_1_patch_mid_coordinates, crop_2_patch_mid_coordinates


def translate_patch_midpoints_and_refine(
    match_matrix: torch.Tensor,
    patch_size: int,
    relative_coordinates: torch.Tensor,
    fine_feature_size: int,
    image_size: int = 640,
    window_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Translates the midpoints of matched patches from one coordinate space to another and applies a refinement step.

    Args:
        match_matrix (torch.Tensor): A tensor representing the (predicted) match matrix.
        patch_size (int): The size of each patch.
        relative_coordinates (torch.Tensor): The (predicted) relative coordinates.
        image_size (int, optional): The size of the image (height, width). Defaults to 640.
        fine_feature_size (int, optional): The size of the fine feature space.
        window_size (int, optional): The size of the window for the relative movement. Defaults to 5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the coordinates of the midpoints for both crops and the refined coordinates for crop 2. All of shape (N, 2).
    """
    # Calculate midpoints of matched patches
    (
        crop_1_patch_mid_coordinates,
        crop_2_patch_mid_coordinates,
    ) = get_patch_mid_coordinates(match_matrix=match_matrix, patch_size=patch_size)

    # Translate midpoints to fine feature space and apply relative refinement
    crop_2_patch_mid_coordinates_fine = translate_fine_to_coarse(
        fine_coordinates=crop_2_patch_mid_coordinates,
        fine_size=image_size,
        coarse_size=fine_feature_size,
    )

    offset = relative_coordinates.cpu() * ((window_size - 1) / 2)

    crop_2_patch_mid_coordinates_refined = translate_coarse_to_fine(
        coarse_coords=crop_2_patch_mid_coordinates_fine.cpu() + offset,
        coarse_size=fine_feature_size,
        fine_size=image_size,
    )

    return (
        crop_1_patch_mid_coordinates,
        crop_2_patch_mid_coordinates,
        crop_2_patch_mid_coordinates_refined,
    )
