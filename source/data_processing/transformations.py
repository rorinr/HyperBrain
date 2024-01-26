import torch


def sample_random_affine_matrix(range_limit: float) -> torch.Tensor:
    """
    Generates a random affine transformation matrix with values uniformly distributed within a specified range.

    Args:
        range_limit (float): The half-width of the range for the random matrix elements,
                             defining the uniform distribution interval from -range_limit to range_limit.

    Returns:
        torch.Tensor: A 3x3 affine transformation matrix with random elements within the specified range.
    """
    affine_matrix = torch.cat(
        [torch.rand((2, 2)).uniform_(-range_limit, range_limit), torch.zeros((2, 1))],
        dim=1,
    ).unsqueeze(0)
    affine_matrix[0, 0, 0] += 1
    affine_matrix[0, 1, 1] += 1
    return affine_matrix


def transform_grid_coordinates(
    grid_coordinates: torch.Tensor, transformation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply an affine transformation to a grid of image coordinates.

    This function transforms each coordinate in the provided grid according to the specified affine transformation matrix.

    Args:
        grid_coordinates (torch.Tensor): A tensor of shape (height, width, 2) representing the grid of image coordinates.
        transformation_matrix (torch.Tensor): A 2x3 affine transformation matrix.

    Returns:
        torch.Tensor: A tensor of the transformed grid coordinates, maintaining the original shape (height, width, 2).
    """
    original_shape = grid_coordinates.shape
    flat_grid_coordinates = grid_coordinates.view(-1, 2)

    # Create homogeneous coordinates by appending a ones column
    ones_column = torch.ones(
        (flat_grid_coordinates.shape[0], 1), device=flat_grid_coordinates.device
    )
    homogeneous_coordinates = torch.cat([flat_grid_coordinates, ones_column], dim=1)

    # Apply the affine transformation
    transformed_coordinates_homogeneous = torch.mm(
        transformation_matrix, homogeneous_coordinates.t()
    ).t()

    # Reshape back to the original grid shape
    transformed_grid_coordinates = transformed_coordinates_homogeneous[:, :2].view(
        original_shape
    )

    return transformed_grid_coordinates


def translate_fine_to_coarse(
    fine_coordinates: torch.Tensor, fine_size: int, coarse_size: int
) -> torch.Tensor:
    """
    Translates coordinates from a fine feature map to a coarse feature map.

    Args:
        fine_coordinates (torch.Tensor): A tensor of shape (N, 2) representing coordinates in the fine map.
        fine_size (int): The size (height/width assuming square) of the fine feature map.
        coarse_size (int): The size (height/width assuming square) of the coarse feature map.

    Returns:
        torch.Tensor: Translated coordinates in the coarse feature map.
    """

    scale_factor = fine_size / coarse_size
    coarse_coords = fine_coordinates.float() / scale_factor

    return coarse_coords.long()


def get_relative_coordinates(
    transformed_coordinates: torch.Tensor,
    reference_coordinates: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Converts absolute coordinates to relative coordinates within a local window.

    Args:
        transformed_coordinates (torch.Tensor): Transformed coordinates in, eg, the fine feature space.
        reference_coordinates (torch.Tensor): Reference coordinates, typically the mid-pixels of patches in crop_2.
        window_size (int): The size of the local window. Defaults to 5.

    Returns:
        torch.Tensor: Relative coordinates within the local window.
    """

    # Calculate the offset
    offset = transformed_coordinates - reference_coordinates

    # Normalize the offset to the -1 to 1 range
    relative_coordinates = offset / ((window_size - 1) / 2)

    return relative_coordinates
