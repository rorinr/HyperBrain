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
