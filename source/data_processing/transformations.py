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
    affine_matrix = torch.cat([torch.rand((2, 2)).uniform_(-range_limit, range_limit), torch.zeros((2, 1))], dim=1).unsqueeze(0)
    affine_matrix[0, 0, 0] += 1
    affine_matrix[0, 1, 1] += 1
    return affine_matrix
