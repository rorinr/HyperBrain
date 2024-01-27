import torch

def fine_loss(coordinates_predicted: torch.Tensor, coordinates_ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean L2 distance (Euclidean distance) between the predicted and ground truth coordinates.

    This function calculates the Euclidean distance for each pair of corresponding coordinates and then 
    returns the mean of these distances. In future iterations, weighting by the inverse standard deviation 
    may be implemented for enhanced accuracy.

    Args:
        coordinates_predicted (torch.Tensor): Tensor containing predicted coordinates, shape (N, 2).
        coordinates_ground_truth (torch.Tensor): Tensor containing ground truth coordinates, shape (N, 2).
        N represents the number of matches.

    Returns:
        torch.Tensor: The mean L2 distance between predicted and ground truth coordinates.
    """
    # Calculate squared differences and sum over coordinate dimension
    squared_differences = (coordinates_ground_truth - coordinates_predicted) ** 2
    l2_distances = torch.sqrt(squared_differences.sum(dim=-1))

    # Return the mean of the L2 distances
    return l2_distances.mean()
