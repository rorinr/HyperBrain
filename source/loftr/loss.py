import torch


def coarse_cross_entropy_loss(
    predicted_confidence: torch.Tensor, ground_truth_confidence: torch.Tensor
) -> torch.Tensor:
    """
    Computes the mean cross-entropy loss for positive and negative matches in confidence scores.

    The function applies point-wise cross-entropy loss for each element in the confidence score tensor.
    It separately computes the loss for positive matches (where ground truth is 1) and negative matches
    (where ground truth is 0), then returns the weighted sum of these mean losses.

    Args:
        predicted_confidence (torch.Tensor): Tensor containing predicted confidence scores, shape (N, P0, P1).
        ground_truth_confidence (torch.Tensor): Tensor containing ground truth for confidence scores, shape (N, P0, P1).
        N represents the number of matches, and P0, P1 are the number of patches in each image.

    Returns:
        torch.Tensor: The weighted sum of mean cross-entropy losses for positive and negative matches.
    """
    # Create masks for positive and negative matches
    positive_mask = ground_truth_confidence == 1
    negative_mask = ground_truth_confidence == 0

    # Clamping the confidence values for numerical stability
    predicted_confidence = torch.clamp(predicted_confidence, 1e-6, 1 - 1e-6)

    # Calculating loss for positive matches
    loss_positive = -torch.log(predicted_confidence[positive_mask])

    # Calculating loss for negative matches
    loss_negative = -torch.log(1 - predicted_confidence[negative_mask])

    # Computing the weighted sum of mean losses for positive and negative matches
    return loss_positive.mean() + loss_negative.mean()

def coarse_focal_loss(predicted_confidence: torch.Tensor, ground_truth_confidence: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    """
    Computes the mean focal loss for positive and negative matches in confidence scores.

    Args:
        predicted_confidence (torch.Tensor): Tensor containing predicted confidence scores, shape (N, P0, P1).
        ground_truth_confidence (torch.Tensor): Tensor containing ground truth for confidence scores, shape (N, P0, P1).
        N represents the number of matches, and P0, P1 are the number of patches in each image.
        alpha (float): Focal loss alpha parameter.
        gamma (float): Focal loss gamma parameter.

    Returns:
        torch.Tensor: The weighted sum of mean focal losses for positive and negative matches.
    """
    positive_mask = ground_truth_confidence == 1
    negative_mask = ground_truth_confidence == 0


    loss_positive = - alpha * torch.pow(1 - predicted_confidence[positive_mask], gamma) * (predicted_confidence[positive_mask]).log()
    loss_negative = - (1-alpha) * torch.pow(predicted_confidence[negative_mask], gamma) * (1-predicted_confidence[negative_mask]).log()

    return loss_positive.mean() + loss_negative.mean()



def fine_l2_loss(
    coordinates_predicted: torch.Tensor, coordinates_ground_truth: torch.Tensor
) -> torch.Tensor:
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
    l2_distances = squared_differences.sum(dim=-1)

    # Return the mean of the L2 distances
    return l2_distances.mean()

def fine_l2_loss_with_standard_deviation(
    coordinates_predicted: torch.Tensor, coordinates_ground_truth: torch.Tensor
) -> torch.Tensor:    
    """
    Computes the mean L2 distance (Euclidean distance) between the predicted and ground truth coordinates.

    This function calculates the Euclidean distance for each pair of corresponding coordinates and then
    returns the mean of these distances.

    Args:
        coordinates_predicted (torch.Tensor): Tensor containing predicted coordinates and their heatmaps deviations. shape (N, 3).
        coordinates_ground_truth (torch.Tensor): Tensor containing ground truth coordinates, shape (N, 2).
        N represents the number of matches.

    Returns:
        torch.Tensor: The mean L2 distance between predicted and ground truth coordinates.
    """

    standard_deviation = coordinates_predicted[:, 2]
    inverse_standard_deviation = 1./torch.clamp(standard_deviation, min=1e-10)
    weight = (inverse_standard_deviation/torch.mean(inverse_standard_deviation)).detach()  # normalize and detach
    l2_distances = ((coordinates_ground_truth - coordinates_predicted[:, :2])**2).sum(-1)

    return (l2_distances*weight).mean()
