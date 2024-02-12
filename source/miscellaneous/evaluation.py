import torch

def compute_euclidean_distances(predicted_matches: torch.Tensor, coordinate_mapping: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean distance between predicted matches and the corresponding ground truth in a vectorized manner.

    Args:
        predicted_matches (torch.Tensor): A tensor of shape (M, 4) containing predicted matches.
                                           Each row is [i, j, k, l], where (i, j) are coordinates in image 1,
                                           and (k, l) are the predicted matching coordinates in image 2.
        coordinate_mapping (torch.Tensor): A 3D tensor of shape (H, W, 2) where each entry [i, j] contains
                                           the ground truth (x, y) coordinates in image 2 for the pixel (i, j)
                                           in image 1.

    Returns:
        torch.Tensor: A tensor of shape (M,) containing the Euclidean distances for each predicted match.
    """
    # Extract (x,y) indices from the first image of the prediction
    predicted_coordinates_image_1 = predicted_matches[:, :2].long()

    # Use indexing to get the ground truth coordinates for (x,y)
    ground_truth_coordinates = coordinate_mapping[predicted_coordinates_image_1[:, 1], predicted_coordinates_image_1[:, 0]]

    # Some pixel coordinates may be in image 1 but not in image 2, so we need to remove them
    pixel_exists_mask = (ground_truth_coordinates != -1).all(dim=1)
    predicted_matches = predicted_matches[pixel_exists_mask]
    ground_truth_coordinates = ground_truth_coordinates[pixel_exists_mask]

    # Extract the predicted (x,y) coordinates for the second image
    predicted_coordinates_image_2 = predicted_matches[:, 2:]

    # Compute the Euclidean distances
    distances = torch.norm(ground_truth_coordinates - predicted_coordinates_image_2, dim=1)

    return distances