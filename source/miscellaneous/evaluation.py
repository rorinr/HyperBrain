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

def count_matches_per_patch(matches: torch.Tensor, 
                                       x_borders: torch.Tensor, 
                                       y_borders: torch.Tensor) -> torch.Tensor:
    """Counts the number of matches in each patch defined by x and y borders.

    This function first calculates the indices of the patches that each match 
    belongs to and then updates the count of matches in each patch accordingly.

    Args:
        matches: A tensor of shape (N, 4) containing match coordinates in the format
            [x1, y1, x2, y2], where x1, y1 are the coordinates in the first image.
        x_borders: A tensor containing the x-coordinates of the patch borders.
        y_borders: A tensor containing the y-coordinates of the patch borders.

    Returns:
        A 2D float tensor of shape (y_patches, x_patches) representing the number 
        of matches in each patch.
    """
    # Calculate the number of patches
    x_patches = len(x_borders) - 1
    y_patches = len(y_borders) - 1
    counts = torch.zeros((y_patches, x_patches), dtype=torch.int32)
    
    # Extract x and y coordinates from the matches
    x_coords = matches[:, 0]
    y_coords = matches[:, 1]
    
    # Find the patch indices for the x and y coordinates
    x_indices = torch.searchsorted(x_borders, x_coords) - 1
    y_indices = torch.searchsorted(y_borders, y_coords) - 1
    
    # Filter out matches that are outside the borders
    valid_indices = (x_indices >= 0) & (x_indices < x_patches) & \
                    (y_indices >= 0) & (y_indices < y_patches)
    
    # Update counts using the valid patch indices
    for x_index, y_index in zip(x_indices[valid_indices], y_indices[valid_indices]):
        counts[y_index, x_index] += 1
    
    return counts.float()

def calculate_entropy(counts: torch.Tensor) -> torch.Tensor:
    """Calculates the entropy of a distribution represented by counts.

    Entropy is a measure of the unpredictability or randomness of a distribution.
    This function converts the counts into probabilities and then computes the 
    entropy of the distribution using the formula: 
    H(P) = -sum(p_i * log2(p_i)), where p_i are the probabilities.

    Args:
        counts: A 2D tensor representing the counts of matches in each patch.

    Returns:
        A scalar tensor representing the entropy of the distribution.

    """
    # Convert counts to probabilities, ensuring float division
    total_matches = torch.sum(counts).float()
    probabilities = counts / total_matches

    # Calculate entropy, handling 0 * log(0) explicitly as 0
    entropy = -torch.sum(torch.where(probabilities > 0, 
                                     probabilities * torch.log2(probabilities), 
                                     torch.tensor(0.0, dtype=torch.float)))
    
    return entropy
