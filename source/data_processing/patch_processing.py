import torch

def get_patch_index(coordinates: torch.Tensor, patch_size: int = 16, num_patches_per_side: int = 40) -> torch.Tensor:
    """
    Calculates the indices of patches in a flattened grid from their coordinates.

    Args:
        coordinates (torch.Tensor): A tensor containing the coordinates of points in the image. Shape: (N, 2).
        patch_size (int): The size of each square patch.
        num_patches_per_side (int): The number of patches per side of the image grid. Defaults to 40.

    Returns:
        torch.Tensor: The indices of the patches corresponding to the given coordinates in the flattened grid.
    """

    # Extract x and y coordinates
    x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]

    # Calculate patch indices along x and y axes
    patch_indices_x = x_coords // patch_size
    patch_indices_y = y_coords // patch_size

    # Combine x and y indices to get the flattened grid index
    flat_patch_indices = patch_indices_x + num_patches_per_side * patch_indices_y

    return flat_patch_indices


def create_match_matrix(crop_coordinate_mapping: torch.Tensor, crop_size: int = 640, patch_size: int = 16) -> torch.Tensor:
    """
    Creates a match matrix that represents patch matches between two image crops.

    Args:
        crop_coordinate_mapping (torch.Tensor): A tensor of shape (H, W, 2) representing pixel-wise mappings from crop 1 to crop 2.
        crop_size (int): The size of the image crops. Defaults to 640.
        patch_size (int): The size of each square patch. Defaults to 16.

    Returns:
        match_matrix (torch.Tensor): A binary matrix of shape (1600, 1600) indicating patch matches.
    """

    # Calculate mid-pixel indices of each patch
    y_indices_mid, x_indices_mid = torch.meshgrid(
        torch.arange(patch_size // 2, crop_size, patch_size), 
        torch.arange(patch_size // 2, crop_size, patch_size), 
        indexing='ij'
    )

    # Extract mid-pixel coordinates for mapping
    mid_pixel_mappings = crop_coordinate_mapping[y_indices_mid, x_indices_mid]

    # Mask to identify valid patches (not [-1, -1])
    valid_patch_mask = (mid_pixel_mappings != -1).all(dim=2)

    # Flatten the indices and filter by the valid patch mask
    x_indices_mid_flat = x_indices_mid.flatten()[valid_patch_mask.flatten()]
    y_indices_mid_flat = y_indices_mid.flatten()[valid_patch_mask.flatten()]
    mid_pixel_mappings_flat = mid_pixel_mappings.view(-1, 2)[valid_patch_mask.flatten()]

    # Calculate patch indices in the original and transformed image
    crop_1_patch_indices = get_patch_index(torch.stack((x_indices_mid_flat, y_indices_mid_flat), dim=1), patch_size)
    crop_2_patch_indices = get_patch_index(mid_pixel_mappings_flat, patch_size)

    # Initialize and populate the match matrix
    match_matrix = torch.zeros((1600, 1600), dtype=torch.int32)
    match_matrix[crop_1_patch_indices, crop_2_patch_indices.long()] = 1

    return match_matrix