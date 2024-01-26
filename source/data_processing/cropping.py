import torch
from typing import Tuple


def crop_image(
    image: torch.Tensor, crop_position: torch.Tensor, crop_size: int
) -> torch.Tensor:
    """
    Crop an image.

    Args:
        image (torch.Tensor): Image of shape (C, H, W) to be cropped.
        crop_position (torch.Tensor): Position of the crop - top left corner in coordinates (x, y).
        crop_size (int): Size of the crop.

    Returns:
        torch.Tensor: Cropped image.
    """
    return image[
        :,
        crop_position[1] : crop_position[1] + crop_size,
        crop_position[0] : crop_position[0] + crop_size,
    ]


def _get_transformed_image_corner_positions(
    coordinate_mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retrieves the corner coordinates of a transformed image grid.

    Given a tensor representing the transformed grid coordinates of an image,
    this function calculates the coordinates of the four corners: top-left, top-right,
    bottom-left, and bottom-right.

    Args:
        coordinate_mapping (torch.Tensor): A tensor representing the transformed
                                                     grid coordinates of an image.

    Returns:
        tuple: A tuple containing the coordinates (x, y) of the top-left, top-right,
               bottom-left, and bottom-right corners of the transformed image grid.
    """
    top_left = coordinate_mapping[0, 0]
    top_right = coordinate_mapping[0, -1]
    bottom_left = coordinate_mapping[-1, 0]
    bottom_right = coordinate_mapping[-1, -1]

    return top_left, top_right, bottom_left, bottom_right


def _calculate_crop_area_bounds(
    top_left: tuple,
    top_right: tuple,
    bottom_left: tuple,
    bottom_right: tuple,
    image_shape: tuple,
) -> Tuple[int, int, int, int]:
    """
    Calculates the boundaries of the crop sampling area based on the provided corner points of the transformed image
    and the original image shape. This ensures that the resulting crop area is within the bounds of the original image,
    preventing any black borders in the cropped image.

    Args:
        top_left (tuple): Coordinates (x, y) of the top left corner of the transformed image.
        top_right (tuple): Coordinates (x, y) of the top right corner of the transformed image.
        bottom_left (tuple): Coordinates (x, y) of the bottom left corner of the transformed image.
        bottom_right (tuple): Coordinates (x, y) of the bottom right corner of the transformed image.
        image_shape (tuple): Shape of the original image in the format (height, width).

    Returns:
        tuple: The minimum and maximum x and y coordinates (min_x, max_x, min_y, max_y) defining the
               boundaries of the crop area within the original image dimensions.

    Note:
        The returned coordinates does not cosider the crop area's width and height, only the boundaries.
    """
    # Calculating the minimum and maximum x coordinates for the crop area
    min_x = max(
        top_left[0], bottom_left[0]
    )  # Dont consider image.shape since top_left=[0,0]
    max_x = min(top_right[0], bottom_right[0], image_shape[1])

    # Calculating the minimum and maximum y coordinates for the crop area
    min_y = max(
        top_left[1], top_right[1]
    )  # Dont consider image.shape since top_left=[0,0]
    max_y = min(bottom_right[1], bottom_left[1], image_shape[0])

    return int(min_x), int(max_x), int(min_y), int(max_y)


def _adjust_crop_area_bounds_by_crop_size_and_shift(
    crop_position_min_x: int,
    crop_position_max_x: int,
    crop_position_min_y: int,
    crop_position_max_y: int,
    crop_size: int,
    max_translation_shift: int,
) -> Tuple[int, int, int, int]:
    """
    Adjusts the boundaries of the crop area to accommodate the specified crop size and maximum translation shift.

    This function modifies the minimum and maximum x and y coordinates of the crop area. It ensures that after applying
    a random translation shift to the crop, and considering its size, the crop remains within the original image boundaries.

    Args:
        crop_position_min_x (int): The minimum x-coordinate of the crop area before adjustment.
        crop_position_max_x (int): The maximum x-coordinate of the crop area before adjustment.
        crop_position_min_y (int): The minimum y-coordinate of the crop area before adjustment.
        crop_position_max_y (int): The maximum y-coordinate of the crop area before adjustment.
        crop_size (int): The size of the crop (assuming a square crop).
        max_translation_shift (int): The maximum shift (in pixels) that can be applied to the crop position.

    Returns:
        Tuple[int, int, int, int]: The adjusted minimum and maximum x and y coordinates of the crop area,
                                   ensuring the crop is contained within the original image bounds after
                                   applying the crop size and translation shift.

    Note:
        The function assumes that the initial crop position coordinates are within the image bounds and that
        the crop size and maximum translation shift are appropriate for the image dimensions.
    """

    crop_position_min_x += max_translation_shift
    crop_position_max_x -= crop_size + max_translation_shift
    crop_position_min_y += max_translation_shift
    crop_position_max_y -= crop_size + max_translation_shift

    return (
        crop_position_min_x,
        crop_position_max_x,
        crop_position_min_y,
        crop_position_max_y,
    )


def sample_crop_coordinates(
    coordinate_mapping: torch.tensor,
    crop_size: int,
    max_translation_shift: int,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Samples the top-left corner coordinates for cropping the original and the transformed images.

    This function calculates the cropping coordinates for both the original and the transformed images.
    It ensures that the crops are within the image boundaries and applies a random shift to the crop position
    in the transformed image for variability.

    Args:
        coordinate_mapping (torch.Tensor):  A grid mapping each pixel from image_1 (index)
                                            to its corresponding location in image_2 (value).
        crop_size (int): The size of the square crop.
        max_translation_shift (int): The maximum translation shift applied to the crop position in the
                                     transformed image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-left corner coordinates of the crop in the
                                           original image and the corresponding crop in the transformed image.

    Notes:
        - The function asserts that the top-left corner of the transformed image is at coordinates [0, 0].
        - It checks if the computed crop positions are within valid ranges before sampling.
        - The crop position in the transformed image is shifted randomly within the specified max_translation_shift.
    """

    image_size = coordinate_mapping.shape[:2]

    # Get the corner positions of the transformed image
    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_transformed_image_corner_positions(coordinate_mapping)
    assert (
        top_left[0] == 0 and top_left[1] == 0
    ), "top_left corner of transformed image is not [0,0]"  # top_left=[0,0] since grid_coordinates_transformed[0,0] = [0,0]

    # Calculate the boundaries of the crop area
    (
        crop_position_min_x,
        crop_position_max_x,
        crop_position_min_y,
        crop_position_max_y,
    ) = _calculate_crop_area_bounds(
        top_left, top_right, bottom_left, bottom_right, image_size
    )

    # Adjust the crop area boundaries to account for the crop size and translation shift
    (
        crop_position_min_x,
        crop_position_max_x,
        crop_position_min_y,
        crop_position_max_y,
    ) = _adjust_crop_area_bounds_by_crop_size_and_shift(
        crop_position_min_x,
        crop_position_max_x,
        crop_position_min_y,
        crop_position_max_y,
        crop_size,
        max_translation_shift,
    )

    # Check if sampling space is valid
    assert (
        crop_position_min_x + 1 <= crop_position_max_x
    ), "crop_position_min_x > crop_position_max_x"
    assert (
        crop_position_min_y + 1 <= crop_position_max_y
    ), "crop_position_min_y > crop_position_max_y"

    # Mask for valid crop positions of transformed image
    mask_x = (coordinate_mapping[:, :, 0] >= crop_position_min_x) & (
        coordinate_mapping[:, :, 0] <= crop_position_max_x
    )  # X positions in range
    mask_y = (coordinate_mapping[:, :, 1] >= crop_position_min_y) & (
        coordinate_mapping[:, :, 1] <= crop_position_max_y
    )  # Y positions in range

    # Combine the masks -> An element of mask is True if the corresponding element
    # in coordinate_mapping is a valid crop position
    mask = mask_x & mask_y

    true_indices = torch.nonzero(
        mask
    ).squeeze()  # Indices of the valid keypoints (that are true in mask) in the image_coordinate_mapping

    # Choose a random valid crop position
    random_index = torch.randint(0, len(true_indices), (1,)).item()

    # Use sampled_index to index into true_indices and find the corresponding crop positions
    crop_position_image_1 = (
        true_indices[random_index, 1].item(),
        true_indices[random_index, 0].item(),
    )  # The crop position in the original image is the same as the index in the image_coordinate_mapping
    crop_position_image_2 = coordinate_mapping[
        true_indices[random_index, 0], true_indices[random_index, 1]
    ]

    # Use max_ranodm_offset to shift the crop position in the transformed image
    # -> makes sure that top left corner of crop is not always the same
    crop_position_image_2 += torch.randint(
        -max_translation_shift, max_translation_shift, (2,)
    )

    return crop_position_image_1, crop_position_image_2.long()


def create_crop_coordinate_mapping(
    image_coordinate_mapping: torch.Tensor,
    crop_position_image_1: tuple,
    crop_position_image_2: tuple,
    crop_size: int,
) -> torch.Tensor:
    """
    Creates a grid mapping between two cropped images from their transformed grid coordinates.

    This function adjusts the grid coordinates of a transformed image (grid_coordinates_transformed)
    to create a mapping grid specific to the cropped regions of the original and transformed images.
    It maps each pixel's position from the crop of the original image (image_1) to its corresponding
    position in the crop of the transformed image (image_2).

    Args:
        image_coordinate_mapping (torch.Tensor): A grid mapping each pixel from image_1 (index) to its corresponding location in image_2 (value).
        crop_start_img1 (tuple): The (x, y) coordinates of the top-left corner of the crop in the original image (image_1).
        crop_start_img2 (tuple): The (x, y) coordinates of the top-left corner of the crop in the transformed image (image_2).
        crop_size (int): The size of the square crop.

    Returns:
        torch.Tensor: A grid mapping each pixel from the crop in image_1 to its corresponding location in the crop of image_2.

    Note:
        The grid is of shape (crop_size, crop_size, 2), where the last dimension represents the x and y coordinates of each pixel.
        The grid is initialized with -1, which is used to mark pixels that fall outside the bounds of the transformed image.
    """

    # Extracting a specific region from image_coordinate_mapping to create the crop mapping.
    # This region corresponds to the area of the original image (image_1) that has been cropped.
    crop_mapping_region = image_coordinate_mapping[
        crop_position_image_1[1] : crop_position_image_1[1] + crop_size,
        crop_position_image_1[0] : crop_position_image_1[0] + crop_size,
    ]

    # Adjusting the coordinates relative to the top-left corner of image_2_crop
    adjusted_crop_mapping_region = crop_mapping_region - crop_position_image_2

    # Filtering out coordinates that fall outside the bounds of image_2_crop
    valid_mask = (adjusted_crop_mapping_region >= 0) & (
        adjusted_crop_mapping_region < crop_size
    )
    valid_mask = valid_mask.all(dim=2)
    crop_coordinate_mapping = torch.where(
        valid_mask.unsqueeze(-1), adjusted_crop_mapping_region, -1
    )  # Mark invalid coordinates with -1

    return crop_coordinate_mapping
