import torch
from typing import Tuple


def _get_transformed_image_corner_positions(
    grid_coordinates_transformed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retrieves the corner coordinates of a transformed image grid.

    Given a tensor representing the transformed grid coordinates of an image,
    this function calculates the coordinates of the four corners: top-left, top-right,
    bottom-left, and bottom-right.

    Args:
        grid_coordinates_transformed (torch.Tensor): A tensor representing the transformed
                                                     grid coordinates of an image.

    Returns:
        tuple: A tuple containing the coordinates (x, y) of the top-left, top-right,
               bottom-left, and bottom-right corners of the transformed image grid.
    """
    top_left = grid_coordinates_transformed[0, 0]
    top_right = grid_coordinates_transformed[0, -1]
    bottom_left = grid_coordinates_transformed[-1, 0]
    bottom_right = grid_coordinates_transformed[-1, -1]

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


def _adjust_crop_area_boundaries(
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
    grid_coordinates_transformed: torch.tensor,
    crop_size: int,
    max_translation_shift: int,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Samples the top-left corner coordinates for cropping the original and the transformed images.

    This function calculates the cropping coordinates for both the original and the transformed images.
    It ensures that the crops are within the image boundaries and applies a random shift to the crop position
    in the transformed image for variability.

    Args:
        crop_size (int): The size of the square crop.
        grid_coordinates_transformed (torch.Tensor): A tensor representing the grid coordinates of the
                                                     transformed image.
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

    image_size = grid_coordinates_transformed.shape[:2]

    # Get the corner positions of the transformed image
    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_transformed_image_corner_positions(grid_coordinates_transformed)
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
    ) = _adjust_crop_area_boundaries(
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

    # Sample random crop position for original image
    original_crop_position_x = torch.randint(
        crop_position_min_x, crop_position_max_x, (1,)
    ).item()
    original_crop_position_y = torch.randint(
        crop_position_min_y, crop_position_max_y, (1,)
    ).item()

    # Find where the original crop position is located in the transformed image
    original_crop_position_transformed = grid_coordinates_transformed[
        original_crop_position_y, original_crop_position_x
    ]

    # Calculate the transformed crop position by shifting the original crop position randomly
    transformed_crop_position_x = (
        original_crop_position_transformed[0]
        + torch.randint(-max_translation_shift, max_translation_shift, (1,)).item()
    )
    transformed_crop_position_y = (
        original_crop_position_transformed[1]
        + torch.randint(-max_translation_shift, max_translation_shift, (1,)).item()
    )

    # Create crop positions
    original_crop_position = torch.tensor(
        [original_crop_position_x, original_crop_position_y]
    )
    transformed_crop_position = torch.tensor(
        [int(transformed_crop_position_x), int(transformed_crop_position_y)]
    )  # Make sure coordinates are integers

    return original_crop_position, transformed_crop_position
