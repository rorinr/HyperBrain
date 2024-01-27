import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import matplotlib.patches as patches
from typing import List

def plot_images_in_row(images: List[torch.Tensor], figsize=(15, 5)) -> None:
    """
    Plots a sequence of images in a row using Matplotlib.

    Args:
        images (list of torch.Tensor): A list of image tensors to be plotted.
                                       Each image tensor should have shape [H, W] or [H, W, C].
        figsize (tuple): The size of the figure. Defaults to (15, 5).
    """

    # Number of images
    n = len(images)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, n, figsize=figsize)

    # If there is only one image, axes will not be an array, so we convert it into one for consistency
    if n == 1:
        axes = [axes]

    # Plot each image
    for ax, image in zip(axes, images):
        # Convert the image to numpy if it's a tensor
        if hasattr(image, 'numpy'):
            image = image.numpy()

        # Use imshow to display the image
        ax.imshow(image)

    plt.show()


def plot_images_with_matches_via_mapping(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    coordinate_mapping: torch.Tensor,
    number_of_matches: int,
    sample_keypoints_randomly: bool = False,
) -> None:
    """
    Plots matches between two images, showing how keypoints from the first image
    are transformed and matched to the second image.

    This function concatenates two images side by side and draws lines between
    matching keypoints. It skips any points that are transformed outside the
    boundaries of the second image. Keypoints can be either regularly spaced or randomly sampled.

    Args:
        image_1 (torch.Tensor): The first image tensor of shape [C, H, W].
        image_2 (torch.Tensor): The second image tensor of the same shape as image_1.
        coordinate_mapping (torch.Tensor): coordinate_mapping (torch.Tensor): A grid mapping each pixel from image
        _1 (index) to its corresponding location in image_2 (value).
        number_of_matches (int): Number of matches to plot.
        sample_keypoints_randomly (bool): If True, keypoints are chosen randomly. Default is False.

    """
    # Concatenate the images for side-by-side comparison
    concatenated_image = torch.cat((image_1, image_2), dim=2)
    plt.imshow(concatenated_image[0], cmap="gray")

    height, width = image_1.shape[1:3]  # Extract height and width from image_1 shape

    if sample_keypoints_randomly:
        # Randomly sample keypoints
        y_coords = np.random.randint(0, height, number_of_matches)
        x_coords = np.random.randint(0, width, number_of_matches)
    else:
        # Calculate step sizes for regular spacing
        step_y = height // number_of_matches
        step_x = width // number_of_matches
        y_coords = np.arange(0, height, step_y)
        x_coords = np.arange(0, width, step_x)

    # Loop over sampled keypoints
    for y, x in zip(y_coords, x_coords):
        transformed_x, transformed_y = coordinate_mapping[y, x]

        # Draw lines only for points within the image boundaries
        if 0 <= transformed_x < width and 0 <= transformed_y < height:
            plt.plot(
                [x, transformed_x + width], [y, transformed_y], c="r", linewidth=0.5
            )

    # Show the plot
    plt.show()


def plot_images_with_matches_via_match_matrix(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    match_matrix: torch.Tensor,
    visualization_mode: str,
    patch_size: int = 16,
    line_frequency: int = 50,
) -> None:
    """
    Visualizes matches between two image crops using a match matrix.

    Args:
        image_1 (torch.Tensor): The first image crop.
        image_2 (torch.Tensor): The second image crop.
        match_matrix (torch.Tensor): A binary matrix indicating matches between patches in the two crops.
        visualization_model (str): Either 'lines' or 'patches'. How to visualize the matches.
        patch_size (int): The size of each square patch. Defaults to 16.
    """

    num_patches_per_side = image_1.shape[1] // patch_size
    image_1_width = image_1.shape[2]

    # Prepare the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.imshow(torch.cat((image_1, image_2), dim=2)[0], cmap="gray")

    match_indices = match_matrix.nonzero()

    # Drawing lines for a subset of matches
    if visualization_mode == "lines":
        for patch_index_1, patch_index_2 in match_indices[::line_frequency]:
            x1, y1 = divmod(patch_index_1.item(), num_patches_per_side)
            x2, y2 = divmod(patch_index_2.item(), num_patches_per_side)
            x1, y1 = (
                x1 * patch_size + patch_size // 2,
                y1 * patch_size + patch_size // 2,
            )
            x2, y2 = (
                x2 * patch_size + patch_size // 2,
                y2 * patch_size + patch_size // 2,
            )
            ax.plot([y1, y2 + image_1_width], [x1, x2], color="red", linewidth=0.5)

    if visualization_mode == "patches":
        # Marking all matches with rectangles
        for patch_index_1, patch_index_2 in match_indices:
            x1, y1 = divmod(patch_index_1.item(), num_patches_per_side)
            x2, y2 = divmod(patch_index_2.item(), num_patches_per_side)
            x1, y1 = x1 * patch_size, y1 * patch_size
            x2, y2 = x2 * patch_size, y2 * patch_size

            # Create a colored rectangle in both image crops
            for rect_x, rect_y in [(y1, x1), (y2 + image_1_width, x2)]:
                rect = patches.Rectangle(
                    (rect_x, rect_y),
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                )
                ax.add_patch(rect)

    plt.show()


def plot_image_with_crop(
    original_image: torch.Tensor,
    crop_image: torch.Tensor,
    crop_position: Tuple[int, int],
) -> None:
    """
    Plots the original image and its cropped section side by side, with a rectangle on the original
    image indicating the crop's location.

    Args:
        original_image (torch.Tensor): The original image tensor of shape (1, height, width).
        crop_image (torch.Tensor): The cropped section of the image of shape (1, crop_height, crop_width).
        crop_position (tuple): The (x, y) coordinates of the top-left corner of the crop in the original image.

    """
    # Convert tensors to numpy for matplotlib compatibility
    original_image_np = original_image.squeeze().numpy()
    crop_image_np = crop_image.squeeze().numpy()

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image
    axs[0].imshow(original_image_np, cmap="gray")
    axs[0].set_title("Original Image")

    # Create a rectangle patch
    crop_x, crop_y = crop_position
    crop_height, crop_width = crop_image.shape[1:3]
    rect = patches.Rectangle(
        (crop_x, crop_y),
        crop_width,
        crop_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    # Add the rectangle to the original image plot
    axs[0].add_patch(rect)

    # Plot the cropped image
    axs[1].imshow(crop_image_np, cmap="gray")
    axs[1].set_title("Crop")

    # Hide axes for both plots
    # axs[0].axis('off')
    # axs[1].axis('off')

    # Show the plot
    plt.show()
