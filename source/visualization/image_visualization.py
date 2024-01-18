import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def plot_images_with_matches(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    grid_coordinates_transformed: torch.Tensor,
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
        grid_coordinates_transformed (torch.Tensor): Tensor of transformed grid
                                                     coordinates, shape [H, W, 2].
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
        transformed_x, transformed_y = grid_coordinates_transformed[y, x]

        # Draw lines only for points within the image boundaries
        if 0 <= transformed_x < width and 0 <= transformed_y < height:
            plt.plot(
                [x, transformed_x + width], [y, transformed_y], c="r", linewidth=0.5
            )

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def plot_image_with_crop(
    original_image: torch.Tensor, 
    crop_image: torch.Tensor, 
    crop_position: Tuple[int, int]
)->None:
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
    axs[0].imshow(original_image_np, cmap='gray')
    axs[0].set_title('Original Image')

    # Create a rectangle patch
    crop_x, crop_y = crop_position
    crop_height, crop_width = crop_image.shape[1:3]
    rect = patches.Rectangle((crop_x, crop_y), crop_width, crop_height, linewidth=1, edgecolor='r', facecolor='none')
    
    # Add the rectangle to the original image plot
    axs[0].add_patch(rect)

    # Plot the cropped image
    axs[1].imshow(crop_image_np, cmap='gray')
    axs[1].set_title('Crop')

    # Hide axes for both plots
    # axs[0].axis('off')
    # axs[1].axis('off')

    # Show the plot
    plt.show()