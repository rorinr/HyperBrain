import torch
import matplotlib.pyplot as plt
import numpy as np


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
