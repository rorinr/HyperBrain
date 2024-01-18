import torch

def generate_keypoints_for_every_pixel(image_shape):
    """
    Generate keypoints for every pixel in an image.

    Parameters:
    - image_shape: A tuple representing the shape of the image (height, width).

    Returns:
    - keypoints: A torch tensor of shape (height*width, 2), where each row is the (x, y) coordinate of a pixel.
    """
    height, width = image_shape
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    keypoints = torch.stack((x_coords.reshape(-1), y_coords.reshape(-1)), dim=1)
    return keypoints