from typing import Tuple, Optional
from PIL import Image
Image.MAX_IMAGE_PIXELS = 9999999999999999
import os

def read_image(
    path: str, size: Optional[Tuple[int, int]] = None, layer: int = None
) -> Image.Image:
    """
    Reads and optionally resizes an image from the given path.

    This function opens an image from the specified path. If a layer is specified for a
    multi-layered image (like a tiff file), it seeks to that layer. If a size is provided,
    the image is resized to the specified size; otherwise, the original size is kept.

    Args:
        path: A string representing the path to the image file.
        size: An optional tuple (width, height) representing the size to which the image
              is resized. If None, the image is not resized. Default is None.
        layer: An optional integer specifying the layer of the image file to be read.
               This is particularly useful for multi-layered file formats like tiff.
               Default is None, which reads the first layer.

    Returns:
        A PIL Image object containing the image, resized if size was provided.
    """
    with Image.open(path) as image:
        if layer is not None:
            image.seek(layer)

        if size is not None:
            image = image.resize(size, Image.LANCZOS)

        return image.copy()

def downscale_image(image: Image.Image, downscale_factor: int) -> Image.Image:
    """
    Downscales the given image by the specified factor.

    Args:
        image: A PIL Image object to be downscaled.
        downscale_factor: An integer representing the factor by which the image should be downscaled.
                          The downscale factor must be a positive integer.

    Returns:
        A new PIL Image object that is downscaled by the given factor.

    Raises:
        ValueError: If downscale_factor is not a positive integer.
    """
    if downscale_factor <= 0:
        raise ValueError("downscale_factor must be a positive integer.")

    # Calculate the new size
    original_width, original_height = image.size
    new_width = original_width // downscale_factor
    new_height = original_height // downscale_factor

    # Resize the image
    downscaled_image = image.resize((new_width, new_height), Image.LANCZOS)

    return downscaled_image

def save_image(image: Image.Image, save_dir: str, filename: str) -> None:
    """
    Saves the given image to the specified directory with the given filename.

    Args:
        image: A PIL Image object to be saved.
        save_dir: A string representing the directory path where the image will be saved.
        filename: A string representing the name of the file under which the image will be saved.

    Raises:
        ValueError: If the directory does not exist and cannot be created.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError as error:
            raise ValueError(f"Failed to create directory {save_dir}. Error: {error}")

    # Construct the full path where the image will be saved
    save_path = os.path.join(save_dir, filename)

    # Save the image
    image.save(save_path)

def downscale_folder(image_folder: str, downscale_factor: int) -> None:
    """
    Downscale all images in the given folder by the specified factor.

    Args:
        image_folder: A string representing the directory path where the images are located.
        downscale_factor: An integer representing the factor by which the images should be downscaled.
                          The downscale factor must be a positive integer.
    """
    downscaled_image_folder = image_folder + f"_downscaled_{downscale_factor}x"

    # Iterate over all files in the folder
    for filename in os.listdir(image_folder):
        # Check if the file is an image
        if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(image_folder, filename)
            image = read_image(image_path)

            # Downscale the image
            downscaled_image = downscale_image(image, downscale_factor)

            # Save the downscaled image
            save_image(downscaled_image, downscaled_image_folder, f"{filename.split('.')[0]}.tif")