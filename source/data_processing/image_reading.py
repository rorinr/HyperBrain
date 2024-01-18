from typing import Tuple, Optional
from PIL import Image

def read_image(path: str, size: Optional[Tuple[int, int]] = None, layer: int = None) -> Image.Image:
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
