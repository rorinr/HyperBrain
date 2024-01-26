import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List
from source.data_processing.image_reading import read_image
from source.data_processing.keypoints import generate_image_grid_coordinates
from source.data_processing.transformations import sample_random_affine_matrix, transform_grid_coordinates, translate_fine_to_coarse, get_relative_coordinates
from source.data_processing.cropping import sample_crop_coordinates, crop_image, create_crop_coordinate_mapping
from source.data_processing.patch_processing import create_match_matrix, get_patch_coordinates
from kornia.geometry.transform import warp_affine
import os


class BrainDataset(Dataset):
    def __init__(
        self,
        images_directory: str,
        train: bool,
        transformation_threshold: float = 0.1,
        crop_size: Tuple[int, int] = (640, 640),
        max_translation_shift: int = 50,
        patch_size: int = 16,
        fine_feature_size: int = 160,
        transform: transforms.transforms.Compose=None
    ) -> None:
        super().__init__()
        self.train = train
        self.image_names = self._get_image_names(images_directory=images_directory)
        self.images_directory = images_directory
        self.transform = transform
        self.transformation_threshold = transformation_threshold
        self.crop_size = crop_size
        self.max_translation_shift = max_translation_shift
        self.patch_size = patch_size
        self.fine_feature_size = fine_feature_size

    def __len__(self) -> int:
        """
        Return the number of images in the dataset -1, since this dataset works with pairs of images.
        """
        return len(self.image_names[:-1])

    def _get_image_names(self, images_directory: str) -> List[str]:
        """
        Load image names from the directory based on training or testing phase.

        Args:
            images_directory (str): Directory path containing images.

        Returns:
            List[str]: A list of image file names.
        """
        image_files = os.listdir(images_directory)
        return image_files[:-2] if self.train else image_files[-2:]

    def _get_images(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a pair of images from the dataset.

        Args:
            index (int): Index of the pair of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the images.
        """
        image_1 = read_image(
            os.path.join(self.images_directory, self.image_names[index])
        )
        image_2 = read_image(
            os.path.join(self.images_directory, self.image_names[index + 1])
        )

        image_1 = transforms.ToTensor()(image_1)
        image_2 = transforms.ToTensor()(image_2)

        return image_1, image_2

    def __getitem__(self, index: int):
        """
        Retrieves a transformed and preprocessed sample from the dataset at the specified index.

        This method performs several steps:
        1. Reads the original images (image_1 and image_2) corresponding to the given index.
        2. Applies transformations (e.g., normalization) to both images.
        3. Applies a random affine transformation to image_2.
        4. Generates and transforms a grid of pixel coordinates to map the correspondence between image_1 and the transformed image_2.
        5. Samples valid crop positions for both images and crops them.
        6. Generates a crop coordinate mapping to identify corresponding pixels between the two cropped images.
        7. Creates a binary match matrix indicating which patches in the two crops match.
        8. Determines the mid-pixels of matched patches and translates these points to the fine feature level.
        9. Computes the relative positions of the transformed mid-pixels of image_1_crop with respect to the mid-pixels of image_2_crop patches in the fine feature level.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - image_1_crop (torch.Tensor): The cropped patch from image_1.
                - image_2_crop (torch.Tensor): The cropped and transformed patch from image_2.
                - match_matrix (torch.Tensor): A binary matrix indicating matches between patches in the two crops.
                - relative_coordinates (torch.Tensor): Relative coordinates of the transformed mid-pixels of image_1 with respect to image_2 at the fine feature level.
        """
        # Read whole image
        image_1, image_2 = self._get_images(index=index)
        image_2_size = image_2.shape[-2:]

        # Apply transformations if passed (eg normlaization)
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # Apply random affine transformation to image 2
        transformation_matrix = sample_random_affine_matrix(
            range_limit=self.transformation_threshold
        )
        image_2_transformed = warp_affine(src=image_2.unsqueeze(0), 
                                          M=transformation_matrix,
                                          dsize=image_2_size,
                                          mode="nearest")
        
        # Generate (height, width, 2) grid of pixel coordinates for image 2
        grid_coordinates = generate_image_grid_coordinates(image_size=image_2_size)

        # Transform the grid coordinates according to the affine transformation
        # Pixel (i,j) of image_1 corresponds to pixel image_coordinate_mapping[i,j] in image_2
        image_coordinate_mapping = transform_grid_coordinates(grid_coordinates=grid_coordinates, 
                                                              transformation_matrix=transformation_matrix[0])
        
        # Sample valid crop positions for image_1 and image_2_transformed
        crop_position_image_1, crop_position_image_2 = sample_crop_coordinates(
            coordinate_mapping=image_coordinate_mapping,
            crop_size=self.crop_size,
            max_translation_shift=self.max_translation_shift)
        
        # Crop the images
        image_1_crop = crop_image(image=image_1, crop_position=crop_position_image_1, crop_size=self.crop_size
        )
        image_2_crop = crop_image(
            image=image_2_transformed, crop_position=crop_position_image_2, crop_size=self.crop_size
        )

        # Generate a mapping between the two crops - interpreted similar to image_coordinate_mapping
        crop_coordinate_mapping = create_crop_coordinate_mapping(
            image_coordinate_mapping=image_coordinate_mapping,
            crop_position_image_1=crop_position_image_1,
            crop_position_image_2=crop_position_image_2,
            crop_size=self.crop_size)

        # Final step of coarse supervision
        # Match matrix can be interpreted as a binary matrix indicating patch matches
        # match_matrix[i,j] = 1 if patch i in image_1_crop matches patch j in image_2_crop
        match_matrix = create_match_matrix(crop_coordinate_mapping=crop_coordinate_mapping, 
                                           crop_size=self.crop_size, 
                                           patch_size=self.patch_size)

        # Beginning fine supvervison
        # Get matched patches in crop 1 and crop 2
        crop_1_patch_indices = match_matrix.nonzero()[:, 0]  # Get all matched patches indices in crop 1
        crop_2_patch_indices = match_matrix.nonzero()[:, 1]  # Get all matched patches indices in crop 2

        # Get mid point of patches in crop 1 and crop 2
        crop_size_half = self.crop_size // 2
        crop_1_patch_mid_indices = get_patch_coordinates(patch_indices=crop_1_patch_indices) + torch.Tensor([crop_size_half,crop_size_half]).long()
        crop_2_patch_mid_indices = get_patch_coordinates(patch_indices=crop_2_patch_indices) + torch.Tensor([crop_size_half,crop_size_half]).long()

        # Translate the mid points of patches in crop 1 and 2 to the fine feature level
        crop_1_patch_mid_indices_fine = translate_fine_to_coarse(fine_coordinates=crop_1_patch_mid_indices, 
                                                                 coarse_size=160, 
                                                                 fine_size=640)
        crop_2_patch_mid_indices_fine = translate_fine_to_coarse(fine_coordinates=crop_2_patch_mid_indices, 
                                                                 coarse_size=160, 
                                                                 fine_size=640)
        
        # Note: An element in crop_2_patch_mid_indices(_fine) doesnt necessarily correspond to the same pixel as the element in crop_1_patch_mid_indices(_fine)
        # It is just the mid point of the matched patch in the crop
        #Therefore we need to find the exact pixel in crop 2 that corresponds to crop_1_patch_mid_indices(_fine)

        # Compute where the mid point of crop 1 went exactly in crop 2 through the affine transformation.
        crop_1_mid_pixels_transformed = crop_coordinate_mapping[crop_1_patch_mid_indices[:, 1], crop_1_patch_mid_indices[:, 0]]
        crop_1_mid_pixels_transformed_fine = translate_fine_to_coarse(crop_1_mid_pixels_transformed, coarse_size=self.fine_feature_size, fine_size=image_2_size[0])

        # Computed the relative position of crop_1_mid_pixels_transformed_fine wrt crop_2_patch_mid_indices_fine
        # At test time we only know the mid point of patches in crop_2 (or in the fine feature level) but not
        # the exact pixel or relative coordinates. Therefore the model needs to predict the relative coordinates.
        # Relative coordinates are [-1, 1] for (top-left, bottom-right) corners of the patch.
        relative_coordinates = get_relative_coordinates(transformed_coordinates=crop_1_mid_pixels_transformed_fine, 
                                                        reference_coordinates=crop_2_patch_mid_indices_fine)

        # Create the final output
        # Note: The output is a tuple of 4 elements. The first two elements are the crops of image 1 and image 2.
        # The third element is the match matrix. The fourth element are the relative coordinates.
        return image_1_crop, image_2_crop, match_matrix, relative_coordinates