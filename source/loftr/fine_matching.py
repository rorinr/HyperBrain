import torch
from torch import nn

from kornia.geometry.subpix import dsnt

class FineMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, fine_image_feature_1: torch.Tensor, fine_image_feature_2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fine feature matching.

        Args:
            fine_image_feature_1 (torch.Tensor): Fine features of the first image. 
                                                 Shape: [number_of_matches, window_size_squared, d_model_fine].
            fine_image_feature_2 (torch.Tensor): Fine features of the second image (unflattened).
                                                 Shape: [number_of_matches, window_size_squared, d_model_fine].

        Returns:
            torch.Tensor: Predicted match coordinates.
        """
        
        # Extract dimensions
        number_of_matches, window_size_squared, d_model_fine = fine_image_feature_1.shape
        window_size = int(window_size_squared ** 0.5)

        # Select mid feature of each window in fine_image_feature_1
        fine_image_feature_1_mid = fine_image_feature_1[:, window_size_squared // 2, :]

        # Compute similarity matrix using einsum
        similarity_matrix = torch.einsum(
            "mc,mrc->mr", fine_image_feature_1_mid, fine_image_feature_2
        )

        # Softmax normalization factor
        softmax_temperature = 1.0/d_model_fine**0.5

        # Compute the heatmap by applying softmax and reshaping
        heatmap = torch.softmax(similarity_matrix * softmax_temperature, dim=1).view(
            -1, window_size, window_size
        )

        # Predict the match coordinates using spatial expectation
        predicted_matches = dsnt.spatial_expectation2d(heatmap[None], normalized_coordinates=True)[0]

        return predicted_matches