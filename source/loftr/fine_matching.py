import torch
from torch import nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    def __init__(
        self, return_standard_deviation: bool = False, clamp_predictions: bool = False
    ) -> None:
        super().__init__()
        self.return_standard_deviation = return_standard_deviation
        self.clamp_predictions = clamp_predictions

    def forward(
        self, fine_image_feature_1: torch.Tensor, fine_image_feature_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for fine feature matching.

        Args:
            fine_image_feature_1 (torch.Tensor): Fine features of the first image.
                                                 Shape: [number_of_matches, window_size_squared, fine_feature_size].
            fine_image_feature_2 (torch.Tensor): Fine features of the second image (unflattened).
                                                 Shape: [number_of_matches, window_size_squared, fine_feature_size].

        Returns:
            torch.Tensor: Predicted match coordinates.
        """

        # Extract dimensions
        (
            number_of_matches,
            window_size_squared,
            fine_feature_size,
        ) = fine_image_feature_1.shape
        window_size = int(window_size_squared**0.5)

        # Select mid feature of each window in fine_image_feature_1
        fine_image_feature_1_mid = fine_image_feature_1[:, window_size_squared // 2, :]

        # Compute similarity matrix using einsum
        similarity_matrix = torch.einsum(
            "mc,mrc->mr", fine_image_feature_1_mid, fine_image_feature_2
        )

        # Softmax normalization factor
        softmax_temperature = 1.0 / fine_feature_size**0.5

        # Compute the heatmap by applying softmax and reshaping
        heatmap = torch.softmax(similarity_matrix * softmax_temperature, dim=1).view(
            -1, window_size, window_size
        )

        # Predict the match coordinates using spatial expectation
        predicted_matches = dsnt.spatial_expectation2d(
            heatmap[None], normalized_coordinates=True
        )[0]

        if self.clamp_predictions:
            # Clip the predicted matches to be within the specified range
            predicted_matches = torch.clamp(predicted_matches, min=-1, max=0.5)

        if not self.return_standard_deviation:
            return predicted_matches

        grid_normalized = create_meshgrid(
            window_size, window_size, True, heatmap.device
        ).reshape(
            1, -1, 2
        )  # [1, window_size_squared, 2]

        # compute std over <x, y>
        variance = (
            torch.sum(
                grid_normalized**2 * heatmap.view(-1, window_size_squared, 1), dim=1
            )
            - predicted_matches**2
        )  # [number_of_matches, 2]
        standard_deviation = torch.sum(
            torch.sqrt(torch.clamp(variance, min=1e-10)), -1
        )  # [number_of_matches]  clamp needed for numerical stability

        return torch.cat([predicted_matches, standard_deviation.unsqueeze(1)], -1)
