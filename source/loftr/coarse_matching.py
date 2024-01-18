import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class CoarseMatching(nn.Module):
    def __init__(self, temperature: float, confidence_threshold: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

    def normalize_features(
        self, coarse_image_feature_1: torch.Tensor, coarse_image_feature_2: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Normalize two feature tensors by dividing each feature by the square root of the last dimension's size.

        This normalization is used to scale features, making them more comparable and
        improving the stability of downstream calculations, such as dot product for similarity measures.

        Args:
            coarse_image_feature_1 (torch.Tensor): The first feature tensor, typically of shape (N, L, C),
                                                    where N is the batch size, L is the number of features,
                                                    and C is the size of each feature.
            coarse_image_feature_2 (torch.Tensor): The second feature tensor, typically of the same shape as the first.

        Returns:
            list of torch.Tensor: A list containing two normalized feature tensors, corresponding to the input tensors.

        Note:
            The normalization is performed by dividing each feature vector in the tensor by the square root of the
            feature's size (the size of the last dimension, C).
        """
        return [
            feature / feature.shape[-1] ** 0.5
            for feature in (coarse_image_feature_1, coarse_image_feature_2)
        ]

    def get_confidence_matrix(
        self, coarse_image_feature_1: torch.Tensor, coarse_image_feature_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a confidence matrix from two sets of image features using a temperature-scaled dot product and dual softmax.

        This function first computes a similarity matrix between each pair of features from two different images using
        a temperature-scaled dot product. It then applies a dual softmax operation to convert the similarity matrix into
        a confidence matrix. The temperature parameter controls the sharpness of the softmax distribution, with lower
        values yielding sharper distributions.

        Args:
            coarse_image_feature_1 (torch.Tensor): The first set of image features, typically of shape (N, L, C),
                                                    where N is the batch size, L is the number of features,
                                                    and C is the size of each feature vector.
            coarse_image_feature_2 (torch.Tensor): The second set of image features, typically of the same shape as the first.
            temperature (float): A scaling factor for controlling the sharpness of the softmax distribution.
                                Lower values result in a sharper distribution.

        Returns:
            torch.Tensor: A confidence matrix of shape (N, L, S), where N is the batch size, L is the number of features
                        in coarse_image_feature_1, and S is the number of features in coarse_image_feature_2. Each element
                        in the matrix represents the confidence of matching a feature in coarse_image_feature_1 with a feature
                        in coarse_image_feature_2.
        Note:
            A specific element of the confidence matrix, denoted as confidence_matrix[i, j], represents the confidence of
            matching the i-th feature in the first set of image features (from coarse_image_feature_1) with the j-th feature
            in the second set of image features (from coarse_image_feature_2).
        """

        # Similarity matrix using dot-product
        similarity_matrix = (
            torch.einsum("nlc,nsc->nls", coarse_image_feature_1, coarse_image_feature_2)
            / self.temperature
        )

        # Confidence matrix using dual softmax
        confidence_matrix = F.softmax(similarity_matrix, 1) * F.softmax(
            similarity_matrix, 2
        )
        return confidence_matrix

    def extract_matching_confidences(
        self, confidence_matrix: torch.Tensor, match_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts the confidence values for matches determined by a mask.

        The function assumes that the mask has been processed with mutual nearest neighbors criteria,
        ensuring that there is at most one 'True' in each row. For each 'True' in the mask, the function
        finds the corresponding confidence value in the confidence_matrix.

        Args:
            confidence_matrix (torch.Tensor): A tensor of shape (batch_size, num_features_1, num_features_2)
                                            representing the confidence of matching features between two images.
            match_mask (torch.Tensor): A boolean tensor of the same shape as confidence_matrix, where 'True'
                                    indicates a match between features.

        Returns:
            dict: A dictionary containing the matching confidence values, and the batch, row, and column indices.
                This indices indicate the position of the coarse matches in the confidence matrix.
        """
        # Extract the maximum values and their indices along the third dimension (columns) of the mask
        max_values_per_row, max_indices_per_row = match_mask.max(dim=2)

        # Find the batch and row indices where the mask has 'True' values
        batch_indices, row_indices = torch.where(max_values_per_row)

        # Use the batch and row indices to locate the column indices of matches in the mask
        column_indices = max_indices_per_row[batch_indices, row_indices]

        # Extract the confidence values for these matches from the confidence matrix
        matching_confidences = confidence_matrix[
            batch_indices, row_indices, column_indices
        ]

        # Create a dictionary with names and their corresponding values
        result = {
            "matching_confidences": matching_confidences,
            "batch_indices": batch_indices,
            "row_indices": row_indices,
            "column_indices": column_indices,
        }

        return result

    def forward(
        self, coarse_image_feature_1: torch.Tensor, coarse_image_feature_2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the CoarseMatching module, processing two sets of
        image features to identify and extract coarse matches.

        The method first normalizes the input feature tensors using the 'normalize_features'
        method. It then computes a confidence matrix for the feature matches using the
        'get_confidence_matrix' method. Finally, it extracts the coarse matches from this
        confidence matrix using the 'get_coarse_matches' method.

        Args:
            coarse_image_feature_1 (torch.Tensor): The first set of image features,
            typically of shape (N, L, C), where N is the batch size, L is the number
            of features, and C is the size of each feature vector.
            coarse_image_feature_2 (torch.Tensor): The second set of image features,
            typically of the same shape as the first.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the extracted coarse
            match information. This includes the confidence values ('matching_confidences')
            and the indices ('batch_indices', 'row_indices', 'column_indices') of the matches
            in the confidence matrix.
        """
        coarse_image_feature_1, coarse_image_feature_2 = self.normalize_features(
            coarse_image_feature_1, coarse_image_feature_2
        )
        confidence_matrix = self.get_confidence_matrix(
            coarse_image_feature_1=coarse_image_feature_1,
            coarse_image_feature_2=coarse_image_feature_2,
        )

        return self.get_coarse_matches(confidence_matrix=confidence_matrix)

    @torch.no_grad()
    def get_coarse_matches(
        self, confidence_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Identifies and extracts coarse matches from a given confidence matrix based on a set confidence threshold and mutual nearest neighbors criteria.
        This function first applies a threshold to the confidence matrix to consider only the matches with a confidence level above the specified threshold.
        It then employs mutual nearest neighbors criteria to refine these matches, ensuring each match is the best choice in both directions (rows and columns).
        Finally, it extracts and returns the confidence values and indices of these selected matches.

        Args:
            confidence_matrix (torch.Tensor): A tensor of shape (batch_size, num_features_1, num_features_2) representing the confidence levels of
            matching features between two sets of image features.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the confidence values (`'matching_confidences'`) and
            the indices (`'batch_indices'`, `'row_indices'`, `'column_indices'`) of the selected coarse matches.
            These indices indicate the positions of the matches in the confidence matrix.

        Note:
            The method uses a confidence threshold defined in the class to filter the initial matches.
            The mutual nearest neighbors criteria is then applied to this filtered set to ensure
            that each selected match is the best in both the row and column dimensions.
            This method is decorated with `@torch.no_grad()` to disable gradient tracking during its execution.
        """

        # 1. Use only confidence values above the threshold
        match_mask = confidence_matrix > self.confidence_threshold

        # TODO: Masking to avoid matches near border

        # 2. Mutual nearest neighbors: find matches in both directions
        match_mask = (
            match_mask
            * (confidence_matrix == confidence_matrix.max(dim=2, keepdim=True)[0])
            * (confidence_matrix == confidence_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. Extract the matching confidence values and their indices
        coarse_matches = self.extract_matching_confidences(
            confidence_matrix=confidence_matrix, match_mask=match_mask
        )

        return coarse_matches, match_mask, confidence_matrix
