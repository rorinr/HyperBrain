import torch
from source.loftr.backbone import ResNetFPN_16_4, ResNetFPN_8_2
from source.loftr.positional_encoding import PositionalEncoding
from source.loftr.transformer import LocalFeatureTransformer
from source.loftr.coarse_matching import CoarseMatching
from source.loftr.fine_matching import FineMatching
from source.loftr.fine_preprocess import FinePreprocess
from source.data_processing.image_reading import read_image
from torchvision.transforms import ToTensor
import h5py
import cv2
import numpy as np
from einops import rearrange
from source.data_processing.keypoints import translate_patch_midpoints_and_refine
from source.data_processing.cropping import crop_image, create_crop_coordinate_mapping
import os
from source.miscellaneous.model_saving import generate_next_id
import json
from typing import List, Dict
from torchvision.transforms import v2



def compute_euclidean_distances(
    predicted_matches: torch.Tensor, coordinate_mapping: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Euclidean distance between predicted matches and the corresponding ground truth in a vectorized manner.

    Args:
        predicted_matches (torch.Tensor): A tensor of shape (M, 4) containing predicted matches.
                                           Each row is [i, j, k, l], where (i, j) are coordinates in image 1,
                                           and (k, l) are the predicted matching coordinates in image 2.
        coordinate_mapping (torch.Tensor): A 3D tensor of shape (H, W, 2) where each entry [i, j] contains
                                           the ground truth (x, y) coordinates in image 2 for the pixel (i, j)
                                           in image 1.

    Returns:
        torch.Tensor: A tensor of shape (M,) containing the Euclidean distances for each predicted match.
    """
    # Extract (x,y) indices from the first image of the prediction
    predicted_coordinates_image_1 = predicted_matches[:, :2].long()

    # Use indexing to get the ground truth coordinates for (x,y)
    ground_truth_coordinates = coordinate_mapping[
        predicted_coordinates_image_1[:, 1], predicted_coordinates_image_1[:, 0]
    ]

    # Some pixel coordinates may be in image 1 but not in image 2, so we need to remove them
    pixel_exists_mask = (ground_truth_coordinates != -1).all(dim=1)
    predicted_matches = predicted_matches[pixel_exists_mask]
    ground_truth_coordinates = ground_truth_coordinates[pixel_exists_mask]

    # Extract the predicted (x,y) coordinates for the second image
    predicted_coordinates_image_2 = predicted_matches[:, 2:]

    # Compute the Euclidean distances
    distances = torch.norm(
        ground_truth_coordinates - predicted_coordinates_image_2, dim=1
    )

    return distances


def count_matches_per_patch(
    matches: torch.Tensor, x_borders: torch.Tensor, y_borders: torch.Tensor
) -> torch.Tensor:
    """Counts the number of matches in each patch defined by x and y borders.

    This function first calculates the indices of the patches that each match
    belongs to and then updates the count of matches in each patch accordingly.

    Args:
        matches: A tensor of shape (N, 4) containing match coordinates in the format
            [x1, y1, x2, y2], where x1, y1 are the coordinates in the first image.
        x_borders: A tensor containing the x-coordinates of the patch borders.
        y_borders: A tensor containing the y-coordinates of the patch borders.

    Returns:
        A 2D float tensor of shape (y_patches, x_patches) representing the number
        of matches in each patch.
    """
    # Calculate the number of patches
    x_patches = len(x_borders) - 1
    y_patches = len(y_borders) - 1
    counts = torch.zeros((y_patches, x_patches), dtype=torch.int32)

    # Extract x and y coordinates from the matches
    x_coords = matches[:, 0]
    y_coords = matches[:, 1]

    # Find the patch indices for the x and y coordinates
    x_indices = torch.searchsorted(x_borders, x_coords) - 1
    y_indices = torch.searchsorted(y_borders, y_coords) - 1

    # Filter out matches that are outside the borders
    valid_indices = (
        (x_indices >= 0)
        & (x_indices < x_patches)
        & (y_indices >= 0)
        & (y_indices < y_patches)
    )

    # Update counts using the valid patch indices
    for x_index, y_index in zip(x_indices[valid_indices], y_indices[valid_indices]):
        counts[y_index, x_index] += 1

    return counts.float()


def calculate_entropy(counts: torch.Tensor) -> torch.Tensor:
    """Calculates the entropy of a distribution represented by counts.

    Entropy is a measure of the unpredictability or randomness of a distribution.
    This function converts the counts into probabilities and then computes the
    entropy of the distribution using the formula:
    H(P) = -sum(p_i * log2(p_i)), where p_i are the probabilities.

    Args:
        counts: A 2D tensor representing the counts of matches in each patch.

    Returns:
        A scalar tensor representing the entropy of the distribution.

    """
    # Convert counts to probabilities, ensuring float division
    total_matches = torch.sum(counts).float()
    probabilities = counts / total_matches

    # Calculate entropy, handling 0 * log(0) explicitly as 0
    entropy = -torch.sum(
        torch.where(
            probabilities > 0,
            probabilities * torch.log2(probabilities),
            torch.tensor(0.0, dtype=torch.float),
        )
    )

    return entropy.item()


def predict_test_image_pair(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    deformation: torch.Tensor,
    backbone: torch.nn.Module,
    positional_encoding: torch.nn.Module,
    coarse_loftr: torch.nn.Module,
    coarse_matcher: torch.nn.Module,
    fine_preprocess: torch.nn.Module,
    fine_loftr: torch.nn.Module,
    fine_matching: torch.nn.Module,
    crop_size: int,
    patch_size: int,
    return_not_refined: bool = False,
    normalize_images: bool = True
):
    """
    Predicts the matching keypoints between two test images in a patch-based manner.

    Args:
        image_1 (torch.Tensor): The first test image.
        image_2 (torch.Tensor): The second test image.
        deformation (torch.Tensor): The deformation tensor.
        backbone (torch.nn.Module): The backbone neural network.
        positional_encoding (torch.nn.Module): The positional encoding module.
        coarse_loftr (torch.nn.Module): The coarse LoFTR module.
        coarse_matcher (torch.nn.Module): The coarse matcher module.
        fine_preprocess (torch.nn.Module): The fine preprocess module.
        fine_loftr (torch.nn.Module): The fine LoFTR module.
        fine_matching (torch.nn.Module): The fine matching module.
        return_not_refined (bool, optional): Whether to return the keypoints without refinement. Defaults to False.
        normalize_images (bool, optional): Whether to normalize the keypoints. Defaults to True.

    Returns:
        torch.Tensor: The keypoints in the first image.
        torch.Tensor: The keypoints in the second image.
        torch.Tensor (optional): The keypoints in the second image without refinement.
    """
    padding = 50

    matches_image_1 = []
    matches_image_2 = []
    matches_image_2_not_refined = []

    if normalize_images:
        normalize = v2.Normalize(mean=[0.594], std=[0.204])

    with torch.no_grad():
        for y in torch.arange(486 + padding, 8000 - crop_size - padding, crop_size):
            for x in torch.arange(496 + padding, 3463 - crop_size - padding, crop_size):
                crop_1 = crop_image(image_1, (x, y), crop_size)

                crop_2_position = deformation[y, x]
                crop_2 = crop_image(image_2, crop_2_position, crop_size)
                if crop_1.shape[-1] != crop_size or crop_1.shape[-2] != crop_size or crop_2.shape[-1] != crop_size or crop_2.shape[-2] != crop_size:
                    continue

                crop_coordinate_mapping = create_crop_coordinate_mapping(
                    deformation,
                    crop_position_image_1=(x, y),
                    crop_position_image_2=crop_2_position,
                    crop_size=crop_size,
                )
                crop_coordinate_mapping = crop_coordinate_mapping.cuda().unsqueeze(0)
                crop_1 = crop_1.cuda().unsqueeze(0)
                crop_2 = crop_2.cuda().unsqueeze(0)

                if normalize_images:
                    crop_1 = normalize(crop_1)
                    crop_2 = normalize(crop_2)

                coarse_image_feature_1, fine_image_feature_1 = backbone(crop_1)
                coarse_image_feature_2, fine_image_feature_2 = backbone(crop_2)

                fine_height_width = fine_image_feature_1.shape[-1]
                coarse_height_width = coarse_image_feature_1.shape[-1]

                coarse_image_feature_1 = positional_encoding(coarse_image_feature_1)
                coarse_image_feature_2 = positional_encoding(coarse_image_feature_2)

                coarse_image_feature_1 = rearrange(
                    coarse_image_feature_1, "n c h w -> n (h w) c"
                )
                coarse_image_feature_2 = rearrange(
                    coarse_image_feature_2, "n c h w -> n (h w) c"
                )

                coarse_image_feature_1, coarse_image_feature_2 = coarse_loftr(
                    coarse_image_feature_1, coarse_image_feature_2
                )

                coarse_matches_predicted = coarse_matcher(
                    coarse_image_feature_1, coarse_image_feature_2
                )
                match_matrix_predicted = coarse_matches_predicted["match_matrix"]

                (
                    fine_image_feature_1_unfold,
                    fine_image_feature_2_unfold,
                ) = fine_preprocess(
                    coarse_image_feature_1=coarse_image_feature_1,
                    coarse_image_feature_2=coarse_image_feature_2,
                    fine_image_feature_1=fine_image_feature_1,
                    fine_image_feature_2=fine_image_feature_2,
                    coarse_matches=coarse_matches_predicted,
                    fine_height_width=fine_height_width,
                    coarse_height_width=coarse_height_width
                )

                fine_image_feature_1_unfold = fine_image_feature_1_unfold.to("cuda")
                fine_image_feature_2_unfold = fine_image_feature_2_unfold.to("cuda")

                fine_image_feature_1_unfold, fine_image_feature_2_unfold = fine_loftr(
                    fine_image_feature_1_unfold, fine_image_feature_2_unfold
                )

                predicted_relative_coordinates = fine_matching(
                    fine_image_feature_1_unfold, fine_image_feature_2_unfold
                )

                match_matrix_predicted = match_matrix_predicted.cpu()
                predicted_relative_coordinates = predicted_relative_coordinates.cpu()

                (
                    crop_1_patch_mid_coordinates,
                    crop_2_patch_mid_coordinates,
                    crop_2_patch_mid_coordinates_refined,
                ) = translate_patch_midpoints_and_refine(
                    match_matrix=match_matrix_predicted,
                    patch_size=patch_size,
                    relative_coordinates=predicted_relative_coordinates,
                    fine_feature_size=fine_image_feature_1.shape[-1]  # 160 or 320. Removed hardcoding lately
                )
                crop_1_patch_mid_coordinates += torch.Tensor([x, y]).long()
                crop_2_patch_mid_coordinates += crop_2_position
                crop_2_patch_mid_coordinates_refined += crop_2_position

                matches_image_1.append(crop_1_patch_mid_coordinates)
                matches_image_2.append(crop_2_patch_mid_coordinates_refined)
                matches_image_2_not_refined.append(crop_2_patch_mid_coordinates)

    matches_image_1 = torch.concatenate(matches_image_1)
    matches_image_2 = torch.concatenate(matches_image_2)
    matches_image_2_not_refined = torch.concatenate(matches_image_2_not_refined)

    if return_not_refined:
        return matches_image_1, matches_image_2, matches_image_2_not_refined

    return matches_image_1, matches_image_2


def evaluate_test_image_pair(
    matches_image_1: torch.Tensor,
    matches_image_2: torch.Tensor,
    deformation: torch.Tensor,
) -> tuple:
    """
    Evaluate the test image pair by computing various metrics.

    Args:
        matches_image_1 (torch.Tensor): Tensor containing matches from image 1.
        matches_image_2 (torch.Tensor): Tensor containing matches from image 2.
        deformation (torch.Tensor): Tensor representing the coordinate mapping.

    Returns:
        Tuple: A tuple containing the following metrics:
            - number_of_matches (int): The total number of matches.
            - average_distance (float): The average distance between predicted matches and ground truth.
            - match_precision (dict): A dictionary of precision values at different pixel thresholds.
            - auc (float): The area under the precision curve.
            - matches_per_patch (torch.Tensor): Tensor containing the number of matches in each patch.
            - entropy (float): The entropy of the distribution of matches per patch.
    """
    matches = torch.column_stack((matches_image_1, matches_image_2))
    number_of_matches = matches.shape[0]

    # Compute the average distance between predicted matches and the ground truth
    average_distance = (
        compute_euclidean_distances(
            predicted_matches=matches.float(), coordinate_mapping=deformation
        )
        .mean()
        .item()
    )

    # Compute precision at different pixel thresholds
    match_precision = {}
    for pixel_threshold in torch.arange(0, 10, 0.01):
        match_precision[pixel_threshold.item()] = (
            (
                compute_euclidean_distances(
                    predicted_matches=matches.float(), coordinate_mapping=deformation
                )
                <= pixel_threshold
            )
            .float()
            .mean()
            .item()
        )

    # Compute the area under the precision curve
    auc = np.trapz(list(match_precision.values()), dx=0.01)

    # Count the number of matches in each patch
    y_borders = torch.arange(486, 8000, 64)
    x_borders = torch.arange(496, 3463, 64)

    matches_per_patch = count_matches_per_patch(matches, x_borders, y_borders)

    # Calculate the entropy of the distribution
    entropy = calculate_entropy(matches_per_patch)

    return (
        number_of_matches,
        average_distance,
        match_precision,
        auc,
        matches_per_patch,
        entropy,
    )


def read_deformation() -> torch.Tensor:
    # Read deformation
    deformation_path = (
        r"C:\Users\robin\Desktop\temp\temp\0524-0525_deformation_low_scale.h5"
    )
    deformation_file = h5py.File(deformation_path, "r")
    deformation = cv2.resize(
        np.array(deformation_file["deformation"]) // 10, (3463, 8000)
    )
    deformation = torch.Tensor(deformation).long()
    deformation = torch.flip(deformation, dims=[-1])

    return deformation


def evaluate_model(
    model_names: List[str],
    confidence_thresholds: List[float],
    block_dimensions: List[list],
    temperatures: List[float],
    patch_size: int,
    crop_size: int
) -> Dict:
    """
    Evaluates multiple models using the given parameters.

    Args:
        model_names (List[str]): A list of model names.
        confidence_thresholds (List[float]): A list of confidence thresholds.
        block_dimensions (List[list]): A list of block dimensions.
        temperatures (List[float]): A list of temperatures.

    Returns:
        Dict[str, dict]: A dictionary containing evaluation metrics for each model.
            The keys are the model names and the values are dictionaries containing
            the evaluation metrics.

    Raises:
        FileNotFoundError: If the model files or evaluation metrics directory is not found.
    """
    # Read test images
    image_1 = read_image(
        r"C:\Users\robin\Desktop\temp\temp\B20_0524_Slice15.tif", size=(3463, 8000)
    )
    image_2 = read_image(
        r"C:\Users\robin\Desktop\temp\temp\B20_0525_Slice15.tif", size=(3668, 7382)
    )
    image_1, image_2 = ToTensor()(image_1), ToTensor()(image_2)

    # Read deformation
    deformation = read_deformation()

    evaluation_metrics_per_model = {}

    for model_name, confidence_threshold, block_dimension, temperature in zip(
        model_names, confidence_thresholds, block_dimensions, temperatures
    ):
        
        if len(block_dimension) == 3:
            fine_feature_size = block_dimension[0]
            backbone = ResNetFPN_8_2(block_dimensions=block_dimension).cuda()

        elif len(block_dimension) == 4:
            fine_feature_size = block_dimension[1]
            backbone = ResNetFPN_16_4(block_dimensions=block_dimension).cuda()
      
        coarse_feature_size = block_dimension[-1]
        backbone.load_state_dict(torch.load(f"../../models/{model_name}/backbone.pt"))

        positional_encoding = PositionalEncoding(coarse_feature_size).cuda()

        coarse_loftr = LocalFeatureTransformer(
            feature_dimension=coarse_feature_size,
            number_of_heads=8,
            layer_names=["self", "cross"] * 4,
        ).cuda()
        coarse_loftr.load_state_dict(
            torch.load(f"../../models/{model_name}/coarse_loftr.pt")
        )

        coarse_matcher = CoarseMatching(
            temperature=temperature, confidence_threshold=confidence_threshold
        ).cuda()

        fine_preprocess = FinePreprocess(
            coarse_feature_size=coarse_feature_size,
            fine_feature_size=fine_feature_size,
            window_size=5,
            use_coarse_context=False,
        ).cuda()

        fine_loftr = LocalFeatureTransformer(
            feature_dimension=fine_feature_size,
            number_of_heads=8,
            layer_names=["self", "cross"],
        ).cuda()
        fine_loftr.load_state_dict(
            torch.load(f"../../models/{model_name}/fine_loftr.pt")
        )

        fine_matching = FineMatching(clamp_predictions=False).cuda()

        matches_image_1, matches_image_2 = predict_test_image_pair(
            image_1=image_1,
            image_2=image_2,
            deformation=deformation,
            backbone=backbone,
            positional_encoding=positional_encoding,
            coarse_loftr=coarse_loftr,
            coarse_matcher=coarse_matcher,
            fine_preprocess=fine_preprocess,
            fine_loftr=fine_loftr,
            fine_matching=fine_matching,
            crop_size=crop_size,
            patch_size=patch_size
        )

        (
            number_of_matches,
            average_distance,
            match_precision,
            auc,
            matches_per_patch,
            entropy,
        ) = evaluate_test_image_pair(matches_image_1, matches_image_2, deformation)

        evaluation_metrics = {
            "confidence_threshold": confidence_threshold,
            "number_of_matches": number_of_matches,
            "average_distance": average_distance,
            "auc": auc,
            "entropy": entropy,
            "matches_per_patch": matches_per_patch.tolist(),
            "match_precision": match_precision,
        }
        evaluation_metrics_per_model[model_name] = evaluation_metrics

        base_path = "../../models"
        final_dir = os.path.join(base_path, f"{model_name}")
        with open(os.path.join(final_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(evaluation_metrics, f)

    return evaluation_metrics_per_model
