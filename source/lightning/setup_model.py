from source.loftr.backbone import ResNetFPN_16_4, ResNetFPN_8_2
from source.loftr.positional_encoding import PositionalEncoding
from source.loftr.transformer import LocalFeatureTransformer
from source.loftr.coarse_matching import CoarseMatching
from source.loftr.fine_matching import FineMatching
from source.loftr.fine_preprocess import FinePreprocess
from source.loftr.loss import *
from typing import List
from einops.einops import rearrange
import lightning as L

class LitLoFTR(L.LightningModule):
    def __init__(self, backbone, positional_encoding, coarse_loftr, coarse_matcher, fine_preprocess, fine_loftr, fine_matching, coarse_loss, fine_loss, alpha = None, gamma = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.coarse_loftr = coarse_loftr
        self.coarse_matcher = coarse_matcher
        self.fine_preprocess = fine_preprocess
        self.fine_loftr = fine_loftr
        self.fine_matching = fine_matching
        self.coarse_loss = coarse_loss
        self.fine_loss = fine_loss
        self.alpha = alpha
        self.gamma = gamma
    
    def training_step(self, batch, batch_idx):
        image_1_crop, image_2_crop, match_matrix, relative_coordinates, coordinate_mapping = batch

        coarse_image_feature_1, fine_image_feature_1 = self.backbone(image_1_crop)
        coarse_image_feature_2, fine_image_feature_2 = self.backbone(image_2_crop)
        coarse_height_width = coarse_image_feature_1.shape[-1]
        fine_height_width = fine_image_feature_1.shape[-1]

        coarse_image_feature_1 = self.positional_encoding(coarse_image_feature_1)
        coarse_image_feature_2 = self.positional_encoding(coarse_image_feature_2)

        coarse_image_feature_1 = rearrange(coarse_image_feature_1, "n c h w -> n (h w) c")
        coarse_image_feature_2 = rearrange(coarse_image_feature_2, "n c h w -> n (h w) c")
        
        coarse_image_feature_1, coarse_image_feature_2 = self.coarse_loftr(coarse_image_feature_1, coarse_image_feature_2)

        coarse_matches = self.coarse_matcher(coarse_image_feature_1, coarse_image_feature_2)

        coarse_matches_ground_truth = {
            "batch_indices": match_matrix.nonzero()[:, 0],
            "row_indices": match_matrix.nonzero()[:, 1],
            "column_indices": match_matrix.nonzero()[:, 2],
        }

        fine_image_feature_1_unfold, fine_image_feature_2_unfold = self.fine_preprocess(
            coarse_image_feature_1=coarse_image_feature_1,
            coarse_image_feature_2=coarse_image_feature_2,
            fine_image_feature_1=fine_image_feature_1,
            fine_image_feature_2=fine_image_feature_2,
            coarse_matches=coarse_matches_ground_truth,
            fine_height_width=fine_height_width,
            coarse_height_width=coarse_height_width,
        )

        fine_image_feature_1_unfold, fine_image_feature_2_unfold = self.fine_loftr(
            fine_image_feature_1_unfold, fine_image_feature_2_unfold
        )

        predicted_relative_coordinates = self.fine_matching(
            fine_image_feature_1_unfold, fine_image_feature_2_unfold
        )

        if self.coarse_loss == "focal":
            coarse_loss_value = coarse_focal_loss(
                predicted_confidence=coarse_matches["confidence_matrix"],
                ground_truth_confidence=match_matrix,
                alpha=self.alpha,
                gamma=self.gamma,
            )

        elif self.coarse_loss == "official_focal":
            coarse_loss_value = coarse_official_focal_loss(
                predicted_confidence=coarse_matches["confidence_matrix"],
                ground_truth_confidence=match_matrix,
                alpha=self.alpha,
                gamma=self.gamma,
            )

        elif self.coarse_loss == "cross_entropy":
            coarse_loss_value = coarse_cross_entropy_loss(
                predicted_confidence=coarse_matches["confidence_matrix"],
                ground_truth_confidence=match_matrix,
            )

        if self.fine_loss == "l2":
            fine_loss_value = fine_l2_loss(
                coordinates_predicted=predicted_relative_coordinates,
                coordinates_ground_truth=relative_coordinates,
            )

        elif self.fine_loss == "l2_std":
            fine_loss_value = fine_l2_loss_with_standard_deviation(
                coordinates_predicted=predicted_relative_coordinates,
                coordinates_ground_truth=relative_coordinates,
            )

        loss = coarse_loss_value + fine_loss_value
        
        metrics = {"coarse_loss": coarse_loss_value, "fine_loss": fine_loss_value}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0007, weight_decay=0.0001)
        return optimizer

def setup_lightning_loftr(block_dimensions: List[int], use_coarse_context: bool, clamp_predictions:bool, temperature: float, coarse_loss:str, fine_loss: str, alpha = None, gamma = None):
    if len (block_dimensions) == 3:
        backbone = ResNetFPN_8_2(block_dimensions=block_dimensions)
    elif len (block_dimensions) == 4:
        backbone = ResNetFPN_16_4(block_dimensions=block_dimensions)

    if backbone._get_name() == "ResNetFPN_8_2":
        fine_feature_size = block_dimensions[0]
    elif backbone._get_name() == "ResNetFPN_16_4":
        fine_feature_size = block_dimensions[1]

    coarse_feature_size = block_dimensions[-1]

    positional_encoding = PositionalEncoding(coarse_feature_size)

    coarse_loftr = LocalFeatureTransformer(
        feature_dimension=coarse_feature_size,
        number_of_heads=8,
        layer_names=["self", "cross"] * 4,
    )

    coarse_matcher = CoarseMatching(temperature=temperature, confidence_threshold=0.2)

    fine_preprocess = FinePreprocess(
        coarse_feature_size=coarse_feature_size,
        fine_feature_size=fine_feature_size,
        window_size=5,
        use_coarse_context=use_coarse_context,
    )

    fine_loftr = LocalFeatureTransformer(
        feature_dimension=fine_feature_size,
        number_of_heads=8,
        layer_names=["self", "cross"],
    )

    use_l2_with_standard_deviation = True if fine_loss == "l2_std" else False
    fine_matching = FineMatching(
        return_standard_deviation=use_l2_with_standard_deviation,
        clamp_predictions=clamp_predictions,
    )

    model = LitLoFTR(
        backbone=backbone,
        positional_encoding=positional_encoding,
        coarse_loftr=coarse_loftr,
        coarse_matcher=coarse_matcher,
        fine_preprocess=fine_preprocess,
        fine_loftr=fine_loftr,
        fine_matching=fine_matching,
        coarse_loss=coarse_loss,
        fine_loss=fine_loss,
        alpha=alpha,
        gamma=gamma,
    )

    return model
