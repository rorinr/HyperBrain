import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

class FinePreprocess(nn.Module):
    def __init__(
        self,
        d_model_coarse: int,
        d_model_fine: int,
        window_size: int,
        use_coarse_context: bool,
    ):
        super().__init__()
        self.d_model_coarse = d_model_coarse
        self.d_model_fine = d_model_fine
        self.window_size = window_size
        self.use_coarse_context = use_coarse_context

        if self.use_coarse_context:
            self.down_projection = nn.Linear(
                d_model_coarse, d_model_fine, bias=True
            )
            self.merge_features = nn.Linear(
                2 * d_model_fine, d_model_fine, bias=True
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, coarse_image_feature_1: torch.Tensor, coarse_image_feature_2: torch.Tensor, fine_image_feature_1: torch.Tensor, fine_image_feature_2: torch.Tensor, coarse_matches: dict, fine_height: int, coarse_height: int) -> torch.Tensor:
        """
        Forward pass of the FineMatching module that processes fine-grained image features.

        This method performs the following operations:
        1. Unfolds (crops) all local windows in the fine feature space.
        2. Selects only the regions of interest based on the coarse matches.
        3. Optionally merges coarse and fine features for enriched feature representation.

        Args:
            coarse_image_feature_1 (torch.Tensor): Coarse feature map of the first image.
            coarse_image_feature_2 (torch.Tensor): Coarse feature map of the second image.
            fine_image_feature_1 (torch.Tensor): Fine feature map of the first image.
            fine_image_feature_2 (torch.Tensor): Fine feature map of the second image.
            coarse_matches (dict): Dictionary containing indices of matched features in the coarse feature maps.
            fine_height (int): Height of the fine feature map, assuming it is square.
            coarse_height (int): Height of the coarse feature map, assuming it is square.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors, each representing the unfolded fine features 
                                            of the first and second images, potentially enriched with coarse feature context.
        """
        
        stride = fine_height // coarse_height

        # If no matches found
        if coarse_matches["batch_indices"].shape[0] == 0:
            feature_0 = torch.empty(0, self.W**2, self.d_model_f)
            feature_1 = torch.empty(0, self.W**2, self.d_model_f)
            return feature_0, feature_1

        # 1. unfold(crop) all local windows -> mid pixels of patches are here in the middle of the window
        # fine_image_feature_1 and 2 are of shape B, d_model_fine, fine_height, fine_width where usually height=width
        fine_image_feature_1_unfold = F.unfold(
            fine_image_feature_1,
            kernel_size=(self.window_size, self.window_size),
            stride=stride,
            padding=self.window_size // 2,
        )  # B, window_size**2*d_model_fine, number_of_patches
        fine_image_feature_1_unfold = rearrange(fine_image_feature_1_unfold, "n (c ww) l -> n l ww c", ww=self.window_size**2
        )  # B, number_of_patches, window_size**2, d_model_fine

        fine_image_feature_2_unfold = F.unfold(
            fine_image_feature_2,
            kernel_size=(self.window_size, self.window_size),
            stride=stride,
            padding=self.window_size // 2,
        )  # B, window_size**2*d_model_fine, number_of_patches
        fine_image_feature_2_unfold = rearrange(fine_image_feature_2_unfold, "n (c ww) l -> n l ww c", ww=self.window_size**2
        )  # B, number_of_patches, window_size**2, d_model_fine

        # 2. select only the predicted matches
        fine_image_feature_1_unfold = fine_image_feature_1_unfold[coarse_matches["batch_indices"], coarse_matches["row_indices"]]  # number_of_matches, window_size**2, d_model_fine
        fine_image_feature_2_unfold = fine_image_feature_2_unfold[coarse_matches["batch_indices"], coarse_matches["column_indices"]]  # number_of_matches, window_size**2, d_model_fine

        if self.use_coarse_context:
            # Combine both coarse feature embedding
            coarse_features_matched = self.down_projection(
                torch.cat(
                    [
                        coarse_image_feature_1[coarse_matches["batch_indices"], coarse_matches["row_indices"]],
                        coarse_image_feature_2[coarse_matches["batch_indices"], coarse_matches["column_indices"]],
                    ],
                    0,
                )  # 2number_of_matches, d_model_coarse
            )  # 2number_of_matches, d_model_fine

            # Merge coarse and fine features
            coarse_fine_features_matched_combined = self.merge_features(
                torch.cat(
                    [
                        torch.cat(
                            [fine_image_feature_1_unfold, fine_image_feature_2_unfold], 0
                        ),  # 2number_of_matches, ww, d_model_fine
                        repeat(
                            coarse_features_matched,
                            "n c -> n ww c",
                            ww=self.window_size**2,
                        ),  # 2number_of_matches, ww, d_model_fine
                    ],
                    -1,
                )  # 2number_of_matches, ww, 2d_model_fine
            )  # 2number_of_matches, ww, d_model_fine

            fine_image_feature_1_unfold, fine_image_feature_2_unfold = torch.chunk(
                coarse_fine_features_matched_combined, 2, dim=0
            )  # each number_of_matches, ww, d_model_fine

        return fine_image_feature_1_unfold, fine_image_feature_2_unfold