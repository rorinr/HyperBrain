import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(
        self,
        coarse_feature_size: int,
        fine_feature_size: int,
        window_size: int,
        use_coarse_context: bool,
    ):
        super().__init__()
        self.coarse_feature_size = coarse_feature_size
        self.fine_feature_size = fine_feature_size
        self.window_size = window_size
        self.use_coarse_context = use_coarse_context

        if self.use_coarse_context:
            self.down_projection = nn.Linear(
                coarse_feature_size, fine_feature_size, bias=True
            )
            self.merge_features = nn.Linear(
                2 * fine_feature_size, fine_feature_size, bias=True
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        coarse_image_feature_1: torch.Tensor,
        coarse_image_feature_2: torch.Tensor,
        fine_image_feature_1: torch.Tensor,
        fine_image_feature_2: torch.Tensor,
        coarse_matches: dict,
        fine_height_width: int,
        coarse_height_width: int,
    ) -> torch.Tensor:
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
            fine_height_width (int): Height of the fine feature map, assuming it is square.
            coarse_height_width (int): Height of the coarse feature map, assuming it is square.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors, each representing the unfolded fine features
                                            of the first and second images, potentially enriched with coarse feature context.
        """

        stride = fine_height_width // coarse_height_width  # usually = 4

        # If no matches found
        if coarse_matches["batch_indices"].shape[0] == 0:
            feature_0 = torch.empty(0, self.window_size**2, self.fine_feature_size, device=coarse_image_feature_1.device)
            feature_1 = torch.empty(0, self.window_size**2, self.fine_feature_size, device=coarse_image_feature_1.device)
            return feature_0, feature_1

        # 1. unfold(crop) all local windows -> mid pixels of patches are here in the middle of the window
        # fine_image_feature_1 and 2 are of shape B, fine_feature_size, fine_height, fine_width where usually height=width
        fine_image_feature_1_unfold = F.unfold(
            fine_image_feature_1,
            kernel_size=(self.window_size, self.window_size),
            stride=stride,
            padding=self.window_size // 2,
        )  # B, window_size**2*fine_feature_size, number_of_patches
        fine_image_feature_1_unfold = rearrange(
            fine_image_feature_1_unfold,
            "n (c ww) l -> n l ww c",
            ww=self.window_size**2,
        )  # B, number_of_patches, window_size**2, fine_feature_size

        fine_image_feature_2_unfold = F.unfold(
            fine_image_feature_2,
            kernel_size=(self.window_size, self.window_size),
            stride=stride,
            padding=self.window_size // 2,
        )  # B, window_size**2*fine_feature_size, number_of_patches
        fine_image_feature_2_unfold = rearrange(
            fine_image_feature_2_unfold,
            "n (c ww) l -> n l ww c",
            ww=self.window_size**2,
        )  # B, number_of_patches, window_size**2, fine_feature_size

        # 2. select only the predicted matches
        fine_image_feature_1_unfold = fine_image_feature_1_unfold[
            coarse_matches["batch_indices"], coarse_matches["row_indices"]
        ]  # number_of_matches, window_size**2, fine_feature_size
        fine_image_feature_2_unfold = fine_image_feature_2_unfold[
            coarse_matches["batch_indices"], coarse_matches["column_indices"]
        ]  # number_of_matches, window_size**2, fine_feature_size

        if self.use_coarse_context:
            # Combine both coarse feature embedding
            coarse_features_matched = self.down_projection(
                torch.cat(
                    [
                        coarse_image_feature_1[
                            coarse_matches["batch_indices"],
                            coarse_matches["row_indices"],
                        ],
                        coarse_image_feature_2[
                            coarse_matches["batch_indices"],
                            coarse_matches["column_indices"],
                        ],
                    ],
                    0,
                )  # 2number_of_matches, coarse_feature_size
            )  # 2number_of_matches, fine_feature_size

            # Merge coarse and fine features
            coarse_fine_features_matched_combined = self.merge_features(
                torch.cat(
                    [
                        torch.cat(
                            [fine_image_feature_1_unfold, fine_image_feature_2_unfold],
                            0,
                        ),  # 2number_of_matches, ww, fine_feature_size
                        repeat(
                            coarse_features_matched,
                            "n c -> n ww c",
                            ww=self.window_size**2,
                        ),  # 2number_of_matches, ww, fine_feature_size
                    ],
                    -1,
                )  # 2number_of_matches, ww, 2fine_feature_size
            )  # 2number_of_matches, ww, fine_feature_size

            fine_image_feature_1_unfold, fine_image_feature_2_unfold = torch.chunk(
                coarse_fine_features_matched_combined, 2, dim=0
            )  # each number_of_matches, ww, fine_feature_size

        return fine_image_feature_1_unfold, fine_image_feature_2_unfold