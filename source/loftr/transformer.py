import copy
from typing import List, Tuple
import torch
import torch.nn as nn
from source.loftr.attention import LinearAttention


class LoFTREncoderLayer(nn.Module):

    """
    Implements an encoder layer for the LoFTR (Local Feature Transformer) model.
    This layer consists of a multi-head linear attention mechanism followed by a
    feed-forward network. The layer processes a query tensor and a context tensor,
    applying attention and then merging the result with the output of a feed-forward
    network.

    The layer uses a multi-head attention mechanism where the number of heads and the
    dimension of each head are configurable. The attention computation is based on the
    Linear Attention implementation.

    Attributes:
        head_dimension (int): The dimensionality of each attention head.
        number_of_heads (int): The number of attention heads.
        q_projection, k_projection, v_projection (nn.Linear): Linear layers for
                                                              projecting query, key,
                                                              and value tensors.
        attention (LinearAttention): The linear attention layer.
        merge (nn.Linear): A linear layer to merge attention outputs.
        mlp (nn.Sequential): A feed-forward network.
        norm1, norm2 (nn.LayerNorm): Layer normalization modules.
    """

    def __init__(self, feature_dimension: int, number_of_heads: int) -> None:
        """
        Initializes the LoFTREncoderLayer with specified feature dimensions and number of heads.

        Args:
            feature_dimension (int): The size of the feature dimension. This value is used to
                                     determine the input and output dimensions of the linear
                                     projections and the normalization layers.
            number_of_heads (int): The number of heads to use in the multi-head attention mechanism.

            Note that the feature_dimension must be divisible by the number_of_heads.
            It ensures that each attention head processes a feature vector of equal size
        """
        super(LoFTREncoderLayer, self).__init__()

        assert (
            feature_dimension % number_of_heads
        ) == 0, "feature_dimension must be divisible by number_of_heads"

        # dimensionality of each head
        self.head_dimension = feature_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # multi-head attention
        # Linear layer for Q, K and V
        self.q_projection = nn.Linear(feature_dimension, feature_dimension, bias=False)
        self.k_projection = nn.Linear(feature_dimension, feature_dimension, bias=False)
        self.v_projection = nn.Linear(feature_dimension, feature_dimension, bias=False)

        # Linear attention layer
        self.attention = LinearAttention()

        # Merge Linear layer after attention
        self.merge = nn.Linear(feature_dimension, feature_dimension, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(feature_dimension * 2, feature_dimension * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(feature_dimension * 2, feature_dimension, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(feature_dimension)
        self.norm2 = nn.LayerNorm(feature_dimension)

    def forward(self, query_tensor: torch.Tensor, context_tensor: torch.Tensor):
        """
        Forward pass of the LoFTREncoderLayer. Processes the input query and context tensors
        through the attention mechanism and feed-forward network.

        The method first applies linear projections to the query (derived from query tensor),
        key and value (derived from context tensor) tensors, then computes attention, and finally
        processes the result through a feed-forward network. The output is a tensor that combines
        the information from the query tensor and the context tensor using the attention mechanism.

        Args:
            query_tensor (torch.Tensor): The query tensor with shape [Batchsize, L, Feature vector size],
                                         where L is the number of features per image.
            context_tensor (torch.Tensor): The context tensor with shape [Batchsize, S, Feature vector size],
                                           where S is the number of features per image.

        Returns:
            torch.Tensor: The output tensor after processing through the attention and feed-forward layers.
                           The shape of the output tensor is [Batchsize, L, Feature vector size].
        """

        batchsize = query_tensor.size(0)
        query, key, value = query_tensor, context_tensor, context_tensor

        # multi-head attention
        # The following view operations split the query, key and value tensors into multiple heads.
        # eg an 192 dimensional feature vector could be split into 8 heads of 24 dimensions each.
        query = self.q_projection(query)
        query = query.view(
            batchsize, -1, self.number_of_heads, self.head_dimension
        )  # [Batchsize, L, Number of heads, Head dimension])]

        key = self.k_projection(key)
        key = key.view(
            batchsize, -1, self.number_of_heads, self.head_dimension
        )  # [Batchsize, S, Number of heads, Head dimension]

        value = self.v_projection(value)
        value = value.view(
            batchsize, -1, self.number_of_heads, self.head_dimension
        )  # [Batchsize, S, Number of heads, Head dimension]

        message = self.attention(
            query, key, value
        )  # [Batchsize, L, Number of heads, Head dimension]
        message = self.merge(
            message.view(batchsize, -1, self.number_of_heads * self.head_dimension)
        )  # [Batchsize, L, Number of heads * Head dimension]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([query_tensor, message], dim=2))
        message = self.norm2(message)

        return query_tensor + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(
        self,
        feature_dimension: int,
        number_of_heads: int,
        layer_names: List[str],
    ) -> None:
        """
        Initializes the LocalFeatureTransformer module, a key component in the LoFTR architecture.

        This module is designed to process pairs of images and extract or transform their local features using
        the LoFTREncoderLayer. It supports both self-attention and cross-attention mechanisms which are crucial
        for understanding the geometric and contextual relationships in image pairs.

        The transformer consists of multiple layers of LoFTREncoderLayer, each capable of either self-attention
        or cross-attention as dictated by the `layer_names` argument.

        Args:
            feature_dimension (int): The size of the feature dimension. It defines the number of features in
                                     each feature vector that will be processed by the transformer., eg 192.
            number_of_heads (int): The number of attention heads in each LoFTREncoderLayer. This number should
                                   divide the `feature_dimension` evenly for proper functioning of the attention mechanism.
            layer_names (list): A list of strings indicating the type of each encoder layer in the transformer.
                                Valid strings are "self" for self-attention and "cross" for cross-attention layers.
                                The order of strings in the list dictates the order of the layers in the transformer.

        The module initializes a sequence of LoFTREncoderLayer instances, each corresponding to an entry in `layer_names`.
        It also sets up the necessary parameters and configurations for the transformer layers.
        """

        super(LocalFeatureTransformer, self).__init__()

        self.feature_dimension = feature_dimension
        self.layer_names = layer_names
        encoder_layer = LoFTREncoderLayer(
            feature_dimension=feature_dimension,
            number_of_heads=number_of_heads,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, feature_image_1, feature_image_2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LocalFeatureTransformer.
        L, S = Number of features per image, eg 1600 (40x40) for 1/16th scale.
        Args:
            feature_image_1 (torch.Tensor): Features of image 1 [Batchsize, L, Feature vector size]
            feature_image_2 (torch.Tensor): Features of image 2 [Batchsize, S, Feature vector size]
        Returns:
            feature_image_1 (torch.Tensor): Features of image 1 [Batchsize, L, Feature vector size]
            feature_image_2 (torch.Tensor): Features of image 2 [Batchsize, S, Feature vector size]
        """

        assert self.feature_dimension == feature_image_1.size(
            2
        ), "the feature number of source and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feature_image_1 = layer(feature_image_1, feature_image_1)
                feature_image_2 = layer(feature_image_2, feature_image_2)

            elif name == "cross":
                feature_image_1 = layer(feature_image_1, feature_image_2)
                feature_image_2 = layer(feature_image_2, feature_image_1)

            else:
                raise KeyError

        return feature_image_1, feature_image_2
