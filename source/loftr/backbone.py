"""
This module defines two neural network architectures based on the ResNet model with Feature Pyramid Networks (FPN), 
namely `ResNetFPN_8_2` and `ResNetFPN_16_4`, and a basic building block for these networks named `BasicBlock`. 

The `BasicBlock` class serves as a foundational component for building deeper network architectures. It consists of 
two convolutional layers, each followed by batch normalization and ReLU activation, and an optional downsampling layer 
to adjust the dimensions when the stride differs from 1. This block forms the building block of the ResNet-based models.

The `ResNetFPN_8_2` class implements a ResNet-FPN architecture where the output resolutions are 1/8 and 1/2 of the 
input resolution. This architecture is designed for tasks requiring feature maps at different scales, providing both 
coarse and fine feature representations. The network includes an initial convolutional layer, followed by a series of 
downsampling layers to reduce spatial resolution and increase channel dimensions, and then upsampling layers to 
combine features at different scales.

Similarly, the `ResNetFPN_16_4` class is another variant of the ResNet-FPN architecture, with output resolutions of 
1/16 and 1/4 of the input resolution. This architecture follows a similar pattern but extends the downsampling further 
to provide an even coarser feature map compared to `ResNetFPN_8_2`.

Both `ResNetFPN_8_2` and `ResNetFPN_16_4` classes include detailed documentation of the forward pass and tensor 
dimension relationships, providing insights into the internal workings of these architectures. Additionally, they 
contain methods to log tensor dimensions at different stages of the network, aiding in understanding and debugging.

This module is designed to be used in image matching tasks, particularly for the LoFTR model.

Dependencies:
    - typing
    - torch
    - torch.nn
    - torch.nn.functional
"""

from typing import List, Dict
from torch import nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    """
    A BasicBlock class for a neural network, inheriting from nn.Module.

    This class represents a basic building block for convolutional neural networks,
    particularly used in architectures like ResNets. It consists of two convolutional layers,
    each followed by batch normalization and ReLU activation. An optional downsampling layer is
    included to match the dimensions when the stride is not equal to 1.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential or None): Downsampling layer if stride is not 1, otherwise None.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolutional layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        This method processes the input tensor through a sequence of operations involving two main convolutional 
        layers. Each convolutional layer is followed by batch normalization and a ReLU activation function. 
        Additionally, if the stride is set to a value other than 1, a downsampling step is applied to the input 
        tensor. This downsampling is necessary to match the spatial dimensions for a subsequent element-wise 
        addition operation with the output of the second convolutional layer.

        Parameters:
        x (Tensor): The input tensor to the BasicBlock. The expected shape of the tensor is (N, in_channels, H, W),
                    where N is the batch size, in_channels is the number of input channels, and H and W are the 
                    height and width of the input.

        Returns:
        Tensor: The output tensor of the BasicBlock after processing. The output retains the same batch size (N) 
                and has potentially altered channel dimensions and spatial dimensions depending on the stride and 
                downsampling operations.
        """
        y = x

        # First layer
        y = self.relu(self.bn1(self.conv1(y)))

        # Second layer
        y = self.bn2(self.conv2(y))

        # Downsampling if stride is not 1 -> match dimensions for element-wise addition
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):
    """
    This class implements a ResNet with Feature Pyramid Networks (FPN) where the output resolutions
    are 1/8 and 1/2 of the input resolution. Each block in the network has 2 layers.

    The network begins with an initial convolution layer, followed by batch normalization and ReLU activation.
    This is followed by a series of downsample layers to progressively reduce the spatial resolution while
    increasing the channel dimensions. The network then includes upsample layers to increase the spatial
    resolution and combine features at different scales.

    Attributes:
        verbose: Whether to print the dimensions of the tensors at different stages of the network.
        initial_dimension (int): Number of channels in the output of the initial convolution layer.
        conv1 (nn.Conv2d): Initial convolution layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the initial convolution layer.
        relu (nn.ReLU): ReLU activation function.
        layer1, layer2, layer3 (nn.Sequential): Layers for downsampling.
        layer3_outconv, layer2_outconv1, layer2_outconv2, layer1_outconv1, layer1_outconv2 (nn.Conv2d or nn.Sequential):
        Layers for processing and upsampling features.

    Args:
        block_dimensions (list[int]): List of channel dimensions for the blocks.

    Returns:
        tuple: A tuple containing two tensors. The first tensor represents coarse features with a
        lower resolution (1/8 of the input), and the second tensor represents fine features with
        a higher resolution (1/2 of the input).
    """

    def __init__(self, block_dimensions: List[int], verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        initial_dimension = block_dimensions[0]

        self.conv1 = nn.Conv2d(
            1, initial_dimension, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dimension)
        self.relu = nn.ReLU(inplace=True)

        # Downsampling
        self.layer1 = self._make_layer(
            initial_dimension, block_dimensions[0], stride=1
        )  # Output = 1/2
        self.layer2 = self._make_layer(
            block_dimensions[0], block_dimensions[1], stride=2
        )  # Output = 1/4
        self.layer3 = self._make_layer(
            block_dimensions[1], block_dimensions[2], stride=2
        )  # Output = 1/8

        # Outputs coarse feature space, red in loftr paper
        self.layer3_outconv = nn.Conv2d(
            block_dimensions[2],
            block_dimensions[2],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Upsampling of layer2's output (just upsampling channels, H and W stay same)
        self.layer2_outconv1 = nn.Conv2d(
            block_dimensions[1],
            block_dimensions[2],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # Process (layer2_outconv1's output + resized layer3_outconv's output)
        self.layer2_outconv2 = nn.Sequential(
            nn.Conv2d(
                block_dimensions[2],
                block_dimensions[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(block_dimensions[2]),
            nn.LeakyReLU(),
            nn.Conv2d(
                block_dimensions[2],
                block_dimensions[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        # Upsampling of layer1's output (just upsampling channels, H and W stay same) so that we can add it with x2_out_2x
        self.layer1_outconv1 = nn.Conv2d(
            block_dimensions[0],
            block_dimensions[1],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # Process (layer1_outconv1's output + resized layer2_outconv2's output)
        self.layer1_outconv2 = nn.Sequential(
            nn.Conv2d(
                block_dimensions[1],
                block_dimensions[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(block_dimensions[1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                block_dimensions[1],
                block_dimensions[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        layer1 = BasicBlock(in_channels, out_channels, stride=stride)
        layer2 = BasicBlock(out_channels, out_channels, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def log_tensor_dimensions(self, tensor_info: Dict[str, torch.Tensor]):
        """
        Logs the dimensions of the tensors in the ResNetFPN_8_2 network.

        Args:
            tensor_info (Dict[str, torch.Tensor]): A dictionary where keys are tensor descriptions
            and values are the tensors.
        """

        for desc, tensor in tensor_info.items():
            print(f"{desc}: {tensor.shape}")

        print("\nTensor Dimension Relationships:")
        print(
            "x0: Output of the initial convolution, batch normalization, and ReLU applied to the input."
        )
        print(
            "x1: Output of layer1, representing features at 1/2 of the input resolution."
        )
        print("x2: Output of layer2, downsampled to 1/4 of the input resolution.")
        print(
            "x3: Output of layer3, further downsampled to 1/8 of the input resolution."
        )
        print("x3_out: Coarse features at 1/8 resolution, output of layer3_outconv.")
        print(
            "x3_out_2x: Upsampled version of x3_out to match the resolution of x2 (1/4 of the input)."
        )
        print(
            "x2_out: Combines upsampled coarse features (x3_out_2x) with the output of layer2, then processed for feature fusion."
        )
        print(
            "x2_out_2x: Upsampled version of x2_out to match the resolution of x1 (1/2 of the input)."
        )
        print(
            "x1_out: Final output, combining features from x1 and upsampled x2_out_2x, representing fine features at 1/2 input resolution."
        )

    def forward(self, x):
        """
        Forward pass of the ResNetFPN_8_2 model.
        Processes the input through a series of convolutional layers, batch normalization, and ReLU activations,
        along with downsampling and upsampling layers to produce feature maps at different scales.

        Args:
            x (Tensor): Input tensor of shape (n, 1, H, W)

        Returns:
            tuple[Tensor, Tensor]: A tuple of two tensors. The first tensor (coarse features) is of shape
            (n, _, H_coarse, W_coarse), and the second tensor (fine features) is of shape
            (n, _, H_fine, W_fine), where H_coarse and W_coarse are 1/8, and H_fine and W_fine are 1/2
            of the original dimensions H and W, respectively.
            Note that _ represents the different number of channels, which is defined by the block_dimensions argument.
        """
        # Half image size, upsample channels
        x0 = self.relu(self.bn1(self.conv1(x)))  # (n, _, H/2, W/2)

        # Downsampling
        x1 = self.layer1(x0)  # (n, _, H/2, W/2)

        x2 = self.layer2(x1)  # (n, _, H/4, W/4)
        x3 = self.layer3(x2)  # (n, _, H/8, W/8)

        # Coarse features
        x3_out = self.layer3_outconv(x3)  # (n, _, H/8, W/8)

        # Upsample coarse features
        x3_out_2x = F.interpolate(
            x3_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )  # (n, _, H/4, W/4)

        # Bring x2 from _ to _ channels. Then add it with x3_out_2x and downscale again
        x2_out = self.layer2_outconv1(x2)  # (n, _, H/4, W/4)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)  # (2, _, H/4, W/4)

        # Upsample from x2_out from (H/4, W/4) to (H/2, W/2)
        x2_out_2x = F.interpolate(
            x2_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )

        x1_out = self.layer1_outconv1(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)  # (n, _, H/2, W/2)

        if self.verbose:
            tensor_info = {
                "x (Input Tensor)": x,
                "x0 (Initial Convolution Output)": x0,
                "x1 (Layer 1 Output)": x1,
                "x2 (Layer 2 Output)": x2,
                "x3 (Layer 3 Output)": x3,
                "x3_out (Layer 3 OutConv Output)": x3_out,
                "x3_out_2x (Upsampled x3_out)": x3_out_2x,
                "x2_out (Layer 2 OutConv Output)": x2_out,
                "x2_out_2x (Upsampled x2_out)": x2_out,
                "x1_out (Layer 1 OutConv Output)": x1_out,
            }
            self.log_tensor_dimensions(tensor_info=tensor_info)

        return x3_out, x1_out


class ResNetFPN_16_4(nn.Module):
    """
    This class implements a ResNet with Feature Pyramid Networks (FPN) where the output resolutions
    are 1/16 and 1/4 of the input resolution. Each block in the network has 2 layers.

    The network begins with an initial convolution layer, followed by batch normalization and ReLU activation.
    This is followed by a series of downsample layers to progressively reduce the spatial resolution while
    increasing the channel dimensions. The network then includes upsample layers to increase the spatial
    resolution and combine features at different scales.

    Attributes:
        verbose: Whether to print the dimensions of the tensors at different stages of the network.
        initial_dimension (int): Number of channels in the output of the initial convolution layer.
        conv1 (nn.Conv2d): Initial convolution layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the initial convolution layer.
        relu (nn.ReLU): ReLU activation function.
        layer1, layer2, layer3 (nn.Sequential): Layers for downsampling.
        layer3_outconv, layer2_outconv1, layer2_outconv2, layer1_outconv1, layer1_outconv2 (nn.Conv2d or nn.Sequential):
        Layers for processing and upsampling features.

    Args:
        block_dimensions (list[int]): List of channel dimensions for the blocks.

    Returns:
        tuple: A tuple containing two tensors. The first tensor represents coarse features with a
        lower resolution (1/8 of the input), and the second tensor represents fine features with
        a higher resolution (1/2 of the input).
    """

    def __init__(self, block_dimensions: List[int], verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose

        initial_dimension = block_dimensions[0]
        self.conv1 = nn.Conv2d(
            1, initial_dimension, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dimension)
        self.relu = nn.ReLU(inplace=True)

        # Downsampling
        self.layer1 = self._make_layer(
            initial_dimension, block_dimensions[0], stride=1
        )  # Output = 1/2
        self.layer2 = self._make_layer(
            block_dimensions[0], block_dimensions[1], stride=2
        )  # Output = 1/4
        self.layer3 = self._make_layer(
            block_dimensions[1], block_dimensions[2], stride=2
        )  # Output = 1/8
        self.layer4 = self._make_layer(
            block_dimensions[2], block_dimensions[3], stride=2
        )  # Output = 1/16

        self.layer4_outconv = nn.Conv2d(
            block_dimensions[3],
            block_dimensions[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Outputs coarse feature space, red in loftr paper
        self.layer3_outconv1 = nn.Conv2d(
            block_dimensions[2],
            block_dimensions[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.layer3_outconv2 = nn.Sequential(
            nn.Conv2d(
                block_dimensions[3],
                block_dimensions[3],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(block_dimensions[3]),
            nn.LeakyReLU(),
            nn.Conv2d(
                block_dimensions[3],
                block_dimensions[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        # Upsampling of layer2's output (just upsampling channels, H and W stay same)
        self.layer2_outconv1 = nn.Conv2d(
            block_dimensions[1],
            block_dimensions[2],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # Process (layer2_outconv1's output + resized layer3_outconv's output)
        self.layer2_outconv2 = nn.Sequential(
            nn.Conv2d(
                block_dimensions[2],
                block_dimensions[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(block_dimensions[2]),
            nn.LeakyReLU(),
            nn.Conv2d(
                block_dimensions[2],
                block_dimensions[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        layer1 = BasicBlock(in_channels, out_channels, stride=stride)
        layer2 = BasicBlock(out_channels, out_channels, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def log_tensor_dimensions(self, tensor_info: Dict[str, torch.Tensor]):
        """
        Logs the dimensions of the tensors.

        Args:
            tensor_info (Dict[str, torch.Tensor]): A dictionary where keys are tensor descriptions
            and values are the tensors.
        """
        for desc, tensor in tensor_info.items():
            print(f"{desc}: {tensor.shape}")

        print("\nTensor Dimension Relationships:")
        print("x0 is the result of the initial convolution applied to the input.")
        print(
            "x1 to x4 are progressively downsampled features, reducing spatial dimensions while increasing channels."
        )
        print(
            "x4_out (coarse features) is the processed output of x4, which is then upsampled to x4_out_2x."
        )
        print("x3_out combines features from x3 and x4_out_2x.")
        print(
            "x3_out_2x is an upsampled version of x3_out, aligning its dimensions with x2 for feature fusion."
        )
        print(
            "x2_out is the final output, merging features from x2 and x3_out_2x, representing fine features."
        )

    def forward(self, x):
        """
        Forward pass of the ResNetFPN_16_4 model.

        Processes the input through a series of convolutional layers, batch normalization, and ReLU activations,
        along with downsampling and upsampling layers to produce feature maps at different scales.

        Args:
            x (Tensor): Input tensor of shape (n, 1, H, W)

        Returns:
            tuple[Tensor, Tensor]: A tuple of two tensors. The first tensor (coarse features) is of shape
            (n, _, H_coarse, W_coarse), and the second tensor (fine features) is of shape
            (n, _, H_fine, W_fine), where H_coarse and W_coarse are 1/16, and H_fine and W_fine are 1/4
            of the original dimensions H and W, respectively.
            Note that _ represents the different number of channels, which is defined by the block_dimensions argument.
        """
        # Half image size, upsample channels
        x0 = self.relu(self.bn1(self.conv1(x)))  # (n, _, H/2, W/2)

        # Downsampling
        x1 = self.layer1(x0)  # (n, _, H/2, W/2)

        x2 = self.layer2(x1)  # (n, _, H/4, W/4)
        x3 = self.layer3(x2)  # (n, _, H/8, W/8)
        x4 = self.layer4(x3)

        # Coarse features
        x4_out = self.layer4_outconv(x4)
        x4_out_2x = F.interpolate(
            x4_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )

        x3_out = self.layer3_outconv1(x3)  # (n, _, H/8, W/8)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        # Upsample coarse features
        x3_out_2x = F.interpolate(
            x3_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )  # (n, _, H/4, W/4)

        # Bring x2 from _ to _ channels. Then add it with x3_out_2x and downscale again
        x2_out = self.layer2_outconv1(x2)  # (n, _, H/4, W/4)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)  # (2, _, H/4, W/4)

        if self.verbose:
            tensor_info = {
                "x (Input Tensor)": x,
                "x0 (Initial Convolution Output)": x0,
                "x1 (Layer 1 Output)": x1,
                "x2 (Layer 2 Output)": x2,
                "x3 (Layer 3 Output)": x3,
                "x4 (Layer 4 Output)": x4,
                "x4_out (Layer 4 OutConv Output)": x4_out,
                "x4_out_2x (Upsampled x4_out)": x4_out_2x,
                "x3_out (Layer 3 OutConv Output)": x3_out,
                "x3_out_2x (Upsampled x3_out)": x3_out_2x,
                "x2_out (Layer 2 OutConv Output)": x2_out,
            }
            self.log_tensor_dimensions(tensor_info=tensor_info)

        return x4_out, x2_out
