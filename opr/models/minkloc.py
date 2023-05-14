"""MinkLoc3D architecture implementation.

Komorowski, Jacek. "Improving Point Cloud Based Place Recognition
with Ranking-based Loss and Large Batch Training." 2022 26th International
Conference on Pattern Recognition (ICPR). IEEE, 2022.

Paper: https://arxiv.org/abs/2203.00972
Code adopted from the original repository: https://github.com/jac99/MinkLoc3Dv2, MIT License
"""
from typing import Tuple, Type, Union

import MinkowskiEngine as ME  # noqa: N817
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from torch import Tensor, nn

from opr.models.base_models import CloudFeatureExtractor
from opr.models.layers.eca import MinkECABasicBlock as ECABasicBlock


class MinkResNetBase(CloudFeatureExtractor):
    """Base ResNet class for sparse tensors with MinkowskiEngine."""

    block: Union[Type[BasicBlock], Type[Bottleneck], Type[ECABasicBlock]]
    layers: Tuple[int, ...] = (1, 1, 1, 1)
    init_dim: int = 64
    planes: Tuple[int, ...] = (64, 128, 256, 512)
    sparse: bool = True

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 3) -> None:
        """Base ResNet class for sparse tensors with MinkowskiEngine.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dimension (int): Number of dimensions. Defaults to 3.
        """
        super().__init__()
        self.dimension = dimension
        assert self.block is not None

        self._network_initialization(in_channels, out_channels, dimension)
        self._weight_initialization()

    def _network_initialization(self, in_channels: int, out_channels: int, dimension: int) -> None:
        self.inplanes = self.init_dim
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, stride=2, dimension=dimension
        )

        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=dimension)

        self.layer1 = self._make_layer(self.block, self.planes[0], self.layers[0], stride=2)
        self.layer2 = self._make_layer(self.block, self.planes[1], self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, self.planes[2], self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, self.planes[3], self.layers[3], stride=2)

        self.conv5 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=dimension
        )
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.glob_avg = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def _weight_initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck], Type[ECABasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        bn_momentum: float = 0.1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.dimension,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.dimension,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.dimension))

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> Tensor:  # noqa: D102
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.glob_avg(x)
        return self.final(x)


class MinkResNetFPNExtractor(MinkResNetBase):
    """Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks."""

    sparse: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        num_top_down: int = 2,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        layers: Tuple[int, ...] = (1, 1, 1, 1),
        planes: Tuple[int, ...] = (64, 128, 64, 32),
    ) -> None:
        """Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks.

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            num_top_down (int): Number of top-down steps for FPN block. Defaults to 2.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Block type name. Defaults to "ECABasicBlock".
            layers (Tuple[int, ...]): Number of layers for each block. Defaults to (1, 1, 1, 1).
            planes (Tuple[int, ...]): Output channel size for each block. Defaults to (64, 128, 64, 32).
        """
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = self._create_resnet_block(block_name=block)
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        MinkResNetBase.__init__(self, in_channels, out_channels, dimension=3)

    def _create_resnet_block(
        self, block_name: str
    ) -> Union[Type[BasicBlock], Type[Bottleneck], Type[ECABasicBlock]]:
        if block_name == "BasicBlock":
            block_module = BasicBlock
        elif block_name == "Bottleneck":
            block_module = Bottleneck
        elif block_name == "ECABasicBlock":
            block_module = ECABasicBlock
        else:
            raise NotImplementedError(f"Unsupported network block: {block_name}")

        return block_module

    def _network_initialization(self, in_channels: int, out_channels: int, dimension: int) -> None:
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=dimension
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=dimension
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - i], self.lateral_dim, kernel_size=1, stride=1, dimension=dimension
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim, self.lateral_dim, kernel_size=2, stride=2, dimension=dimension
                )
            )
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - self.num_top_down],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=dimension,
                )
            )
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[0], self.lateral_dim, kernel_size=1, stride=1, dimension=dimension
                )
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger kernel)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])

        return x



class MinkResNetFPNExtractorOneHotSemantic(MinkResNetBase):
    """Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks."""

    sparse: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        num_top_down: int = 2,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        layers: Tuple[int, ...] = (1, 1, 1, 1),
        planes: Tuple[int, ...] = (64, 128, 64, 32),
    ) -> None:
        """Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks.

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            num_top_down (int): Number of top-down steps for FPN block. Defaults to 2.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Block type name. Defaults to "ECABasicBlock".
            layers (Tuple[int, ...]): Number of layers for each block. Defaults to (1, 1, 1, 1).
            planes (Tuple[int, ...]): Output channel size for each block. Defaults to (64, 128, 64, 32).
        """
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = self._create_resnet_block(block_name=block)
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        MinkResNetBase.__init__(self, in_channels, out_channels, dimension=3)

    def _create_resnet_block(
        self, block_name: str
    ) -> Union[Type[BasicBlock], Type[Bottleneck], Type[ECABasicBlock]]:
        if block_name == "BasicBlock":
            block_module = BasicBlock
        elif block_name == "Bottleneck":
            block_module = Bottleneck
        elif block_name == "ECABasicBlock":
            block_module = ECABasicBlock
        else:
            raise NotImplementedError(f"Unsupported network block: {block_name}")

        return block_module

    def _network_initialization(self, in_channels: int, out_channels: int, dimension: int) -> None:
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=dimension
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=dimension
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - i], self.lateral_dim, kernel_size=1, stride=1, dimension=dimension
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim, self.lateral_dim, kernel_size=2, stride=2, dimension=dimension
                )
            )
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - self.num_top_down],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=dimension,
                )
            )
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[0], self.lateral_dim, kernel_size=1, stride=1, dimension=dimension
                )
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger kernel)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])

        return x
