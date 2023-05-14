"""Interfaces and meta-models definitions."""
from typing import Dict, Optional, Union

import MinkowskiEngine as ME  # noqa: N817
from torch import Tensor, nn


class ImageFeatureExtractor(nn.Module):
    """Interface class for image feature extractor module."""

    def __init__(self):
        """Interface class for image feature extractor module."""
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageHead(nn.Module):
    """Interface class for image head module."""

    def __init__(self):
        """Interface class for image head module."""
        super().__init__()

    def forward(self, feature_map: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageModule(nn.Module):
    """Meta-module for image branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: ImageFeatureExtractor,
        head: ImageHead,
    ):
        """Meta-module for image branch.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class CloudFeatureExtractor(nn.Module):
    """Interface class for cloud feature extractor module."""

    def __init__(self):
        """Interface class for cloud feature extractor module."""
        super().__init__()

    def forward(self, cloud: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        raise NotImplementedError()


class CloudHead(nn.Module):
    """Interface class for cloud head module."""

    def __init__(self):
        """Interface class for cloud head module."""
        super().__init__()

    def forward(self, feature_map: ME.SparseTensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class CloudModule(nn.Module):
    """Meta-module for cloud branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: CloudFeatureExtractor,
        head: CloudHead,
    ):
        """Meta-module for cloud branch.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.

        Raises:
            ValueError: If incompatible cloud backbone and head are given.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: ME.SparseTensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class FusionModule(nn.Module):
    """Interface class for fusion module."""

    def __init__(self):
        """Interface class for fusion module."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class MultiImageModule(nn.Module):
    """Module to work with multiple images with late fusion."""

    def __init__(self, image_module: ImageModule, fusion_module: FusionModule) -> None:
        """Module to work with multiple images with late fusion.

        Args:
            image_module (ImageModule): Module to process each image.
            fusion_module (FusionModule): Module to fuse descriptors of each image.
        """
        super().__init__()
        self.image_module = image_module
        self.fusion_module = fusion_module

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        x_dict = {}
        for key in data:
            if key.startswith("images_"):
                x_dict[key] = self.image_module(data[key])
        x = self.fusion_module(x_dict)
        return x


class ComposedModel(nn.Module):
    """Composition model for multimodal architectures."""

    def __init__(
        self,
        image_module: Optional[Union[ImageModule, MultiImageModule]] = None,
        semantic_module: Optional[ImageModule] = None,
        semantic_mk_module: Optional[CloudModule] = None,
        cloud_module: Optional[CloudModule] = None,
        fusion_module: Optional[FusionModule] = None,
    ) -> None:
        """Composition model for multimodal architectures.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            semantic_module (ImageModule, optional): Semantic modality branch. Defaults to None.
            semantic_module (CloudModule, optional): Minkovski semantic modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            fusion_module (FusionModule, optional): Module to fuse different modalities. Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.semantic_mk_module = semantic_mk_module
        self.cloud_module = cloud_module
        self.fusion_module = fusion_module

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        out_dict: Dict[str, Tensor] = {}

        if self.image_module is not None and isinstance(self.image_module, ImageModule):
            out_dict["image"] = self.image_module(batch["images"])
        elif self.image_module is not None and isinstance(self.image_module, MultiImageModule):
            out_dict["image"] = self.image_module(batch)

        if self.semantic_module is not None and isinstance(self.semantic_module, ImageModule):
            out_dict["semantic"] = self.semantic_module(batch["semantics"])

        if self.semantic_mk_module is not None and isinstance(self.semantic_mk_module, CloudModule):
            mk_semantic_data = ME.SparseTensor(features=batch["sem_features"], coordinates=batch["sem_coordinates"])
            out_dict["semantic_mk"] = self.semantic_mk_module(mk_semantic_data)

        if self.cloud_module is not None:
            cloud = ME.SparseTensor(features=batch["features"], coordinates=batch["coordinates"])
            out_dict["cloud"] = self.cloud_module(cloud)

        if self.fusion_module is not None:
            out_dict["fusion"] = self.fusion_module(out_dict)

        return out_dict
