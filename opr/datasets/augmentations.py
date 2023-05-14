"""Data augmentation pipelines.

Point cloud augmentations adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""
import math
import random
from typing import Optional, Tuple

import albumentations as A  # noqa: N812
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from scipy.linalg import expm, norm
from torch import Tensor
from torchvision import transforms


class OheHotTransform:
    """Rotate by one of the given angles."""
    def __call__(self, image):
        onehot = torch.squeeze(F.one_hot(torch.from_numpy(image).long(), 65)) #! Magic number
        onehot = onehot.permute(2, 0, 1).float()
        return {'image':onehot} 

class DefaultImageTransform:
    """Default image augmentation pipeline."""

    def __init__(self, train: bool = False, resize: Optional[Tuple[int, int]] = None) -> None:
        """Default image augmentation pipeline.

        Args:
            train (bool): If not train, only normalization will be applied. Defaults to False.
            resize (Tuple[int, int], optional): Target size in (W, H) format. Defaults to None.
        """
        if train:
            transform_list = [
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                    ],
                    p=0.2,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=True),
                A.CoarseDropout(max_width=96, max_height=66, min_width=32, min_height=22, max_holes=1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        else:
            transform_list = [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]

        if resize is not None:
            transform_list = [A.Resize(height=resize[1], width=resize[0])] + transform_list

        self.transform = A.Compose(transform_list)

    def __call__(self, img: np.ndarray) -> Tensor:
        """Applies transformations to the given image.

        Args:
            img (np.ndarray): The image in the cv2 format.

        Returns:
            Tensor: Augmented PyTorch tensor in the channel-first format.
        """
        return self.transform(image=img)["image"]


class DefaultSemanticTransform:
    """Default semantic mask augmentation pipeline."""

    def __init__(self, train: bool = False, resize: Optional[Tuple[int, int]] = None) -> None:
        """Default semantic mask augmentation pipeline.

        Args:
            train (bool): If not train, only normalization will be applied. Defaults to False.
            resize (Tuple[int, int], optional): Target size in (W, H) format. Defaults to None.
        """
        if train:
            transform_list = [
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),

                A.OneOf(
                    [
                        A.CoarseDropout(max_width=96,
                                        max_height=66,
                                        min_width=32,
                                        min_height=22,
                                        max_holes=1, p=0.5),
                        A.CoarseDropout(max_width=30,
                                        max_height=30,
                                        min_width=10,
                                        min_height=10,
                                        max_holes=10, p=0.5),
                        A.GridDropout (ratio=0.05,
                                     unit_size_min=4,
                                     unit_size_max=30,
                                     p=0.5)
                    ],
                    p=0.2,
                ),
                
                A.Normalize(mean=(0.,), std=(1.,)),
                ToTensorV2(),
            ]
        else:
            transform_list = [
                A.Normalize(mean=(0.,), std=(1.,)),
                ToTensorV2(),
            ]

        if resize is not None:
            transform_list = [A.Resize(height=resize[1], width=resize[0])] + transform_list

        self.transform = A.Compose(transform_list)

    def __call__(self, img: np.ndarray) -> Tensor:
        """Applies transformations to the given semantic mask.

        Args:
            img (np.ndarray): The semantic mask (single channel image) in the cv2 format.

        Returns:
            Tensor: Augmented PyTorch tensor in the channel-first format.
        """
        return self.transform(image=img)["image"]


class OneHotSemanticTransform:
    """One-Hot semantic mask augmentation pipeline."""

    def __init__(self, train: bool = False, resize: Optional[Tuple[int, int]] = None) -> None:
        """One-Hot semantic mask augmentation pipeline.

        Args:
            train (bool): If not train, only normalization will be applied. Defaults to False.
            resize (Tuple[int, int], optional): Target size in (W, H) format. Defaults to None.
        """
        if train:
            transform_list = [
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),

                A.OneOf(
                    [
                        A.CoarseDropout(max_width=96,
                                        max_height=66,
                                        min_width=32,
                                        min_height=22,
                                        max_holes=1, p=0.5),
                        A.CoarseDropout(max_width=30,
                                        max_height=30,
                                        min_width=10,
                                        min_height=10,
                                        max_holes=10, p=0.5),
                        A.GridDropout (ratio=0.05,
                                     unit_size_min=4,
                                     unit_size_max=30,
                                     p=0.5)
                    ],
                    p=0.2,
                ),
                
                OheHotTransform(),
                # ToTensorV2(),
            ]
        else:
            transform_list = [
                OheHotTransform(),
                # ToTensorV2(),
            ]

        if resize is not None:
            transform_list = [A.Resize(height=resize[1], width=resize[0])] + transform_list

        self.transform = A.Compose(transform_list)

    # def _channel(self, image):
    #     num_tags = len(stuff_classes)
    #     image_shape = image.shape

    #     height, width = image_shape[0], image_shape[1]
    #     new_image = np.zeros([height, width, num_tags])

    #     for i in range(height):
    #         for j in range(width - 1):

    #             if not (stuff_classes[image[i, j]] in blacklist):
    #                 new_image[i, j, image[i, j]] = 1

    #     return new_image

    def __call__(self, img: np.ndarray) -> Tensor:
        """Applies transformations to the given semantic mask.

        Args:
            img (np.ndarray): The semantic mask (single channel image) in the cv2 format.

        Returns:
            Tensor: Augmented PyTorch tensor in the channel-first format.
        """
        return self.transform(image=img)["image"]

class DefaultCloudTransform:
    """Default point cloud augmentation pipeline."""

    def __init__(self, train: bool = False) -> None:
        """Default point cloud augmentation pipeline.

        Args:
            train (bool): If False, no transforms will be applied. Defaults to False.
        """
        if train:
            self.transform = transforms.Compose(
                [
                    JitterPoints(sigma=0.001, clip=0.002),
                    RemoveRandomPoints(r=(0.0, 0.1)),
                    RandomTranslation(max_delta=0.01),
                    RemoveRandomBlock(p=0.4),
                ]
            )
        else:
            self.transform = transforms.Compose([])

    def __call__(self, pointcloud: Tensor) -> Tensor:
        """Apply the transformations to the given point cloud.

        Args:
            pointcloud (Tensor): The coordinates tensor.

        Returns:
            Tensor: Augmented coordinates tensor.
        """
        return self.transform(pointcloud)


class DefaultCloudSetTransform:
    """Default point cloud set augmentation pipeline."""

    def __init__(self, train: bool = False) -> None:
        """Default point cloud set augmentation pipeline.

        Note:
            This is how augmentation for the whole batch was implemented in MinkLoc method.

        Args:
            train (bool): If False, no transforms will be applied. Defaults to False.
        """
        if train:
            self.transform = transforms.Compose(
                [
                    RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
                    RandomFlip([0.25, 0.25, 0.0]),
                ]
            )
        else:
            self.transform = transforms.Compose([])

    def __call__(self, pointcloud: Tensor) -> Tensor:
        """Apply the transformations to the given point cloud.

        Args:
            pointcloud (Tensor): The coordinates tensor.

        Returns:
            Tensor: Augmented coordinates tensor.
        """
        return self.transform(pointcloud)


# NOTE: The latter is the raw code taken from https://github.com/jac99/MinkLocMultimodal, MIT License
# TODO: Format code properly, add typing and remove temporary flake8 and mypy disablers
# flake8: noqa
# mypy: ignore-errors


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, "sum(p) must be in (0, 1] range, is: {}".format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta  # Rotation around axis
        self.max_theta2 = max_theta2  # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(
                np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5)
            )
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.0):
        assert 0 < p <= 1.0
        assert sigma > 0.0

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        sample_shape = (e.shape[0],)
        if self.p < 1.0:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64)

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n * r), replace=False)  # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)  # Fronto-parallel cuboid to remove
            mask = (
                (x < coords[..., 0])
                & (coords[..., 0] < x + w)
                & (y < coords[..., 1])
                & (coords[..., 1] < y + h)
            )
            coords[mask] = torch.zeros_like(coords[mask])
        return coords
