"""NCLT dataset implementation."""
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultSemanticTransform,
    OneHotSemanticTransform
)
from opr.datasets.base import BaseDataset


class NCLTDataset(BaseDataset):
    """NCLT dataset implementation."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud"),
        images_subdir: Optional[Union[str, Path]] = "lb3_small/Cam5",
        semantic_subdir: Optional[Union[str, Path]] = "lb3_segmentation_small/Cam5",
        mink_quantization_size: Optional[float] = 0.5,
        coords_limit: Tuple[int, int] = (-100, 100),
    ) -> None:
        """NCLT dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "image", "cloud").
            images_subdir (Union[str, Path], optional): Images subdirectory path.
                Defaults to "lb3_small/Cam5".
            mink_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.5.
            coords_limit (Tuple[int, int]): Lower and upper limits for pointcloud's coordinates.
                Defaults to (-100, 100).

        Raises:
            ValueError: If images_subdir is undefined.
        """
        super().__init__(dataset_root, subset, modalities)

        if "image" in self.modalities:
            if images_subdir:
                self.images_subdir = Path(images_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_subdir' is set to None"
                )

        if "semantic" in self.modalities:
            if semantic_subdir:
                self.semantic_subdir = Path(semantic_subdir)
            else:
                raise ValueError(
                    "Given 'semantic' in 'modalities' argument, but 'semantic_subdir' is set to None"
                )
        
        if "semantic_mk" in self.modalities:
            if semantic_subdir:
                self.semantic_subdir = Path(semantic_subdir)
            else:
                raise ValueError(
                    "Given 'semantic_mk' in 'modalities' argument, but 'semantic_subdir' is set to None"
                )

        if "cloud" in self.modalities:
            self.clouds_subdir = Path("velodyne_data")

        self.mink_quantization_size = mink_quantization_size
        self.coords_limit = coords_limit

        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.semantic_transform = DefaultSemanticTransform(train=(self.subset == "train"))
        self.semantic_mk_transform = OneHotSemanticTransform(train=(self.subset == "train"))
        self.cloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        self.cloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float32))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            im_filepath = track_dir / self.images_subdir / f"{row['image']}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image"] = im

        # TODO: implement multi-camera setup better?
        for n in range(6):
            if f"image_cam{n}" in self.modalities:
                im_filepath = track_dir / f"lb3_small/Cam{n}" / f"{row['image']}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[f"image_cam{n}"] = im

        if "semantic" in self.modalities and self.semantic_subdir is not None:
            im_filepath = track_dir / self.semantic_subdir / f"{row['image']}.png" # image id is equal to semantic mask id~
            im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.semantic_transform(im)
            data["semantic"] = im
        
        if "semantic_mk" in self.modalities and self.semantic_subdir is not None:
            im_filepath = track_dir / self.semantic_subdir / f"{row['image']}.png"
            im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
            im = self.semantic_mk_transform(im)
            data["semantic_mk"] = im

        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['pointcloud']}.bin"
            pc = self._load_pc(pc_filepath)
            pc = self.cloud_transform(pc)
            data["cloud"] = pc
        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        in_range_idx = np.all(
            np.logical_and(self.coords_limit[0] <= pc, pc <= self.coords_limit[1]),  # select points in range
            axis=1,
        )
        pc = pc[in_range_idx]
        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        return pc_tensor

class NCLTDataset_OneHotSemantic(BaseDataset):
    """NCLT dataset implementation."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud"),
        images_subdir: Optional[Union[str, Path]] = "lb3_small/Cam5",
        semantic_subdir: Optional[Union[str, Path]] = "lb3_segmentation_small/Cam5",
        mink_quantization_size: Optional[float] = 0.5,
        coords_limit: Tuple[int, int] = (-100, 100),
    ) -> None:
        """NCLT dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "image", "cloud").
            images_subdir (Union[str, Path], optional): Images subdirectory path.
                Defaults to "lb3_small/Cam5".
            mink_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.5.
            coords_limit (Tuple[int, int]): Lower and upper limits for pointcloud's coordinates.
                Defaults to (-100, 100).

        Raises:
            ValueError: If images_subdir is undefined.
        """
        super().__init__(dataset_root, subset, modalities)

        if "image" in self.modalities:
            if images_subdir:
                self.images_subdir = Path(images_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_subdir' is set to None"
                )

        if "semantic" in self.modalities:
            if semantic_subdir:
                self.semantic_subdir = Path(semantic_subdir)
            else:
                raise ValueError(
                    "Given 'semantic' in 'modalities' argument, but 'semantic_subdir' is set to None"
                )

        if "cloud" in self.modalities:
            self.clouds_subdir = Path("velodyne_data")

        self.mink_quantization_size = mink_quantization_size
        self.coords_limit = coords_limit

        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.semantic_transform = OneHotSemanticTransform(train=(self.subset == "train"))
        self.cloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        self.cloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float32))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            im_filepath = track_dir / self.images_subdir / f"{row['image']}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image"] = im

        # TODO: implement multi-camera setup better?
        for n in range(6):
            if f"image_cam{n}" in self.modalities:
                im_filepath = track_dir / f"lb3_small/Cam{n}" / f"{row['image']}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[f"image_cam{n}"] = im

        if "semantic" in self.modalities and self.semantic_subdir is not None:
            im_filepath = track_dir / self.semantic_subdir / f"{row['image']}.png" # image id is equal to semantic mask id~
            im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.semantic_transform(im)
            data["semantic"] = im
        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['pointcloud']}.bin"
            pc = self._load_pc(pc_filepath)
            pc = self.cloud_transform(pc)
            data["cloud"] = pc
        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        in_range_idx = np.all(
            np.logical_and(self.coords_limit[0] <= pc, pc <= self.coords_limit[1]),  # select points in range
            axis=1,
        )
        pc = pc[in_range_idx]
        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        return pc_tensor