"""NCLT dataset implementation."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import MinkowskiEngine as ME  # type: ignore
import numpy as np
import torch
from torch import Tensor

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultSemanticTransform,
)
from opr.datasets.base import BasePlaceRecognitionDataset
from opr.utils import in_sorted_array


class NCLTDataset(BasePlaceRecognitionDataset):
    """NCLT dataset implementation."""

    _images_dirname: str
    _masks_dirname: str
    _pointclouds_dirname: str
    _pointcloud_quantization_size: Optional[Union[float, Tuple[float, float, float]]]
    _max_point_distance: Optional[float]
    _spherical_coords: bool
    _use_intensity_values: bool
    _valid_data: Tuple[str, ...] = (
        "image_lb3_Cam0",
        "image_lb3_Cam1",
        "image_lb3_Cam2",
        "image_lb3_Cam3",
        "image_lb3_Cam4",
        "image_lb3_Cam5",
        "pointcloud_lidar",
        "mask_lb3_Cam0",
        "mask_lb3_Cam1",
        "mask_lb3_Cam2",
        "mask_lb3_Cam3",
        "mask_lb3_Cam4",
        "mask_lb3_Cam5",
        # TODO: add text embeddings data
    )

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"],
        data_to_load: Union[str, Tuple[str, ...]],
        positive_threshold: float = 10.0,
        negative_threshold: float = 50.0,
        images_dirname: str = "",
        masks_dirname: str = "segmentation_masks",
        pointclouds_dirname: str = "velodyne_data",
        pointcloud_quantization_size: Optional[Union[float, Tuple[float, float, float]]] = 0.01,
        max_point_distance: Optional[float] = None,
        spherical_coords: bool = False,
        use_intensity_values: bool = False,
    ) -> None:
        """NCLT dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            data_to_load (Union[str, Tuple[str, ...]]): The list of data to load.
                Check the documentation for the list of available data.
            positive_threshold (float): The UTM distance threshold value for positive samples.
                Defaults to 10.0.
            negative_threshold (float): The UTM distance threshold value for negative samples.
                Defaults to 50.0.
            images_dirname (str): Images directory name. It should be specified explicitly
                if custom preprocessing was done. Defaults to "images".
            masks_dirname (str): Masks directory name. It should be specified explicitly
                if custom preprocessing was done. Defaults to "segmentation_masks".
            pointclouds_dirname (str): Point clouds directory name. It should be specified
                explicitly if custom preprocessing was done. Defaults to "velodyne_data".
            pointcloud_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.
            max_point_distance (float, optional): The maximum distance of points from the origin.
                Defaults to None.
            spherical_coords (bool): Whether to use spherical coordinates for point clouds.
                Defaults to False.
            use_intensity_values (bool): Whether to use intensity values for point clouds. Defaults to False.

        Raises:
            ValueError: If data_to_load contains invalid data source names.
            FileNotFoundError: If images, masks or pointclouds directory does not exist.
        """
        # TODO: ^ docstring is also not DRY -> it is almost the same as in Oxford dataset
        super().__init__(dataset_root, subset, data_to_load, positive_threshold, negative_threshold)

        if any(elem not in self._valid_data for elem in self.data_to_load):
            raise ValueError(f"Invalid data_to_load argument. Valid data list: {self._valid_data!r}")

        # TODO: review legacy code:
        # if "chonky" in self.modalities:  # ! It's a bit tricky but idk how to do it better now
        #     self.modalities.append("image")
        #     self.modalities.append("semantic")

        _track_name = self.dataset_df.iloc[0]["track"]

        if any(elem.startswith("image") for elem in self.data_to_load):
            self._images_dirname = images_dirname
            if not (self.dataset_root / _track_name / self._images_dirname).exists():
                raise FileNotFoundError(f"Images directory {self._images_dirname!r} does not exist.")

        if any(elem.startswith("mask") for elem in self.data_to_load):
            self._masks_dirname = masks_dirname
            if not (self.dataset_root / _track_name / self._masks_dirname).exists():
                raise FileNotFoundError(f"Masks directory {self._masks_dirname!r} does not exist.")

        if "pointcloud_lidar" in self.data_to_load:
            self._pointclouds_dirname = pointclouds_dirname
            if not (self.dataset_root / _track_name / self._pointclouds_dirname).exists():
                raise FileNotFoundError(
                    f"Pointclouds directory {self._pointclouds_dirname!r} does not exist."
                )

        # TODO: review upper code ^ and apply DRY principle

        # TODO: images and masks transforms should be performed simualtenously via Albumentations
        # TODO: transforms should be passed to init function as arguments
        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.semantic_transform = DefaultSemanticTransform(train=(self.subset == "train"))
        self.pointcloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        self.pointcloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))

        self._pointcloud_quantization_size = pointcloud_quantization_size
        self._max_point_distance = max_point_distance
        self._spherical_coords = spherical_coords
        self._use_intensity_values = use_intensity_values

        # TODO: review legacy code:
        # if text_embs_dir == "tfidf_pca":
        #     tracks = [i for i in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, i))]
        #     df_dict = {}
        #     for track in tracks:
        #         track_path = os.path.join(dataset_root, track)
        #         df_dict[track] = {
        #             f"cam{n}": pd.read_csv(os.path.join(track_path, f"descriptions_Cam{n}.csv"))
        #             for n in range(1, 6)
        #         }
        #     self.descriptoins_dict = df_dict

        #     # load tfidf and pca
        #     self.vectorizer, self.pca = self._load_tfidf_pca()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # noqa: D105
        data = {"idx": torch.tensor(idx)}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float64))
        track_dir = self.dataset_root / str(row["track"])

        for data_source in self.data_to_load:
            if data_source.startswith("image_"):
                cam_name = data_source[6:]  # remove "image_" prefix
                cam_name, cam_number = cam_name.split("_")
                image_ts = int(row[cam_name])
                im_filepath = (
                    track_dir / self._images_dirname / f"{cam_name}_small" / cam_number / f"{image_ts}.png"
                )  # TODO: get rid of _small
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[data_source] = im
            elif data_source.startswith("mask_"):
                cam_name = data_source[5:]  # remove "mask_" prefix
                cam_name, cam_number = cam_name.split("_")
                image_ts = int(row[cam_name])
                mask_filepath = (
                    track_dir / self._masks_dirname / f"{cam_name}_small" / cam_number / f"{image_ts}.png"
                )  # TODO: get rid of _small
                mask = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
                mask = self.semantic_transform(mask)
                data[data_source] = mask
            elif data_source == "pointcloud_lidar":
                pc_filepath = track_dir / self._pointclouds_dirname / f"{row['pointcloud']}.bin"
                pointcloud = self._load_pc(pc_filepath)
                data[f"{data_source}_coords"] = self.pointcloud_transform(pointcloud[:, :3])
                if self._spherical_coords:
                    # TODO: implement conversion to spherical coords
                    raise NotImplementedError("Spherical coords are not implemented yet.")
                if self._use_intensity_values:
                    data[f"{data_source}_feats"] = pointcloud[:, 3].unsqueeze(1)
                else:
                    data[f"{data_source}_feats"] = torch.ones_like(pointcloud[:, :1])

        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        if self._max_point_distance is not None:
            pc = pc[np.linalg.norm(pc, axis=1) < self._max_point_distance]
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        return pc_tensor

    # TODO: this is the same collate_fn as in Oxford -> refactor to DRY principle
    def collate_fn(self, data_list: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.

        Raises:
            ValueError: if data source is not supported.
        """
        result: Dict[str, Tensor] = {}
        indices = torch.stack([e["idx"] for e in data_list], dim=0).tolist()
        for data_key in data_list[0].keys():
            if data_key == "idx":
                continue
            elif data_key == "utm":
                result["utms"] = torch.stack([e["utm"] for e in data_list], dim=0)
            elif data_key.startswith("image_"):
                result[f"images_{data_key[6:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key.startswith("mask_"):
                result[f"masks_{data_key[5:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key == "pointcloud_lidar_coords":
                coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
                feats_list = [e["pointcloud_lidar_feats"] for e in data_list]
                n_points = [int(e.shape[0]) for e in coords_list]
                coords_tensor = torch.cat(list(coords_list), dim=0).unsqueeze(0)  # (1,batch_size*n_points,3)
                if self.pointcloud_set_transform is not None:
                    # Apply the same transformation on all dataset elements
                    clouds = self.pointcloud_set_transform(coords_tensor)
                coords_list = torch.split(clouds.squeeze(0), split_size_or_sections=n_points, dim=0)
                cf_list = [
                    ME.utils.sparse_quantize(
                        coordinates=c, features=f, quantization_size=self._pointcloud_quantization_size
                    )
                    for c, f in zip(coords_list, feats_list)
                ]
                coords_list = [e[0] for e in cf_list]
                feats_list = [e[1] for e in cf_list]
                result["pointclouds_lidar_coords"] = ME.utils.batched_coordinates(coords_list)
                result["pointclouds_lidar_feats"] = torch.cat(feats_list)
            elif data_key == "pointcloud_lidar_feats":
                continue
            else:
                raise ValueError(f"Unknown data key: {data_key!r}")

        positives_mask = torch.tensor(
            [[in_sorted_array(e, self.positives_index[label]) for e in indices] for label in indices]
        )
        negatives_mask = torch.tensor(
            [[not in_sorted_array(e, self.nonnegative_index[label]) for e in indices] for label in indices]
        )

        return result, positives_mask, negatives_mask

    # def _load_tfidf_pca(self, base_savepath="./opr/datasets/"):
    #     vectorizer_savepath = os.path.join(base_savepath, "vectorizer.joblib")
    #     pca_savepath = os.path.join(base_savepath, "pca.joblib")

    #     vectorizer = load(vectorizer_savepath)
    #     pca = load(pca_savepath)
    #     return vectorizer, pca

    # def tfidf_pca_text_transform(self, text):
    #     vect_data = self.vectorizer.transform([text]).toarray()
    #     pca_data = self.pca.transform(vect_data)
    #     pca_data = torch.tensor(pca_data, dtype=torch.float32)
    #     return pca_data
