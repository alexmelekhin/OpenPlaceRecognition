"""Batch hard triplet miner implementation.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from typing import Dict, List, Sequence, Tuple, Union, Optional

import torch
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import BaseMiner
from torch import Tensor


class HardTripletMiner:
    """Batch hard triplet miner.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    # valid_modalities = ("image", "cloud", "fusion")

    def __init__(
        self,
        distance: Union[LpDistance, CosineSimilarity],
        modalities: Union[List[str], Sequence[str]] = ("image",),
    ) -> None:
        """Batch hard triplet miner.

        Args:
            modalities: Union[List[str], Sequence[str]]: Modalities to use
            distance (Union[LpDistance, CosineSimilarity]): Distance function to use.
        """
        self.distance = distance
        self.modalities = modalities

    def __call__(
        self, embeddings: Dict[str, Tensor], positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[Dict[str, Tuple[Tensor, Tensor, Tensor]], Dict[str, Dict[str, Union[int, float]]]]:
        """Mine hard triplets from given batch of embeddings. For each element in batch triplet will be mined.

        Args:
            embeddings (Dict[str, Tensor]): Dictionary with model output embeddings.
            positives_mask (Tensor): Binary mask of positive elements in batch.
            negatives_mask (Tensor): Binary mask of negative elements in batch.

        Returns:
            Tuple[Dict[str, Tuple[Tensor, Tensor, Tensor]], Dict[str, Dict[str, Union[int, float]]]]:
                Hard triplets dict {"modality": (a, p, n)} and stats dict {"modality": {"stat": value}}.
        """
        hard_triplets = {}
        stats = {}
        for key, values in embeddings.items():
            if key in self.modalities and values is not None:
                assert values.dim() == 2
                d_embeddings = values.detach()
                with torch.no_grad():
                    hard_triplets[key], stats[key] = self._mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets, stats

    def _mine(
        self, embeddings, positives_mask, negatives_mask
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, Union[int, float]]]:
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = self._get_max_per_row(
            dist_mat, positives_mask
        )
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = self._get_min_per_row(
            dist_mat, negatives_mask
        )
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        stats = {}
        stats["max_pos_pair_dist"] = torch.max(hardest_positive_dist).item()
        stats["max_neg_pair_dist"] = torch.max(hardest_negative_dist).item()
        stats["mean_pos_pair_dist"] = torch.mean(hardest_positive_dist).item()
        stats["mean_neg_pair_dist"] = torch.mean(hardest_negative_dist).item()
        stats["min_pos_pair_dist"] = torch.min(hardest_positive_dist).item()
        stats["min_neg_pair_dist"] = torch.min(hardest_negative_dist).item()
        return (a, p, n), stats

    def _get_max_per_row(self, mat, mask) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        non_zero_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = 0
        return torch.max(mat_masked, dim=1), non_zero_rows

    def _get_min_per_row(self, mat, mask) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        non_inf_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = float("inf")
        return torch.min(mat_masked, dim=1), non_inf_rows

class MinersHolder:
    """Holder for miners for differnet modalities"""
    def __init__(
        self,
        modalities: Union[List[str], Sequence[str]] = ("image",),
        **kwargs
    ) -> None:
        self.modalities = modalities

        self.miners = {}
        for m in modalities:
            if f"{m}_miner" not in kwargs:
                raise ValueError(f"{m} modality is pased as argument, but {m}_miner is not configured in triple_margin.yaml")
            if isinstance(kwargs[f"{m}_miner"], HardTripletMinerDist):
                self.miners[f"{m}"] = kwargs[f"{m}_miner"] 
            else:
                raise ValueError(f"Miner {m}_miner is not instance of HardTripletMinerDist")


class HardTripletMinerDist(BaseMiner):
    """Batch hard triplet miner.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    # valid_modalities = ("image", "cloud", "fusion")

    def __init__(
        self,
        distance: Union[LpDistance, CosineSimilarity],
        **kwargs
    ) -> None:
        """Batch hard triplet miner.

        Args:
            modalities: Union[List[str], Sequence[str]]: Modalities to use
            distance (Union[LpDistance, CosineSimilarity]): Distance function to use.
        """
        super().__init__(**kwargs)
        self.distance = distance

        self.stats = None

    def mine(
        self, fake_embeddings, _, __, ____
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, Union[int, float]]]:
        # Based on pytorch-metric-learning implementation
        flat_mask_size = fake_embeddings.shape[0]
        embeddings, positives_mask, negatives_mask = torch.split(fake_embeddings, 
                                                                 [fake_embeddings.shape[1]-flat_mask_size*2,        flat_mask_size, flat_mask_size], dim=1)
        positives_mask = positives_mask.cpu().bool()
        negatives_mask = negatives_mask.cpu().bool()
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = self._get_max_per_row(
            dist_mat, positives_mask
        )
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = self._get_min_per_row(
            dist_mat, negatives_mask
        )
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        stats = {}
        stats["max_pos_pair_dist"] = torch.max(hardest_positive_dist).item()
        stats["max_neg_pair_dist"] = torch.max(hardest_negative_dist).item()
        stats["mean_pos_pair_dist"] = torch.mean(hardest_positive_dist).item()
        stats["mean_neg_pair_dist"] = torch.mean(hardest_negative_dist).item()
        stats["min_pos_pair_dist"] = torch.min(hardest_positive_dist).item()
        stats["min_neg_pair_dist"] = torch.min(hardest_negative_dist).item()

        self.stats = stats
        return a, p, n

    def _get_max_per_row(self, mat, mask) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        non_zero_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = 0
        return torch.max(mat_masked, dim=1), non_zero_rows

    def _get_min_per_row(self, mat, mask) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        non_inf_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = float("inf")
        return torch.min(mat_masked, dim=1), non_inf_rows