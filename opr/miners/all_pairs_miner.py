from typing import Dict, Tuple, Union

import torch
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from torch import Tensor


class AllPairsMiner:
    valid_modalities = ("image", "cloud", "semantic", "text", "fusion")

    def __init__(self, distance: Union[LpDistance, CosineSimilarity]) -> None:
        self.distance = distance

    def __call__(
        self, embeddings: Dict[str, Tensor], positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[
        Dict[str, Tuple[Tensor, Tensor, Tensor, Tensor]], Dict[str, Dict[str, Union[int, float]]]
    ]:
        a1, p = self.get_index_pairs(positives_mask)
        a2, n = self.get_index_pairs(negatives_mask)
        indices_tuple = {}

        stats = {}
        for key, values in embeddings.items():
            if key in self.valid_modalities and values is not None:
                assert values.dim() == 2
                indices_tuple[key] = (a1, p, a2, n)
                d_embeddings = values.detach()
                with torch.no_grad():
                    stats[key] = self.get_distance_stats(d_embeddings, indices_tuple[key])

        return indices_tuple, stats

    def get_index_pairs(self, mask: Tensor) -> Tuple[Tensor, Tensor]:
        mask = torch.triu(mask, diagonal=1)  # set values on and below main diagonal to False
        row_indices, col_indices = torch.where(mask)
        return row_indices, col_indices

    def get_distance_stats(
        self, embeddings: Tensor, indices_tuple: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Dict:
        a1, p, a2, n = indices_tuple
        dist_mat = self.distance(embeddings)

        stats = {}
        if len(p) > 0:
            positive_distances = dist_mat[a1, p]
            stats["max_pos_pair_dist"] = torch.max(positive_distances).item()
            stats["mean_pos_pair_dist"] = torch.mean(positive_distances).item()
            stats["min_pos_pair_dist"] = torch.min(positive_distances).item()
        if len(n) > 0:
            negative_distances = dist_mat[a2, n]
            stats["max_neg_pair_dist"] = torch.max(negative_distances).item()
            stats["mean_neg_pair_dist"] = torch.mean(negative_distances).item()
            stats["min_neg_pair_dist"] = torch.min(negative_distances).item()

        return stats
