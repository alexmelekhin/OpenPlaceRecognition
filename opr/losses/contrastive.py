from typing import Dict, List, Literal, Sequence, Tuple, Union

import torch
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer, MeanReducer, SumReducer
from torch import Tensor, nn

from opr.miners.all_pairs_miner import AllPairsMiner
from opr.miners.hard_triplet_miner import HardTripletMiner


class MultimodalContrastiveLoss(nn.Module):
    # TODO: docstring https://github.com/pytorch/pytorch/wiki/torch.nn-Module-Documentation-Style-Guide

    valid_modalities = ("image", "cloud", "semantic", "text", "fusion")

    def __init__(
        self,
        pos_margin: float,
        neg_margin: float,
        distance: Union[LpDistance, CosineSimilarity],
        miner: Union[AllPairsMiner, HardTripletMiner],
        reducer: Union[AvgNonZeroReducer, MeanReducer, SumReducer],
        modalities: Union[
            Literal["image", "cloud", "fusion"], Sequence[Literal["image", "cloud", "fusion"]]
        ] = ("image",),
        weights: Union[float, Sequence[float]] = 1.0,
    ) -> None:
        super().__init__()
        self.pos_margin = pos_margin

        if isinstance(modalities, str):
            modalities = tuple([modalities])
        if not set(modalities).issubset(self.valid_modalities):
            raise ValueError(f"Invalid modalities argument: '{modalities}' not in {self.valid_modalities}")
        self.modalities = modalities

        if isinstance(weights, float):
            weights = tuple([weights])
        if len(weights) != len(self.modalities):
            raise ValueError(f"Incorrect len(weights) = {len(weights)}, len(modalities) = {len(modalities)}")
        self.w = weights

        if isinstance(distance, (LpDistance, CosineSimilarity)):
            self.distance_fn = distance
        else:
            raise ValueError(f"Incorrect distance_fn = {distance}")

        if isinstance(miner, (AllPairsMiner, HardTripletMiner)):
            self.miner_fn = miner
        else:
            raise ValueError(f"Incorrect miner_fn = {miner}")

        if isinstance(reducer, (AvgNonZeroReducer, MeanReducer, SumReducer)):
            self.reducer_fn = reducer
        else:
            raise ValueError(f"Incorrect reducer_fn = {reducer}")

        self.loss_fn = {}
        for key in self.modalities:
            self.loss_fn[key] = ContrastiveLoss(
                pos_margin, neg_margin, distance=self.distance_fn, reducer=self.reducer_fn, collect_stats=True
            )

    def forward(
        self,
        model_output: Dict[str, Tensor],
        positives_mask: Tensor,
        negatives_mask: Tensor,
    ) -> Tuple[Tensor, Dict, Dict]:
        loss: Dict[str, Tensor] = {}
        losses: List[Tensor] = []
        stats = {}

        indices_tuple, miner_stats = self.miner_fn(model_output, positives_mask, negatives_mask)

        for i, key in enumerate(self.modalities):
            if key not in model_output.keys():
                raise KeyError(f"No key {key} in model_output.keys() = {model_output.keys()}")
            loss[key] = self.loss_fn[key](model_output[key], indices_tuple=indices_tuple[key])
            losses.append(loss[key] * self.w[i])

            if isinstance(self.miner_fn, AllPairsMiner):
                num_pairs = len(indices_tuple[key][0]) + len(indices_tuple[key][2])
            elif isinstance(self.miner_fn, HardTripletMiner):
                num_pairs = len(indices_tuple[key][0]) * 2  # each triplet is divided into two pairs
            else:
                raise ValueError(f"Unknown miner_fn, can't compute number of pairs")

            stats[key] = {
                "loss": loss[key].item(),
                "avg_embedding_norm": self.loss_fn[key].distance.final_avg_query_norm,
                "num_pairs": num_pairs,
            }
            if isinstance(self.reducer_fn, AvgNonZeroReducer):
                stats[key]["num_non_zero_positive_pairs"] = (
                    self.loss_fn[key].reducer.reducers["pos_loss"].num_past_filter
                )
                stats[key]["num_non_zero_negative_pairs"] = (
                    self.loss_fn[key].reducer.reducers["neg_loss"].num_past_filter
                )
                if self.pos_margin > 0:
                    stats[key]["non_zero_rate"] = (
                        stats[key]["num_non_zero_positive_pairs"] + stats[key]["num_non_zero_negative_pairs"]
                    ) / stats[key]["num_pairs"]
                else:
                    stats[key]["non_zero_rate"] = stats[key]["num_non_zero_negative_pairs"] / len(
                        indices_tuple[key][2]
                    )

            # TODO: calculate max loss triplets somehow?
        total_loss = torch.stack(losses).sum()
        stats["total_loss"] = total_loss.item()
        return total_loss, stats, miner_stats
