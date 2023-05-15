"""Multimodal triplet margin loss implementation.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from typing import Any, Dict, List, Literal, Sequence, Tuple, Union

import torch
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer, MeanReducer, SumReducer
from pytorch_metric_learning.utils import distributed as pml_dist
from torch import Tensor, nn

from opr.miners import HardTripletMiner, MinersHolder


class MultimodalTripletMarginLoss(nn.Module):
    """Triplet margin loss implementation for multimodal models.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    valid_modalities = ("image", "cloud", "fusion")

    def __init__(
        self,
        margin: float,
        distance: Union[LpDistance, CosineSimilarity],
        miner: HardTripletMiner,
        reducer: Union[AvgNonZeroReducer, MeanReducer, SumReducer],
        swap: bool = False,
        modalities: Union[
            Literal["image", "cloud", "fusion"], Sequence[Literal["image", "cloud", "fusion"]]
        ] = ("image",),
        weights: Union[float, Sequence[float]] = 1.0,
    ) -> None:
        """Triplet margin loss implementation for multimodal models.

        Args:
            margin (float): Margin hyperparameter.
            distance (Union[LpDistance, CosineSimilarity]): Distance function for loss calculation.
            miner (HardTripletMiner): Miner function to form triplets.
            reducer (Union[AvgNonZeroReducer, MeanReducer, SumReducer]): Reducer function. Will be used
                to reduce loss values to scalar.
            swap (bool): Use the positive-negative distance instead of anchor-negative distance,
                if it violates the margin more. Defaults to False.
            modalities (Union[Literal["image", "cloud", "fusion"],
                Sequence[Literal["image", "cloud", "fusion"]]]): For which modalities loss will be calculated.
                Defaults to ("image",).
            weights (Union[float, Sequence[float]]): Weight for each given modality. Defaults to 1.0.

        Raises:
            ValueError: If given incorrect modalities.
            ValueError: If number of weights not equal to number of modalities.
            ValueError: If incorrect distance function given.
            ValueError: If incorrect miner function given.
            ValueError: If incorrect reducer function given.
        """
        super().__init__()

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

        if isinstance(miner, HardTripletMiner):
            self.miner_fn = miner
        else:
            raise ValueError(f"Incorrect miner_fn = {miner}")

        if isinstance(reducer, (AvgNonZeroReducer, MeanReducer, SumReducer)):
            self.reducer_fn = reducer
        else:
            raise ValueError(f"Incorrect reducer_fn = {reducer}")

        self.loss_fn = {}
        for key in self.modalities:
            self.loss_fn[key] = TripletMarginLoss(
                margin=margin,
                swap=swap,
                distance=self.distance_fn,
                reducer=self.reducer_fn,
                collect_stats=True,
            )

    def forward(  # noqa: D102
        self,
        model_output: Dict[str, Tensor],
        positives_mask: Tensor,
        negatives_mask: Tensor,
    ) -> Tuple[Tensor, Dict, Dict]:
        loss: Dict[str, Tensor] = {}
        losses: List[Tensor] = []
        stats: Dict[str, Any] = {}

        hard_triplets, miner_stats = self.miner_fn(model_output, positives_mask, negatives_mask)

        for i, key in enumerate(self.modalities):
            if key not in model_output.keys():
                raise KeyError(f"No key {key} in model_output.keys() = {model_output.keys()}")
            loss[key] = self.loss_fn[key](model_output[key], indices_tuple=hard_triplets[key])
            losses.append(loss[key] * self.w[i])

            stats[key] = {
                "loss": loss[key].item(),
                "avg_embedding_norm": self.loss_fn[key].distance.final_avg_query_norm,
                "num_triplets": len(hard_triplets[key][0]),
            }
            if isinstance(self.reducer_fn, AvgNonZeroReducer):
                stats[key]["num_non_zero_triplets"] = float(self.loss_fn[key].reducer.num_past_filter)
                try:
                    stats[key]["non_zero_rate"] = (
                        stats[key]["num_non_zero_triplets"] / stats[key]["num_triplets"]
                    )
                except ZeroDivisionError:
                    print("WARNING: encoutered batch with 'num_triplets' == 0.")
                    stats[key]["non_zero_rate"] = 1.0

            # TODO: calculate max loss triplets somehow?
        total_loss = torch.stack(losses).sum()
        stats["total_loss"] = total_loss.item()
        return total_loss, stats, miner_stats


class MultimodalTripletMarginLossDist(nn.Module):
    """Triplet margin loss implementation for multimodal models.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    valid_modalities = ("image", "cloud", "fusion")

    def __init__(
        self,
        margin: float,
        distance: Union[LpDistance, CosineSimilarity],
        miner: MinersHolder,
        reducer: Union[AvgNonZeroReducer, MeanReducer, SumReducer],
        swap: bool = False,
        modalities: Union[
            Literal["image", "cloud", "fusion"], Sequence[Literal["image", "cloud", "fusion"]]
        ] = ("image",),
        weights: Union[float, Sequence[float]] = 1.0,
        distributed: bool = False
    ) -> None:
        """Triplet margin loss implementation for multimodal models.

        Args:
            margin (float): Margin hyperparameter.
            distance (Union[LpDistance, CosineSimilarity]): Distance function for loss calculation.
            miner (HardTripletMiner): Miner function to form triplets.
            reducer (Union[AvgNonZeroReducer, MeanReducer, SumReducer]): Reducer function. Will be used
                to reduce loss values to scalar.
            swap (bool): Use the positive-negative distance instead of anchor-negative distance,
                if it violates the margin more. Defaults to False.
            modalities (Union[Literal["image", "cloud", "fusion"],
                Sequence[Literal["image", "cloud", "fusion"]]]): For which modalities loss will be calculated.
                Defaults to ("image",).
            weights (Union[float, Sequence[float]]): Weight for each given modality. Defaults to 1.0.

        Raises:
            ValueError: If given incorrect modalities.
            ValueError: If number of weights not equal to number of modalities.
            ValueError: If incorrect distance function given.
            ValueError: If incorrect miner function given.
            ValueError: If incorrect reducer function given.
        """
        super().__init__()

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

        self.miner_fn = {}
        if isinstance(miner, MinersHolder):
            self.miner_fn = miner.miners
        else:
            raise ValueError(f"Incorrect miner_fn = {miner}")

        if isinstance(reducer, (AvgNonZeroReducer, MeanReducer, SumReducer)):
            self.reducer_fn = reducer
        else:
            raise ValueError(f"Incorrect reducer_fn = {reducer}")

        self.loss_fn = {}
        for key in self.modalities:
            self.loss_fn[key] = TripletMarginLoss(
                margin=margin,
                swap=swap,
                distance=self.distance_fn,
                reducer=self.reducer_fn,
                collect_stats=True,
            )
        
        if distributed:
            for key in self.modalities:
                self.loss_fn[key] = pml_dist.DistributedMinerWrapper(self.loss_fn[key])
                self.miner_fn[key] = pml_dist.DistributedMinerWrapper(self.miner_fn[key])


    def forward(  # noqa: D102
        self,
        model_output: Dict[str, Tensor],
        positives_mask: Tensor,
        negatives_mask: Tensor,
    ) -> Tuple[Tensor, Dict, Dict]:
        loss: Dict[str, Tensor] = {}
        losses: List[Tensor] = []
        stats: Dict[str, Any] = {}

        #? hard_triplets, miner_stats = self.miner_fn(model_output, positives_mask, negatives_mask)

        hard_triplets = {}
        miner_stats = {}
        for key, values in model_output.items():
            if key in self.modalities and values is not None:
                assert values.dim() == 2
                d_embeddings = values.detach()
                with torch.no_grad():
                    fake_embeddings = torch.cat([d_embeddings, positives_mask.to(d_embeddings.device), negatives_mask.to(d_embeddings.device)], dim=1)
                    fake_labels = torch.zeros(d_embeddings.shape[0])
                    hard_triplets[key]= self.miner_fn[key](fake_embeddings, fake_labels, fake_embeddings, fake_labels)
                    miner_stats[key] = self.miner_fn[key].stats



        for i, key in enumerate(self.modalities):
            if key not in model_output.keys():
                raise KeyError(f"No key {key} in model_output.keys() = {model_output.keys()}")
            loss[key] = self.loss_fn[key](model_output[key], indices_tuple=hard_triplets[key])
            losses.append(loss[key] * self.w[i])

            stats[key] = {
                "loss": loss[key].item(),
                "avg_embedding_norm": self.loss_fn[key].distance.final_avg_query_norm,
                "num_triplets": len(hard_triplets[key][0]),
            }
            if isinstance(self.reducer_fn, AvgNonZeroReducer):
                stats[key]["num_non_zero_triplets"] = float(self.loss_fn[key].reducer.num_past_filter)
                try:
                    stats[key]["non_zero_rate"] = (
                        stats[key]["num_non_zero_triplets"] / stats[key]["num_triplets"]
                    )
                except ZeroDivisionError:
                    print("WARNING: encoutered batch with 'num_triplets' == 0.")
                    stats[key]["non_zero_rate"] = 1.0

            # TODO: calculate max loss triplets somehow?
        total_loss = torch.stack(losses).sum()
        stats["total_loss"] = total_loss.item()
        return total_loss, stats, miner_stats