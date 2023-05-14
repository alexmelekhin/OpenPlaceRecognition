"""Training script."""
from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from opr.datasets.dataloader_factory import make_dataloaders
from opr.testing import test
from opr.training import epoch_loop
from opr.utils import flatten_dict, set_seed


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Summary of training script.

    Args:
        cfg (DictConfig): config to train with
    """
    if not cfg.general.debug and not cfg.wandb.disabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            settings=wandb.Settings(start_method="thread"),
            config=config_dict,
        )
        wandb.save(f"configs/{wandb.run.name}.yaml")
        run_name = wandb.run.name
    else:
        run_name = "debug"

    checkpoints_dir = (
        Path(cfg.general.checkpoints_dir) / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{run_name}"
    )
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True)

    set_seed(seed=cfg.general.seed, make_deterministic=False)  # we cannot use determenistic operators here :(
    print(f"=> Seed: {cfg.general.seed}")

    print("=> Instantiating model...")
    model = instantiate(cfg.model)

    print("=> Instantiating loss...")
    loss_fn = instantiate(cfg.loss)

    print("=> Making dataloaders...")
    dataloaders = make_dataloaders(
        dataset_cfg=cfg.dataset.dataset,
        batch_sampler_cfg=cfg.dataset.sampler,
        num_workers=cfg.dataset.num_workers,
    )

    print("=> Instantiating optimizer...")
    params_list = []
    # modalities = list(set([m.split("_")[0] for m in cfg.general.modalities]))
    modalities = list(cfg.general.modalities)
    
    for modality in modalities:
        params_list.append(
            {
                "params": getattr(model, f"{modality}_module").parameters(),
                "lr": cfg.optimizer.learning_rates[f"{modality}_lr"],
            }
        )
    optimizer = instantiate(cfg.optimizer.fn, params=params_list)
    print("Instantiating scheduler...")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    model = model.to(cfg.general.device)

    # ==========> TRAIN LOOP:

    best_recall_at_1 = 0.0

    for epoch in range(cfg.general.epochs):
        print(f"\n\n=====> Epoch {epoch+1}:")
        # TODO: resolve mypy typing here
        train_batch_size = dataloaders["train"].batch_sampler.batch_size  # type: ignore
        val_batch_size = dataloaders["val"].batch_sampler.batch_size  # type: ignore

        print("\n=> Training:\n")

        train_stats, train_rate_non_zero = epoch_loop(
            dataloader=dataloaders["train"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            phase="train",
            device=cfg.general.device,
        )

        print(f"\ntrain_rate_non_zero = {train_rate_non_zero}")

        batch_expansion_th = cfg.general.batch_expansion_th
        if batch_expansion_th is not None:
            if batch_expansion_th == 1.0:
                print("Batch expansion rate is set to every epoch. Increasing batch size.")
                # TODO: resolve mypy typing here
                dataloaders["train"].batch_sampler.expand_batch()  # type: ignore
            elif train_rate_non_zero is None:
                print(
                    "\nWARNING: 'batch_expansion_th' was set, but 'train_rate_non_zero' is None. ",
                    "The batch size was not expanded.",
                )
            elif train_rate_non_zero < batch_expansion_th:
                print(
                    "Average non-zero triplet ratio is less than threshold: ",
                    f"{train_rate_non_zero} < {batch_expansion_th}",
                )
                # TODO: resolve mypy typing here
                dataloaders["train"].batch_sampler.expand_batch()  # type: ignore

        print("\n=> Validating:\n")

        val_stats, val_rate_non_zero = epoch_loop(
            dataloader=dataloaders["val"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            phase="val",
            device=cfg.general.device,
        )

        print(f"\nval_rate_non_zero = {val_rate_non_zero}")

        print("\n=> Testing:\n")

        recall_at_n, recall_at_one_percent, mean_top1_distance = test(
            model=model,
            descriptor_key=cfg.general.test_modality,
            dataloader=dataloaders["test"],
            device=cfg.general.device,
        )

        stats_dict = {}
        stats_dict["test"] = {
            "mean_top1_distance": mean_top1_distance,
            "recall_at_1%": recall_at_one_percent,
            "recall_at_1": recall_at_n[0],
            "recall_at_3": recall_at_n[2],
            "recall_at_5": recall_at_n[4],
            "recall_at_10": recall_at_n[9],
        }
        stats_dict["train"] = train_stats
        stats_dict["train"]["batch_size"] = train_batch_size
        stats_dict["val"] = val_stats
        stats_dict["val"]["batch_size"] = val_batch_size

        # wandb.log({'Test recall_at_1': recall_at_n[0],
        #            'Test Mean Recall@1%': recall_at_one_percent,
        #            'Test Mean top-1 distance': mean_top1_distance,
        #            'batch_size': train_stats['batch_size'],
        #            'train_total_loss': train_stats['total_loss'],
        #            'val_total_loss': val_stats['total_loss']})

        # saving checkpoints
        checkpoint_dict = {
            "epoch": epoch + 1,
            "config": cfg,
            "stats_dict": stats_dict,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint_dict, checkpoints_dir / f"epoch_{epoch+1}.pth")
        # wandb logging
        if not cfg.general.debug and not cfg.wandb.disabled:
            wandb.log(flatten_dict(stats_dict))
            wandb.save(str((checkpoints_dir / f"epoch_{epoch+1}.pth").relative_to(".")))
        if recall_at_n[0] > best_recall_at_1:
            print("Recall@1 improved!")
            torch.save(checkpoint_dict, checkpoints_dir / "best.pth")
            best_recall_at_1 = recall_at_n[0]
            if not cfg.general.debug and not cfg.wandb.disabled:
                wandb.save(str((checkpoints_dir / "best.pth").relative_to(".")))


if __name__ == "__main__":
    train()
