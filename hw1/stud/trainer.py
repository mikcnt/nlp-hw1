from typing import Any, Dict, List, Optional

import torch
from torch import nn
from tqdm import tqdm

import wandb
from stud.utils import Checkpoint


def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> List[torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def fit(
    epochs: int,
    device: str,
    save_wandb: bool,
    model: nn.Module,
    criterion: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    scheduler: Any = None,
    checkpoint: Optional[Checkpoint] = None,
    verbose: int = 2,
) -> None:

    # keep track of losses and accuracies
    losses = {"train": [], "val": []}
    accuracies = {"train": [], "val": []}

    if verbose > 0:
        print("Starting training.")

    # training loop
    for epoch in range(1, epochs + 1):
        # instantiate loss and accuracy each epoch
        losses_train = 0
        train_total_instances, train_correct_instances = 0, 0

        model.train()
        train_iterator = (
            tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} (TRAIN)", leave=False)
            if verbose > 1
            else train_dl
        )
        for batch in train_iterator:
            # send batch to device
            batch = batch_to_device(batch, device)

            pred = model(batch)

            # compute loss and backprop
            loss_train = criterion(pred, batch["label"])
            loss_train.backward()
            opt.step()
            opt.zero_grad()
            losses_train += loss_train.item()

            pred = torch.round(pred)
            # number of predictions
            train_total_instances += pred.shape[0]
            # number of correct predictions
            train_correct_instances += (batch["label"] == pred).int().sum().item()

        model.eval()
        with torch.no_grad():
            losses_val = 0
            val_total_instances, val_correct_instances = 0, 0

            valid_iterator = (
                tqdm(valid_dl, desc=f"Epoch {epoch}/{epochs} (VALID)", leave=False)
                if verbose > 1
                else valid_dl
            )
            for batch in valid_iterator:
                # send batch to device
                batch = batch_to_device(batch, device)

                # compute predictions
                pred_val = model(batch)

                # compute loss (validation step => no backprop)
                loss_val = criterion(pred_val, batch["label"])
                losses_val += loss_val.item()

                pred_val = torch.round(pred_val)
                # number of predictions
                val_total_instances += pred_val.shape[0]
                # number of correct predictions
                val_correct_instances += (batch["label"] == pred_val).int().sum().item()

        if scheduler is not None:
            scheduler.step()

        # compute accuracy (train + val)
        loss_train = losses_train / len(train_dl)
        loss_val = losses_val / len(valid_dl)

        acc_train = train_correct_instances / train_total_instances
        acc_val = val_correct_instances / val_total_instances

        # log losses and accuracies
        losses["train"].append(loss_train)
        losses["val"].append(loss_val)

        accuracies["train"].append(acc_train)
        accuracies["val"].append(acc_val)

        # store checkpoint
        if checkpoint:
            checkpoint.save(model, opt, epoch, losses, accuracies)

        if verbose > 1:
            print(
                f"Epoch {epoch} \t T. Loss = {loss_train:.4f}, V. Loss = {loss_val:.4f}, T. Accuracy {acc_train:.3f}, V. Accuracy {acc_val:.3f}."
            )

        if save_wandb:
            # save losses on wandb
            wandb.log(
                {
                    "Train loss": loss_train,
                    "Train Accuracy": acc_train,
                    "Val loss": loss_val,
                    "Val Accuracy": acc_val,
                }
            )

    if verbose > 0:
        print("Training finished.")
    return losses, accuracies
