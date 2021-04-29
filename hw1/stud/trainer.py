from typing import Any, Dict, List, Optional
import copy

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from tqdm import tqdm

import wandb

def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> List[torch.Tensor]:
    """Move all elements of batch (i.e., `sentence1`, `pos1` etc.) to device (i.e. gpu if available)."""
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
    save_checkpoint: bool = False,
    verbose: int = 2,
    evaluate: bool = False,
) -> None:
    """Trainer function."""
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
        if save_checkpoint:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": losses,
                    "accuracies": accuracies
                },
                f"checkpoints/{str(epoch).zfill(3)}.pth",
            )

        if verbose > 1:
            print(
                f"Epoch {epoch} \t T. Loss = {loss_train:.4f}, V. Loss = {loss_val:.4f}, T. Accuracy {acc_train:.3f}, V. Accuracy {acc_val:.3f}."
            )

        # update best_model if it is better than previous ones
        if evaluate:
            if epoch == 1 or acc_val > max(accuracies["val"][:-1]):
                best_model = copy.deepcopy(model)

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

    # if evaluate, save all useful plots
    if evaluate:
        if verbose > 0:
            print("Starting evaluation.")

        # get predictions
        best_model = best_model.to("cpu")
        best_model.eval()
        predictions = torch.cat(
            [best_model(batch).round() for batch in valid_dl], dim=0
        )
        predictions = predictions.reshape(-1).detach().numpy()

        # get ground truths
        ground_truths = torch.cat([batch["label"] for batch in valid_dl], dim=0)
        ground_truths = ground_truths.reshape(-1).detach().numpy()

        # confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        # normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # save confusion matrix plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        ConfusionMatrixDisplay(cm, display_labels=["False", "True"]).plot(
            cmap="Blues", ax=ax
        )
        plt.savefig("outputs/confusion_matrix.png")

        # save losses and accuracies plots
        train_loss = losses["train"]
        val_loss = losses["val"]
        train_acc = accuracies["train"]
        val_acc = accuracies["val"]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.plot(train_loss, label="Train loss")
        ax.plot(val_loss, label="Val loss")
        ax.legend()
        plt.savefig("outputs/losses.png")

        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.plot(train_acc, label="Train acc")
        ax.plot(val_acc, label="Val acc")
        ax.legend()
        plt.savefig("outputs/accuracies.png")

    return losses, accuracies
