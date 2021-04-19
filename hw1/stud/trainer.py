from typing import List, Optional
import torch
from torch import nn
from tqdm import tqdm

from utils import Checkpoint

def batch_to_device(batch: List[torch.Tensor], device: str) -> List[torch.Tensor]:
    return [x.to(device) for x in batch]

def fit(
    epochs: int,
    model: nn.Module,
    criterion: nn.Module,
    opt: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    checkpoint: Optional[Checkpoint] = None,
    verbose: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
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
        d_train, n_train = 0, 0

        model.train()
        train_iterator = tqdm(
            train_dl, desc=f"Epoch {epoch}/{epochs} (TRAIN)", leave=False
        ) if verbose > 1 else train_dl
        for batch in train_iterator:
            # send batch to device
            batch = batch_to_device(batch, device)
            batch_x = batch[:-1]
            batch_y = batch[-1]

            pred = model(*batch_x)

            # compute loss and backprop
            loss_train = criterion(pred, batch_y)
            loss_train.backward()
            opt.step()
            opt.zero_grad()
            losses_train += loss_train.item()

            pred = torch.round(pred)
            # number of predictions
            d_train += pred.shape[0]
            # number of correct predictions
            n_train += (batch_y == pred).int().sum().item()

        model.eval()
        with torch.no_grad():
            losses_val = 0
            d_val, n_val = 0, 0

            valid_iterator = tqdm(
                valid_dl, desc=f"Epoch {epoch}/{epochs} (VALID)", leave=False
            ) if verbose > 1 else valid_dl
            for batch in valid_iterator:
                # send batch to device
                batch = batch_to_device(batch, device)
                batch_x = batch[:-1]
                batch_y = batch[-1]

                # compute predictions
                pred_val = model(*batch_x)

                # compute loss (validation step => no backprop)
                loss_val = criterion(pred_val, batch_y)
                losses_val += loss_val.item()

                pred_val = torch.round(pred_val)
                # number of predictions
                d_val += pred_val.shape[0]
                # number of correct predictions
                n_val += (batch_y == pred_val).int().sum().item()

        # compute accuracy (train + val)
        loss_train = losses_train / d_train
        loss_val = losses_val / d_val

        acc_train = n_train / d_train
        acc_val = n_val / d_val

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
    
    if verbose > 0:
        print("Training finished.")
    return losses, accuracies