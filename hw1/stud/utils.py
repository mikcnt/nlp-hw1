from typing import Dict
import pickle
import os
from tqdm import tqdm

import torch
from torch import nn

# Saving data with pickle
def save_pickle(data: Dict, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

# To create embedding dictionary from `txt` file (e.g., GloVe)
def create_embedding_dictionary(path: str) -> Dict[str, torch.Tensor]:
    word_vectors = dict()
    with open(path) as f:
        for i, line in tqdm(enumerate(f)):

            word, *vector = line.strip().split(' ')
            vector = torch.tensor([float(c) for c in vector])

            word_vectors[word] = vector
    return word_vectors

# To save / load checkpoints
class Checkpoint:
    def __init__(self, path, resume=False):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.resume = resume

    def load(self, model: nn.Module, optimizer, id_path=""):
        if (not self.resume) and id_path == "":
            raise RuntimeError()
        if self.resume:
            id_path = sorted(os.listdir(self.path))[-1]
        self.checkpoint = torch.load(
            os.path.join(self.path, id_path), map_location=lambda storage, loc: storage
        )
        if self.checkpoint == None:
            raise RuntimeError("Checkpoint empty.")
        epoch = self.checkpoint["epoch"]
        model.load_state_dict(self.checkpoint["model_state_dict"])
        optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        losses = self.checkpoint["losses"]
        accuracies = self.checkpoint["accuracies"]
        return (model, optimizer, epoch, losses, accuracies)

    def save(self, model, optimizer, epoch, losses, accuracies):
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "losses": losses,
            "accuracies": accuracies
        }
        checkpoint_name = "{}.pth".format(str(epoch).zfill(3))
        complete_path = os.path.join(self.path, checkpoint_name)
        torch.save(model_checkpoint, complete_path)
        return

    def load_just_model(self, model, id_path=""):
        if self.resume:
            id_path = sorted(os.listdir(self.path))[-1]
        self.checkpoint = torch.load(
            os.path.join(self.path, id_path), map_location=lambda storage, loc: storage
        )
        if self.checkpoint == None:
            raise RuntimeError("Checkpoint empty.")
        model.load_state_dict(self.checkpoint["model_state_dict"])
        return model