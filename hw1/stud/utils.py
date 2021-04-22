import pickle
import os
from collections import defaultdict, Counter
import jsonlines
import torch
from torch import nn
from tqdm import tqdm

import wandb

from typing import Dict, Tuple, List

from datasets.data_processing import preprocess

# save/load pickle
def save_pickle(data: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# save/load models checkpoints
class Checkpoint:
    def __init__(self, path: str, resume=False):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.resume = resume

    def load(self, model, optimizer, id_path=""):
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
            "accuracies": accuracies,
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


# load pretrained embedding dictionaries
def embeddings_dictionary(
    embedding_path: str, skip_first=False
) -> Dict[str, torch.Tensor]:
    word_vectors = dict()
    num_lines = sum(1 for line in open(embedding_path, "r"))
    with open(embedding_path) as f:
        for i, line in tqdm(enumerate(f), total=num_lines):

            if i == 0 and skip_first:
                continue
            # for tests
            # if i == 10000:
            #     break
            word, *vector = line.strip().split(" ")
            vector = torch.tensor([float(c) for c in vector])

            word_vectors[word] = vector
    return word_vectors


def index_dictionary(
    word_vectors: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, int], List[torch.Tensor]]:
    word_index = dict()
    vectors_store = []

    # pad token, index = 0
    vectors_store.append(torch.rand(300))

    # unk token, index = 1
    vectors_store.append(torch.rand(300))

    # save index for each word
    for word, vector in word_vectors.items():
        # skip unk token if present
        if word == "<unk>":
            continue
        word_index[word] = len(vectors_store)
        vectors_store.append(vector)

    word_index = defaultdict(
        lambda: 1, word_index
    )  # default dict returns 1 (unk token) when unknown word
    vectors_store = torch.stack(vectors_store)
    return word_index, vectors_store


def word_vectors_most_common(dataset_path, word_vectors, threshold=1):
    if threshold == 0:
        return word_vectors
    vocabulary_count = Counter()
    with jsonlines.open(dataset_path, "r") as f:
        for line in f.iter():
            s1 = line["sentence1"]
            s2 = line["sentence2"]

            # preprocessing
            s1 = preprocess(s1)
            s2 = preprocess(s2)

            t1 = s1.split()
            t2 = s2.split()

            vocabulary_count.update(t1 + t2)

    return {
        word: vector
        for word, vector in word_vectors.items()
        if vocabulary_count[word] >= threshold
    }


def config_wandb(args, model: nn.Module) -> None:
    """Save on wandb current training settings."""
    # initialize wandb remote repo
    wandb.init(project="nlp-hw1")

    # wandb config hyperparameters
    config = wandb.config

    # dataset parameters
    config.remove_stopwords = args.remove_stopwords
    config.remove_digits = args.remove_digits
    config.target_window = args.target_window
    
    # general parameters
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.model_type = args.model_type
    config.vocab_threshold = args.vocab_threshold

    # mlp parameters
    if args.model_type == 'MLP':
        config.mlp_n_features = args.mlp_n_features
        config.mlp_num_layers = args.mlp_num_layers
        config.mlp_n_hidden = args.mlp_n_hidden
    
    # lstm parameters
    if args.model_type == 'LSTM':
        config.sentence_embedding_size = args.sentence_embedding_size
        config.sentence_n_hidden = args.sentence_n_hidden
        config.sentence_num_layers = args.sentence_num_layers
        config.sentence_bidirectional = args.sentence_bidirectional
        config.sentence_dropout = args.sentence_dropout
        config.use_pos = args.use_pos
        config.pos_embedding_size = args.pos_embedding_size
        config.pos_vocab_size = args.pos_vocab_size
        config.pos_n_hidden = args.pos_n_hidden
        config.pos_num_layers = args.pos_num_layers
        config.pos_bidirectional = args.pos_bidirectional
        config.pos_dropout = args.pos_dropout

    # parameter for wandb update
    config.log_interval = 1

    # save model parameters
    wandb.watch(model, log="all")
    return
