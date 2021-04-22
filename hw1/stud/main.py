import string
import random
import numpy as np
import torch
from torch import nn

from dataclasses import dataclass

from utils import (
    embeddings_dictionary,
    index_dictionary,
    Checkpoint,
    word_vectors_most_common,
)
from datasets.manual_embedding import AverageEmbedder, WeightedAverageEmbedder
from datasets.mlp_dataset import EmbeddedDataset
from datasets.lstm_dataset import IndicesDataset
from datasets.pos import pos_all_tags
from models import MlpClassifier, LstmClassifier
from trainer import fit

# constants
MODEL_TYPE = "LSTM"

# LSTM args
@dataclass
class LstmArgs:
    sentence_embedding_size = 300
    sentence_n_hidden = 512
    sentence_num_layers = 2
    sentence_bidirectional = True
    sentence_dropout = 0.3

    use_pos = True
    pos_embedding_size = 300
    pos_vocab_size = len(pos_all_tags)
    pos_n_hidden = 512
    pos_num_layers = 2
    pos_bidirectional = True
    pos_dropout = 0.3


if __name__ == "__main__":
    # seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset paths
    train_path = "../../data/train.jsonl"
    dev_path = "../../data/dev.jsonl"

    # pretrained embedding path
    embedding_path = "../../embeddings/glove.6B.300d.txt"

    # create vocabulary with pretrained embeddings
    print("Reading pretrained embeddings file...")
    word_vectors = embeddings_dictionary(
        embedding_path=embedding_path, skip_first=False
    )

    word_vectors = word_vectors_most_common(train_path, word_vectors, 1)

    # create dictionary from word to index and respective list of embedding tensors
    word_index, vectors_store = index_dictionary(word_vectors)

    # create random string of 20 characters to mark target word
    # so that we don't lose it during the preprocessing steps in the dataset creation
    marker = "".join(random.choices(string.ascii_lowercase, k=20))

    # create train and validation datasets
    train_dataset = IndicesDataset(
        dataset_path=train_path,
        word_index=word_index,
        marker=marker,
        neigh_width=None,
    )
    val_dataset = IndicesDataset(
        dataset_path=dev_path,
        word_index=word_index,
        marker=marker,
        neigh_width=None,
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # select device ('gpu' if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate loss
    criterion = nn.BCELoss()

    # select either MLP or LSTM as model
    if MODEL_TYPE == "MLP":
        model = MlpClassifier(
            n_features=300,
            vectors_store=vectors_store,
            num_layers=4,
            hidden_dim=1024,
            activation=nn.functional.relu,
        ).to(device)
    elif MODEL_TYPE == "LSTM":
        lstm_args = LstmArgs()
        model = LstmClassifier(vectors_store, lstm_args).to(device)

    # instantiate loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # to save/load checkpoints during training
    # checkpoint = Checkpoint(path="checkpoints/rnn")

    epochs = 50

    # start training
    losses, accuracies = fit(
        epochs,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        checkpoint=None,
        device=device,
    )
