import string
import random
import numpy as np
import torch
from torch import nn

from utils import embeddings_dictionary, index_dictionary, Checkpoint
from datasets.manual_embedding import AverageEmbedder, WeightedAverageEmbedder
from datasets.mlp_dataset import EmbeddedDataset
from datasets.lstm_dataset import IndicesDataset
from models import MLP, LSTMClassifier
from trainer import fit

# constants
MODEL_TYPE = "MLP"

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
    word_vectors["<unk>"] = torch.rand(300)

    # create dictionary from word to index and respective list of embedding tensors
    word_index, vectors_store = index_dictionary(word_vectors)

    # create random string of 20 characters to mark target word
    # so that we don't lose it during the preprocessing steps in the dataset creation
    marker = "".join(random.choices(string.ascii_lowercase, k=20))

    # select dataset according to model selection
    if MODEL_TYPE == "MLP":
        embedder = WeightedAverageEmbedder(word_vectors, 1, 0)
        train_dataset = EmbeddedDataset(
            dataset_path=train_path, marker=marker, embedder=embedder, neigh_width=None
        )
        val_dataset = EmbeddedDataset(
            dataset_path=dev_path, marker=marker, embedder=embedder, neigh_width=None
        )
    elif MODEL_TYPE == "LSTM":
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
        model = MLP(
            n_features=300,
            num_layers=3,
            hidden_dim=1024,
            activation=nn.functional.relu,
        ).to(device)
    elif MODEL_TYPE == "LSTM":
        model = LSTMClassifier(
            vectors_store=vectors_store,
            n_hidden=1024,
            num_layers=2,
            bidirectional=True,
            lstm_dropout=0.3,
        ).to(device)

    # instantiate loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # to save/load checkpoints during training
    # checkpoint = Checkpoint(path="checkpoints/rnn")

    epochs = 20

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