import string
import random
import numpy as np
import torch
from torch import nn

from dataclasses import dataclass

import wandb
from utils import (
    embeddings_dictionary,
    index_dictionary,
    Checkpoint,
    word_vectors_most_common,
    config_wandb,
)
from datasets.manual_embedding import AverageEmbedder, WeightedAverageEmbedder
from datasets.mlp_dataset import EmbeddedDataset
from datasets.lstm_dataset import IndicesDataset
from datasets.pos import pos_all_tags
from models import MlpClassifier, LstmClassifier
from trainer import fit


@dataclass
class Args:
    # wandb
    save_wandb = False

    # general parameters
    num_epochs = 15
    batch_size = 64
    lr = 0.0001
    weight_decay = 0.0001
    vocab_threshold = 0

    # dataset parameters
    remove_stopwords = True
    remove_digits = True
    remove_target_word = False
    target_window = None

    # model parameters
    model_type = "MLP"
    use_pos = False

    # MLP Parameters
    if model_type == "MLP":
        mlp_n_features = 300
        mlp_num_layers = 2
        mlp_n_hidden = 512
        mlp_dropout = 0.3

    # LSTM Parameters
    if model_type == "LSTM":
        sentence_embedding_size = 300
        sentence_n_hidden = 512
        sentence_num_layers = 2
        sentence_bidirectional = True
        sentence_dropout = 0.3

    # POS parameters
    if use_pos:
        pos_embedding_size = 300
        pos_vocab_size = len(pos_all_tags)
        if model_type == "LSTM":
            pos_n_hidden = 256
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

    args = Args()

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

    # remove words with low frequency (if `args.vocab_threshold` != 0)
    word_vectors = word_vectors_most_common(
        train_path, word_vectors, args.vocab_threshold
    )

    # create dictionary from word to index and respective list of embedding tensors
    word_index, vectors_store = index_dictionary(word_vectors)

    # create random string of 20 characters to mark the target word
    # so that we don't lose it during the preprocessing steps in the dataset creation
    marker = "".join(random.choices(string.ascii_lowercase, k=20))

    # create train and validation datasets
    train_dataset = IndicesDataset(
        dataset_path=train_path,
        word_index=word_index,
        marker=marker,
        args=args,
    )
    val_dataset = IndicesDataset(
        dataset_path=dev_path,
        word_index=word_index,
        marker=marker,
        args=args,
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # select device ('gpu' if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate loss
    criterion = nn.BCELoss()

    # select either MLP or LSTM as model
    if args.model_type == "MLP":
        model = MlpClassifier(vectors_store, args).to(device)
    elif args.model_type == "LSTM":
        model = LstmClassifier(vectors_store, args).to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = None  # torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

    # to save/load checkpoints during training
    checkpoint = None  # Checkpoint(path="checkpoints/rnn")

    # save current training on wandb
    if args.save_wandb:
        config_wandb(args, model)

    # start training
    losses, accuracies = fit(
        epochs=args.num_epochs,
        device=device,
        save_wandb=args.save_wandb,
        model=model,
        criterion=criterion,
        train_dl=train_loader,
        valid_dl=val_loader,
        opt=optimizer,
        scheduler=scheduler,
        checkpoint=checkpoint,
        verbose=2,
    )
