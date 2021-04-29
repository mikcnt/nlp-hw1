import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stud.datasets.dataset import WiCDataset, read_data
from stud.datasets.pos import pos_all_tags

from stud.models.mlp import MlpClassifier
from stud.models.lstm import LstmClassifier
from stud.models.bilinear import BilinearClassifier
from stud.models.lstm_bilinear import LstmBilinearClassifier

from stud.trainer import fit
from stud.utils import (
    config_wandb,
    embeddings_dictionary,
    index_dictionary,
    word_vectors_most_common,
    save_pickle,
)

# tweak the parameters of the class `Args` to test different
# architectures and parameters
@dataclass
class Args:
    # if in evaluation mode, save losses plots, accuracies plots, confusion matrix
    evaluate = False
    # wandb
    save_wandb = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # general parameters
    num_epochs = 15
    batch_size = 64
    lr = 0.0001
    weight_decay = 0.0001
    vocab_threshold = 0
    sentence_embedding_size = 300

    # dataset parameters
    save_labels = True
    remove_stopwords = True
    remove_digits = True
    remove_target_word = False
    target_window = None

    # model parameters
    model_type = "BILINEARLSTM"
    use_pretrained_embeddings = True
    use_pos = False

    # MLP Parameters
    if model_type == "MLP":
        mlp_num_layers = 0
        mlp_n_hidden = 512
        mlp_dropout = 0.3

    # LSTM Parameters
    if model_type == "LSTM":
        linear_dropout = 0.3
        sentence_n_hidden = 512
        sentence_num_layers = 2
        sentence_bidirectional = True
        sentence_dropout = 0.3

    # LSTM with bilinear Parameters
    if model_type == "BILINEARLSTM":
        linear_dropout = 0.3
        sentence_n_hidden = 300
        sentence_num_layers = 2
        sentence_bidirectional = True
        sentence_dropout = 0.3

    if model_type == "BILINEAR":
        bi_n_features = 300
        bi_n_hidden = 400
        bi_dropout = 0.3

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
    embedding_path = "embeddings/glove.6B.300d.txt"

    # create vocabulary with pretrained embeddings
    print("Reading pretrained embeddings file...")
    word_vectors = embeddings_dictionary(
        embedding_path=embedding_path, skip_first=False, testing_mode=False,
    )

    # remove words with low frequency (if `args.vocab_threshold` != 0)
    word_vectors = word_vectors_most_common(
        train_path, word_vectors, args.vocab_threshold
    )

    # create dictionary from word to index and respective list of embedding tensors
    word_index, vectors_store = index_dictionary(word_vectors)

    # create train and validation datasets
    train_data = read_data(train_path)
    val_data = read_data(dev_path)

    train_dataset = WiCDataset(
        data=train_data,
        word_index=word_index,
        args=args,
    )
    val_dataset = WiCDataset(
        data=val_data,
        word_index=word_index,
        args=args,
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # instantiate loss
    criterion = nn.BCELoss()

    # select model type
    # mean aggregation model with concatenation
    if args.model_type == "MLP":
        model = MlpClassifier(vectors_store, args).to(args.device)
    # sequence encoding with concatenation
    elif args.model_type == "LSTM":
        model = LstmClassifier(vectors_store, args).to(args.device)
    # mean aggregation model with bilinear layer
    elif args.model_type == "BILINEAR":
        model = BilinearClassifier(vectors_store, args).to(args.device)
    # LSTM with bilinear layer
    elif args.model_type == "BILINEARLSTM":
        model = LstmBilinearClassifier(vectors_store, args).to(args.device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # instantiate scheduler (if None, no scheduler is used during training)
    scheduler = None  # torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
    # to save/load checkpoints during training (if None, no checkpoints are saved)
    save_checkpoint = True

    # save current training on wandb
    if args.save_wandb:
        config_wandb(args, model)

    # start training
    losses, accuracies = fit(
        epochs=args.num_epochs,
        device=args.device,
        save_wandb=args.save_wandb,
        model=model,
        criterion=criterion,
        train_dl=train_loader,
        valid_dl=val_loader,
        opt=optimizer,
        scheduler=scheduler,
        save_checkpoint=save_checkpoint,
        verbose=2,
        evaluate=args.evaluate,
    )