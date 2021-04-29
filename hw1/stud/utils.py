import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import jsonlines
import torch
from torch import nn
from tqdm import tqdm

import wandb
from stud.datasets.data_processing import preprocess


# save/load pickle
def save_pickle(data: dict, path: str) -> None:
    """Save object as pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> dict:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def embeddings_dictionary(
    embedding_path: str,
    skip_first=False,
    testing_mode=False,
) -> Dict[str, torch.Tensor]:
    """Load pretrained embedding dictionaries."""
    word_vectors = dict()
    num_lines = sum(1 for line in open(embedding_path, "r"))
    with open(embedding_path) as f:
        for i, line in tqdm(enumerate(f), total=num_lines):
            # if pretrained embedding has 1 header line
            if i == 0 and skip_first:
                continue
            # break immediately if in testing mode to save time
            # all tokens will therefore be considered as <unk>
            if testing_mode and i == 10000:
                break
            word, *vector = line.strip().split(" ")
            vector = torch.tensor([float(c) for c in vector])

            word_vectors[word] = vector
    return word_vectors


def index_dictionary(
    word_vectors: Dict[str, torch.Tensor], embedding_size=300
) -> Tuple[Dict[str, int], torch.Tensor]:
    """Create word2index dictionary and list of vectors containing respective embedding."""
    word_index = dict()
    # pad token -> index = 0; unk token -> index = 1
    vectors_store = [torch.rand(embedding_size), torch.rand(embedding_size)]

    # save index for each word
    for word, vector in word_vectors.items():
        word_index[word] = len(vectors_store)
        vectors_store.append(vector)

    # default dict returns 1 (unk token) when unknown word is encountered
    word_index = defaultdict(lambda: 1, word_index)
    vectors_store = torch.stack(vectors_store)
    return word_index, vectors_store


def word_vectors_most_common(
    dataset_path: str, word_vectors: Dict[str, torch.Tensor], threshold: int = 1
) -> Dict[str, torch.Tensor]:
    """Modify word vectors vocabulary according to a threshold.
    If `threshold == 0`, all words are kept (even the ones not contained in the
    dataset itself!). Otherwise, just the words with frequency > threshold are kept."""
    if threshold == 0:
        return word_vectors
    # compute frequency of words in the dataset
    vocabulary_count = Counter()
    with jsonlines.open(dataset_path, "r") as f:
        # notice that all sentences in the dataset go through the same process
        # that we do in the dataset creation. This way, we don't risk having words
        # in the vocabulary that don't actually appear in the dataset
        for line in f.iter():
            s1 = line["sentence1"]
            s2 = line["sentence2"]

            # preprocessing
            s1 = preprocess(s1)
            s2 = preprocess(s2)

            t1 = s1.split()
            t2 = s2.split()

            vocabulary_count.update(t1 + t2)

    # return words with selected frequency
    return {
        word: vector
        for word, vector in word_vectors.items()
        if vocabulary_count[word] >= threshold
    }


def text_length(sentences: torch.Tensor) -> torch.Tensor:
    """Compute number of non padded indices in tensor."""
    # search first zero
    lengths = (sentences == 0).int().argmax(axis=1)
    # length 0 only if sentence has max length
    # => replace 0 with max length
    lengths[lengths == 0] = sentences.shape[-1]
    return lengths


# Weight and biases logs
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
    config.remove_target_word = args.remove_target_word

    # general parameters
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.model_type = args.model_type
    config.vocab_threshold = args.vocab_threshold

    # pos parameters
    config.use_pos = args.use_pos
    if args.use_pos:
        config.pos_embedding_size = args.pos_embedding_size
        config.pos_vocab_size = args.pos_vocab_size
        if args.model_type == "LSTM":
            config.pos_n_hidden = args.pos_n_hidden
            config.pos_num_layers = args.pos_num_layers
            config.pos_bidirectional = args.pos_bidirectional
            config.pos_dropout = args.pos_dropout

    # mlp parameters
    if args.model_type == "MLP":
        config.mlp_n_features = args.mlp_n_features
        config.mlp_num_layers = args.mlp_num_layers
        config.mlp_n_hidden = args.mlp_n_hidden
        config.mlp_dropout = args.mlp_dropout

    # lstm parameters
    if args.model_type == "LSTM":
        config.sentence_embedding_size = args.sentence_embedding_size
        config.sentence_dropout = args.sentence_dropout
        config.sentence_n_hidden = args.sentence_n_hidden
        config.sentence_num_layers = args.sentence_num_layers
        config.sentence_bidirectional = args.sentence_bidirectional
        config.sentence_dropout = args.sentence_dropout

    if args.model_type == "BILINEAR":
        config.bi_n_features = args.bi_n_features
        config.bi_n_hidden = args.bi_n_hidden
        config.bi_n_dropout = args.bi_dropout

    # parameter for wandb update
    config.log_interval = 1

    # save model parameters
    wandb.watch(model, log="all")
    return
